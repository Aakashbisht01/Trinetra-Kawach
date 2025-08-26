# video/video_processor.py
# This processor handles the live video stream for real-time anomaly detection.

import os
import time
from collections import deque
import threading

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from .video_utils import VideoUtils
from .video_pose import VideoPose
from .model_loader import VideoModelLoader

class VideoProcessor:
    """
    Multi-stage video pipeline for women's safety:
      1. YOLO to detect people (Who & Where).
      2. X3D to classify actions (What are they doing?).
      3. Live display + 'q' to quit.
    """

    PREFERRED_T = 16
    MIN_KERNEL_T = 16

    def __init__(
        self,
        history_length: int = 16,
        buffer_duration_sec: int = 15,
        fps: int = 30,
        source: int = 0,
        model_name: str = "facebookresearch/pytorchvideo:main:x3d_m",
        output_dir: str = "logs/video_clips",
    ):
        # IO / Pose
        self.video_utils = VideoUtils(source=source)
        self.video_pose = VideoPose()
        self.fps = fps
        
        # YOLO Model (Stage 1)
        self.yolo_model = YOLO("yolov8n.pt")
        
        # X3D Model (Stage 2)
        self.model_loader = VideoModelLoader(model_name=model_name)
        self.model = self.model_loader.get_model()
        self.device = self.model_loader.get_device()

        # The X3D fine-tuned classes for your project
        self.class_labels = ["Normal_activity", "Fighting", "Harassment", "Fall", "Run-away"]
        
        # Buffers
        self.history_length = self.PREFERRED_T
        self.model_buffer = deque(maxlen=self.history_length)
        self.frame_buffer = deque(maxlen=int(buffer_duration_sec * self.fps))

        # Recording
        self.recording = False
        self.video_writer = None
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Normalization (ImageNet-style)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1, 1)
        self.target_h = 224
        self.target_w = 224

        # Threading
        self.anomaly_probability = 0.0
        self.anomaly_class_name = ""
        self.buffer_lock = threading.Lock()
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)

    def _inference_loop(self):
        while True:
            if len(self.model_buffer) == self.history_length:
                with self.buffer_lock:
                    frames_to_process = list(self.model_buffer)
                
                probability, class_name = self._predict_action(frames_to_process)
                self.anomaly_probability = probability
                self.anomaly_class_name = class_name
            
            time.sleep(0.01)

    def _predict_action(self, frames):
        if len(frames) == 0:
            return 0.0, "None"
        
        # Step 1: Detect people with YOLO (only on the most recent frame for speed)
        yolo_results = self.yolo_model(frames[-1], verbose=False)
        person_detections = [
            result for result in yolo_results[0].boxes if self.yolo_model.names[int(result.cls)] == "person"
        ]
        
        if not person_detections:
            return 0.0, "No Person"
        
        # Step 2: Classify action with X3D using the full clip
        clip_tensor = self._preprocess_clip(frames, self.PREFERRED_T)
        
        try:
            with torch.no_grad():
                logits = self.model(clip_tensor)
            probs = torch.softmax(logits, dim=-1)[0]
            
            max_prob, predicted_idx = torch.max(probs, dim=-1)
            predicted_class = self.class_labels[predicted_idx.item()]
            
            return max_prob.item(), predicted_class
        except Exception as e:
            print(f"X3D inference failed: {e}")
            return 0.0, "Model Error"
            
    def _temporal_sample(self, frames_list, T):
        n = len(frames_list)
        if n == T:
            return np.stack(frames_list, axis=0)
        
        indices = np.linspace(0, n - 1, T).round().astype(np.int32)
        return np.stack([frames_list[i] for i in indices], axis=0)
        
    def _preprocess_clip(self, frames_list, desired_T: int):
        clip_np = self._temporal_sample(frames_list, desired_T)
        clip_np_rgb = np.ascontiguousarray(clip_np[..., ::-1])
        clip_tensor = torch.from_numpy(clip_np_rgb).to(self.device, dtype=torch.float32)
        clip_tensor = clip_tensor.permute(3, 0, 1, 2).unsqueeze(0)
        clip_tensor = clip_tensor / 255.0
        
        clip_tensor = F.interpolate(
            clip_tensor,
            size=(desired_T, self.target_h, self.target_w),
            mode="trilinear",
            align_corners=False,
        )
        clip_tensor = (clip_tensor - self.mean) / self.std
        return clip_tensor

    def start_recording(self, frame):
        if self.recording:
            return
        print("Starting video clip recording...")
        self.recording = True
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.output_dir, f"anomaly_{timestamp}.mp4")
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(filename, fourcc, self.fps, (width, height))
        for buffered_frame in self.frame_buffer:
            if isinstance(buffered_frame, np.ndarray):
                self.video_writer.write(buffered_frame)

    def stop_recording(self):
        if self.recording and self.video_writer:
            print("Stopping video clip recording...")
            self.recording = False
            self.video_writer.release()
            self.video_writer = None

    def process_frame(self, frame):
        with self.buffer_lock:
            self.model_buffer.append(frame)
            self.frame_buffer.append(frame)
        return frame

    def run_live_stream(self, callback_function, high_anomaly_threshold=0.7):
        if not self.video_utils.start_capture():
            print("Error: Could not start video capture.")
            return
            
        is_recording_active = False
        frames_recorded = 0
        self.inference_thread.start()
        
        try:
            while True:
                frame = self.video_utils.get_frame()
                if frame is None:
                    break
                
                annotated_frame = self.process_frame(frame)
                current_prob = self.anomaly_probability
                current_class = self.anomaly_class_name
                
                callback_function(annotated_frame, current_prob, current_class)
                
                if current_prob >= high_anomaly_threshold and not is_recording_active:
                    print(f"High anomaly detected: {current_class}. Initiating recording.")
                    self.start_recording(annotated_frame)
                    is_recording_active = True
                    frames_recorded = 0
                
                if is_recording_active:
                    if isinstance(annotated_frame, np.ndarray) and self.video_writer:
                        self.video_writer.write(annotated_frame)
                    frames_recorded += 1
                    if frames_recorded >= len(self.frame_buffer):
                        self.stop_recording()
                        is_recording_active = False
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    if is_recording_active:
                        self.stop_recording()
                    break
        finally:
            self.video_utils.release()
            cv2.destroyAllWindows()

