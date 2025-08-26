# live_video_test.py
# This is the upgraded, more robust version for live testing.
# It uses a state machine and advanced heuristics to provide reliable,
# real-world distress detection and minimize false alarms.

import cv2
import numpy as np
import joblib
import json
import os
from collections import deque
import time

# We need to import the VideoPose class from your existing video folder.
from video.video_pose import VideoPose

# --- Configuration ---
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'video_action_detector.pkl')
LABELS_PATH = os.path.join(MODEL_DIR, 'video_action_labels.json')

# --- Advanced Heuristics & State Machine Configuration ---

# We'll analyze movement over a buffer of 15 frames (about half a second).
POSE_BUFFER_SIZE = 15 

# Heuristic thresholds for detecting signs of distress.
JERK_THRESHOLD = 0.3       # Measures erratic, violent movement.
Y_COLLAPSE_THRESHOLD = 0.15 # Measures a sudden drop in height (a fall).

# State Machine Configuration
ALERT_CONFIRMATION_TIME = 2.0  # Seconds of continuous alert signals to confirm distress.
COOLDOWN_TIME = 5.0            # Seconds to wait before detecting a new event.

class StateManager:
    """A simple state machine to manage the detection logic."""
    def __init__(self):
        self.state = "NORMAL"
        self.alert_start_time = 0
        self.cooldown_end_time = 0
        self.alert_level = 0.0 # A value from 0.0 to 1.0

    def update(self, is_distress_signal):
        """Updates the state based on the latest signal."""
        current_time = time.time()

        # Handle the COOLDOWN state
        if self.state == "COOLDOWN":
            if current_time > self.cooldown_end_time:
                print("Cooldown finished. Returning to NORMAL state.")
                self.state = "NORMAL"
            return

        # Handle the NORMAL state
        if self.state == "NORMAL":
            if is_distress_signal:
                # First sign of trouble, enter ALERT state.
                print("Potential distress detected. Entering ALERT state...")
                self.state = "ALERT_TRIGGERED"
                self.alert_start_time = current_time
                self.alert_level = 0.1
            else:
                self.alert_level = 0.0

        # Handle the ALERT_TRIGGERED state
        elif self.state == "ALERT_TRIGGERED":
            if is_distress_signal:
                # Continue seeing distress, increase alert level.
                time_in_alert = current_time - self.alert_start_time
                self.alert_level = min(time_in_alert / ALERT_CONFIRMATION_TIME, 1.0)
                
                if time_in_alert >= ALERT_CONFIRMATION_TIME:
                    # Distress confirmed!
                    print("DISTRESS CONFIRMED. Entering COOLDOWN.")
                    self.state = "COOLDOWN"
                    self.cooldown_end_time = current_time + COOLDOWN_TIME
                    self.alert_level = 1.0
            else:
                # Distress signal stopped, go back to normal.
                print("Alert cancelled. Returning to NORMAL state.")
                self.state = "NORMAL"
                self.alert_level = 0.0

def run_live_test():
    """
    Initializes and runs the main live prediction loop.
    """
    print("--- Live Video Test (Advanced Heuristics) ---")
    
    # --- Load Model and Initialize ---
    try:
        classifier = joblib.load(MODEL_PATH)
        with open(LABELS_PATH, 'r') as f:
            class_labels = json.load(f)
    except Exception as e:
        print(f"❌ Could not load model files. Error: {e}")
        return

    pose_detector = VideoPose()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not open webcam.")
        return
        
    pose_buffer = deque(maxlen=POSE_BUFFER_SIZE)
    state_manager = StateManager()

    print("\nStarting webcam feed. Press 'q' to quit.")

    # --- Main Loop ---
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        annotated_frame, landmarks = pose_detector.detect_pose(frame)
        
        is_potential_distress = False

        if landmarks:
            pose_buffer.append(landmarks.landmark)

            if len(pose_buffer) == POSE_BUFFER_SIZE:
                # --- Heuristic 1: Fall Detection ---
                hip_y = [(p[31].y + p[32].y) / 2 for p in pose_buffer]
                if (hip_y[-1] - hip_y[0]) > Y_COLLAPSE_THRESHOLD:
                    is_potential_distress = True

                # --- Heuristic 2: Violent Motion Detection ---
                start_pose, mid_pose, end_pose = pose_buffer[0], pose_buffer[POSE_BUFFER_SIZE // 2], pose_buffer[-1]
                keypoints_indices = [15, 16, 0] # Wrists and Nose

                max_jerk = 0
                for i in keypoints_indices:
                    v1_x, v1_y = mid_pose[i].x - start_pose[i].x, mid_pose[i].y - start_pose[i].y
                    v2_x, v2_y = end_pose[i].x - mid_pose[i].x, end_pose[i].y - mid_pose[i].y
                    jerk_magnitude = np.sqrt((v2_x - v1_x)**2 + (v2_y - v1_y)**2)
                    if jerk_magnitude > max_jerk:
                        max_jerk = jerk_magnitude
                
                if max_jerk > JERK_THRESHOLD:
                    is_potential_distress = True

        # Update our state machine with the latest signal
        state_manager.update(is_potential_distress)

        # --- Display Logic ---
        display_status = "NORMAL"
        status_color = (0, 255, 0) # Green

        if state_manager.state == "ALERT_TRIGGERED":
            display_status = "ALERT"
            status_color = (0, 165, 255) # Orange
        elif state_manager.state == "COOLDOWN":
            display_status = "DISTRESS CONFIRMED"
            status_color = (0, 0, 255) # Red

        # Draw the main status text
        cv2.putText(annotated_frame, f"STATUS: {display_status}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3, cv2.LINE_AA)

        # Draw the alert level bar
        bar_width = int(state_manager.alert_level * (frame.shape[1] - 20))
        cv2.rectangle(annotated_frame, (10, frame.shape[0] - 30), (frame.shape[1] - 10, frame.shape[0] - 10), (100, 100, 100), -1)
        cv2.rectangle(annotated_frame, (10, frame.shape[0] - 30), (10 + bar_width, frame.shape[0] - 10), status_color, -1)
        cv2.putText(annotated_frame, "Alert Level", (15, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Live Action Recognition', annotated_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
    pose_detector.close()
    print("--- Test finished. ---")

if __name__ == '__main__':
    run_live_test()
