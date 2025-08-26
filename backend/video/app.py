import sys
import cv2
import threading
import logging
import time
import os
import csv
import numpy as np
from collections import deque
from datetime import datetime
from flask import Flask, Response
from flask_socketio import SocketIO
from flask_cors import CORS
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import socket

# --- Import your project modules ---
from video.video_pose import VideoPose
from audio.audio_processor import main as run_audio_processor

# --- Email Configuration ---
SENDER_EMAIL = "youvegotspam98@gmail.com"
SENDER_PASSWORD = "gxtt qqva lglj eoqp"  # Use an App Password
RECIPIENT_EMAIL = "amansharmagamer8@gmail.com"

# --- Global variables ---
output_frame = None
lock = threading.Lock()
current_alert_status = {
    "video": {"label": "Normal", "probability": 0.0},
    "audio": {"label": "Normal", "probability": 0.0},
    "fused_alert": "NORMAL"
}

# --- Flask App and SocketIO Setup ---
app = Flask(__name__)
CORS(app) 
socketio = SocketIO(app, cors_allowed_origins="*")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Video Processing & Recording Configuration ---
POSE_BUFFER_SIZE = 15 
JERK_THRESHOLD = 0.3
Y_COLLAPSE_THRESHOLD = 0.15
ALERT_CONFIRMATION_TIME = 2.0
COOLDOWN_TIME = 5.0
RECORDING_DURATION_SEC = 10
OUTPUT_DIR = "recordings"
CSV_LOG_FILE = os.path.join(OUTPUT_DIR, "alert_log.csv")

# --- Email Sending Function ---
def get_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def send_alert_email(timestamp, alert_details, video_path, screenshot_path):
    logging.info("Attempting to send alert email...")
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECIPIENT_EMAIL
        msg['Subject'] = f"HIGH ALERT Detected by Trinetra Kavach at {timestamp}"

        body = f"""
        <html><body>
            <h2>Trinetra Kavach - High Priority Alert</h2>
            <p>A high-priority distress event was detected by the surveillance system.</p>
            <h3>Event Details:</h3>
            <ul>
                <li><b>Time:</b> {timestamp}</li>
                <li><b>System IP Address:</b> {get_ip_address()}</li>
                <li><b>Video Trigger:</b> {alert_details['video']['label']}</li>
                <li><b>Audio Trigger:</b> {alert_details['audio']['label']}</li>
            </ul>
            <p>A video recording and a screenshot of the event are attached for your review.</p>
        </body></html>
        """
        msg.attach(MIMEText(body, 'html'))

        # Attach screenshot and video
        for path in [screenshot_path, video_path]:
            with open(path, "rb") as attachment:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename= {os.path.basename(path)}")
            msg.attach(part)

        logging.info("Connecting to SMTP server...")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        # --- NEW: Enable detailed debug output for SMTP ---
        server.set_debuglevel(1) 
        server.starttls()
        logging.info("Logging into email server...")
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        text = msg.as_string()
        logging.info(f"Sending email to {RECIPIENT_EMAIL}...")
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, text)
        server.quit()
        logging.info(f"Successfully sent alert email.")

    except Exception as e:
        logging.error(f"Failed to send email: {e}")

class StateManager:
    def __init__(self):
        self.state = "NORMAL"
        self.alert_start_time = 0
        self.cooldown_end_time = 0
        self.alert_level = 0.0

    def update(self, is_distress_signal):
        current_time = time.time()
        if self.state == "COOLDOWN" and current_time > self.cooldown_end_time:
            self.state = "NORMAL"
        
        if self.state == "NORMAL":
            if is_distress_signal:
                self.state = "ALERT_TRIGGERED"
                self.alert_start_time = current_time
                self.alert_level = 0.1
            else:
                self.alert_level = 0.0
        elif self.state == "ALERT_TRIGGERED":
            if is_distress_signal:
                time_in_alert = current_time - self.alert_start_time
                self.alert_level = min(time_in_alert / ALERT_CONFIRMATION_TIME, 1.0)
                if time_in_alert >= ALERT_CONFIRMATION_TIME:
                    self.state = "COOLDOWN"
                    self.cooldown_end_time = current_time + COOLDOWN_TIME
                    self.alert_level = 1.0
            else:
                self.state = "NORMAL"
                self.alert_level = 0.0
        return self.state, self.alert_level

def run_video_thread():
    global output_frame, current_alert_status, lock
    
    logging.info("Initializing video processor...")
    pose_detector = VideoPose()
    state_manager = StateManager()
    pose_buffer = deque(maxlen=POSE_BUFFER_SIZE)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.critical("ERROR: Could not open webcam.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30 

    is_recording = False
    recording_end_time = 0
    video_writer = None
    email_data_to_send = None

    logging.info("Starting live video stream processing.")
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        annotated_frame, landmarks = pose_detector.detect_pose(frame)
        is_potential_distress = False

        if landmarks:
            pose_buffer.append(landmarks.landmark)
            if len(pose_buffer) == POSE_BUFFER_SIZE:
                hip_y = [(p[23].y + p[24].y) / 2 for p in pose_buffer]
                if (hip_y[-1] - hip_y[0]) > Y_COLLAPSE_THRESHOLD: is_potential_distress = True

                start_pose, mid_pose, end_pose = pose_buffer[0], pose_buffer[POSE_BUFFER_SIZE // 2], pose_buffer[-1]
                keypoints_indices = [15, 16, 0]
                max_jerk = 0
                for i in keypoints_indices:
                    v1_x, v1_y = mid_pose[i].x - start_pose[i].x, mid_pose[i].y - start_pose[i].y
                    v2_x, v2_y = end_pose[i].x - mid_pose[i].x, end_pose[i].y - mid_pose[i].y
                    jerk_magnitude = np.sqrt((v2_x - v1_x)**2 + (v2_y - v1_y)**2)
                    if jerk_magnitude > max_jerk: max_jerk = jerk_magnitude
                if max_jerk > JERK_THRESHOLD: is_potential_distress = True
        
        video_state, alert_level = state_manager.update(is_potential_distress)
        
        with lock:
            current_alert_status["video"] = {"label": video_state, "probability": float(alert_level)}
            fuse_alerts()
            
            alert_type = current_alert_status["fused_alert"]
            if (alert_type == "HIGH_ALERT" or alert_type == "MEDIUM_ALERT") and not is_recording:
                is_recording = True
                recording_end_time = time.time() + RECORDING_DURATION_SEC
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                
                screenshot_path = os.path.join(OUTPUT_DIR, f"alert_{timestamp}.jpg")
                cv2.imwrite(screenshot_path, frame)
                
                video_path = os.path.join(OUTPUT_DIR, f"alert_{timestamp}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
                
                # --- NEW: Add IP address to CSV log ---
                with open(CSV_LOG_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, get_ip_address(), alert_type, current_alert_status['video']['label'], current_alert_status['audio']['label'], video_path, screenshot_path])
                
                logging.warning(f"{alert_type}! Saving evidence to {OUTPUT_DIR}")

                if alert_type == "HIGH_ALERT":
                    email_data_to_send = {"timestamp": timestamp, "details": current_alert_status.copy(), "video_path": video_path, "screenshot_path": screenshot_path}

            if is_recording:
                video_writer.write(frame)
                if time.time() >= recording_end_time:
                    is_recording = False
                    video_writer.release()
                    video_writer = None
                    logging.info("Recording finished.")

                    if email_data_to_send:
                        logging.info("High alert confirmed. Preparing to send email...")
                        email_thread = threading.Thread(target=send_alert_email, args=(email_data_to_send["timestamp"], email_data_to_send["details"], email_data_to_send["video_path"], email_data_to_send["screenshot_path"]), daemon=True)
                        email_thread.start()
                        email_data_to_send = None

            status_color = (0, 255, 0)
            if current_alert_status["fused_alert"] == "LOW_ALERT": status_color = (0, 165, 255)
            elif current_alert_status["fused_alert"] == "MEDIUM_ALERT": status_color = (0, 255, 255)
            elif current_alert_status["fused_alert"] == "HIGH_ALERT": status_color = (0, 0, 255)
            
            cv2.putText(annotated_frame, f"STATUS: {current_alert_status['fused_alert']}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3, cv2.LINE_AA)
            output_frame = annotated_frame.copy()
            
    cap.release()
    pose_detector.close()

def on_audio_data_received(label: str, probability: float):
    global current_alert_status
    with lock:
        current_alert_status["audio"] = {"label": label, "probability": float(probability)}

def fuse_alerts():
    global current_alert_status
    
    VIDEO_WEIGHT = 0.6
    AUDIO_WEIGHT = 0.4
    HIGH_ALERT_THRESHOLD = 0.75
    MEDIUM_ALERT_THRESHOLD = 0.55
    LOW_ALERT_THRESHOLD = 0.40
    
    video_prob = current_alert_status["video"]["probability"]
    audio_prob = 0.0
    
    if current_alert_status["audio"]["label"].lower() not in ["normal", "silence", "noise", "background noise", "speech"]:
        audio_prob = current_alert_status["audio"]["probability"]

    fused_score = (video_prob * VIDEO_WEIGHT) + (audio_prob * AUDIO_WEIGHT)

    if fused_score >= HIGH_ALERT_THRESHOLD:
        current_alert_status["fused_alert"] = "HIGH_ALERT"
    elif fused_score >= MEDIUM_ALERT_THRESHOLD:
        current_alert_status["fused_alert"] = "MEDIUM_ALERT"
    elif fused_score >= LOW_ALERT_THRESHOLD:
        current_alert_status["fused_alert"] = "LOW_ALERT"
    else:
        current_alert_status["fused_alert"] = "NORMAL"
        
    socketio.emit('update_status', current_alert_status)

def generate_video_frames():
    global output_frame, lock
    while True:
        time.sleep(0.03)
        with lock:
            if output_frame is None: continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag: continue
        
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate_video_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/test_email")
def test_email():
    """A simple route to manually test the email functionality."""
    logging.info("Manual email test triggered.")
    try:
        # Create dummy files for the test
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        dummy_screenshot = os.path.join(OUTPUT_DIR, "test_screenshot.jpg")
        dummy_video = os.path.join(OUTPUT_DIR, "test_video.mp4")
        cv2.imwrite(dummy_screenshot, np.zeros((100, 100, 3), dtype=np.uint8))
        with open(dummy_video, 'w') as f: f.write("dummy video")

        test_details = {"video": {"label": "MANUAL_TEST"}, "audio": {"label": "MANUAL_TEST"}}
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        send_alert_email(timestamp, test_details, dummy_video, dummy_screenshot)
        return "Test email sent! Please check your inbox and terminal for debug info.", 200
    except Exception as e:
        logging.error(f"Manual email test failed: {e}")
        return f"Failed to send test email. Check server logs for details. Error: {e}", 500

@socketio.on('connect')
def handle_connect(auth=None):
    logging.info('Client connected to WebSocket.')
    with lock:
        socketio.emit('update_status', current_alert_status)

def main():
    # --- NEW: Add IPAddress to CSV header ---
    if not os.path.exists(CSV_LOG_FILE):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(CSV_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "IPAddress", "AlertType", "VideoTrigger", "AudioTrigger", "VideoPath", "ScreenshotPath"])

    video_thread = threading.Thread(target=run_video_thread, daemon=True)
    video_thread.start()
    
    audio_thread = threading.Thread(target=run_audio_processor, args=(on_audio_data_received,), daemon=True)
    audio_thread.start()

    logging.info("Starting Flask server... Your React app can now connect.")
    socketio.run(app, host="0.0.0.0", port=5000)

if __name__ == "__main__":
    main()
