# video/video_pose.py

# This is our Pose Detective. 🕵️‍♀️
# Its special skill is finding the human skeleton (pose landmarks) in any
# video frame we give it, using Google's MediaPipe technology.

import cv2
import mediapipe as mp

class VideoPose:
    """
    A class that wraps up all the complex work of MediaPipe's pose detection
    into one easy-to-use tool.
    """
    def __init__(self):
        """Initializes the Pose Detective and gets its tools ready."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # We'll use bright colors to draw the skeleton so it's easy to see.
        self.landmark_style = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=2  # Green dots for joints
        )
        self.connection_style = self.mp_drawing.DrawingSpec(
            color=(0, 0, 255), thickness=2, circle_radius=2  # Red lines for bones
        )

    def detect_pose(self, frame):
        """
        Analyzes a single frame to find a person's pose.

        Args:
            frame: An image from OpenCV.

        Returns:
            A tuple with two things:
            - annotated_frame: A copy of the frame with the skeleton drawn on top.
            - pose_landmarks: The raw data of where all the joints are.
        """
        # MediaPipe needs the image in RGB format, but OpenCV gives it in BGR.
        # It's like flipping a negative to make a photo.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # The detective does its work and finds the pose.
        results = self.pose.process(rgb_frame)

        # We make a copy to draw on, so we don't mess up the original image.
        annotated_frame = frame.copy()
        
        # If a pose was found, we draw the skeleton on our copy.
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=results.pose_landmarks,
                connections=self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.landmark_style,
                connection_drawing_spec=self.connection_style,
            )

        return annotated_frame, results.pose_landmarks

    def close(self):
        """Lets the detective pack up its tools when we're done."""
        self.pose.close()

if __name__ == '__main__':
    # A quick test to see our Pose Detective in action with a webcam.
    print("--- Testing the Pose Detective ---")
    print("Look at your webcam and press 'q' in the window to quit.")
    
    cap = cv2.VideoCapture(0)
    pose_detector = VideoPose()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Hmm, can't get a frame from the webcam.")
            continue

        annotated_frame, landmarks = pose_detector.detect_pose(frame)
        
        cv2.imshow('Pose Detective Test', annotated_frame)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    pose_detector.close()