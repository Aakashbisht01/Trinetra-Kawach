# video/video_utils.py

# This is our video toolbox. 🧰
# It contains useful helper functions for both loading your dataset
# and running the live webcam demo.

import os
import cv2
import numpy as np
from tqdm import tqdm

class VideoUtils:
    """
    A collection of handy tools for video-related tasks.
    """
    def __init__(self, source=0):
        """
        Initializes the toolbox. The 'source' is for the webcam.
        """
        self.source = source
        self.cap = None

    def load_image_sequence(self, folder_path):
        """
        Loads a folder of images in the correct order to create a video clip.
        
        Args:
            folder_path (str): The path to the folder with all the video frames.

        Returns:
            list: A list of frames, ready for analysis.
        """
        if not os.path.isdir(folder_path):
            print(f"Error: I can't find a folder at '{folder_path}'")
            return []

        # Find all the images and sort them by name to keep them in order.
        image_files = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])

        frames = []
        for image_file in image_files:
            frame = cv2.imread(image_file)
            if frame is not None:
                frames.append(frame)
        
        return frames

    def start_capture(self):
        """Initializes and turns on the webcam."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            print(f"Error: Could not turn on the webcam: {self.source}")
            return False
        return True

    def get_frame(self):
        """Takes a single snapshot from the webcam."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def release(self):
        """Turns off the webcam."""
        if self.cap:
            self.cap.release()

    @staticmethod
    def save_frame_as_image(frame, filename):
        """Saves a single frame as an image file."""
        cv2.imwrite(filename, frame)

if __name__ == '__main__':
    # A quick test to see if our toolbox can read your dataset.
    print("--- Testing the Video Toolbox ---")
    
    # IMPORTANT: Make sure this path points to one of your video frame folders.
    # For example: 'data/video_samples/distress/your_video_frame_folder'
    test_folder = 'data/video_samples/distress/Fighting001_x264' 
    
    utils = VideoUtils()
    frames = utils.load_image_sequence(test_folder)
    
    if frames:
        print(f"✅ Awesome! I loaded {len(frames)} frames from '{test_folder}'.")
        # Let's see the first frame to make sure it looks right.
        cv2.imshow("First Frame Test", frames[0])
        print("Displaying the first frame. Press any key in the window to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"❌ Hmm, I couldn't load any frames from '{test_folder}'.")
        print("   Double-check that the folder path is correct and has images inside.")
