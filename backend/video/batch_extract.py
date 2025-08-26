import os
import glob
import cv2   # ✅ FIX: you forgot this import


# --- CONFIGURATION ---
processing_jobs = [
    {
        "input_video": "data/raw_videos/normal",   # folder with normal videos
        "output_folder": "data/video_samples/normal"
    },
    {
        "input_video": "data/raw_videos/fighting", # folder with fighting videos
        "output_folder": "data/video_samples/distress"
    }
]
# --- END OF CONFIGURATION ---


def extract_frames(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    print(f"➡ Trying to open: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"❌ Error: Could not open video file at {video_path}")
        return

    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    frame_count = 0
    print(f"--- Starting frame extraction for: {video_basename} ---")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_filename = os.path.join(output_folder, f"{video_basename}_frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"✅ Done. Extracted {frame_count} frames from '{video_basename}' to '{output_folder}'.\n")


def process_folder(input_folder, output_folder):
    # Define allowed video extensions
    video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.AVI", "*.MOV", "*.MKV")
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_folder, ext)))

    if not video_files:
        print(f"⚠ No video files found in '{input_folder}'")
        return

    for video_file in video_files:
        abs_path = os.path.abspath(video_file)  # ✅ ensure absolute path
        extract_frames(abs_path, output_folder)


if __name__ == "__main__":
    print("🚀 Starting batch video processing...\n")
    for job in processing_jobs:
        input_path = job.get("input_video")
        output_dir = job.get("output_folder")

        if os.path.isdir(input_path):
            process_folder(input_path, output_dir)
        elif os.path.isfile(input_path):
            abs_path = os.path.abspath(input_path)  # ✅ ensure absolute path
            extract_frames(abs_path, output_dir)
        else:
            print(f"❌ Error: Path not found '{input_path}'")

    print("\n✅ All processing jobs complete.")
