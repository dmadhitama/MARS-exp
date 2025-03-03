"""
Script to extract frames from video files.
This script processes video files from multiple subjects, extracts all frames,
and optionally resizes them to specified dimensions.
"""

# Step 1: Import required libraries
import os
import cv2
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Step 2: Define frame extraction function
def extract_frames(video_path, output_folder, target_size):
    """
    Extract all frames from a video file and save them as JPEG images.
    
    Args:
        video_path: Path to the video file
        output_folder: Folder where extracted frames will be saved
        target_size: Optional tuple (width, height) for resizing frames
    """
    # Step 2.1: Get video name and create output folder
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_folder, exist_ok=True)
    
    # Step 2.2: Open video file and get frame count
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Step 2.3: Process each frame in the video
    with tqdm(total=total_frames, desc=f"Extracting {video_name}", unit="frame") as pbar:
        frame_count = 0
        while True:
            # Step 2.3.1: Read the next frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Step 2.3.2: Resize frame if target size is specified
            if target_size:
                frame = cv2.resize(frame, target_size)
            
            # Step 2.3.3: Save frame as JPEG image
            output_path = os.path.join(output_folder, f"{video_name}_frame_{frame_count:06d}.jpg")
            cv2.imwrite(output_path, frame)
            
            # Step 2.3.4: Update counter and progress bar
            frame_count += 1
            pbar.update(1)
    
    # Step 2.4: Release video capture object
    cap.release()

# Step 3: Define function to process all videos for a subject
def process_subject(subject_dir, target_size):
    """
    Process all video files for a single subject.
    
    Args:
        subject_dir: Directory containing the subject's video files
        target_size: Optional tuple (width, height) for resizing frames
    """
    # Step 3.1: Iterate through all files in the subject directory
    for file in os.listdir(subject_dir):
        # Step 3.2: Process only MP4 files
        if file.endswith('.mp4'):
            video_path = os.path.join(subject_dir, file)
            # Step 3.3: Create output folder with same name as video file (without extension)
            output_folder = os.path.join(subject_dir, os.path.splitext(file)[0])
            # Step 3.4: Extract frames from the video
            extract_frames(video_path, output_folder, target_size)

# Step 4: Main function to coordinate processing
def main(args):
    """
    Main function to coordinate the processing of all subjects.
    
    Args:
        args: Command-line arguments
    """
    # Step 4.1: Define base directory containing all subject folders
    base_dir = "/home/ubuntu/gdrive/workspace/blurred_videos/"
    
    # Step 4.2: Set target size for frame resizing if specified
    target_size = (args.width, args.height) if args.width and args.height else None
    
    # Step 4.3: Get list of all subject directories
    subject_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # Step 4.4: Process all subjects in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        # Step 4.4.1: Submit processing tasks for all subjects
        futures = [executor.submit(process_subject, subject_dir, target_size) for subject_dir in subject_dirs]
        
        # Step 4.4.2: Monitor task completion with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing subjects"):
            future.result()

# Step 5: Parse command-line arguments and run main function
if __name__ == "__main__":
    # Step 5.1: Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Extract frames from videos and optionally resize them.")
    parser.add_argument("--width", type=int, help="Target width for resized frames")
    parser.add_argument("--height", type=int, help="Target height for resized frames")
    args = parser.parse_args()
    
    # Step 5.2: Validate arguments
    if (args.width and not args.height) or (args.height and not args.width):
        parser.error("Both --width and --height must be provided if resizing is desired.")
    
    # Step 5.3: Run the main function
    main(args)