import os
import cv2
import argparse
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def extract_frames(video_path, output_folder, target_size):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    with tqdm(total=total_frames, desc=f"Extracting {video_name}", unit="frame") as pbar:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if target_size:
                frame = cv2.resize(frame, target_size)
            
            output_path = os.path.join(output_folder, f"{video_name}_frame_{frame_count:06d}.jpg")
            cv2.imwrite(output_path, frame)
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()

def process_subject(subject_dir, target_size):
    for file in os.listdir(subject_dir):
        if file.endswith('.mp4'):
            video_path = os.path.join(subject_dir, file)
            output_folder = os.path.join(subject_dir, os.path.splitext(file)[0])
            extract_frames(video_path, output_folder, target_size)

def main(args):
    base_dir = "/home/ubuntu/gdrive/workspace/blurred_videos/"
    
    target_size = (args.width, args.height) if args.width and args.height else None
    
    subject_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_subject, subject_dir, target_size) for subject_dir in subject_dirs]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing subjects"):
            future.result()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos and optionally resize them.")
    parser.add_argument("--width", type=int, help="Target width for resized frames")
    parser.add_argument("--height", type=int, help="Target height for resized frames")
    args = parser.parse_args()
    
    if (args.width and not args.height) or (args.height and not args.width):
        parser.error("Both --width and --height must be provided if resizing is desired.")
    
    main(args)