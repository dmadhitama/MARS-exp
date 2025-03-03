"""
Script to prepare MRI image data for training and testing (Modified Version).
This script processes video frames from two camera angles and corresponding pose labels,
concatenates the videos, and splits the data into training and testing sets using sklearn.
"""

# Step 1: Import required libraries
import os  
import cv2  
import numpy as np  
import pickle  
from sklearn.model_selection import train_test_split  
from tqdm import tqdm
from loguru import logger

# Step 2: Define helper functions
# Step 2.1: Function to resize video frames
def resize_frame(frame, target_size=(224, 224)):
    """
    Resize a video frame to the target dimensions.
    
    Args:
        frame: Input video frame
        target_size: Target dimensions (width, height)
    
    Returns:
        Resized frame
    """
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
  
# Step 2.2: Function to load video frames
def load_video(video_path, target_size=(224, 224)):
    """
    Load all frames from a video file and resize them.
    
    Args:
        video_path: Path to the video file
        target_size: Target dimensions for frame resizing
    
    Returns:
        Numpy array of video frames
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    with tqdm(total=total_frames, desc=f"Loading {os.path.basename(video_path)}", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(resize_frame(frame, target_size))
            pbar.update(1)
    cap.release()
    
    if not frames:
        logger.warning(f"No frames were read from {video_path}")
    
    return np.array(frames)
  
# Step 2.3: Function to concatenate two videos
def concatenate_videos(video1_path, video2_path):  
    """
    Load two videos and concatenate them along the last axis.
    
    Args:
        video1_path: Path to the first video file
        video2_path: Path to the second video file
    
    Returns:
        Concatenated video frames
    """
    video1 = load_video(video1_path)  
    video2 = load_video(video2_path)  
    if video1.shape[0] != video2.shape[0]:  
        logger.warning(f"Warning: Video frame counts do not match for {video1_path} and {video2_path}")  
    return np.concatenate((video1, video2), axis=-1)
  
# Step 2.4: Function to load and process label files
def load_labels(label_path):  
    """
    Load and process label files.
    
    Args:
        label_path: Path to the label file (.cpl)
    
    Returns:
        Tuple of (processed labels, available frames range)
    """
    with open(label_path, "rb") as f:  
        cpl = pickle.load(f)  
    refined_gt_kps = cpl["refined_gt_kps"]  
    rgb_avail_frames = cpl["rgb_avail_frames"]  
    refined_gt_kps = refined_gt_kps.reshape(refined_gt_kps.shape[0], -1)  
    return refined_gt_kps, rgb_avail_frames  
  
# Step 2.5: Function to process a subject's data
def process_subject(subject_dir, label_path):
    """
    Process video and label data for a single subject.
    
    Args:
        subject_dir: Directory containing the subject's video files
        label_path: Path to the subject's label file
    
    Returns:
        Tuple of (concatenated video data, corresponding labels)
    """
    print(subject_dir, label_path)
    # Step 2.5.1: Extract filename and construct paths to video files
    filename = os.path.basename(subject_dir).split("_")[0]
    video1_path = os.path.join(subject_dir, f"{filename}_color0.mp4")
    video2_path = os.path.join(subject_dir, f"{filename}_color1.mp4")
    
    logger.debug(f"Processing videos: {video1_path} and {video2_path}")
    
    # Step 2.5.2: Concatenate videos from two camera angles
    try:
        concatenated_video = concatenate_videos(video1_path, video2_path)
        logger.debug(f"Concatenated video shape: {concatenated_video.shape}")
    except Exception as e:
        logger.error(f"Error concatenating videos for {subject_dir}: {str(e)}")
        return None, None

    # Step 2.5.3: Load and process label data
    try:
        labels, rgb_avail_frames = load_labels(label_path)
        logger.debug(f"Loaded labels shape: {labels.shape}, rgb_avail_frames: {rgb_avail_frames}")
    except Exception as e:
        logger.error(f"Error loading labels for {subject_dir}: {str(e)}")
        return None, None

    # Step 2.5.4: Trim videos and labels to available frames range
    start_idx, end_idx = rgb_avail_frames
    concatenated_video = concatenated_video[start_idx:end_idx]
    labels = labels[start_idx:end_idx]
    
    logger.debug(f"Processed {subject_dir}: video shape {concatenated_video.shape}, labels shape {labels.shape}")
    return concatenated_video, labels
  
# Step 3: Main function for data preparation
def main():
    """
    Main function to prepare the dataset.
    """
    logger.info("Starting data preparation process")
    
    # Step 3.1: Define input and output directories
    video_base_dir = "/home/ubuntu/gdrive/workspace/blurred_videos/"
    label_base_dir = "/home/ubuntu/gdrive/workspace/dataset_release/aligned_data/pose_labels"
    out_dir = "/home/ubuntu/MARS-exp/mri_rgb_rede/"

    test_size = 0.2
    
    # Step 3.2: Initialize lists to store all videos and labels
    all_videos = []
    all_labels = []
    
    # Step 3.3: Get list of all subject directories
    subjects = [d for d in os.listdir(video_base_dir) if os.path.isdir(os.path.join(video_base_dir, d))]
    logger.info(f"Found {len(subjects)} subjects to process")

    # Step 3.4: Process each subject
    for subject_dir in subjects:
        subject_path = os.path.join(video_base_dir, subject_dir)
        label_path = os.path.join(label_base_dir, f"{subject_dir.split('_')[0]}_all_labels.cpl")
        
        logger.debug(f"Processing subject: {subject_dir}")
        # Step 3.4.1: Process subject's video and label data
        video_data, label_data = process_subject(subject_path, label_path)
        # Step 3.4.2: Add to collection if processing was successful
        if video_data is not None and label_data is not None:
            all_videos.append(video_data)
            all_labels.append(label_data)
        else:
            logger.warning(f"Skipping subject {subject_dir} due to processing errors")

    # Step 3.5: Concatenate all processed data
    if all_videos:  
        all_videos = np.concatenate(all_videos, axis=0)  
    if all_labels:  
        all_labels = np.concatenate(all_labels, axis=0)  
      
    logger.info(f"Total frames in all videos: {all_videos.shape[0]}")  
    logger.info(f"Total frames in all labels: {all_labels.shape[0]}")  
      
    # Step 3.6: Verify data integrity
    if all_videos.shape[0] != all_labels.shape[0]:  
        raise ValueError("Mismatch between total frames in videos and labels")  
      
    # Step 3.7: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        all_videos, 
        all_labels, 
        test_size=test_size, 
        random_state=42
    ) 

    # Step 3.8: Save processed data
    logger.info("Saving processed data")
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)
    
    logger.success("Data preparation completed successfully")

# Step 4: Main execution block
if __name__ == "__main__":  
    # Step 4.1: Configure logging
    logger.add("data_preparation.log", rotation="10 MB")
    # Step 4.2: Run the main function
    main()
