"""
Script to prepare MRI image data for training and testing.
This script processes video frames and corresponding pose labels, resizes the frames,
and splits them into training and testing sets.
"""

# Step 1: Import required libraries
import numpy as np
import os
import pickle
import cv2
from tqdm import tqdm
from loguru import logger
import threading
from queue import Queue

# Step 2: Define helper functions
# Step 2.1: Function to verify matching filenames
def check_if_filename_same(video_folder, labels_filename):
    """
    Verify that video folder and label filenames correspond to the same sample.
    Compares the first part of the names (before the underscore).
    """
    assert video_folder.split("_")[0] == labels_filename.split("_")[0]
    logger.info(f"Filenames {video_folder} and {labels_filename} match.")

# Step 2.2: Function to resize video frames
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

# Step 2.3: Function to create or append data to a file
def create_or_append_to_file(file_path, data):
    """
    Create a new file or append data to an existing file.
    
    Args:
        file_path: Path to the file
        data: Data to save or append
    """
    if not os.path.exists(file_path):
        # Create new file if it doesn't exist
        np.save(file_path, data)
    else:
        # Append to existing file
        existing_data = np.load(file_path)
        updated_data = np.concatenate((existing_data, data), axis=-1)
        np.save(file_path, updated_data)

# Step 2.4: Function to print dataset information
def print_dataset_info(data_tr_file, labels_tr_file, data_tt_file, labels_tt_file):
    """
    Print information about the dataset (shapes of training and testing data/labels).
    
    Args:
        data_tr_file: Path to training data file
        labels_tr_file: Path to training labels file
        data_tt_file: Path to testing data file
        labels_tt_file: Path to testing labels file
    """
    logger.info(f"Training data shape: {np.load(data_tr_file, mmap_mode='r').shape}")
    logger.info(f"Training labels shape: {np.load(labels_tr_file, mmap_mode='r').shape}")
    logger.info(f"Testing data shape: {np.load(data_tt_file, mmap_mode='r').shape}")
    logger.info(f"Testing labels shape: {np.load(labels_tt_file, mmap_mode='r').shape}")

# Step 2.5: Function to read video frames in a separate thread
def video_reader(video_file, frame_queue, rgb_avail_frames, target_size):
    """
    Read video frames and put them in a queue.
    
    Args:
        video_file: Path to the video file
        frame_queue: Queue to store frames
        rgb_avail_frames: Range of frames to extract [start, end]
        target_size: Target dimensions for frame resizing
    """
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if rgb_avail_frames[0] <= frame_count <= rgb_avail_frames[1]:
            frame = resize_frame(frame, target_size)
            frame_queue.put((frame_count, frame))
        frame_count += 1
    cap.release()
    frame_queue.put(None)  # Signal end of video

# Step 3: Define main function for data preparation
def prepare_video_data_labels(video_dir, labels_dir, output_dir, test_size=0.2, target_size=(224, 224), chunk_size=1000):
    """
    Prepare video data and labels for training and testing.
    
    Args:
        video_dir: Directory containing video folders
        labels_dir: Directory containing label files (.cpl)
        output_dir: Directory where processed data will be saved
        test_size: Proportion of data to use for testing (default: 0.2 or 20%)
        target_size: Target dimensions for frame resizing (default: 224x224)
        chunk_size: Number of frames to process at once (default: 1000)
    """
    # Step 3.1: List and sort all video folders and label files
    video_folders = sorted([d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))])
    labels_list = sorted([d for d in os.listdir(labels_dir) if d.endswith(".cpl")])
    assert len(video_folders) == len(labels_list), "Number of video folders and labels is not the same."

    # Step 3.2: Define output file paths
    data_tr_file = os.path.join(output_dir, "data_tr.npy")
    labels_tr_file = os.path.join(output_dir, "labels_tr.npy")
    data_tt_file = os.path.join(output_dir, "data_tt.npy")
    labels_tt_file = os.path.join(output_dir, "labels_tt.npy")

    # Step 3.3: Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 3.4: Process each video folder and corresponding label file
    for i, (video_folder, label_file) in enumerate(zip(video_folders, labels_list)):
        logger.info(f"Processing video folder {video_folder}")
        
        # Step 3.4.1: Verify that video folder and label file match
        check_if_filename_same(video_folder, label_file)
        
        # Step 3.4.2: Load label file
        with open(os.path.join(labels_dir, label_file), 'rb') as f:
            cpl = pickle.load(f)
        
        # Step 3.4.3: Extract available frames range and reshape labels
        rgb_avail_frames = cpl['rgb_avail_frames']
        Y = cpl['refined_gt_kps'].reshape(-1, cpl['refined_gt_kps'].shape[1]*cpl['refined_gt_kps'].shape[2])

        # Step 3.4.4: Get list of video files in the folder
        video_folder_path = os.path.join(video_dir, video_folder)
        video_files = sorted([os.path.join(video_folder_path, f) for f in os.listdir(video_folder_path) if f.endswith(".mp4")])
        
        # Step 3.4.5: Create queue for frame processing
        frame_queue = Queue(maxsize=100)  # Adjust queue size as needed

        # Step 3.4.6: Initialize buffers for frames and labels
        frames_buffer = []
        labels_buffer = []

        # Step 3.4.7: Process each video file
        for video_file in video_files:
            # Step 3.4.7.1: Start thread to read video frames
            reader_thread = threading.Thread(target=video_reader, args=(video_file, frame_queue, rgb_avail_frames, target_size))
            reader_thread.start()

            # Step 3.4.7.2: Process frames from the queue
            pbar = tqdm(total=rgb_avail_frames[1] - rgb_avail_frames[0] + 1, desc=f"Processing {os.path.basename(video_file)}")
            while True:
                item = frame_queue.get()
                if item is None:
                    break
                frame_count, frame = item
                
                # Step 3.4.7.3: Add frame and corresponding label to buffers
                frames_buffer.append(frame)
                labels_buffer.append(Y[frame_count])
                
                # Step 3.4.7.4: Save buffered data when chunk size is reached
                if len(frames_buffer) >= chunk_size:
                    # Step 3.4.7.5: Split into training and testing sets
                    if i < (len(video_folders)*(1-test_size)):
                        # Add to training set
                        create_or_append_to_file(data_tr_file, np.array(frames_buffer))
                        create_or_append_to_file(labels_tr_file, np.array(labels_buffer))
                    else:
                        # Add to testing set
                        create_or_append_to_file(data_tt_file, np.array(frames_buffer))
                        create_or_append_to_file(labels_tt_file, np.array(labels_buffer))
                    frames_buffer = []
                    labels_buffer = []
                
                pbar.update(1)
            
            pbar.close()
            reader_thread.join()

        # Step 3.4.8: Save any remaining frames in the buffer
        if frames_buffer:
            if i < (len(video_folders)*(1-test_size)):
                create_or_append_to_file(data_tr_file, np.array(frames_buffer))
                create_or_append_to_file(labels_tr_file, np.array(labels_buffer))
            else:
                create_or_append_to_file(data_tt_file, np.array(frames_buffer))
                create_or_append_to_file(labels_tt_file, np.array(labels_buffer))

        # Step 3.4.9: Clean up memory
        del Y

        # Step 3.4.10: Print current dataset information
        print(f"Data size after video folder {video_folder} processed.")
        print_dataset_info(data_tr_file, labels_tr_file, data_tt_file, labels_tt_file)

# Step 4: Main execution block
if __name__ == "__main__":
    # Step 4.1: Define input and output directories
    video_dir = "/home/ubuntu/gdrive/workspace/blurred_videos"
    labels_dir = "/home/ubuntu/gdrive/workspace/dataset_release/aligned_data/pose_labels/"
    out_dir = "/home/ubuntu/MARS-exp/mri_rgb_rede/"
    target_size = (224, 224)
    chunk_size = 200

    # Step 4.2: Run the data preparation function with error handling
    try:
        prepare_video_data_labels(video_dir, labels_dir, out_dir, target_size=target_size, chunk_size=chunk_size)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")