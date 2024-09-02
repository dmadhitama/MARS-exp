import os  
import cv2  
import numpy as np  
import pickle  
from sklearn.model_selection import train_test_split  
from tqdm import tqdm
from loguru import logger


def resize_frame(frame, target_size=(224, 224)):
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
  
def load_video(video_path, target_size=(224, 224)):
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
  
def concatenate_videos(video1_path, video2_path):  
    video1 = load_video(video1_path)  
    video2 = load_video(video2_path)  
    if video1.shape[0] != video2.shape[0]:  
        logger.warning(f"Warning: Video frame counts do not match for {video1_path} and {video2_path}")  
    return np.concatenate((video1, video2), axis=-1)
  
def load_labels(label_path):  
    with open(label_path, "rb") as f:  
        cpl = pickle.load(f)  
    refined_gt_kps = cpl["refined_gt_kps"]  
    rgb_avail_frames = cpl["rgb_avail_frames"]  
    refined_gt_kps = refined_gt_kps.reshape(refined_gt_kps.shape[0], -1)  
    return refined_gt_kps, rgb_avail_frames  
  
def process_subject(subject_dir, label_path):
    print(subject_dir, label_path)
    filename = os.path.basename(subject_dir).split("_")[0]
    video1_path = os.path.join(subject_dir, f"{filename}_color0.mp4")
    video2_path = os.path.join(subject_dir, f"{filename}_color1.mp4")
    
    logger.debug(f"Processing videos: {video1_path} and {video2_path}")
    
    try:
        concatenated_video = concatenate_videos(video1_path, video2_path)
        logger.debug(f"Concatenated video shape: {concatenated_video.shape}")
    except Exception as e:
        logger.error(f"Error concatenating videos for {subject_dir}: {str(e)}")
        return None, None

    try:
        labels, rgb_avail_frames = load_labels(label_path)
        logger.debug(f"Loaded labels shape: {labels.shape}, rgb_avail_frames: {rgb_avail_frames}")
    except Exception as e:
        logger.error(f"Error loading labels for {subject_dir}: {str(e)}")
        return None, None

    start_idx, end_idx = rgb_avail_frames
    concatenated_video = concatenated_video[start_idx:end_idx]
    labels = labels[start_idx:end_idx]
    
    logger.debug(f"Processed {subject_dir}: video shape {concatenated_video.shape}, labels shape {labels.shape}")
    return concatenated_video, labels
  
def main():
    logger.info("Starting data preparation process")
    video_base_dir = "/home/ubuntu/gdrive/workspace/blurred_videos/"
    label_base_dir = "/home/ubuntu/gdrive/workspace/dataset_release/aligned_data/pose_labels"
    out_dir = "/home/ubuntu/MARS-exp/mri_rgb_rede/"

    test_size = 0.2
    
    all_videos = []
    all_labels = []
    
    subjects = [d for d in os.listdir(video_base_dir) if os.path.isdir(os.path.join(video_base_dir, d))]
    logger.info(f"Found {len(subjects)} subjects to process")

    for subject_dir in subjects:
        subject_path = os.path.join(video_base_dir, subject_dir)
        label_path = os.path.join(label_base_dir, f"{subject_dir.split('_')[0]}_all_labels.cpl")
        
        logger.debug(f"Processing subject: {subject_dir}")
        video_data, label_data = process_subject(subject_path, label_path)
        if video_data is not None and label_data is not None:
            all_videos.append(video_data)
            all_labels.append(label_data)
        else:
            logger.warning(f"Skipping subject {subject_dir} due to processing errors")

      
    if all_videos:  
        all_videos = np.concatenate(all_videos, axis=0)  
    if all_labels:  
        all_labels = np.concatenate(all_labels, axis=0)  
      
    logger.info(f"Total frames in all videos: {all_videos.shape[0]}")  
    logger.info(f"Total frames in all labels: {all_labels.shape[0]}")  
      
    if all_videos.shape[0] != all_labels.shape[0]:  
        raise ValueError("Mismatch between total frames in videos and labels")  
      
    X_train, X_test, y_train, y_test = train_test_split(
        all_videos, 
        all_labels, 
        test_size=test_size, 
        random_state=42
    ) 

    logger.info("Saving processed data")
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)
    
    logger.success("Data preparation completed successfully")

  
if __name__ == "__main__":  
    logger.add("data_preparation.log", rotation="10 MB")
    main()  
