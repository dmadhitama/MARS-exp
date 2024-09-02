import numpy as np
import os
import pickle
import cv2
from tqdm import tqdm
from loguru import logger
import threading
from queue import Queue

def check_if_filename_same(video_folder, labels_filename):
    assert video_folder.split("_")[0] == labels_filename.split("_")[0]
    logger.info(f"Filenames {video_folder} and {labels_filename} match.")

def resize_frame(frame, target_size=(224, 224)):
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

def create_or_append_to_file(file_path, data):
    if not os.path.exists(file_path):
        np.save(file_path, data)
    else:
        existing_data = np.load(file_path)
        updated_data = np.concatenate((existing_data, data), axis=-1)
        np.save(file_path, updated_data)

def print_dataset_info(data_tr_file, labels_tr_file, data_tt_file, labels_tt_file):
    logger.info(f"Training data shape: {np.load(data_tr_file, mmap_mode='r').shape}")
    logger.info(f"Training labels shape: {np.load(labels_tr_file, mmap_mode='r').shape}")
    logger.info(f"Testing data shape: {np.load(data_tt_file, mmap_mode='r').shape}")
    logger.info(f"Testing labels shape: {np.load(labels_tt_file, mmap_mode='r').shape}")

def video_reader(video_file, frame_queue, rgb_avail_frames, target_size):
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

def prepare_video_data_labels(video_dir, labels_dir, output_dir, test_size=0.2, target_size=(224, 224), chunk_size=1000):
    video_folders = sorted([d for d in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, d))])
    labels_list = sorted([d for d in os.listdir(labels_dir) if d.endswith(".cpl")])
    assert len(video_folders) == len(labels_list), "Number of video folders and labels is not the same."

    data_tr_file = os.path.join(output_dir, "data_tr.npy")
    labels_tr_file = os.path.join(output_dir, "labels_tr.npy")
    data_tt_file = os.path.join(output_dir, "data_tt.npy")
    labels_tt_file = os.path.join(output_dir, "labels_tt.npy")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, (video_folder, label_file) in enumerate(zip(video_folders, labels_list)):
        logger.info(f"Processing video folder {video_folder}")
        check_if_filename_same(video_folder, label_file)
        
        with open(os.path.join(labels_dir, label_file), 'rb') as f:
            cpl = pickle.load(f)
        
        rgb_avail_frames = cpl['rgb_avail_frames']
        Y = cpl['refined_gt_kps'].reshape(-1, cpl['refined_gt_kps'].shape[1]*cpl['refined_gt_kps'].shape[2])

        video_folder_path = os.path.join(video_dir, video_folder)
        video_files = sorted([os.path.join(video_folder_path, f) for f in os.listdir(video_folder_path) if f.endswith(".mp4")])
        
        frame_queue = Queue(maxsize=100)  # Adjust queue size as needed

        frames_buffer = []
        labels_buffer = []

        for video_file in video_files:
            reader_thread = threading.Thread(target=video_reader, args=(video_file, frame_queue, rgb_avail_frames, target_size))
            reader_thread.start()

            pbar = tqdm(total=rgb_avail_frames[1] - rgb_avail_frames[0] + 1, desc=f"Processing {os.path.basename(video_file)}")
            while True:
                item = frame_queue.get()
                if item is None:
                    break
                frame_count, frame = item
                
                frames_buffer.append(frame)
                labels_buffer.append(Y[frame_count])
                
                if len(frames_buffer) >= chunk_size:
                    if i < (len(video_folders)*(1-test_size)):
                        create_or_append_to_file(data_tr_file, np.array(frames_buffer))
                        create_or_append_to_file(labels_tr_file, np.array(labels_buffer))
                    else:
                        create_or_append_to_file(data_tt_file, np.array(frames_buffer))
                        create_or_append_to_file(labels_tt_file, np.array(labels_buffer))
                    frames_buffer = []
                    labels_buffer = []
                
                pbar.update(1)
            
            pbar.close()
            reader_thread.join()

        # Save any remaining frames in the buffer
        if frames_buffer:
            if i < (len(video_folders)*(1-test_size)):
                create_or_append_to_file(data_tr_file, np.array(frames_buffer))
                create_or_append_to_file(labels_tr_file, np.array(labels_buffer))
            else:
                create_or_append_to_file(data_tt_file, np.array(frames_buffer))
                create_or_append_to_file(labels_tt_file, np.array(labels_buffer))

        del Y

        print(f"Data size after video folder {video_folder} processed.")
        print_dataset_info(data_tr_file, labels_tr_file, data_tt_file, labels_tt_file)

if __name__ == "__main__":
    video_dir = "/home/ubuntu/gdrive/workspace/blurred_videos"
    labels_dir = "/home/ubuntu/gdrive/workspace/dataset_release/aligned_data/pose_labels/"
    out_dir = "/home/ubuntu/MARS-exp/mri_rgb_rede/"
    target_size = (224, 224)
    chunk_size = 200

    try:
        prepare_video_data_labels(video_dir, labels_dir, out_dir, target_size=target_size, chunk_size=chunk_size)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")