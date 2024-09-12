import os
import pickle
import cv2
from loguru import logger
from tqdm import tqdm
import multiprocessing
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split

label_dirs = "/home/ubuntu/gdrive/workspace/dataset_release/aligned_data/pose_labels/"
video_dirs = "/home/ubuntu/gdrive/workspace/blurred_videos/"

def read_frame(frame_path):
    return cv2.imread(frame_path)

# collect data and label path for each subject into a dictionary
def get_pair_path(label_dirs, video_dirs):
    labels = os.listdir(label_dirs)
    frame_dirs = os.listdir(video_dirs)
    data_label_dict = {}
    for frame_dir in frame_dirs:
        frame_dir_path = os.path.join(video_dirs, frame_dir)
        subject_name = frame_dir.split('_')[0]
        if subject_name not in data_label_dict:
            data_label_dict[subject_name] = {}
        for label in labels:
            if subject_name == label.split('_')[0]:
                data_label_dict[subject_name]["labels_path"] = os.path.join(label_dirs, label)
                break
        frames_subdirs = [os.path.join(frame_dir_path, f) for f in os.listdir(frame_dir_path) if os.path.isdir(os.path.join(frame_dir_path, f))]
        data_label_dict[subject_name]["frames_dirs"] = frames_subdirs
    return data_label_dict

# load labels
# for subject_name, data in data_label_dict.items():
def parse_data_label(frames_dirs, label_path, num_workers=4):
    with open(label_path, "rb") as f:
        cpl = pickle.load(f)
    labels = cpl["refined_gt_kps"]
    idx_start, idx_end = cpl["rgb_avail_frames"]

    all_frames = []
    with multiprocessing.Pool(num_workers) as pool:
        all_frame_paths = []
        for frame_dir in frames_dirs:
            frame_files = sorted(os.listdir(frame_dir))
            valid_frames = frame_files[idx_start:idx_end+1]
            frame_paths = [os.path.join(frame_dir, frame_file) for frame_file in valid_frames]
            all_frame_paths.extend(frame_paths)
        
        frames = list(tqdm(
            pool.imap(read_frame, all_frame_paths),
            total=len(all_frame_paths),
            desc=f"Processing all frames"
        ))

        all_frames = frames
        
        # all_frames.extend(frames)

    # Ensure that the number of frames matches the number of labels
    assert len(all_frames) == labels.shape[0], "Number of frames doesn't match number of labels"

    return all_frames, labels

class MARSDataset(Dataset):
    def __init__(self, label_dirs, video_dirs, num_workers=16):
        self.data_label_dict = get_pair_path(label_dirs, video_dirs)
        self.num_workers = num_workers
        self.frames = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        for subject_name, data in self.data_label_dict.items():
            frames_dirs = data["frames_dirs"]
            label_path = data["labels_path"]
            X, y = parse_data_label(frames_dirs, label_path, num_workers=self.num_workers)
            self.frames.extend(X)
            self.labels.extend(y)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = torch.from_numpy(self.frames[idx]).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(self.labels[idx]).float()
        return frame, label
    

# Define the CNN model  
class CNN(nn.Module):  
    def __init__(self, in_shape, n_keypoints):  
        super(CNN, self).__init__()  
        self.conv1 = nn.Conv2d(in_shape[0], 16, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  
        self.dropout = nn.Dropout(0.3)  
        self.batch_norm = nn.BatchNorm2d(32)  
        self.flatten = nn.Flatten()  
        self.fc1 = nn.Linear(32 * in_shape[1] * in_shape[2], 512)  
        self.fc2 = nn.Linear(512, n_keypoints)  
        self.fc_bn = nn.BatchNorm1d(512)  
        self.fc_dropout = nn.Dropout(0.4)  
  
    def forward(self, x):  
        x = torch.relu(self.conv1(x))  
        x = self.dropout(x)  
        x = torch.relu(self.conv2(x))  
        x = self.dropout(x)  
        x = self.batch_norm(x)  
        x = self.flatten(x)  
        x = torch.relu(self.fc1(x))  
        x = self.fc_bn(x)  
        x = self.fc_dropout(x)  
        x = self.fc2(x)  
        return x  

# num_workers = 8
# for subject_name, data in data_label_dict.items():
#     frames_dirs = data["frames_dirs"]
#     label_path = data["labels_path"]
#     X, y = parse_data_label(frames_dirs, label_path, num_workers=num_workers)
#     print(X[0].shape)
#     print(y[0].shape)
#     break


if __name__ == "__main__":
    label_dirs = "/home/ubuntu/gdrive/workspace/dataset_release/aligned_data/pose_labels/"
    video_dirs = "/home/ubuntu/gdrive/workspace/blurred_videos/"
    
    data_label_dict = get_pair_path(label_dirs, video_dirs)

    # split data into train and test using test_size = 0.2
    train_data, test_data = train_test_split(data_label_dict, test_size=0.2, random_state=42)

    dataset = MARSDataset(label_dirs, video_dirs)
    print(f"Dataset size: {len(dataset)}")
    
    sample_frame, sample_label = dataset[0]
    print(f"Sample frame shape: {sample_frame.shape}")
    print(f"Sample label shape: {sample_label.shape}")