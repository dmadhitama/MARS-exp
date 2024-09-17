import os
import pickle
import cv2
from loguru import logger
from tqdm import tqdm
import multiprocessing
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.cuda.amp as amp
from torch.utils.data import Sampler

from torchvision import transforms
import turbojpeg
from functools import partial

import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn import metrics
import matplotlib.pyplot as plt

def read_frame(frame_path):
    return cv2.imread(frame_path)

# collect data and label path for each subject into a dictionary
def process_frame_dir(args):
    frame_dir, video_dirs, label_dirs = args
    logger.info(f"Processing {frame_dir}")
    frame_dir_path = os.path.join(video_dirs, frame_dir)
    subject_name = frame_dir.split('_')[0]
    result = {subject_name: {"frames_dirs": {}}}

    for label in os.listdir(label_dirs):
        if subject_name == label.split('_')[0]:
            result[subject_name]["labels_path"] = os.path.join(label_dirs, label)
            break

    frames_subdirs = [
        os.path.join(frame_dir_path, f) 
        for f in os.listdir(frame_dir_path) 
        if os.path.isdir(os.path.join(frame_dir_path, f))
    ]

    for frames_subdir in frames_subdirs:
        result[subject_name]["frames_dirs"][frames_subdir] = [
            os.path.join(frames_subdir, f) 
            for f in os.listdir(frames_subdir) 
            if (f.endswith(".jpg") or f.endswith(".png")) and "subject" in f
        ]

    return result

def get_pair_path(label_dirs, video_dirs):
    frame_dirs = os.listdir(video_dirs)
    
    with multiprocessing.Pool() as pool:
        results = list(tqdm(
            pool.imap(process_frame_dir, [(fd, video_dirs, label_dirs) for fd in frame_dirs]),
            total=len(frame_dirs),
            desc="Processing frame directories"
        ))

    data_label_dict = {}
    for result in results:
        data_label_dict.update(result)

    return data_label_dict

# load labels
# for subject_name, data in data_label_dict.items():
def parse_data_label(frames_dirs, label_path, num_workers=4):
    with open(label_path, "rb") as f:
        cpl = pickle.load(f)
    labels = cpl["refined_gt_kps"]
    try:
        rgb0_paths, rgb1_paths = list(frames_dirs.keys())
    except Exception as ex:
        logger.error(f"Error: {ex}")
        logger.error(list(frames_dirs.keys()))
        return None, None

    rgb0_frames = frames_dirs[rgb0_paths]
    rgb1_frames = frames_dirs[rgb1_paths]
    assert len(rgb0_frames) == len(rgb1_frames) == labels.shape[0], \
        f"Length of frames and labels do not match: {len(rgb0_frames)} != {len(rgb1_frames)} != {labels.shape[0]}"
    data_path = []
    label = []

    for i in tqdm(range(len(rgb0_frames))):
        data_path.append((rgb0_frames[i], rgb1_frames[i]))
        label.append(labels[i])

    return data_path, label
    
class MARSDatasetChunked(Dataset):
    def __init__(
            self, 
            label_dirs, 
            video_dirs, 
            num_workers=8, 
            cache_dir=None, 
            chunk_size=1000,
            target_size=(128, 128),  # Reduced size
            dtype=np.float16  # Lower precision
        ):
        self.data_label_dict = get_pair_path(label_dirs, video_dirs)
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self.frames = []
        self.labels = []
        self._load_data()
        self.target_size = target_size
        self.dtype = dtype

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.preprocessed_file = os.path.join(self.cache_dir, 'preprocessed_frames.npz')
            if not os.path.exists(self.preprocessed_file):
                self._preprocess_and_cache()
            try:
                with np.load(self.preprocessed_file, mmap_mode='r') as data:
                    self.frames = data['frames']
            except Exception as e:
                print(f"Error loading npz: {e}")
                print("Falling back to in-memory processing")
                self._preprocess_and_cache()
        else:
            self._preprocess_and_cache()

        print(f"Total frames: {len(self.frames)}")
        print(f"Total labels: {len(self.labels)}")


    def _load_data(self):
        for subject_name, data in self.data_label_dict.items():
            print(f"Processing {subject_name}")
            frames_dirs = data["frames_dirs"]
            label_path = data["labels_path"]
            X, y = parse_data_label(frames_dirs, label_path, num_workers=self.num_workers)
            if X is None or y is None:
                continue
            self.frames.extend(X)
            self.labels.extend(y)

        print(f"Loaded {len(self.frames)} frames and {len(self.labels)} labels")

    def _preprocess_and_cache(self):
        total_frames = len(self.frames)
        chunk_size = min(10000, total_frames)  # Process 10000 frames at a time, or less if total_frames is smaller
        
        if self.cache_dir:
            if not os.path.exists(self.preprocessed_file):
                with np.load(self.preprocessed_file) as data:
                    self.frames = data['frames']
                return

        processed_frames = []
        
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            for chunk_start in range(0, total_frames, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_frames)
                chunk = self.frames[chunk_start:chunk_end]
                
                preprocessed_chunk = list(tqdm(
                    pool.imap(partial(self._preprocess_frames, target_size=self.target_size), chunk),
                    total=len(chunk),
                    desc=f"Preprocessing frames {chunk_start}-{chunk_end}"
                ))
                
                processed_frames.extend(preprocessed_chunk)
                
                # Save intermediate results
                if self.cache_dir and (chunk_end == total_frames or len(processed_frames) >= 50000):
                    temp_file = os.path.join(self.cache_dir, f'temp_preprocessed_frames_{chunk_start}.npz')
                    np.savez_compressed(temp_file, frames=np.array(processed_frames, dtype=self.dtype))
                    processed_frames = []  # Clear processed_frames to free up memory
        
        # Combine all temporary files into one
        if self.cache_dir:
            temp_files = [f for f in os.listdir(self.cache_dir) if f.startswith('temp_preprocessed_frames_')]
            combined_frames = []
            for temp_file in temp_files:
                with np.load(os.path.join(self.cache_dir, temp_file)) as data:
                    combined_frames.append(data['frames'])
            
            final_frames = np.concatenate(combined_frames)
            np.savez_compressed(self.preprocessed_file, frames=final_frames)
            
            # Remove temporary files
            for temp_file in temp_files:
                os.remove(os.path.join(self.cache_dir, temp_file))
            
            self.frames = final_frames
        else:
            self.frames = np.array(processed_frames, dtype=self.dtype)

    def __getitem__(self, idx):
        if idx >= len(self.frames):
            raise IndexError(f"Index {idx} is out of bounds for dataset with {len(self.frames)} items")
        
        frame = torch.from_numpy(self.frames[idx].astype(np.float32))
        label = torch.from_numpy(self.labels[idx]).float().view(51)
        return frame, label

class ChunkSampler(Sampler):
    def __init__(self, data_source, chunk_size):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.num_samples = len(data_source)

    def __iter__(self):
        indices = list(range(self.num_samples))
        np.random.shuffle(indices)
        for i in range(0, self.num_samples, self.chunk_size):
            yield from indices[i:min(i + self.chunk_size, self.num_samples)]

    def __len__(self):
        return self.num_samples
    

# Define the CNN model  
class CNN(nn.Module):  
    def __init__(self, in_shape, n_keypoints):  
        super(CNN, self).__init__()  
        self.conv1 = nn.Conv2d(in_shape[0], 16, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  
        self.dropout = nn.Dropout(0.3)  
        self.layer_norm = nn.LayerNorm([32, in_shape[1], in_shape[2]])  # Replace BatchNorm2d
        self.flatten = nn.Flatten()  
        self.fc1 = nn.Linear(32 * in_shape[1] * in_shape[2], 512)  
        self.fc2 = nn.Linear(512, n_keypoints)  
        self.fc_ln = nn.LayerNorm(512)  # Replace BatchNorm1d
        self.fc_dropout = nn.Dropout(0.4)  
  
    def forward(self, x):  
        x = torch.relu(self.conv1(x))  
        x = self.dropout(x)  
        x = torch.relu(self.conv2(x))  
        x = self.dropout(x)  
        x = self.layer_norm(x)  # Use LayerNorm instead of BatchNorm
        x = self.flatten(x)  
        x = torch.relu(self.fc1(x))  
        x = self.fc_ln(x)  # Use LayerNorm instead of BatchNorm
        x = self.fc_dropout(x)  
        x = self.fc2(x)  
        return x  


if __name__ == "__main__":
    label_dirs = "/home/ubuntu/gdrive/workspace/dataset_release/aligned_data/pose_labels/"
    video_dirs = "/home/ubuntu/gdrive/workspace/blurred_videos/"
    cache_dir = '/home/ubuntu/MARS-cache'
    output_direct = 'model_mri_pytorch-RGB-e150-160/'

    # Set PYTORCH_CUDA_ALLOC_CONF
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Reduce batch size further if needed
    batch_size = 256
    accumulation_steps = 4  # Accumulate gradients over 4 batches
    target_size = (160, 160)

    chunk_size = 1000

    # Define epochs and iterations
    epochs = 150
    iters = 5

    # Create the full dataset
    full_dataset = MARSDatasetChunked(
        label_dirs, 
        video_dirs, 
        num_workers=8, 
        cache_dir=cache_dir,
        chunk_size=chunk_size,
        target_size=target_size,
        dtype=np.float16
    )

    # Split the dataset into train and test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    dataset_train, dataset_test = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    # Enable pin_memory for faster data transfer to GPU
    pin_memory = torch.cuda.is_available()

    # Create DataLoaders
    train_loader = DataLoader(
        dataset_train, 
        batch_size=batch_size, 
        # shuffle=True,
        pin_memory=pin_memory,
        num_workers=8,
        prefetch_factor=2,
        sampler=ChunkSampler(dataset_train, chunk_size)
    )
    test_loader = DataLoader(
        dataset_test, 
        batch_size=batch_size, 
        # shuffle=False,
        pin_memory=pin_memory,
        num_workers=8,
        prefetch_factor=2,
        sampler=ChunkSampler(dataset_test, chunk_size)
    )

    # Initialize the result array
    paper_result_list = []

    # Define the output directory
    if not os.path.exists(output_direct):
        os.makedirs(output_direct)

    n_keypoints = 51  # Assuming 17 keypoints with x, y, z coordinates

    # Repeat i iteration to get the average result
    for iter in range(iters):
        print(f"Iteration {iter}")
        # Instantiate the model
        model = CNN(dataset_train[0][0].shape, n_keypoints)
        model = model.cuda() if torch.cuda.is_available() else model

        score_min = float('inf')

        # Define optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=0.0001)#, betas=(0.5, 0.999))
        criterion = nn.MSELoss()
        scaler = amp.GradScaler()

        # Initialize lists to store metrics
        train_mae_history = []
        val_mae_history = []
        train_loss_history = []
        val_loss_history = []

        # Training loop
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            train_mae = 0
            print(f"Epoch {epoch+1}/{epochs}")
            for i, (batch_features, batch_labels) in enumerate(tqdm(train_loader)):
                batch_features, batch_labels = batch_features.cuda(non_blocking=True), batch_labels.cuda(non_blocking=True) if torch.cuda.is_available() else (batch_features, batch_labels)

                with amp.autocast():
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)

                scaler.scale(loss).backward()
                
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += loss.item()
                train_mae += torch.mean(torch.abs(outputs - batch_labels)).item()

            train_loss /= len(train_loader)
            train_mae /= len(train_loader)
            train_loss_history.append(train_loss)
            train_mae_history.append(train_mae)

            # Validation loop (using test set as validation)
            model.eval()
            val_loss = 0
            val_mae = 0
            print("Validation")
            with torch.no_grad():
                for batch_features, batch_labels in tqdm(test_loader):
                    batch_features, batch_labels = batch_features.cuda(non_blocking=True), batch_labels.cuda(non_blocking=True) if torch.cuda.is_available() else (batch_features, batch_labels)
                    with amp.autocast():
                        outputs = model(batch_features)
                        val_loss += criterion(outputs, batch_labels).item()
                        val_mae += torch.mean(torch.abs(outputs - batch_labels)).item()

            val_loss /= len(test_loader)
            val_mae /= len(test_loader)
            val_loss_history.append(val_loss)
            val_mae_history.append(val_mae)
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Train MAE: {train_mae:.4f}, Validation MAE: {val_mae:.4f}')
             
  
        # Save and print the metrics  
        model.eval()  
        train_loss = 0  
        with torch.no_grad():  
            for batch_features, batch_labels in train_loader:  
                batch_features, batch_labels = batch_features.cuda(), batch_labels.cuda() if torch.cuda.is_available() else (batch_features, batch_labels)  
                outputs = model(batch_features)  
                train_loss += criterion(outputs, batch_labels).item()  
        train_loss /= len(train_loader)  
        print('Train Loss = ', train_loss)  
    
        test_loss = 0  
        result_test = []  
        with torch.no_grad():  
            for batch_features, batch_labels in test_loader:  
                batch_features, batch_labels = batch_features.cuda(), batch_labels.cuda() if torch.cuda.is_available() else (batch_features, batch_labels)  
                outputs = model(batch_features)  
                test_loss += criterion(outputs, batch_labels).item()  
                result_test.append(outputs.cpu().numpy())  
        test_loss /= len(test_loader)  
        result_test = np.concatenate(result_test, axis=0)  
        print('Test Loss = ', test_loss)  
    
        # Plot accuracy  
        plt.figure(figsize=(15, 15))  
        plt.plot(train_mae_history)  
        plt.plot(val_mae_history)  
        plt.title('Model MAE')  
        plt.ylabel('MAE')  
        plt.xlabel('Epoch')  
        plt.grid()  
        plt.legend(['Train', 'Validation'], loc='upper left')  
        plt.savefig(output_direct + f"/acc-{iter}.png")  
    
        # Plot loss  
        plt.figure(figsize=(10, 10))  
        plt.plot(train_loss_history)  
        plt.plot(val_loss_history)  
        plt.title('Model loss')  
        plt.ylabel('Loss')  
        plt.xlabel('Epoch')  
        plt.grid()  
        plt.legend(['Train', 'Validation'], loc='upper left')  
        plt.savefig(output_direct + f"/loss-{iter}.png")  

        # Before calculating metrics, move tensors to CPU and convert to NumPy arrays
        batch_labels = batch_labels.cpu().numpy()
        outputs = outputs.cpu().numpy()
    
        # Error for each axis  
        print("mae for x is", metrics.mean_absolute_error(batch_labels[:, 0:17], outputs[:, 0:17]))  
        print("mae for y is", metrics.mean_absolute_error(batch_labels[:, 17:34], outputs[:, 17:34]))  
        print("mae for z is", metrics.mean_absolute_error(batch_labels[:, 34:51], outputs[:, 34:51]))  
    
        # Matrix transformation for the final all 17 points mae  
        x_mae = metrics.mean_absolute_error(batch_labels[:, 0:17], outputs[:, 0:17], multioutput='raw_values')  
        y_mae = metrics.mean_absolute_error(batch_labels[:, 17:34], outputs[:, 17:34], multioutput='raw_values')  
        z_mae = metrics.mean_absolute_error(batch_labels[:, 34:51], outputs[:, 34:51], multioutput='raw_values')  
    
        all_17_points_mae = np.concatenate((x_mae, y_mae, z_mae)).reshape(3, 17)  
        avg_17_points_mae = np.mean(all_17_points_mae, axis=0)  
        avg_17_points_mae_xyz = np.mean(all_17_points_mae, axis=1).reshape(1, 3)  
        all_17_points_mae_Transpose = all_17_points_mae.T  
    
        # Matrix transformation for the final all 17 points rmse  
        x_rmse = metrics.mean_squared_error(batch_labels[:, 0:17], outputs[:, 0:17], multioutput='raw_values', squared=False)  
        y_rmse = metrics.mean_squared_error(batch_labels[:, 17:34], outputs[:, 17:34], multioutput='raw_values', squared=False)  
        z_rmse = metrics.mean_squared_error(batch_labels[:, 34:51], outputs[:, 34:51], multioutput='raw_values', squared=False)  
    
        all_17_points_rmse = np.concatenate((x_rmse, y_rmse, z_rmse)).reshape(3, 17)  
        avg_17_points_rmse = np.mean(all_17_points_rmse, axis=0)  
        avg_17_points_rmse_xyz = np.mean(all_17_points_rmse, axis=1).reshape(1, 3)  
        all_17_points_rmse_Transpose = all_17_points_rmse.T  
    
        # Merge the mae and rmse  
        all_17_points_maermse_Transpose = np.concatenate((all_17_points_mae_Transpose, all_17_points_rmse_Transpose), axis=1) * 100  
        avg_17_points_maermse_Transpose = np.concatenate((avg_17_points_mae_xyz, avg_17_points_rmse_xyz), axis=1) * 100  
    
        # Concatenate the array, the final format is the same as shown in paper. First 17 rows each joint, the final row is the average  
        paper_result_maermse = np.concatenate((all_17_points_maermse_Transpose, avg_17_points_maermse_Transpose), axis=0)  
        paper_result_maermse = np.around(paper_result_maermse, 2)  
        # Reorder the columns to make it xmae, xrmse, ymae, yrmse, zmae, zrmse, avgmae, avgrmse  
        paper_result_maermse = paper_result_maermse[:, [0, 3, 1, 4, 2, 5]]  
        # Append each iteration's result  
        paper_result_list.append(paper_result_maermse)  
        # Save the best model so far  
        if test_loss < score_min:  
            torch.save(model.state_dict(), output_direct + 'MARS.pth')  
            score_min = test_loss  
    
    # Average the result for all iterations  
    mean_paper_result_list = np.mean(paper_result_list, axis=0)  
    mean_mae = np.mean(
        np.dstack(
            (
                mean_paper_result_list[:, 0], 
                mean_paper_result_list[:, 2], 
                mean_paper_result_list[:, 4]
            )
        ).reshape(18, 3), axis=1
    )  
    mean_rmse = np.mean(
        np.dstack(
            (
                mean_paper_result_list[:, 1], 
                mean_paper_result_list[:, 3], 
                mean_paper_result_list[:, 5]
            )
        ).reshape(18, 3), axis=1
    )  
    mean_paper_result_list = np.concatenate(
        (
            np.mean(paper_result_list, axis=0), 
            mean_mae.reshape(18, 1), 
            mean_rmse.reshape(18, 1)
        ), axis=1
    )  
    
    # Export the Accuracy  
    output_path = output_direct + "Accuracy"  
    output_filename = output_path + "/MARS_accuracy"  
    if not os.path.exists(output_path):  
        os.makedirs(output_path)  
    np.save(output_filename + ".npy", mean_paper_result_list)  
    np.savetxt(output_filename + ".txt", mean_paper_result_list, fmt='%.2f')  
    # Create DataFrame for the results  
    columns = ['X_MAE', 'X_RMSE', 'Y_MAE', 'Y_RMSE', 'Z_MAE', 'Z_RMSE', 'Avg_MAE', 'Avg_RMSE']  
    df_results = pd.DataFrame(mean_paper_result_list, columns=columns)  
    # Save DataFrame to CSV  
    output_filename = output_path + "/MARS_accuracy"  
    df_results.to_csv(output_filename + ".csv", index=False) 
