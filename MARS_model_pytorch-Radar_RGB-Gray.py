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
    frame_dir, video_dirs, label_dirs, radar_dirs = args
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

    # collect radar data
    radar_dir_path = os.path.join(radar_dirs, f"{subject_name}_featuremap.npy")
    if os.path.exists(radar_dir_path):
        result[subject_name]["radar_dirs"] = radar_dir_path
    else:
        raise ValueError(f"Radar data not found for {subject_name}. Please check the radar data directory. Exiting...")

    return result

def get_pair_path(label_dirs, video_dirs, radar_dirs):
    frame_dirs = os.listdir(video_dirs)

    with multiprocessing.Pool() as pool:
        results = list(tqdm(
            pool.imap(process_frame_dir, [(fd, video_dirs, label_dirs, radar_dirs) for fd in frame_dirs]),
            total=len(frame_dirs),
            desc="Processing frame directories"
        ))

    data_label_dict = {}
    for result in results:
        data_label_dict.update(result)

    return data_label_dict

# load labels
# for subject_name, data in data_label_dict.items():
def parse_data_label(frames_dirs, label_path, radar_path, num_workers=4):
    with open(label_path, "rb") as f:
        cpl = pickle.load(f)
    labels = cpl["refined_gt_kps"]

    radar_data = np.load(radar_path)
    
    try:
        rgb0_paths, rgb1_paths = list(frames_dirs.keys())
    except Exception as ex:
        logger.error(f"Error: {ex}")
        logger.error(list(frames_dirs.keys()))
        return None, None

    rgb0_frames = frames_dirs[rgb0_paths]
    rgb1_frames = frames_dirs[rgb1_paths]
    # adjust the length of exceeded frames to match the length of labels (for radar data)
    assert len(rgb0_frames) == len(rgb1_frames) == labels.shape[0], \
        f"Length of frames and labels do not match: {len(rgb0_frames)} != {len(rgb1_frames)} != {labels.shape[0]}"
    data_path = []
    radar = []
    label = []

    # use the shortest length between frames and radar data
    min_length = min(len(rgb0_frames), len(radar_data))

    for i in tqdm(range(min_length)):
        data_path.append((rgb0_frames[i], rgb1_frames[i]))
        label.append(labels[i])
        radar.append(radar_data[i])
    return data_path, radar, label
    
class MARSDatasetChunked(Dataset):
    def __init__(
            self, 
            label_dirs, 
            video_dirs, 
            radar_dirs,
            num_workers=8, 
            cache_dir=None, 
            chunk_size=1000,
            target_size=(128, 128),
            radar_size=(14, 14, 5),
        ):
        self.data_label_dict = get_pair_path(label_dirs, video_dirs, radar_dirs)
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self.frames = []
        self.radar = []
        self.labels = []
        self._load_data() # load data into memory
        self.target_size = target_size
        self.radar_size = radar_size

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.preprocessed_radar_file = os.path.join(self.cache_dir, 'preprocessed_radar.npy')
            self.preprocessed_file = os.path.join(self.cache_dir, 'preprocessed_frames_gray.npy')
            if not os.path.exists(self.preprocessed_radar_file):
                self._preprocess_and_cache_radar()
            if not os.path.exists(self.preprocessed_file):
                self._preprocess_and_cache()

            # load memmap for radar
            try:
                self.radar = np.memmap(
                    self.preprocessed_radar_file, 
                    dtype='float32', 
                    mode='r', 
                    shape=(len(self.radar), self.radar_size[0], self.radar_size[1], self.radar_size[2])
                )
            except Exception as e:
                print(f"Error loading memmap: {e}")
                print("Falling back to in-memory processing")
                self._preprocess_and_cache_radar()

            # load memmap for frames
            try:
                self.frames = np.memmap(
                    self.preprocessed_file, 
                    dtype='float32', 
                    mode='r', 
                    shape=(len(self.frames), 2, self.target_size[0], self.target_size[1])
                )
            except Exception as e:
                print(f"Error loading memmap: {e}")
                print("Falling back to in-memory processing")
                self._preprocess_and_cache()

        else:
            self._preprocess_and_cache_radar()
            self._preprocess_and_cache()

        print(f"Total frames: {len(self.frames)}")
        print(f"Total labels: {len(self.labels)}")
        print(f"Total radar: {len(self.radar)}")
    def _load_data(self):
        for subject_name, data in self.data_label_dict.items():
            print(f"Processing {subject_name}")
            frames_dirs = data["frames_dirs"]
            label_path = data["labels_path"]
            radar_path = data["radar_dirs"]
            X_image, X_radar, y = parse_data_label(frames_dirs, label_path, radar_path, num_workers=self.num_workers)
            if X_image is None or X_radar is None or y is None:
                continue
            self.frames.extend(X_image)
            self.radar.extend(X_radar)
            self.labels.extend(y)

        print(f"Loaded {len(self.frames)} frames and {len(self.labels)} labels")

    def _preprocess_and_cache_radar(self):
        print("Starting radar data preprocessing and caching")
        total_frames_radar = len(self.radar)
        shape = (total_frames_radar, self.radar_size[0], self.radar_size[1], self.radar_size[2])

        if self.cache_dir:
            try:
                print(f"Creating memmap file: {self.preprocessed_radar_file}")
                radar_mmap = np.memmap(self.preprocessed_radar_file, dtype='float32', mode='w+', shape=shape)
            except Exception as e:
                print(f"Error creating memmap: {e}")
                print("Falling back to in-memory processing")
                radar_mmap = np.zeros(shape, dtype='float32')
        else:
            radar_mmap = np.zeros(shape, dtype='float32')

        for chunk_start in range(0, total_frames_radar, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_frames_radar)
            print(f"Processing radar chunk {chunk_start}-{chunk_end}")
            chunk = self.radar[chunk_start:chunk_end]
            # Reshape or convert if necessary
            chunk = np.array(chunk).reshape(-1, self.radar_size[0], self.radar_size[1], self.radar_size[2]).astype(np.float32)
            radar_mmap[chunk_start:chunk_end] = chunk

        if self.cache_dir:
            try:
                radar_mmap.flush()
            except Exception as e:
                print(f"Error flushing memmap: {e}")
        
        self.radar = radar_mmap
        print("Radar data preprocessing and caching completed")

    def _preprocess_and_cache(self):
        total_frames = len(self.frames)
        shape = (total_frames, 2, self.target_size[0], self.target_size[1])
        
        if self.cache_dir:
            try:
                frames_mmap = np.memmap(self.preprocessed_file, dtype='float32', mode='w+', shape=shape)
            except Exception as e:
                print(f"Error creating memmap: {e}")
                print("Falling back to in-memory processing")
                frames_mmap = np.zeros(shape, dtype='float32')
        else:
            frames_mmap = np.zeros(shape, dtype='float32')

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            for chunk_start in range(0, total_frames, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, total_frames)
                chunk = self.frames[chunk_start:chunk_end]
                preprocessed_chunk = list(tqdm(
                    pool.imap(partial(self._preprocess_frames, target_size=self.target_size), chunk),
                    total=len(chunk),
                    desc=f"Preprocessing frames {chunk_start}-{chunk_end}"
                ))
                frames_mmap[chunk_start:chunk_end] = preprocessed_chunk

        if self.cache_dir:
            try:
                frames_mmap.flush()
            except Exception as e:
                print(f"Error flushing memmap: {e}")
        
        self.frames = frames_mmap

    @staticmethod
    def _preprocess_frames(frame_pair, target_size):
        jpeg = turbojpeg.TurboJPEG()
        frame0 = MARSDatasetChunked._read_and_transform(frame_pair[0], jpeg, target_size)
        frame1 = MARSDatasetChunked._read_and_transform(frame_pair[1], jpeg, target_size)
        return np.concatenate((frame0, frame1), axis=0)

    @staticmethod
    def _read_and_transform(frame_path, jpeg, target_size):
        with open(frame_path, 'rb') as in_file:
            frame = jpeg.decode(in_file.read(), pixel_format=turbojpeg.TJPF_GRAY)
        frame = cv2.resize(frame, target_size)
        return frame.reshape(1, *target_size).astype(np.float32) / 255.0

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        if idx >= len(self.frames):
            raise IndexError(f"Index {idx} is out of bounds for dataset with {len(self.frames)} items")
        
        frame = torch.from_numpy(self.frames[idx])
        radar = torch.from_numpy(self.radar[idx])
        label = torch.from_numpy(self.labels[idx]).float().view(51)
        return frame, radar, label

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
    def __init__(self, rgb_shape, radar_shape, n_keypoints):
        super(CNN, self).__init__()
        
        # RGB branch
        self.rgb_conv1 = nn.Conv2d(rgb_shape[0], 16, kernel_size=3, padding=1)
        self.rgb_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.rgb_dropout = nn.Dropout(0.3)
        self.rgb_layer_norm = nn.LayerNorm([32, rgb_shape[1], rgb_shape[2]])
        self.rgb_flatten = nn.Flatten()
        
        # Radar branch
        self.radar_conv1 = nn.Conv2d(radar_shape[2], 16, kernel_size=3, padding=1)
        self.radar_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.radar_dropout = nn.Dropout(0.3)
        self.radar_layer_norm = nn.LayerNorm([32, radar_shape[0], radar_shape[1]])
        self.radar_flatten = nn.Flatten()
        
        # Combined fully connected layers
        combined_features = 32 * rgb_shape[1] * rgb_shape[2] + 32 * radar_shape[0] * radar_shape[1]
        self.fc1 = nn.Linear(combined_features, 512)
        self.fc2 = nn.Linear(512, n_keypoints)
        self.fc_ln = nn.LayerNorm(512)
        self.fc_dropout = nn.Dropout(0.4)

    def forward(self, rgb, radar):
        # RGB branch
        rgb = torch.relu(self.rgb_conv1(rgb))
        rgb = self.rgb_dropout(rgb)
        rgb = torch.relu(self.rgb_conv2(rgb))
        rgb = self.rgb_dropout(rgb)
        rgb = self.rgb_layer_norm(rgb)
        rgb = self.rgb_flatten(rgb)
        
        # Radar branch
        radar = radar.permute(0, 3, 1, 2)  # Change from (B, H, W, C) to (B, C, H, W)
        radar = torch.relu(self.radar_conv1(radar))
        radar = self.radar_dropout(radar)
        radar = torch.relu(self.radar_conv2(radar))
        radar = self.radar_dropout(radar)
        radar = self.radar_layer_norm(radar)
        radar = self.radar_flatten(radar)
        
        # Combine features
        combined = torch.cat((rgb, radar), dim=1)
        
        # Fully connected layers
        x = torch.relu(self.fc1(combined))
        x = self.fc_ln(x)
        x = self.fc_dropout(x)
        x = self.fc2(x)
        return x
    
def custom_collate(batch):
    frames, radars, labels = zip(*batch)
    frames = torch.stack(frames).float()
    radars = torch.stack(radars).float()
    labels = torch.stack(labels).float()
    return frames, radars, labels


if __name__ == "__main__":
    label_dirs = "/home/ubuntu/gdrive/workspace/dataset_release/aligned_data/pose_labels/"
    video_dirs = "/home/ubuntu/gdrive/workspace/blurred_videos/"
    radar_dirs = "/home/ubuntu/gdrive/workspace/dataset_release/features/radar/"
    cache_dir = '/home/ubuntu/MARS-RGB-Radar-cache-npy'
    output_direct = 'model_mri_pytorch-RGB-Radar-Gray-e200-160/'

    # Set PYTORCH_CUDA_ALLOC_CONF
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    # Reduce batch size further if needed
    batch_size = 128
    accumulation_steps = 4  # Accumulate gradients over 4 batches
    target_size = (160, 160) 
    radar_size = (14, 14, 5)
    chunk_size = 1000
    num_workers = 8
    prefetch_factor = 4

    # Define epochs and iterations
    epochs = 200
    iters = 1

    full_dataset = MARSDatasetChunked(
        label_dirs, 
        video_dirs, 
        radar_dirs,
        num_workers=num_workers, 
        cache_dir=cache_dir,
        chunk_size=chunk_size,
        target_size=target_size,
        radar_size=radar_size
    )

    # Split the dataset into train and test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    dataset_train, dataset_test = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(
        dataset_train, 
        batch_size=batch_size, 
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        sampler=ChunkSampler(dataset_train, chunk_size),
        collate_fn=custom_collate
    )
    test_loader = DataLoader(
        dataset_test, 
        batch_size=batch_size, 
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=True,
        sampler=ChunkSampler(dataset_test, chunk_size),
        collate_fn=custom_collate
    )

    # Initialize the result array
    paper_result_list = []

    # Define the output directory
    if not os.path.exists(output_direct):
        os.makedirs(output_direct)

    rgb_shape = (2, target_size[0], target_size[1])  # Assuming 2 channels for RGB (grayscale for two frames)
    radar_shape = radar_size
    n_keypoints = 51  # Assuming 17 keypoints with x, y, z coordinates

    # Repeat i iteration to get the average result
    for iter in range(iters):
        print(f"Iteration {iter}")
        # Instantiate the model
        model = CNN(rgb_shape, radar_shape, n_keypoints)
        model = model.cuda() if torch.cuda.is_available() else model

        best_val_loss = float('inf')
        best_model_state = None

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
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                for i, (batch_rgb, batch_radar, batch_labels) in enumerate(tqdm(train_loader)):
                    batch_rgb, batch_radar, batch_labels = batch_rgb.cuda(non_blocking=True), batch_radar.cuda(non_blocking=True), batch_labels.cuda(non_blocking=True) if torch.cuda.is_available() else (batch_rgb, batch_radar, batch_labels)
        
                    with amp.autocast():
                        outputs = model(batch_rgb, batch_radar)
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
                    for batch_rgb, batch_radar, batch_labels in tqdm(test_loader):
                        batch_rgb, batch_radar, batch_labels = batch_rgb.cuda(non_blocking=True), batch_radar.cuda(non_blocking=True), batch_labels.cuda(non_blocking=True) if torch.cuda.is_available() else (batch_rgb, batch_radar, batch_labels)
                        with amp.autocast():
                            outputs = model(batch_rgb, batch_radar)
                            val_loss += criterion(outputs, batch_labels).item()
                            val_mae += torch.mean(torch.abs(outputs - batch_labels)).item()

                val_loss /= len(test_loader)
                val_mae /= len(test_loader)
                val_loss_history.append(val_loss)
                val_mae_history.append(val_mae)
                print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Train MAE: {train_mae:.4f}, Validation MAE: {val_mae:.4f}')

                # Save the best model so far  
                if val_loss < best_val_loss:
                    best_model_state = model.state_dict()
                    best_val_loss = val_loss
                    torch.save(best_model_state, output_direct + 'best_model.pth')
                
                torch.cuda.synchronize()
                if epoch % 5 == 0:
                    torch.cuda.empty_cache()

        # After the training loop, load the best model
        model.load_state_dict(best_model_state)

        # Perform full evaluation on the test set
        model.eval()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for batch_rgb, batch_radar, batch_labels in tqdm(test_loader, desc="Evaluating best model"):
                batch_rgb, batch_radar, batch_labels = batch_rgb.cuda(), batch_radar.cuda(), batch_labels.cuda() if torch.cuda.is_available() else (batch_rgb, batch_radar, batch_labels)
                outputs = model(batch_rgb, batch_radar)
                all_outputs.append(outputs.cpu().numpy())
                all_labels.append(batch_labels.cpu().numpy())

        all_outputs = np.concatenate(all_outputs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
    
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
        # all_labels = all_labels.cpu().numpy()
        # all_outputs = all_outputs.cpu().numpy()
    
        # Error for each axis  
        print("mae for x is", metrics.mean_absolute_error(all_labels[:, 0:17], all_outputs[:, 0:17]))  
        print("mae for y is", metrics.mean_absolute_error(all_labels[:, 17:34], all_outputs[:, 17:34]))  
        print("mae for z is", metrics.mean_absolute_error(all_labels[:, 34:51], all_outputs[:, 34:51]))  
    
        # Matrix transformation for the final all 17 points mae  
        x_mae = metrics.mean_absolute_error(all_labels[:, 0:17], all_outputs[:, 0:17], multioutput='raw_values')  
        y_mae = metrics.mean_absolute_error(all_labels[:, 17:34], all_outputs[:, 17:34], multioutput='raw_values')  
        z_mae = metrics.mean_absolute_error(all_labels[:, 34:51], all_outputs[:, 34:51], multioutput='raw_values')  
    
        all_17_points_mae = np.concatenate((x_mae, y_mae, z_mae)).reshape(3, 17)  
        avg_17_points_mae = np.mean(all_17_points_mae, axis=0)  
        avg_17_points_mae_xyz = np.mean(all_17_points_mae, axis=1).reshape(1, 3)  
        all_17_points_mae_Transpose = all_17_points_mae.T  
    
        # Matrix transformation for the final all 17 points rmse  
        x_rmse = metrics.mean_squared_error(all_labels[:, 0:17], all_outputs[:, 0:17], multioutput='raw_values', squared=False)  
        y_rmse = metrics.mean_squared_error(all_labels[:, 17:34], all_outputs[:, 17:34], multioutput='raw_values', squared=False)  
        z_rmse = metrics.mean_squared_error(all_labels[:, 34:51], all_outputs[:, 34:51], multioutput='raw_values', squared=False)  
    
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
    print("Writing accuracy report...")
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
    print("Done!")