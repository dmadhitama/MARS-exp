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
from torchvision import transforms

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
    
class MARSDataset(Dataset):
    def __init__(self, label_dirs, video_dirs, num_workers=4):
        self.data_label_dict = get_pair_path(label_dirs, video_dirs)
        self.num_workers = num_workers
        self.frames = []
        self.labels = []
        self._load_data()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),  # Adjust size as needed
            transforms.ToTensor(),
        ])

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

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # read and concatenate two frames
        frame0 = read_frame(self.frames[idx][0])
        frame1 = read_frame(self.frames[idx][1])
        frame0 = self.transform(frame0)
        frame1 = self.transform(frame1)
        frame = np.concatenate((frame0, frame1), axis=1)

        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        label = torch.from_numpy(self.labels[idx]).float()
        # reshape label from 17*3 to 51 
        label = label.view(51)
        return frame, label
    

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

    # Create the full dataset
    full_dataset = MARSDataset(label_dirs, video_dirs)
    
    # Split the dataset into train and test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    dataset_train, dataset_test = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # Set PYTORCH_CUDA_ALLOC_CONF
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Reduce batch size further if needed
    batch_size = 256
    accumulation_steps = 4  # Accumulate gradients over 4 batches
    # Enable pin_memory for faster data transfer to GPU
    pin_memory = torch.cuda.is_available()

    # Create DataLoaders
    train_loader = DataLoader(
        dataset_train, 
        batch_size=batch_size, 
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=16
    )
    test_loader = DataLoader(
        dataset_test, 
        batch_size=batch_size, 
        shuffle=False,
        pin_memory=pin_memory,
        num_workers=16
    )

    # Initialize the result array
    paper_result_list = []

    # Define epochs and iterations
    epochs = 150
    iters = 10

    # Define the output directory
    output_direct = 'model_mri_pytorch-RGB/'
    if not os.path.exists(output_direct):
        os.makedirs(output_direct)

    n_keypoints = 51  # Assuming 17 keypoints with x, y, z coordinates

    # Repeat i iteration to get the average result
    for i in range(iters):
        print(f"Iteration {i}")
        # Instantiate the model
        model = CNN(dataset_train[0][0].shape, n_keypoints)
        model = model.cuda() if torch.cuda.is_available() else model

        score_min = float('inf')

        # Define optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
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
            for batch_features, batch_labels in tqdm(train_loader):
                batch_features, batch_labels = batch_features.cuda(), batch_labels.cuda() if torch.cuda.is_available() else (batch_features, batch_labels)
                optimizer.zero_grad()

                with amp.autocast():
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
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
                    batch_features, batch_labels = batch_features.cuda(), batch_labels.cuda() if torch.cuda.is_available() else (batch_features, batch_labels)
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
        plt.savefig(output_direct + f"/acc-{i}.png")  
    
        # Plot loss  
        plt.figure(figsize=(10, 10))  
        plt.plot(train_loss_history)  
        plt.plot(val_loss_history)  
        plt.title('Model loss')  
        plt.ylabel('Loss')  
        plt.xlabel('Epoch')  
        plt.grid()  
        plt.legend(['Train', 'Validation'], loc='upper left')  
        plt.savefig(output_direct + f"/loss-{i}.png")  
    
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
