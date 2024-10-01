import os
from loguru import logger
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.cuda.amp as amp

import numpy as np
import pandas as pd
import torch.optim as optim
from sklearn import metrics
import matplotlib.pyplot as plt

from module.models.CNNbaseline import CNNSingleInput, CNNMultiInput
from module.models.CNNmod import CNNSingleInputMod, CNNMultiInputMod
from module.models.CNNKAN import CNNKANSingleInput, CNNKANMultiInput

from helpers.preprocess import MARSDatasetChunked, ChunkSampler
from helpers.args import run_multipurpose

if __name__ == "__main__":
    # label_dirs = "/home/ubuntu/gdrive/workspace/dataset_release/aligned_data/pose_labels/"
    # video_dirs = "/home/ubuntu/gdrive/workspace/blurred_videos/"
    # radar_dirs = "/home/ubuntu/gdrive/workspace/dataset_release/features/radar/"
    # cache_dir = '/home/ubuntu/MARS-RGB-Radar-cache-npy'
    # output_direct = 'model_mri_mod-RGB-Radar-e200/'

    # input_type = "both"  # Can be "video", "radar", or "both"
    # kan = False

    # # Reduce batch size further if needed
    # batch_size = 64
    # accumulation_steps = 8
    # target_size = (160, 160) 
    # radar_size = (14, 14, 5)
    # chunk_size = 1000
    # num_workers = 8
    # prefetch_factor = 4

    # # Define epochs and iterations
    # epochs = 200
    # iters = 1

    (
        label_dirs, 
        video_dirs, 
        radar_dirs, 
        cache_dir, 
        output_direct, 
        input_type, 
        kan, 
        batch_size, 
        accumulation_steps,
        target_size, 
        radar_size, 
        chunk_size, 
        num_workers, 
        prefetch_factor, 
        epochs, 
        iters
    ) = run_multipurpose()

    # Set PYTORCH_CUDA_ALLOC_CONF
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    # Modify custom_collate function based on input_type
    def custom_collate(batch):
        frames, radars, labels = zip(*batch)
        labels = torch.stack(labels).float()
        if input_type == "video":
            frames = torch.stack(frames).float()
            return frames, labels
        elif input_type == "radar":
            radars = torch.stack(radars).float()
            return radars, labels
        else:  # both
            frames = torch.stack(frames).float()
            radars = torch.stack(radars).float()
            return frames, radars, labels

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
        # Instantiate the model based on input_type
        if input_type == "video":
            if kan:
                model = CNNKANSingleInput(rgb_shape, n_keypoints)
            else:
                # model = CNNSingleInput(rgb_shape, n_keypoints)
                model = CNNSingleInputMod(rgb_shape, n_keypoints)
        elif input_type == "radar":
            if kan:
                model = CNNKANSingleInput((radar_shape[2], radar_shape[0], radar_shape[1]), n_keypoints)
            else:
                # model = CNNSingleInput((radar_shape[2], radar_shape[0], radar_shape[1]), n_keypoints)
                model = CNNSingleInputMod((radar_shape[2], radar_shape[0], radar_shape[1]), n_keypoints)
        else:  # both
            if kan:
                model = CNNKANMultiInput(rgb_shape, radar_shape, n_keypoints)
            else:
                # model = CNNMultiInput(rgb_shape, radar_shape, n_keypoints)
                model = CNNMultiInputMod(rgb_shape, radar_shape, n_keypoints)
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
                for i, batch in enumerate(tqdm(train_loader)):
                    if input_type == "both":
                        batch_rgb, batch_radar, batch_labels = [b.cuda(non_blocking=True) if torch.cuda.is_available() else b for b in batch]
                    else:
                        batch_data, batch_labels = [b.cuda(non_blocking=True) if torch.cuda.is_available() else b for b in batch]
        
                    with amp.autocast():
                        if input_type == "both":
                            outputs = model(batch_rgb, batch_radar)
                        else:
                            outputs = model(batch_data)
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
                    for batch in tqdm(test_loader):
                        if input_type == "both":
                            batch_rgb, batch_radar, batch_labels = [b.cuda(non_blocking=True) if torch.cuda.is_available() else b for b in batch]
                        else:
                            batch_data, batch_labels = [b.cuda(non_blocking=True) if torch.cuda.is_available() else b for b in batch]
                        with amp.autocast():
                            if input_type == "both":
                                outputs = model(batch_rgb, batch_radar)
                            else:
                                outputs = model(batch_data)
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
            for batch in tqdm(test_loader, desc="Evaluating best model"):
                if input_type == "both":
                    batch_rgb, batch_radar, batch_labels = [b.cuda() if torch.cuda.is_available() else b for b in batch]
                    outputs = model(batch_rgb, batch_radar)
                else:
                    batch_data, batch_labels = [b.cuda() if torch.cuda.is_available() else b for b in batch]
                    outputs = model(batch_data)
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