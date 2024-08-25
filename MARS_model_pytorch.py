import os  
import numpy as np  
import matplotlib.pyplot as plt  
import torch  
import torch.nn as nn  
import torch.optim as optim  
from torch.utils.data import DataLoader, TensorDataset  
from sklearn import metrics  
import pandas as pd
  
# Set the directory  
path = os.getcwd()  
os.chdir(path)  
  
# Load the feature and labels  
featuremap_train = np.load('dataset_release/mri_radar_rede/data_tr.npy')  
featuremap_validate = np.load('dataset_release/mri_radar_rede/data_tt.npy')  
featuremap_test = np.load('dataset_release/mri_radar_rede/data_tt.npy')  
labels_train = np.load('dataset_release/mri_radar_rede/labels_tr.npy')  
labels_validate = np.load('dataset_release/mri_radar_rede/labels_tt.npy')  
labels_test = np.load('dataset_release/mri_radar_rede/labels_tt.npy')  
  
# Convert numpy arrays to PyTorch tensors  
featuremap_train = torch.tensor(featuremap_train, dtype=torch.float32)  
featuremap_validate = torch.tensor(featuremap_validate, dtype=torch.float32)  
featuremap_test = torch.tensor(featuremap_test, dtype=torch.float32)  
labels_train = torch.tensor(labels_train, dtype=torch.float32)  
labels_validate = torch.tensor(labels_validate, dtype=torch.float32)  
labels_test = torch.tensor(labels_test, dtype=torch.float32)  
  
# Create DataLoader  
batch_size = 256  
train_dataset = TensorDataset(featuremap_train, labels_train)  
validate_dataset = TensorDataset(featuremap_validate, labels_validate)  
test_dataset = TensorDataset(featuremap_test, labels_test)  
  
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  
validate_loader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)  
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  
  
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
  
# Initialize the result array  
paper_result_list = []  
  
# Define epochs  
epochs = 150
iters = 10 
  
# Define the output directory  
output_direct = 'model_mri_pytorch/'  
if not os.path.exists(output_direct):  
    os.makedirs(output_direct)  
  
n_keypoints = 51  

  
# Repeat i iteration to get the average result  
for i in range(iters):  
    # Instantiate the model  
    model = CNN(featuremap_train[0].shape, n_keypoints)  
    model = model.cuda() if torch.cuda.is_available() else model  

    score_min = 10
  
    # Define optimizer and loss function  
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))  
    criterion = nn.MSELoss()  
  
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
        for batch_features, batch_labels in train_loader:  
            batch_features, batch_labels = batch_features.cuda(), batch_labels.cuda() if torch.cuda.is_available() else (batch_features, batch_labels)  
            optimizer.zero_grad()  
            outputs = model(batch_features)  
            loss = criterion(outputs, batch_labels)  
            loss.backward()  
            optimizer.step()  
            train_loss += loss.item()  
            train_mae += torch.mean(torch.abs(outputs - batch_labels)).item()  
  
        train_loss /= len(train_loader)  
        train_mae /= len(train_loader)  
        train_loss_history.append(train_loss)  
        train_mae_history.append(train_mae)  
  
        # Validation loop  
        model.eval()  
        val_loss = 0  
        val_mae = 0  
        with torch.no_grad():  
            for batch_features, batch_labels in validate_loader:  
                batch_features, batch_labels = batch_features.cuda(), batch_labels.cuda() if torch.cuda.is_available() else (batch_features, batch_labels)  
                outputs = model(batch_features)  
                val_loss += criterion(outputs, batch_labels).item()  
                val_mae += torch.mean(torch.abs(outputs - batch_labels)).item()  
  
        val_loss /= len(validate_loader)  
        val_mae /= len(validate_loader)  
        val_loss_history.append(val_loss)  
        val_mae_history.append(val_mae)  
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss}, Validation Loss: {val_loss}, Train MAE: {train_mae}, Validation MAE: {val_mae}')  
  
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
    print("mae for x is", metrics.mean_absolute_error(labels_test[:, 0:17], result_test[:, 0:17]))  
    print("mae for y is", metrics.mean_absolute_error(labels_test[:, 17:34], result_test[:, 17:34]))  
    print("mae for z is", metrics.mean_absolute_error(labels_test[:, 34:51], result_test[:, 34:51]))  
  
    # Matrix transformation for the final all 17 points mae  
    x_mae = metrics.mean_absolute_error(labels_test[:, 0:17], result_test[:, 0:17], multioutput='raw_values')  
    y_mae = metrics.mean_absolute_error(labels_test[:, 17:34], result_test[:, 17:34], multioutput='raw_values')  
    z_mae = metrics.mean_absolute_error(labels_test[:, 34:51], result_test[:, 34:51], multioutput='raw_values')  
  
    all_17_points_mae = np.concatenate((x_mae, y_mae, z_mae)).reshape(3, 17)  
    avg_17_points_mae = np.mean(all_17_points_mae, axis=0)  
    avg_17_points_mae_xyz = np.mean(all_17_points_mae, axis=1).reshape(1, 3)  
    all_17_points_mae_Transpose = all_17_points_mae.T  
  
    # Matrix transformation for the final all 17 points rmse  
    x_rmse = metrics.mean_squared_error(labels_test[:, 0:17], result_test[:, 0:17], multioutput='raw_values', squared=False)  
    y_rmse = metrics.mean_squared_error(labels_test[:, 17:34], result_test[:, 17:34], multioutput='raw_values', squared=False)  
    z_rmse = metrics.mean_squared_error(labels_test[:, 34:51], result_test[:, 34:51], multioutput='raw_values', squared=False)  
  
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