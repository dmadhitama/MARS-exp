import torch
import torch.nn as nn

class CNNSingleInputMod(nn.Module):
    def __init__(self, in_shape, n_keypoints):
        super(CNNSingleInputMod, self).__init__()
        
        self.conv1 = nn.Conv2d(in_shape[0], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)
        
        self.flatten_size = 256 * (in_shape[1]//8) * (in_shape[2]//8)
        
        self.fc1 = nn.Linear(self.flatten_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, n_keypoints)
        
        self.fc_ln1 = nn.LayerNorm(1024)
        self.fc_ln2 = nn.LayerNorm(512)
        
        self.activation = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        # For radar data, we need to permute the dimensions
        if x.shape[1] == 14 and x.shape[3] == 5:
            x = x.permute(0, 3, 1, 2)  # Change from (B, H, W, C) to (B, C, H, W)
        
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = self.dropout2(x)
        
        x = self.activation(self.conv3(x))
        x = self.pool(x)
        x = self.dropout2(x)
        
        x = self.activation(self.conv4(x))
        x = self.dropout3(x)
        
        x = torch.flatten(x, 1)
        
        x = self.activation(self.fc1(x))
        x = self.fc_ln1(x)
        x = self.dropout3(x)
        
        x = self.activation(self.fc2(x))
        x = self.fc_ln2(x)
        x = self.dropout3(x)
        
        x = self.fc3(x)
        return x
    
class CNNMultiInputMod(nn.Module):
    def __init__(self, rgb_shape, radar_shape, n_keypoints):
        super(CNNMultiInputMod, self).__init__()
        
        # RGB branch
        self.rgb_conv1 = nn.Conv2d(rgb_shape[0], 32, kernel_size=3, padding=1)
        self.rgb_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.rgb_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.rgb_conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Radar branch
        self.radar_conv1 = nn.Conv2d(radar_shape[2], 32, kernel_size=3, padding=1)
        self.radar_conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.radar_conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.radar_conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.4)
        
        self.rgb_flatten_size = 256 * (rgb_shape[1]//8) * (rgb_shape[2]//8)
        self.radar_flatten_size = 256 * (radar_shape[0]//8) * (radar_shape[1]//8)
        
        combined_features = self.rgb_flatten_size + self.radar_flatten_size
        
        self.fc1 = nn.Linear(combined_features, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, n_keypoints)
        
        self.fc_ln1 = nn.LayerNorm(1024)
        self.fc_ln2 = nn.LayerNorm(512)
        
        self.activation = nn.LeakyReLU(0.1)
        
    def forward(self, rgb, radar):
        # RGB branch
        rgb = self.activation(self.rgb_conv1(rgb))
        rgb = self.pool(rgb)
        rgb = self.dropout1(rgb)
        
        rgb = self.activation(self.rgb_conv2(rgb))
        rgb = self.pool(rgb)
        rgb = self.dropout2(rgb)
        
        rgb = self.activation(self.rgb_conv3(rgb))
        rgb = self.pool(rgb)
        rgb = self.dropout2(rgb)
        
        rgb = self.activation(self.rgb_conv4(rgb))
        rgb = self.dropout3(rgb)
        
        rgb = torch.flatten(rgb, 1)
        
        # Radar branch
        radar = radar.permute(0, 3, 1, 2)  # Change from (B, H, W, C) to (B, C, H, W)
        radar = self.activation(self.radar_conv1(radar))
        radar = self.pool(radar)
        radar = self.dropout1(radar)
        
        radar = self.activation(self.radar_conv2(radar))
        radar = self.pool(radar)
        radar = self.dropout2(radar)
        
        radar = self.activation(self.radar_conv3(radar))
        radar = self.pool(radar)
        radar = self.dropout2(radar)
        
        radar = self.activation(self.radar_conv4(radar))
        radar = self.dropout3(radar)
        
        radar = torch.flatten(radar, 1)
        
        # Combine features
        combined = torch.cat((rgb, radar), dim=1)
        
        # Fully connected layers
        x = self.activation(self.fc1(combined))
        x = self.fc_ln1(x)
        x = self.dropout3(x)
        
        x = self.activation(self.fc2(x))
        x = self.fc_ln2(x)
        x = self.dropout3(x)
        
        x = self.fc3(x)
        return x