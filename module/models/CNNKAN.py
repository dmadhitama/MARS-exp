import torch
import torch.nn as nn
from module.layers.kan import KANLinear

class CNNKANSingleInput(nn.Module):  
    def __init__(self, in_shape, n_keypoints):  
        super(CNNKANSingleInput, self).__init__()  
        self.conv1 = nn.Conv2d(in_shape[0], 16, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  
        self.dropout = nn.Dropout(0.3)  
        self.layer_norm = nn.LayerNorm([32, in_shape[1], in_shape[2]])  # Replace BatchNorm2d
        self.flatten = nn.Flatten()  
        self.fc1 = KANLinear(32 * in_shape[1] * in_shape[2], 512)  
        self.fc2 = KANLinear(512, n_keypoints)  
        self.fc_ln = nn.LayerNorm(512)  # Replace BatchNorm1d
        self.fc_dropout = nn.Dropout(0.4)  
  
    def forward(self, x):
        # For radar data, we need to permute the dimensions
        if x.shape[1] == 14 and x.shape[3] == 5:
            x = x.permute(0, 3, 1, 2)  # Change from (B, H, W, C) to (B, C, H, W)
        x = torch.relu(self.conv1(x))  
        x = self.dropout(x)  
        x = torch.relu(self.conv2(x))  
        x = self.dropout(x)  
        x = self.layer_norm(x)  
        x = self.flatten(x)  
        x = torch.relu(self.fc1(x))  
        x = self.fc_ln(x)  
        x = self.fc_dropout(x)  
        x = self.fc2(x)  
        return x  
    
class CNNKANMultiInput(nn.Module):
    def __init__(self, rgb_shape, radar_shape, n_keypoints):
        super(CNNKANMultiInput, self).__init__()
        
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
        self.fc1 = KANLinear(combined_features, 512)
        self.fc2 = KANLinear(512, n_keypoints)
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