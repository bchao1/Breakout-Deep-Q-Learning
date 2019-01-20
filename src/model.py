import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, action_size = 3):
        super(Network, self).__init__()
        
        self.action_size = action_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = 4, out_channels = 32, kernel_size = 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1),
            nn.ReLU(),
        )
        
        self.dense = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_size)
        )
    
    def forward(self, frame):
        h = self.conv(frame)
        h = h.view(h.shape[0], -1)
        return self.dense(h)