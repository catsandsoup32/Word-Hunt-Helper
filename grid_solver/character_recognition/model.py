import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )

    def forward(self, x):
        return F.relu(self.conv(x))

class SmallCNN(nn.Module): # Short single layer conv blocks
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.b1 = ConvBlock(32, 64,  1)
        self.b2 = ConvBlock(64, 128, 2)
        self.b3 = ConvBlock(128, 128, 1)
        self.b4 = ConvBlock(128, 256, 2)
        self.b5 = ConvBlock(256, 256, 1)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_out = nn.Linear(256, 26)

    def forward(self, x):
        x = self.conv1(x)
        x = self.b5(self.b4(self.b3(self.b2(self.b1(x)))))
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_out(x)
        return x
    
def logits_to_class(pred: torch.Tensor):
    idx = torch.argmax(pred)
    return chr(ord('a') + idx)