import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置设备：如果有 MPS 则使用 MPS，否则使用 CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"当前使用设备: {device}")

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class Net(nn.Module):
    def __init__(self, width, height, n_res_blocks=10):
        super().__init__()
        self.width, self.height = width, height
        self.conv_block = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResBlock(64) for _ in range(n_res_blocks)])
        
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * width * height, width * height)
        
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.value_fc = nn.Sequential(
            nn.Linear(width * height, 64), 
            nn.ReLU(), 
            nn.Linear(64, 1), 
            nn.Tanh()
        )

    def forward(self, state):
        x = self.conv_block(state)
        x = self.res_blocks(x)
        
        p = self.policy_conv(x).view(-1, 2 * self.width * self.height)
        p = self.policy_fc(p)
        
        v = self.value_conv(x).view(-1, self.width * self.height)
        v = self.value_fc(v)
        return F.log_softmax(p, dim=1), v