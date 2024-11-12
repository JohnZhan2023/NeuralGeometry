import torch
import torch.nn as nn

class SimplifiedRasterModel(nn.Module):
    def __init__(self):
        super(SimplifiedRasterModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 通道数从32减到16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出尺寸：16x32x32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 通道数从64减到32
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 输出尺寸：32x16x16
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 16 * 16, 64),  # 输入尺寸减小
            nn.ReLU(),
            nn.Linear(64, 2),  # 输出类别数为2
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 32 * 16 * 16)
        x = self.fc_layers(x)
        return x
