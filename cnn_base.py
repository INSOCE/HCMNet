import torch
import torch.nn as nn

class cnn_base(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super(cnn_base, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, stride=1, padding=1),  # 修改输入通道为1
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.pool = nn.AdaptiveMaxPool1d(1)
        # 分类器：全连接层
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 特征提取
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        # 展平
        x = x.view(x.size(0), -1)  # 展平为一维

        # 分类
        x = self.fc1(x)
        x = self.fc2(x)
        return x