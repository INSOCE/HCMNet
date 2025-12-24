from modules.mamba2 import Mamba2_1d
from modules.nlcca import nlcca
import torch.nn as nn


class HCMnet(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super(HCMnet, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Sequential(  # layer1
                nn.Conv1d(input_channels, 32, kernel_size=32, stride=1, padding=16),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.Conv1d(32, 32, kernel_size=32, stride=1, padding=16),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=8, stride=1)
            ),
            nn.Sequential(  # layer2
                nn.Conv1d(32, 64, kernel_size=16, stride=1, padding=8),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.Conv1d(64, 64, kernel_size=16, stride=1, padding=8),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=8, stride=1)
            ),
            nn.Sequential(  # layer3
                nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=4, stride=2)
            ),
            nlcca(channel=128),  # nlcca
            Mamba2_1d(128, 256, 128)  # mamba2

        )

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_in):
        x = self.feature_extractor[:-1](x_in)  # layer1, layer2, layer3, ncca
        x_res1 = x
        x = self.feature_extractor[-1](x)  # mamba2
        x = x_res1 + x

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        x_out = self.classifier(x)
        return x_out