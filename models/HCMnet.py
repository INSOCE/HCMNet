from mamba_ssm import Mamba2
from modules.nlcca import nlcca
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class BidirectionalMamba2(nn.Module):
    def __init__(self, cin, cmid, cout, d_state=128):
        super().__init__()
        assert cmid % 64 == 0

        mamba_args = {
            'd_state': d_state,
            'd_conv': 4,
            'expand': 2,
            'headdim': 64,
            'chunk_size': 64
        }

        self.fc_in = nn.Linear(cin, cmid, bias=False)

        self.mamba2_for = Mamba2(d_model=cmid, **mamba_args)
        self.mamba2_back = Mamba2(d_model=cmid, **mamba_args)

        self.fc_out = nn.Linear(cmid, cout, bias=False)

    def forward(self, x):
        l = x.shape[2]

        pad_len = (64 - l % 64) % 64
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))

        x = rearrange(x, 'b c l -> b l c')

        x = self.fc_in(x)

        x1 = self.mamba2_for(x)
        if isinstance(x1, tuple):
            x1 = x1[0]

        x_back = x.flip(dims=[1])
        x2 = self.mamba2_back(x_back)
        if isinstance(x2, tuple):
            x2 = x2[0]
        x2 = x2.flip(dims=[1])

        x = x1 + x2

        x = self.fc_out(x)

        x = rearrange(x, 'b l c -> b c l')

        if pad_len > 0:
            x = x[:, :, :l]

        return x


class HCMnet(nn.Module):

    def __init__(self, num_classes, input_channels=1):
        super(HCMnet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=32, stride=1, padding=16),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Conv1d(32, 32, kernel_size=32, stride=1, padding=16),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=8, stride=1)
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=16, stride=1, padding=8),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Conv1d(64, 64, kernel_size=16, stride=1, padding=8),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=8, stride=1)
        )

        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2)
        )

        self.nlcca = nlcca(channel=128)

        self.mamba2 = BidirectionalMamba2(
            cin=128,
            cmid=256,
            cout=128,
            d_state=128
        )

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x_in):
        x = self.layer1(x_in)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.nlcca(x)
        x_res1 = x

        x = self.mamba2(x)

        x = x_res1 + x

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x_out = self.fc2(x)

        return x_out