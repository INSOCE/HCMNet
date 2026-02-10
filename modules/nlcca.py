import torch
import torch.nn as nn
import math


class NCA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(NCA, self).__init__()
        self.feature_channel = channel
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv_end = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.soft = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        zy = y.permute(0, 2, 1)
        zg = torch.matmul(zy, y)
        batch = zg.shape[0]
        v = zg.squeeze(-1).permute(1, 0).expand((self.feature_channel, batch))
        v = v.unsqueeze_(-1).permute(1, 2, 0)

        atten = self.conv(y.transpose(1, 2))
        atten = atten + v
        atten = self.conv_end(atten)
        atten = atten.transpose(1, 2)

        atten_score = self.soft(atten)
        return x * atten_score

class nlcca(nn.Module):
    def __init__(self, channel):
        super(nlcca, self).__init__()

        self.ca = NCA(channel=channel)
        self.dconv5 = nn.Conv1d(channel, channel, kernel_size=5, padding=2, groups=channel)
        self.dconv7 = nn.Conv1d(channel, channel, kernel_size=7, padding=3, groups=channel)
        self.dconv11 = nn.Conv1d(channel, channel, kernel_size=11, padding=5, groups=channel)
        self.dconv21 = nn.Conv1d(channel, channel, kernel_size=21, padding=10, groups=channel)
        self.conv = nn.Conv1d(channel, channel, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        x_c = self.ca(inputs)

        x_init = self.dconv5(x_c)
        x_1 = self.dconv7(x_init)
        x_2 = self.dconv11(x_init)
        x_3 = self.dconv21(x_init)

        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * x_c
        out = inputs + self.conv(out)
        return out
