import torch
from torch import nn


class TemporalBlock(nn.Module):
    def __init__(self, cin, cout):
        super(TemporalBlock, self).__init__()
        self.conv = nn.Conv2d(cin, cout, (7, 1), bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.bnf = nn.BatchNorm1d(cout)
        self.bnx = nn.BatchNorm1d(cout)
        self.bny = nn.BatchNorm1d(cout)

    def bn(self, x):
        x[:, :, :, 0] = self.bnf(x[:, :, :, 0])
        x[:, :, :, 1] = self.bnx(x[:, :, :, 1])
        x[:, :, :, 2] = self.bny(x[:, :, :, 2])
        return x

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class FusionBlock(nn.Module):
    def __init__(self, cin, cout):
        super(FusionBlock, self).__init__()
        self.conv = nn.Conv2d(cin, cout, (1, 3), bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Classifier(nn.Module):
    def __init__(self, cin):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(cin * 4, cin // 4)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(cin // 4, 4)

    def forward(self, x1, x2, x3, x4):
        x = torch.cat((x1, x2, x3, x4), dim=1)  # [B,256] -> [B,512]
        x = self.fc1(x)  # [B,512] -> [B,128]
        x = self.relu(x)
        x = self.fc2(x)  # [B,128] -> [B,2]
        return x


class PostLinker(nn.Module):
    def __init__(self):
        super(PostLinker, self).__init__()
        self.TemporalModule_1 = nn.Sequential(
            TemporalBlock(1, 8),
            TemporalBlock(8, 16),
            TemporalBlock(16, 32)
        )
        self.TemporalModule_2 = nn.Sequential(
            TemporalBlock(1, 8),
            TemporalBlock(8, 16),
            TemporalBlock(16, 32)
        )
        self.TemporalModule_3 = nn.Sequential(
            TemporalBlock(1, 8),
            TemporalBlock(8, 16),
            TemporalBlock(16, 32)
        )
        self.TemporalModule_4 = nn.Sequential(
            TemporalBlock(1, 8),
            TemporalBlock(8, 16),
            TemporalBlock(16, 32)
        )
        self.FusionBlock_1 = FusionBlock(32, 32)
        self.FusionBlock_2 = FusionBlock(32, 32)
        self.FusionBlock_3 = FusionBlock(32, 32)
        self.FusionBlock_4 = FusionBlock(32, 32)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = Classifier(32)

    def forward(self, x1, x2, x3, x4):
        x1 = x1[:, :, :, :3]
        x2 = x2[:, :, :, :3]
        x3 = x3[:, :, :, :3]
        x4 = x4[:, :, :, :3]
        x1 = self.TemporalModule_1(x1)
        x2 = self.TemporalModule_2(x2)
        x3 = self.TemporalModule_3(x3)
        x4 = self.TemporalModule_4(x4)
        x1 = self.FusionBlock_1(x1)
        x2 = self.FusionBlock_2(x2)
        x3 = self.FusionBlock_3(x3)
        x4 = self.FusionBlock_4(x4)
        x1 = self.pooling(x1).squeeze(-1).squeeze(-1)
        x2 = self.pooling(x2).squeeze(-1).squeeze(-1)
        x3 = self.pooling(x3).squeeze(-1).squeeze(-1)
        x4 = self.pooling(x4).squeeze(-1).squeeze(-1)
        y = self.classifier(x1, x2, x3, x4)
        if not self.training:
            y = torch.softmax(y, dim=1)
        return y


if __name__ == '__main__':
    x1 = torch.ones((2, 1, 30, 3))
    x2 = torch.ones((2, 1, 30, 3))
    m = PostLinker()
    y = m(x1, x2)
    print(y)
