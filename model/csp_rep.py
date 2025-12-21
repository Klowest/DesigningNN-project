from model.rep_conv import RepConv
import torch.nn as nn
import torch

class RepBlock(nn.Module):
    def __init__(self, c1, c2, n=1, act=True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.conv1 = RepConv(c1, c2, act=act)
        self.blocks = nn.Sequential(*[RepConv(c2, c2, act=act) for i in range(n-1)]) if n > 1 else None

    def forward(self, x):
        x1 = self.conv1(x)
        out = x1 if self.blocks is None else self.blocks(x1)
        out = out if self.blocks is None else out + x           # добавил скип
        return out

class CSPRepLayer(nn.Module):
    def __init__(self, c1, c2, n=1, act=True):
        super().__init__()
        self.act = act
        self.c2 = int(c2 * 0.5)

        self.relu = nn.ReLU()
        self.cv1 = nn.Conv2d(c1, self.c2, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(self.c2)

        self.cv2 = nn.Conv2d(c1, self.c2, 1, 1, 0)
        self.bn2 = nn.BatchNorm2d(self.c2)

        self.m = RepBlock(self.c2, self.c2, n=n, act=act)

        self.cv3 = nn.Conv2d(2 * self.c2, c2, 1, 1, 0)
        self.bn3 = nn.BatchNorm2d(c2)

    def forward(self, x):
        x1 = self.bn1(self.cv1(x))
        x1 = self.relu(x1) if (self.act) else x1

        x2 = self.bn2(self.cv2(x))
        x2 = self.relu(x2) if (self.act) else x2
        x2 = self.m(x2)

        out = self.bn3(self.cv3(torch.cat([x1, x2], 1)))
        out = self.relu(out) if (self.act) else out
        
        return out