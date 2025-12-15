from rep_conv import RepConv
import torch.nn as nn
import torch

class RepBlock(nn.Module):
    def __init__(self, c1, c2, n=1, act=True):
        super().__init__()
        self.conv1 = RepConv(c1, c2, act=act)
        self.blocks = nn.Sequential(*[RepConv(c2, c2, act=act) for i in range(n-1)]) if n > 1 else nn.Identity()

    def forward(self, x):
        return self.blocks(self.conv1(x))

class CSPRepLayer(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5, act=True):
        super().__init__()
        self.c2 = int(c2 * e)
        self.cv1 = RepConv(c1, self.c2, 1, 1, 0, act=act)
        self.cv2 = RepConv(c1, self.c2, 1, 1, 0, act=act)
        self.m = RepBlock(self.c2, self.c2, n=n, act=act)
        self.cv3 = RepConv(2 * self.c2, c2, 1, 1, 0, act=act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x2 = self.m(x2)
        return self.cv3(torch.cat([x1, x2], 1))