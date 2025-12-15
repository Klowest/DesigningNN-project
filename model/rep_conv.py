import torch
import torch.nn as nn

class RepConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, act=True):
        super().__init__()
        self.c1, self.c2, self.k, self.s, self.p = c1, c2, k, s, p
        self.groups = g
        self.act = nn.ReLU() if act else nn.Identity()
        
        self.rbr_dense = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False),
            nn.BatchNorm2d(c2)
        )
        self.rbr_1x1 = nn.Sequential(
            nn.Conv2d(c1, c2, 1, s, 0, groups=g, bias=False),
            nn.BatchNorm2d(c2)
        )
        if c1 == c2 and s == 1:
            self.rbr_identity = nn.BatchNorm2d(c2)
        else:
            self.rbr_identity = None

    def forward(self, x):
        out = self.rbr_dense(x)
        out += self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out += self.rbr_identity(x)
        return self.act(out)