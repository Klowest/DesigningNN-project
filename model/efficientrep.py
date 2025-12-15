import torch.nn as nn
from rep_conv import RepConv
from csp_rep import CSPRepLayer

class EfficientRep(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512], n=[1, 2, 4, 2], act=True):
        super().__init__()
        c1, c2, c3, c4 = channels
        n2, n3, n4, n5 = n

        self.stem = RepConv(3, c1, 3, 2, 1, act=act) # 640 -> 320
        self.stage1 = nn.Sequential(
            RepConv(c1, c1, 3, 2, 1, act=act),       # 320 -> 160
            CSPRepLayer(c1, c1, n=n2, act=act)
        )
        self.stage2 = nn.Sequential(
            RepConv(c1, c2, 3, 2, 1, act=act),       # 160 -> 80
            CSPRepLayer(c2, c2, n=n3, act=act)
        )
        self.stage3 = nn.Sequential(
            RepConv(c2, c3, 3, 2, 1, act=act),       # 80 -> 40
            CSPRepLayer(c3, c3, n=n4, act=act)
        )
        self.stage4 = nn.Sequential(
            RepConv(c3, c4, 3, 2, 1, act=act),       # 40 -> 20
            CSPRepLayer(c4, c4, n=n5, act=act)
        )

    def forward(self, x):
        x = self.stem(x)       # 320x320
        C2 = self.stage1(x)    # 160x160
        C3 = self.stage2(C2)   # 80x80
        C4 = self.stage3(C3)   # 40x40
        C5 = self.stage4(C4)   # 20x20
        return [C2, C3, C4, C5]