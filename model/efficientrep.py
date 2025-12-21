import torch.nn as nn
from model.rep_conv import RepConv
from model.csp_rep import CSPRepLayer

# выбор n делается таким, чтобы не терять много деталей на большои разрешении, а в остальном балансировать.
# Стадия 3 при этом важнее остальных, поэтому больше блоков
class EfficientRep(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512], n=[2, 3, 4, 2], act=True): # прибавил 1 к 0 и 1
        super().__init__()
        c1, c2, c3, c4 = channels
        n1, n2, n3, n4 = n

        self.stem = RepConv(3, c1, s=2, act=act) # 640 -> 320
        self.stage1 = nn.Sequential(
            RepConv(c1, c1, s=2, act=act),       # 320 -> 160
            CSPRepLayer(c1, c1, n=n1, act=act)
        )
        self.stage2 = nn.Sequential(
            RepConv(c1, c2, s=2, act=act),       # 160 -> 80
            CSPRepLayer(c2, c2, n=n2, act=act)
        )
        self.stage3 = nn.Sequential(
            RepConv(c2, c3, s=2, act=act),       # 80 -> 40
            CSPRepLayer(c3, c3, n=n3, act=act)
        )
        self.stage4 = nn.Sequential(
            RepConv(c3, c4, s=2, act=act),       # 40 -> 20
            CSPRepLayer(c4, c4, n=n4, act=act)
        )

    def forward(self, x):
        x = self.stem(x)       # 320x320
        C2 = self.stage1(x)    # 160x160
        C3 = self.stage2(C2)   # 80x80
        C4 = self.stage3(C3)   # 40x40
        C5 = self.stage4(C4)   # 20x20
        return [C2, C3, C4, C5]