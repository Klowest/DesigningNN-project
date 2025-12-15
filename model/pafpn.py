import torch.nn as nn
import torch.nn.functional as F
import torch
from rep_conv import RepConv
from csp_rep import CSPRepLayer

class EnhancedPAFPN(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512], act=True):
        super().__init__()
        c2, c3, c4, c5 = channels
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.reduce5 = RepConv(c5, c4, 1, 1, 0, act=act)
        self.top_down4 = CSPRepLayer(c4*2, c4, n=1, act=act)
        
        self.reduce4 = RepConv(c4, c3, 1, 1, 0, act=act)
        self.top_down3 = CSPRepLayer(c3*2, c3, n=1, act=act)
        
        self.reduce3 = RepConv(c3, c2, 1, 1, 0, act=act)
        self.top_down2 = CSPRepLayer(c2*2, c2, n=1, act=act)

    def forward(self, feats):
        C2, C3, C4, C5 = feats

        P5 = self.reduce5(C5)
        P4 = self.top_down4(torch.cat([C4, self.up(P5)], 1))

        P4_reduced = self.reduce4(P4)
        P3 = self.top_down3(torch.cat([C3, self.up(P4_reduced)], 1))

        P3_reduced = self.reduce3(P3)
        P2 = self.top_down2(torch.cat([C2, self.up(P3_reduced)], 1))

        return [P2, P3, P4, C5]