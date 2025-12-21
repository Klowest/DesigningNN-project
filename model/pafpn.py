import torch.nn as nn
import torch.nn.functional as F
import torch
from model.rep_conv import RepConv
from model.csp_rep import CSPRepLayer

# в целом эта идея была в yolov3 (FPN - top-bottom), yolov4 PANet, далее в YOLOv6.. модификация - CSPRepLayer.
class EnhancedPAFPN(nn.Module):
    def __init__(self, channels=[64, 128, 256, 512], act=True):
        super().__init__()
        c2, c3, c4, c5 = channels

        self.act_fn = nn.ReLU() if act else nn.Identity()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        # уменьшение каналов
        self.cv5 = nn.Conv2d(c5, c4, 1)
        self.bn5 = nn.BatchNorm2d(c4)
        self.cv4 = nn.Conv2d(c4, c3, 1)
        self.bn4 = nn.BatchNorm2d(c3)
        self.cv3 = nn.Conv2d(c3, c2, 1)
        self.bn3 = nn.BatchNorm2d(c2)

        self.top_down4 = CSPRepLayer(c4 * 2, c4, n=2, act=act)  # были все 2
        self.top_down3 = CSPRepLayer(c3 * 2, c3, n=3, act=act)
        self.top_down2 = CSPRepLayer(c2 * 2, c2, n=3, act=act)

        # увеличение каналов
        self.cv_down3 = nn.Conv2d(c2, c3, 3, 2, 1)
        self.bn_down3 = nn.BatchNorm2d(c3)
        self.cv_down4 = nn.Conv2d(c3, c4, 3, 2, 1)
        self.bn_down4 = nn.BatchNorm2d(c4)
        self.cv_down5 = nn.Conv2d(c4, c5, 3, 2, 1)
        self.bn_down5 = nn.BatchNorm2d(c5)

        self.bottom_up3 = CSPRepLayer(c3 * 2, c3, n=3, act=act)
        self.bottom_up4 = CSPRepLayer(c4 * 2, c4, n=2, act=act)
        self.bottom_up5 = CSPRepLayer(c5 * 2, c5, n=1, act=act)

    def forward(self, feats):
        C2, C3, C4, C5 = feats

        # уменьшение каналов
        P5 = self.act_fn(self.bn5(self.cv5(C5)))    # 256
        P4 = self.top_down4(torch.cat([C4, self.up(P5)], dim=1))    # 256
        P4_reduced = self.act_fn(self.bn4(self.cv4(P4)))    # 128
        P3 = self.top_down3(torch.cat([C3, self.up(P4_reduced)], dim=1))    # 128

        P3_reduced = self.act_fn(self.bn3(self.cv3(P3)))    # 64
        P2 = self.top_down2(torch.cat([C2, self.up(P3_reduced)], dim=1))    # 64
        
        # увеличение каналов
        P2_down = self.act_fn(self.bn_down3(self.cv_down3(P2)))     # 128
        P3_out = self.bottom_up3(torch.cat([P3, P2_down], dim=1))   # 128

        P3_out_down = self.act_fn(self.bn_down4(self.cv_down4(P3_out))) # 256
        P4_out = self.bottom_up4(torch.cat([P4, P3_out_down], dim=1))   # 256

        P4_out_down = self.act_fn(self.bn_down5(self.cv_down5(P4_out))) # 512
        P5_out = self.bottom_up5(torch.cat([C5, P4_out_down], dim=1))   # 512
        
        return [P2, P3_out, P4_out, P5_out]

    # Более быстрый вариант:

    # P5_up = self.up(P5)  # 40×40
    # P4_enhanced = self.top_down4(torch.cat([C4, P5_up], 1))

    # P4_up = self.up(P4_enhanced)  # 80×80
    # P3_enhanced = self.top_down3(torch.cat([C3, P4_up], 1))

    # P3_up = self.up(P3_enhanced)  # 160×160
    # P2_enhanced = self.top_down2(torch.cat([C2, P3_up], 1))

    # # И добавить downsample для C5 → P5_out (хотя бы один шаг!)
    # P5_out = self.bottom_up5(torch.cat([P5, self.down(P4_enhanced)], 1))

    # return [P2_enhanced, P3_enhanced, P4_enhanced, P5_out]