import torch.nn as nn
import torch
from rep_conv import RepConv

class DetectHead(nn.Module):
    def __init__(self, in_channels=[64, 128, 256, 512], nc=5, stride=[4, 8, 16, 32]):
        super().__init__()
        self.nc = nc      # количество классов
        self.no = nc + 4  # + координаты box
        self.stride = stride
        
        self.heads = nn.ModuleList()
        for c in in_channels:
            self.heads.append(
                nn.Sequential(
                    RepConv(c, c, 3, 1, 1, act=True),
                    RepConv(c, c, 3, 1, 1, act=True),
                    nn.Conv2d(c, self.no, 1)
                )
            )

    def forward(self, feats):
        outputs = []
        for feat, head, stride in zip(feats, self.heads, self.stride):
            out = head(feat)
            B, _, H, W = out.shape
            out = out.view(B, self.no, H*W).permute(0, 2, 1)

            xy = out[..., :2].sigmoid() * stride
            wh = (out[..., 2:4].sigmoid() + 1e-8).log() * stride
            cls = out[..., 4:].sigmoid()
            out = torch.cat([xy, wh, cls], dim=-1)
            outputs.append(out)
        return torch.cat(outputs, 1)