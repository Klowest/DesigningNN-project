import torch.nn as nn
import torch
from efficientrep import EfficientRep
from head import DetectHead
from pafpn import EnhancedPAFPN

class YOLOEfficient(nn.Module):
    def __init__(self, nc=5, imgsz=640):
        super().__init__()
        self.nc = nc
        self.backbone = EfficientRep(channels=[64, 128, 256, 512], act=True)
        self.neck = EnhancedPAFPN(channels=[64, 128, 256, 512])
        self.head = DetectHead(
            in_channels=[64, 128, 256, 512],
            nc=nc,
            stride=[4, 8, 16, 32]
        )

    def forward(self, x):
        feats = self.backbone(x)
        neck_feats = self.neck(feats)
        return self.head(neck_feats)

def create_yolo_model(nc=5, imgsz=640):
    model = YOLOEfficient(nc=nc, imgsz=imgsz)
    return model