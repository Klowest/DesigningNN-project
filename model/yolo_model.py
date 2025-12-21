import torch.nn as nn
import torch
from model.efficientrep import EfficientRep
from model.head import DetectHead
from model.pafpn import EnhancedPAFPN

class YOLOEfficient(nn.Module):
    def __init__(self, nc=5, imgsz=640):
        super().__init__()
        self.nc_external = nc       # сколько передано
        self.nc_internal = nc + 1   # с objects
        self.backbone = EfficientRep(channels=[64, 128, 256, 512], act=True)
        self.neck = EnhancedPAFPN(channels=[64, 128, 256, 512])
        self.head = DetectHead(
            imgsz=imgsz,
            in_channels=[64, 128, 256, 512],
            nc=self.nc_internal,
            stride=[4, 8, 16, 32]
        )

    def forward(self, x):
        feats = self.backbone(x)
        neck_feats = self.neck(feats)
        return self.head(neck_feats)

def create_yolo_model(nc=5, imgsz=640):
    model = YOLOEfficient(nc=nc, imgsz=imgsz)
    return model