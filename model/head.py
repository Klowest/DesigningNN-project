import torch.nn as nn
import torch
from model.rep_conv import RepConv

class DetectHead(nn.Module):
    def __init__(self, imgsz, in_channels=[64, 128, 256, 512], nc=5, stride=[4, 8, 16, 32]):
        super().__init__()
        self.imgsz = imgsz
        self.nc = nc      # количество классов
        self.no = nc + 4  # + координаты box
        self.stride = stride
        self.grids = {}
        
        self.heads = nn.ModuleList()
        for c in in_channels:
            self.heads.append(
                nn.Sequential(
                    RepConv(c, c, s=1, act=True),
                    RepConv(c, c, s=1, act=True),
                    nn.Conv2d(c, self.no, 1)
                )
            )
        
        # инициализация маленькими значениями, чтобы не было больших значений по началу
        for head in self.heads:
            final_conv = head[-1]

            nn.init.constant_(final_conv.bias, 0.0)
            nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)

            b = final_conv.bias.data
            b[:4] = 0.0

    def forward(self, feats):
        outputs = []
        for feat, head, stride in zip(feats, self.heads, self.stride):
            out = head(feat)
            B, _, H, W = out.shape
            

            # сетка для перевода в абсолютные коодринаты
            if (H, W) not in self.grids or self.grids[(H, W)].device != out.device:
                gy, gx = torch.meshgrid(
                    torch.arange(H, device=out.device),
                    torch.arange(W, device=out.device),
                    indexing='ij'
                )
                self.grids[(H, W)] = torch.stack([gx, gy], dim=-1).float()  # [H, W, 2]
            grid = self.grids[(H, W)]

            reg = out[:, :4]    # [B, 4, H, W]
            cls = out[:, 4:]    # [B, nc, H, W] НЕТ сигмоиды, потому что предаолагается использование BCEWithLogitsLoss()

            reg = reg.permute(0, 2, 3, 1)  # [B, H, W, 4]
            cls = cls.permute(0, 2, 3, 1)  # [B, H, W, nc]

            xy = (reg[..., :2].sigmoid() + grid) * stride   # [B, H, W, 2]
            wh_raw = reg[..., 2:4].sigmoid() # [B, H, W, 2]
            wh_scale = min(10 * stride, self.imgsz / 2)
            wh = wh_raw * wh_scale
            
            out = torch.cat([xy, wh, cls], dim=-1)  # [B, H, W, no]
            out = out.reshape(B, -1, self.no)       # [B, H*W, no]
            outputs.append(out)

        return torch.cat(outputs, 1)