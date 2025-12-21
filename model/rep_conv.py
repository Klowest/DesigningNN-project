import torch
import torch.nn as nn

class RepConv(nn.Module):
    def __init__(self, c1, c2, s=1, act=True):
        super().__init__()
        k = 3
        p = 1
        self.c1, self.c2, self.k, self.s, self.p = c1, c2, k, s, p
        self.act = nn.ReLU() if act else nn.Identity()
        
        self.rbr_dense = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2)
        )
        self.rbr_1x1 = nn.Sequential(
            nn.Conv2d(c1, c2, 1, s, 0, bias=False),
            nn.BatchNorm2d(c2)
        )
        if c1 == c2 and s == 1:
            self.rbr_identity = nn.BatchNorm2d(c2)
        else:
            self.rbr_identity = None

    def forward(self, x):
        if hasattr(self, "reparam_conv"):
            return self.act(self.reparam_conv(x))
        else:
            out = self.rbr_dense(x)
            out += self.rbr_1x1(x)
            if self.rbr_identity is not None:
                out += self.rbr_identity(x)
            return self.act(out)
    
    # преобразование трёх веток в одну
    def reparameterize(self):
        if hasattr(self, "reparam_conv"):
            return  # уже преобразован

        # 1. Получаем эквивалентные веса и смещения для каждой ветки
        def _fuse_conv_bn(conv, bn):
            w = conv.weight
            # b = torch.zeros_like(bn.running_mean)  # если bias не ноль, то использовать
            
            # Эквивалентный вес после BN: W' = γ / σ * W
            w_fused = w * (bn.weight / torch.sqrt(bn.running_var + bn.eps)).reshape(-1, 1, 1, 1)
            # Эквивалентный bias: b' = γ/σ * (b - μ) + β = -γ/σ * μ + β
            b_fused = bn.bias - bn.weight * bn.running_mean / torch.sqrt(bn.running_var + bn.eps)
            return w_fused, b_fused

        # Веса и смещения dense-ветки (3×3)
        w_dense, b_dense = _fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        # Размер: [c2, c1, 3, 3], [c2]

        # Веса и смещения 1×1-ветки
        w_1x1, b_1x1 = _fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        # Размер: [c2, c1, 1, 1], [c2]

        # Начинаем с нулевого ядра 3×3
        w_final = torch.zeros_like(w_dense)  # [c2, c1, 3, 3]
        b_final = b_dense.clone()            # [c2]

        # Добавляем dense ветку (она — основная)
        w_final += w_dense

        # Добавляем 1×1 в центр 3×3
        w_final[:, :, 1:2, 1:2] += w_1x1  # [c2, c1, 1, 1] → в позицию (1,1)

        # Добавляем identity-ветку (если есть)
        if self.rbr_identity is not None:
            # BN(identity) = affine: x ↦ γ/σ * x + (β - γ/σ * μ)
            # Это эквивалентно свёртке 1×1 с ядром = diag(γ/σ), но только если c1 == c2
            assert self.c1 == self.c2
            gamma = self.rbr_identity.weight
            beta = self.rbr_identity.bias
            mu = self.rbr_identity.running_mean
            var = self.rbr_identity.running_var
            eps = self.rbr_identity.eps

            scale = gamma / torch.sqrt(var + eps)  # [c2]
            bias_id = beta - scale * mu           # [c2]

            # Добавляем δ-импульс по диагонали: W[i, i, 1, 1] += scale[i]
            idx = torch.arange(self.c2)
            w_final[idx, idx, 1, 1] += scale
            b_final += bias_id

        else:
            b_final += b_1x1  # т.к. dense уже включён, осталось прибавить bias от 1x1

        # Создаём новый conv
        self.reparam_conv = nn.Conv2d(
            self.c1, self.c2, 3, self.s, 1, bias=True
        )
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final

        # Удаляем старые ветки (освобождаем память)
        for attr in ["rbr_dense", "rbr_1x1", "rbr_identity"]:
            delattr(self, attr)