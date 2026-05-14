%%writefile model.py
"""CrossModalChangeNet architecture"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, ic: int, oc: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
            nn.Dropout2d(dropout),
            nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(True),
        )
    def forward(self, x): return self.net(x)


class Down(nn.Module):
    def __init__(self, ic: int, oc: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(ic, oc, dropout))
    def forward(self, x): return self.net(x)


class Up(nn.Module):
    def __init__(self, ic: int, skip_c: int, oc: int, dropout: float = 0.1):
        super().__init__()
        self.up   = nn.ConvTranspose2d(ic, ic//2, 2, stride=2)
        self.conv = DoubleConv(ic//2 + skip_c, oc, dropout)
    def forward(self, x, skip):
        x  = self.up(x)
        dh = skip.size(2) - x.size(2); dw = skip.size(3) - x.size(3)
        x  = F.pad(x, [dw//2, dw-dw//2, dh//2, dh-dh//2])
        return self.conv(torch.cat([skip, x], 1))


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, hidden), nn.ReLU(True),
            nn.Linear(hidden, channels), nn.Sigmoid(),
        )
    def forward(self, x):
        return x * self.se(x).view(x.size(0), -1, 1, 1)


class FusionBlock(nn.Module):
    def __init__(self, eo_c: int, sar_c: int, out_c: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(eo_c + sar_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(True),
        )
        self.se = SEBlock(out_c)
    def forward(self, eo_feat, sar_feat):
        if eo_feat.shape[2:] != sar_feat.shape[2:]:
            sar_feat = F.interpolate(sar_feat, size=eo_feat.shape[2:],
                                     mode="bilinear", align_corners=False)
        fused = torch.cat([eo_feat, sar_feat], dim=1)
        return self.se(self.proj(fused))


class CrossModalChangeNet(nn.Module):
    def __init__(self, base_channels: int = 64, dropout: float = 0.1, in_channels: int = 4):
        super().__init__()
        c = base_channels

        self.eo_enc1 = DoubleConv(3,   c,    dropout)
        self.eo_enc2 = Down(c,    c*2, dropout)
        self.eo_enc3 = Down(c*2,  c*4, dropout)
        self.eo_enc4 = Down(c*4,  c*8, dropout)
        self.eo_bot  = Down(c*8, c*16, dropout)

        self.sar_enc1 = DoubleConv(1,   c//2, dropout)
        self.sar_enc2 = Down(c//2, c,   dropout)
        self.sar_enc3 = Down(c,    c*2, dropout)
        self.sar_enc4 = Down(c*2,  c*4, dropout)
        self.sar_bot  = Down(c*4,  c*8, dropout)

        self.fuse_bot  = FusionBlock(c*16, c*8, c*16)
        self.fuse_enc4 = FusionBlock(c*8,  c*4, c*8)
        self.fuse_enc3 = FusionBlock(c*4,  c*2, c*4)
        self.fuse_enc2 = FusionBlock(c*2,  c,   c*2)
        self.fuse_enc1 = FusionBlock(c,    c//2,c)

        self.dec4 = Up(c*16, c*8, c*8, dropout)
        self.dec3 = Up(c*8,  c*4, c*4, dropout)
        self.dec2 = Up(c*4,  c*2, c*2, dropout)
        self.dec1 = Up(c*2,  c,   c,   dropout)

        self.head = nn.Sequential(
            nn.Conv2d(c, c//2, 3, padding=1), nn.ReLU(True),
            nn.Dropout2d(dropout), nn.Conv2d(c//2, 1, 1),
        )

    def _encode_eo(self, x):
        e1 = self.eo_enc1(x); e2 = self.eo_enc2(e1)
        e3 = self.eo_enc3(e2); e4 = self.eo_enc4(e3)
        b  = self.eo_bot(e4)
        return e1, e2, e3, e4, b

    def _encode_sar(self, x):
        s1 = self.sar_enc1(x); s2 = self.sar_enc2(s1)
        s3 = self.sar_enc3(s2); s4 = self.sar_enc4(s3)
        sb = self.sar_bot(s4)
        return s1, s2, s3, s4, sb

    def forward(self, x):
        eo  = x[:, :3]; sar = x[:, 3:]
        e1, e2, e3, e4, eb = self._encode_eo(eo)
        s1, s2, s3, s4, sb = self._encode_sar(sar)
        fb = self.fuse_bot (eb, sb); f4 = self.fuse_enc4(e4, s4)
        f3 = self.fuse_enc3(e3, s3); f2 = self.fuse_enc2(e2, s2)
        f1 = self.fuse_enc1(e1, s1)
        d = self.dec4(fb, f4); d = self.dec3(d, f3)
        d = self.dec2(d, f2);  d = self.dec1(d, f1)
        return self.head(d)


def build_model(cfg: dict):
    return CrossModalChangeNet(
        base_channels=cfg.get("base_channels", 64),
        dropout=cfg.get("dropout", 0.1),
        in_channels=cfg.get("in_channels", 4),
    )

print("✓ model.py created")

