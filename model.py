import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.net(x)


class HREncoder(nn.Module):
    def __init__(self, in_channels, embed_dim=256):
        super().__init__()
        self.blocks = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x):
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)


class TSEncoder(nn.Module):
    def __init__(self, in_channels, seq_len, embed_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x):
        if x.ndim == 3:
            x = x.permute(0, 2, 1)
        x = self.conv(x).squeeze(-1)
        return self.proj(x)


class ProjectionHead(nn.Module):
    def __init__(self, embed_dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ForestModel(nn.Module):
    def __init__(self, hr_channels, ts_channels, ts_len, embed_dim=256, num_classes=15):
        super().__init__()
        self.hr_enc = HREncoder(hr_channels, embed_dim)
        self.ts_enc = TSEncoder(ts_channels, ts_len, embed_dim)
        self.proj_head = ProjectionHead(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def encode_hr(self, x_hr):
        return self.hr_enc(x_hr)

    def forward(self, x_hr, x_ts):
        hr_feat = self.hr_enc(x_hr)
        ts_feat = self.ts_enc(x_ts)
        fused = torch.cat([hr_feat, ts_feat], dim=-1)
        logits = self.head(fused)
        return F.softmax(logits, dim=-1)

    def forward_with_proj(self, x_hr):
        feat = self.hr_enc(x_hr)
        return self.proj_head(feat)
