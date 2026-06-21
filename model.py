import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class HREncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(4, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x):
        return self.proj(self.net(x).flatten(1))


class TSEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # x_ts: (B, 8, 10, 6, 6) → flatten spatial per timestep → (B, 8, 360)
        self.input_dim = 10 * 6 * 6
        self.gru = nn.GRU(self.input_dim, 512, num_layers=2, batch_first=True, dropout=0.2)
        self.proj = nn.Linear(512, embed_dim)

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 8, -1)
        _, h = self.gru(x)
        return self.proj(h[-1])


class ProjectionHead(nn.Module):
    def __init__(self, embed_dim=256, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ForestModel(nn.Module):
    def __init__(self, embed_dim=256, num_classes=15):
        super().__init__()
        self.hr_enc = HREncoder(embed_dim)
        self.ts_enc = TSEncoder(embed_dim)
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
        return F.softmax(self.head(torch.cat([hr_feat, ts_feat], dim=-1)), dim=-1)

    def forward_with_proj(self, x_hr):
        return self.proj_head(self.hr_enc(x_hr))
