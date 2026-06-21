import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class HREncoder(nn.Module):
    def __init__(self, embed_dim=256, backbone="efficientnet_b0"):
        super().__init__()
        feat_dims = {
            "efficientnet_b0": 1280,
            "resnet18": 512,
            "resnet50": 2048,
        }
        self.backbone = timm.create_model(
            backbone, pretrained=True, in_chans=4, num_classes=0, global_pool="avg"
        )
        self.proj = nn.Sequential(
            nn.Linear(feat_dims[backbone], embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        return self.proj(self.backbone(x))


class TSEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.spatial_enc = nn.Sequential(
            nn.Conv2d(10, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.gru = nn.GRU(64, 256, num_layers=2, batch_first=True, dropout=0.2)
        self.proj = nn.Linear(256, embed_dim)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.spatial_enc(x).flatten(1)
        x = x.view(B, T, 64)
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
    def __init__(self, embed_dim=256, num_classes=15, backbone="efficientnet_b0"):
        super().__init__()
        self.hr_enc = HREncoder(embed_dim, backbone)
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

    def get_param_groups(self, lr_backbone=1e-5, lr_head=3e-4):
        backbone_params = list(self.hr_enc.backbone.parameters())
        other_params = (
            list(self.hr_enc.proj.parameters())
            + list(self.ts_enc.parameters())
            + list(self.head.parameters())
            + list(self.proj_head.parameters())
        )
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": other_params, "lr": lr_head},
        ]
