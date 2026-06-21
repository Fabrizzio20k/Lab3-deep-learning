import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_pretrained_hrencoder(embed_dim):
    try:
        import timm
        backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, in_chans=4, num_classes=0, global_pool="avg"
        )
        out_dim = backbone.num_features
        return backbone, out_dim
    except Exception:
        pass

    try:
        from torchvision.models import resnet18, ResNet18_Weights
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3] = old_conv.weight.mean(1)
        backbone.conv1 = new_conv
        backbone.fc = nn.Identity()
        return backbone, 512
    except Exception:
        return None, None


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
        backbone, out_dim = _make_pretrained_hrencoder(embed_dim)
        if backbone is not None:
            self.backbone = backbone
            self.use_pretrained = True
            print(f"HREncoder: using pretrained backbone (out_dim={out_dim})")
        else:
            self.backbone = nn.Sequential(
                ConvBlock(4, 64),
                ConvBlock(64, 128),
                ConvBlock(128, 256),
                ConvBlock(256, 512),
                ConvBlock(512, 512),
                nn.AdaptiveAvgPool2d(1),
            )
            out_dim = 512
            self.use_pretrained = False
            print("HREncoder: using custom CNN (no pretrained)")
        self.proj = nn.Sequential(
            nn.Linear(out_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        feat = self.backbone(x)
        if self.use_pretrained:
            return self.proj(feat)
        return self.proj(feat.flatten(1))

    def get_backbone_params(self):
        if self.use_pretrained:
            return list(self.backbone.parameters())
        return []

    def get_head_params(self):
        return list(self.proj.parameters())


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
    def __init__(self, embed_dim=256, num_classes=15, **kwargs):
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

    def get_param_groups(self, lr_backbone=1e-5, lr_head=3e-4):
        backbone_params = self.hr_enc.get_backbone_params()
        other_params = (
            self.hr_enc.get_head_params()
            + list(self.ts_enc.parameters())
            + list(self.head.parameters())
            + list(self.proj_head.parameters())
        )
        if backbone_params:
            return [
                {"params": backbone_params, "lr": lr_backbone},
                {"params": other_params, "lr": lr_head},
            ]
        return [{"params": other_params, "lr": lr_head}]
