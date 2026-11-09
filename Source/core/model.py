"""Ban sao nguyen van ViT25DTriPlanar (Baseline 3) va TriPlanarViTClinicalFusion (v1/v2) tu notebook de app chay doc lap, khong can repo trifuse.
Dau vao: slices (B, T=27, C=3, 224, 224), tab (B, 7) hoac None."""
import torch
import torch.nn as nn
import timm


class TriPlanarViTClinicalFusion(nn.Module):
    """Mo hinh de xuat v1/v2: ViT 2.5D (anh MRI tri-planar) + du lieu lam sang,
    ket hop bang gated residual fusion (v2) hoac concat (v1).
    """

    def __init__(self, n_classes: int = 3, n_tab_features: int = 7, pretrained: bool = True,
                 backbone: str = "vit_small_patch16_224", dropout: float = 0.3,
                 fusion_mode: str = "gated", d_fusion: int = 256):
        super().__init__()
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, in_chans=3
        )
        d_img = self.encoder.num_features
        self.drop = nn.Dropout(dropout)
        self.fusion_mode = fusion_mode
        self.n_tab_features = n_tab_features

        self.img_proj = nn.Sequential(
            nn.Linear(d_img, d_fusion), nn.LayerNorm(d_fusion), nn.GELU()
        )
        self.tab_proj = nn.Sequential(
            nn.Linear(n_tab_features, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, d_fusion), nn.LayerNorm(d_fusion), nn.GELU()
        )

        if fusion_mode == "gated":
            self.gate = nn.Linear(2 * d_fusion, d_fusion)
            nn.init.constant_(self.gate.bias, -2.0)
            head_in = d_fusion
        else:
            head_in = 2 * d_fusion
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, n_classes))
        self.last_gate = None

    def freeze_backbone(self, freeze: bool = True):
        for p in self.encoder.parameters():
            p.requires_grad = not freeze

    def unfreeze_last_blocks(self, n: int):
        params = list(self.encoder.parameters())
        for p in params[-n:]:
            p.requires_grad = True

    def forward(self, slices, tab=None):
        B, T, C, H, W = slices.shape
        feats = self.encoder(slices.reshape(B * T, C, H, W))
        feats = feats.reshape(B, T, -1).mean(dim=1)

        if tab is None:
            tab = torch.zeros(B, self.n_tab_features, device=slices.device)

        zi = self.img_proj(feats)
        zt = self.tab_proj(tab)

        if self.fusion_mode == "gated":
            g = torch.sigmoid(self.gate(torch.cat([zi, zt], dim=-1)))
            z = zi + g * zt
            self.last_gate = g.detach()
        else:
            z = torch.cat([zi, zt], dim=-1)
            self.last_gate = None

        return self.head(self.drop(z))


class ViT25DTriPlanar(nn.Module):
    def __init__(self, n_tab_features=7, num_classes=3, pretrained=True, **kwargs):
        super().__init__()
        import timm
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
        in_features = self.backbone.num_features
        
        self.n_tab_features = n_tab_features
        if n_tab_features > 0:
            self.tab_mlp = nn.Sequential(
                nn.Linear(n_tab_features, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32)
            )
            fc_in = in_features + 32
        else:
            self.tab_mlp = None
            fc_in = in_features
            
        self.classifier = nn.Linear(fc_in, num_classes)

    def forward(self, x, tab=None):
        b = x.shape[0]
        
        if x.dim() == 5:
            b, n_views, c, h, w = x.shape
            x = x.view(-1, c, h, w)
            
            # Chia nhỏ 54 ảnh thành từng batch 9 ảnh để tránh OOM GPU
            chunk_size = 9
            feats_list = []
            for i in range(0, x.shape[0], chunk_size):
                chunk = x[i:i + chunk_size]
                feats_list.append(self.backbone(chunk))
                
            feats = torch.cat(feats_list, dim=0)
            feats = feats.view(b, n_views, -1).mean(dim=1)
        else:
            feats = self.backbone(x)
            
        if self.n_tab_features > 0:
            if tab is None:
                tab = torch.zeros(b, self.n_tab_features, device=x.device)
            tab_feats = self.tab_mlp(tab)
            feats = torch.cat([feats, tab_feats], dim=1)
            
        out = self.classifier(feats)
        return out


class TriPlanarViTImageOnly(TriPlanarViTClinicalFusion):
    """Doi chung cho v1/v2: kien truc, so tham so va cau hinh GIONG HET, nhung BO QUA dau vao lam sang (tab luon = 0).
    Cung ten thuoc tinh (encoder, img_proj, tab_proj, gate, head) nen trainer xu ly y het v2."""

    def forward(self, slices, tab=None):
        return super().forward(slices, None)
