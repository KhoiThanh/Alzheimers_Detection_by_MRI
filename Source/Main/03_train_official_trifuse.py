#!/usr/bin/env python3
"""
03_train_official_trifuse.py
============================
Huấn luyện mô hình chính thức TriFuse-AD cho bài toán phân loại
3 giai đoạn Alzheimer (CN / VMD / AD) trên tập OASIS-1.

Kiến trúc TriFuse-AD:
  ┌─────────────────────────────────────────────────────┐
  │  27 lát cắt (9 Axial, 9 Coronal, 9 Sagittal)       │
  │       ↓  Shared ConvNeXt-Tiny Backbone              │
  │  27 token đặc trưng (768-dim)                       │
  │       + Plane Embedding (3 planes)                  │
  │       + Position Embedding (9 positions/plane)      │
  │  [CLS] token prepend                                │
  │       ↓  Transformer Encoder (2 layers)             │
  │  z_mri = output[CLS]  (768-dim)                     │
  └─────────────────────────────────────────────────────┘
             ↓
  ┌─────────────────────────┐
  │  7 clinical features    │
  │  TabularMLPBranch       │
  │  z_tab  (128-dim)       │
  └─────────────────────────┘
             ↓
  GatedResidualFusion:
    g = σ(Linear([z_mri ; z_tab]))   gate (768-dim)
    z_fused = LayerNorm(z_mri + g ⊙ W_t(z_tab))
             ↓
  Linear Classifier → 3 logits (CN / VMD / AD)

Protocol:
  • 5-Fold × 3 Seeds = 15 lượt chạy
  • AdamW (lr=1e-4, wd=1e-2) + CosineAnnealingLR
  • Class-Balanced Focal Loss (gamma=2) + label smoothing
  • AMP FP16 (torch.cuda.amp)
  • Checkpoint best Validation Macro-F1

Tác giả  : KLTN Project
Ngày tạo : 2026-10-03
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys
import json
import math
import copy
import gc
import inspect
import logging
import warnings
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.checkpoint import checkpoint

# AMP: forward-compatible import (torch >= 2.4 moved GradScaler)
try:
    from torch.amp import GradScaler, autocast          # PyTorch >= 2.4
except ImportError:
    from torch.cuda.amp import GradScaler, autocast     # PyTorch < 2.4

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix,
)

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# ĐƯỜNG DẪN & CẤU HÌNH TOÀN CỤC
# ─────────────────────────────────────────────────────────────────────────────
# Có thể chạy trên máy khác bằng TRIFUSE_DATA_ROOT. Thư mục này phải chứa:
#   processed_data\manifest.csv
#   processed_data\kfold_splits.json
#   processed_data\triplanar_2d\
BASE_DATA_DIR  = Path(os.environ.get(
    "TRIFUSE_DATA_ROOT",
    r"d:\Document\KLTN\raw\OASIS_DATA\data_processed\data",
))
TRIPLANAR_DIR  = BASE_DATA_DIR / "processed_data" / "triplanar_2d"
KFOLD_JSON     = BASE_DATA_DIR / "processed_data" / "kfold_splits.json"
MANIFEST_CSV   = BASE_DATA_DIR / "processed_data" / "manifest.csv"

MODEL_VARIANT  = os.environ.get("TRIFUSE_VARIANT", "gated").strip().lower()
if MODEL_VARIANT not in {
    "gated", "concat", "mri_only", "weighted_ce",
    "axial_only", "mean_pool", "no_metadata",
}:
    raise ValueError(
        "TRIFUSE_VARIANT must be one of: gated, concat, mri_only, "
        "weighted_ce, axial_only, mean_pool, no_metadata"
    )

OUTPUT_ROOT    = Path(os.environ.get(
    "TRIFUSE_OUTPUT_ROOT", r"d:\Document\KLTN\outputs_main"
))
CKPT_DIR       = OUTPUT_ROOT / "checkpoints" / "trifuse"
LOG_DIR        = OUTPUT_ROOT / "logs"
PLOT_DIR       = OUTPUT_ROOT / "plots"
RESULT_JSON    = OUTPUT_ROOT / "baseline_results.json"

def _read_resume_from() -> Optional[Tuple[int, int]]:
    value = os.environ.get("TRIFUSE_RESUME_FROM", "").strip()
    if not value:
        return None
    parts = [part.strip() for part in value.split(",", maxsplit=1)]
    if len(parts) != 2:
        raise ValueError(
            "TRIFUSE_RESUME_FROM must use the format 'seed,fold', "
            "for example '2024,4'"
        )
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(
            "TRIFUSE_RESUME_FROM must contain integer seed and fold"
        ) from exc


# Leave unset for a complete protocol. Set TRIFUSE_RESUME_FROM=seed,fold
# to continue from that checkpoint without editing this file.
RESUME_FROM = _read_resume_from()
# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    # Data
    "n_slices"          : 27,           # 9 Axial + 9 Coronal + 9 Sagittal
    "n_slices_per_plane": 9,
    "img_size"          : 224,
    "clinical_dim"      : 7,
    "num_classes"       : 3,

    # Architecture
    "backbone"          : "convnext_tiny",
    "mri_feat_dim"      : 768,          # ConvNeXt-Tiny output channels
    "tab_hidden_dims"   : [64, 128],    # 7 → 64 → 128
    "tab_out_dim"       : 128,
    "n_transformer_layers": 2,
    "n_attn_heads"      : 8,
    "transformer_dropout": 0.1,
    "mlp_dropout"       : 0.3,
    "fused_dim"         : 768,          # = mri_feat_dim (GatedResidualFusion output)
    "backbone_chunk_size": 1,           # Giảm đỉnh VRAM khi fine-tune 27 lát cắt

    # Training
    "seeds"             : [42, 123, 2024],
    "n_folds"           : 5,
    "epochs"            : 60,
    "batch_size"        : 1,            # 27 lát cắt/sample; phù hợp GPU 8 GB
    "gradient_accumulation_steps": 4,  # effective batch size = 4
    "lr"                : 1e-4,
    "weight_decay"      : 1e-2,
    "warmup_epochs"     : 5,
    "early_stop_patience": 15,
    "label_smoothing"   : 0.05,
    "focal_gamma"       : 2.0,
    "class_balanced_beta": 0.999,
    "backbone_freeze_epochs": 7,
    "augmentation_rotation_degrees": 7.0,
    "augmentation_contrast": 0.10,

    # Device
    "fp16"              : True,
    "num_workers"       : 0,            # Windows: 0 để tránh lỗi
    # Windows may fail to allocate page-locked buffers after long runs.
    # Non-blocking CUDA copies remain valid; disable pinning for reliable resume.
    "pin_memory"       : False,
    "model_variant"    : MODEL_VARIANT,
    "axial_only"       : MODEL_VARIANT == "axial_only",
    "mean_pool"        : MODEL_VARIANT == "mean_pool",
    "no_metadata"      : MODEL_VARIANT == "no_metadata",
    # Ablation theo bài báo: thay Class-Balanced Focal bằng Weighted CE.
    "weighted_ce"      : MODEL_VARIANT == "weighted_ce",
}

# Các override này phục vụ các đợt retrain có kiểm soát. Khi không đặt biến
# môi trường, cấu hình lịch sử phía trên vẫn được giữ nguyên.
CFG["transformer_dropout"] = float(os.environ.get(
    "TRIFUSE_TRANSFORMER_DROPOUT", CFG["transformer_dropout"]
))
CFG["mlp_dropout"] = float(os.environ.get(
    "TRIFUSE_MLP_DROPOUT", CFG["mlp_dropout"]
))
CFG["weight_decay"] = float(os.environ.get(
    "TRIFUSE_WEIGHT_DECAY", CFG["weight_decay"]
))
CFG["backbone_freeze_epochs"] = int(os.environ.get(
    "TRIFUSE_FREEZE_EPOCHS", CFG["backbone_freeze_epochs"]
))
CFG["pretrained_backbone"] = os.environ.get(
    "TRIFUSE_PRETRAINED_BACKBONE", "1"
).strip().lower() not in {"0", "false", "no", "off"}
CFG["fp16"] = os.environ.get(
    "TRIFUSE_FP16", "1"
).strip().lower() not in {"0", "false", "no", "off"}

CLINICAL_FEATURES = ['age', 'sex', 'education', 'ses', 'etiv', 'nwbv', 'asf']
CLASS_NAMES = ['CN', 'VMD', 'AD']

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
def setup_logger(name: str, log_file: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    # File handler
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# JSON ENCODER (numpy-safe)
# ─────────────────────────────────────────────────────────────────────────────
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)


def build_autocast_context(device: torch.device, enabled: bool):
    """Compatibility wrapper for torch.autocast API across PyTorch versions."""
    try:
        params = inspect.signature(autocast).parameters
    except (TypeError, ValueError):
        return autocast(enabled=enabled)

    kwargs = {"enabled": enabled}
    if "device_type" in params:
        kwargs["device_type"] = device.type
    return autocast(**kwargs)


def build_grad_scaler(device: torch.device, enabled: bool):
    """Compatibility wrapper for GradScaler across PyTorch versions."""
    try:
        params = inspect.signature(GradScaler).parameters
    except (TypeError, ValueError):
        return GradScaler(enabled=enabled)

    if "device" in params:
        return GradScaler(device=device.type, enabled=enabled)
    return GradScaler(enabled=enabled)


# ─────────────────────────────────────────────────────────────────────────────# DATASET
# ─────────────────────────────────────────────────────────────────────────────
class TriFuseDataset(Dataset):
    """
    Dataset cho TriFuse-AD.

    Mỗi sample trả về:
        mri_slices  : Tensor (27, 3, 224, 224) – 27 lát cắt RGB (grayscale replicated)
        clinical    : Tensor (7,)               – 7 đặc trưng lâm sàng đã chuẩn hóa
        label       : Tensor scalar int64
    """

    def __init__(
        self,
        subject_ids: List[str],
        labels: List[int],
        clinical_features: np.ndarray,
        triplanar_dir: Path,
        augment: bool = False,
    ):
        self.subject_ids = subject_ids
        self.labels      = labels
        self.clinical    = torch.tensor(clinical_features, dtype=torch.float32)
        self.triplanar_dir = triplanar_dir
        self.augment     = augment
        self.img_size    = CFG["img_size"]

    def __len__(self) -> int:
        return len(self.subject_ids)

    def _load_triplanar(self, subject_id: str) -> torch.Tensor:
        """
        Tải file .npy shape (27, 224, 224) float32, chuyển thành (27, 3, 224, 224).
        Grayscale → RGB bằng cách replicate channel.
        """
        npy_path = self.triplanar_dir / f"{subject_id}.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"Không tìm thấy: {npy_path}")

        slices = np.load(str(npy_path), mmap_mode="r")  # (27, H, W)
        assert slices.shape == (27, self.img_size, self.img_size), \
            f"Shape sai: {slices.shape}"

        # Keep the mmap read-only and normalize in Torch to avoid an extra
        # writable NumPy copy at the DataLoader boundary.
        slices_tensor = torch.from_numpy(slices).float()
        s_min = slices_tensor.amin(dim=(1, 2), keepdim=True)
        s_max = slices_tensor.amax(dim=(1, 2), keepdim=True)
        denom = (s_max - s_min).clamp_min(1e-8)
        normalized = (slices_tensor - s_min) / denom
        del slices, slices_tensor, s_min, s_max, denom

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
            1, 3, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
            1, 3, 1, 1
        )

        # (27, H, W) → (27, 3, H, W) – replicate channel
        tensor = normalized.unsqueeze(1).expand(-1, 3, -1, -1)
        tensor = (tensor - mean) / std
        del normalized

        if self.augment:
            tensor = self._augment(tensor)

        return tensor

    def _augment(self, tensor: torch.Tensor) -> torch.Tensor:
        """Augmentation nhẹ: xoay ±7° và chỉnh tương phản, không lật ảnh."""
        max_angle = CFG["augmentation_rotation_degrees"]
        angle = (torch.rand(1).item() * 2.0 - 1.0) * max_angle
        radians = math.radians(angle)
        affine = tensor.new_tensor([
            [math.cos(radians), -math.sin(radians), 0.0],
            [math.sin(radians), math.cos(radians), 0.0],
        ]).unsqueeze(0).expand(tensor.shape[0], -1, -1)
        grid = F.affine_grid(
            affine, tensor.size(), align_corners=False
        )
        tensor = F.grid_sample(
            tensor, grid, mode="bilinear", padding_mode="border",
            align_corners=False,
        )
        contrast = 1.0 + (
            (torch.rand(1).item() * 2.0 - 1.0)
            * CFG["augmentation_contrast"]
        )
        return tensor * contrast

    def __getitem__(self, idx: int):
        mri = self._load_triplanar(self.subject_ids[idx])

        label    = torch.tensor(self.labels[idx], dtype=torch.long)
        clinical = self.clinical[idx]

        return mri, clinical, label


# ─────────────────────────────────────────────────────────────────────────────
# MODEL: BACKBONE (ConvNeXt-Tiny)
# ─────────────────────────────────────────────────────────────────────────────
class ConvNeXtTinyBackbone(nn.Module):
    """
    Shared ConvNeXt-Tiny feature extractor.
    Input : (B, 3, 224, 224)
    Output: (B, 768)   – Global Average Pooled feature vector
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        if TIMM_AVAILABLE:
            # Dùng timm nếu có
            self.backbone = timm.create_model(
                "convnext_tiny",
                pretrained=pretrained,
                num_classes=0,          # remove classifier head
                global_pool="avg",      # GAP → (B, 768)
            )
            self.feat_dim = self.backbone.num_features   # 768
        else:
            # Fallback: torchvision ConvNeXt-Tiny
            try:
                from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
                weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
                _model  = convnext_tiny(weights=weights)
                # Bỏ avgpool + classifier, tự thêm GAP
                self.backbone = nn.Sequential(
                    _model.features,
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(1),
                )
                self.feat_dim = 768
            except ImportError:
                raise ImportError(
                    "Cần cài timm hoặc torchvision >= 0.12. "
                    "Chạy: pip install timm"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) → (B, feat_dim)"""
        return self.backbone(x)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL: SlicePlaneTransformer
# ─────────────────────────────────────────────────────────────────────────────
class SlicePlaneTransformer(nn.Module):
    """
    SlicePlaneTransformer – xử lý 27 lát cắt tri-planar qua Transformer.

    Workflow:
      1. Shared ConvNeXt-Tiny extract 27 token vectors (768-dim)
      2. Cộng Plane Embedding + Position Embedding (learnable)
      3. Prepend [CLS] token
      4. Transformer Encoder 2 lớp
      5. Output: z_mri = CLS token (768-dim)
    """

    def __init__(
        self,
        feat_dim: int = 768,
        n_slices: int = 27,
        n_planes: int = 3,
        n_pos_per_plane: int = 9,
        n_transformer_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        pretrained_backbone: bool = True,
        backbone_chunk_size: int = 1,
        axial_only: bool = False,
        mean_pool: bool = False,
    ):
        super().__init__()

        self.feat_dim        = feat_dim
        self.n_slices        = n_slices
        self.n_planes        = n_planes
        self.n_pos_per_plane = n_pos_per_plane
        self.backbone_chunk_size = max(1, backbone_chunk_size)
        self.axial_only = axial_only
        self.mean_pool = mean_pool

        # ── Shared Backbone ───────────────────────────────────────────────────
        self.backbone = ConvNeXtTinyBackbone(pretrained=pretrained_backbone)
        assert self.backbone.feat_dim == feat_dim, \
            f"backbone.feat_dim={self.backbone.feat_dim} ≠ feat_dim={feat_dim}"

        # ── Embeddings ────────────────────────────────────────────────────────
        # Plane embedding: 0=Axial, 1=Coronal, 2=Sagittal
        self.plane_embedding = nn.Embedding(n_planes, feat_dim)
        # Position embedding: 0..8 (vị trí lát cắt trong mặt phẳng)
        self.pos_embedding   = nn.Embedding(n_pos_per_plane, feat_dim)

        # ── [CLS] token (learnable) ───────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Transformer Encoder ───────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=n_heads,
            dim_feedforward=feat_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,            # Pre-LN (ổn định hơn)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers,
            norm=nn.LayerNorm(feat_dim),
        )

        # Dropout cuối
        self.dropout = nn.Dropout(dropout)

    def _build_plane_ids(self, device) -> torch.Tensor:
        """
        Tạo plane_id cho 27 lát cắt:
          [0]*9 (Axial) + [1]*9 (Coronal) + [2]*9 (Sagittal)
        """
        return torch.tensor(
            [0] * 9 + [1] * 9 + [2] * 9, dtype=torch.long, device=device
        )

    def _build_pos_ids(self, device) -> torch.Tensor:
        """
        Tạo pos_id cho 27 lát cắt: [0..8, 0..8, 0..8]
        """
        return torch.tensor(
            list(range(9)) * 3, dtype=torch.long, device=device
        )

    def forward(self, slices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slices: (B, 27, 3, H, W)
        Returns:
            z_mri : (B, feat_dim)  – CLS token output
        """
        B, N, C, H, W = slices.shape
        assert N == self.n_slices, f"Expected {self.n_slices} slices, got {N}"
        if self.axial_only:
            slices = slices[:, :9]
            N = 9

        # ── 1. Backbone: xử lý từng lát cắt ──────────────────────────────────
        # Reshape: (B*27, 3, H, W) → backbone → (B*27, 768) → (B, 27, 768)
        slices_flat = slices.reshape(B * N, C, H, W)

        # Process small chunks and checkpoint each chunk to keep fine-tuning
        # memory bounded on GPUs that cannot hold all 27 slice activations.
        if self.training:
            token_chunks = [
                checkpoint(
                    self.backbone,
                    chunk,
                    use_reentrant=False,
                )
                for chunk in slices_flat.split(self.backbone_chunk_size, dim=0)
            ]
        else:
            token_chunks = [
                self.backbone(chunk)
                for chunk in slices_flat.split(self.backbone_chunk_size, dim=0)
            ]
        tokens = torch.cat(token_chunks, dim=0)      # (B*27, 768)
        tokens = tokens.view(B, N, self.feat_dim)    # (B, 27, 768)

        # ── 2. Plane + Position Embedding ─────────────────────────────────────
        plane_ids = self._build_plane_ids(slices.device)[:N]
        pos_ids   = self._build_pos_ids(slices.device)[:N]

        plane_emb = self.plane_embedding(plane_ids)    # (27, 768)
        pos_emb   = self.pos_embedding(pos_ids)        # (27, 768)

        tokens = tokens + plane_emb.unsqueeze(0) + pos_emb.unsqueeze(0)
        tokens = self.dropout(tokens)

        # Mean pooling ablation: bỏ CLS token và Transformer.
        if self.mean_pool:
            return tokens.mean(dim=1)

        # ── 3. Prepend [CLS] token ────────────────────────────────────────────
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, 768)
        tokens     = torch.cat([cls_tokens, tokens], dim=1)  # (B, 28, 768)

        # ── 4. Transformer Encoder ────────────────────────────────────────────
        out = self.transformer(tokens)   # (B, 28, 768)

        # ── 5. Lấy CLS output ────────────────────────────────────────────────
        z_mri = out[:, 0, :]             # (B, 768)
        return z_mri


# ─────────────────────────────────────────────────────────────────────────────
# MODEL: TabularMLPBranch
# ─────────────────────────────────────────────────────────────────────────────
class TabularMLPBranch(nn.Module):
    """
    MLP Branch xử lý 7 đặc trưng lâm sàng → z_tab (128-dim).

    Architecture: 7 → 64 → 128
    Mỗi lớp: Linear → LayerNorm → GELU → Dropout
    """

    def __init__(
        self,
        in_dim: int = 7,
        hidden_dims: List[int] = [64, 128],
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev   = in_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h

        self.network = nn.Sequential(*layers)
        self.out_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 7) → (B, 128)"""
        return self.network(x)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL: GatedResidualFusion
# ─────────────────────────────────────────────────────────────────────────────
class GatedResidualFusion(nn.Module):
    """
    Gated Residual Fusion Layer.

    Công thức:
        gate   = σ( Linear([z_mri ; z_tab], out_dim=mri_dim) )
        z_tab' = W_t(z_tab)      # project 128 → 768
        z_fused = LayerNorm( z_mri + gate ⊙ z_tab' )

    Thiết kế theo nghiên cứu về multimodal fusion với soft gating:
    gate học cách điều tiết mức độ ảnh hưởng của thông tin lâm sàng
    lên đặc trưng MRI – đặc biệt phù hợp khi dữ liệu lâm sàng có nhiễu.
    """

    def __init__(self, mri_dim: int = 768, tab_dim: int = 128):
        super().__init__()

        # Projection: tab_dim → mri_dim
        self.W_t = nn.Linear(tab_dim, mri_dim, bias=False)

        # Gate network: concat [z_mri(768) ; z_tab(128)] → gate(768)
        self.gate_net = nn.Sequential(
            nn.Linear(mri_dim + tab_dim, mri_dim),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(mri_dim)

    def forward(
        self, z_mri: torch.Tensor, z_tab: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z_mri : (B, 768)
            z_tab : (B, 128)
        Returns:
            z_fused : (B, 768)
        """
        # Gate
        gate    = self.gate_net(torch.cat([z_mri, z_tab], dim=-1))  # (B, 768)

        # Project tab to mri space
        z_tab_p = self.W_t(z_tab)                                   # (B, 768)

        # Gated residual
        z_fused = self.norm(z_mri + gate * z_tab_p)                 # (B, 768)
        return z_fused


# ─────────────────────────────────────────────────────────────────────────────
# MODEL: TriFuse-AD (Full Model)
# ─────────────────────────────────────────────────────────────────────────────
class TriFuseAD(nn.Module):
    """
    TriFuse-AD: Mô hình chính thức phân loại Alzheimer 3 giai đoạn.

    Forward Input:
        mri_slices  : (B, 27, 3, 224, 224)
        clinical    : (B, 7)
    Forward Output:
        logits      : (B, 3)
    """

    def __init__(self, cfg: dict = CFG):
        super().__init__()

        self.mri_branch = SlicePlaneTransformer(
            feat_dim             = cfg["mri_feat_dim"],
            n_slices             = cfg["n_slices"],
            n_planes             = 3,
            n_pos_per_plane      = cfg["n_slices_per_plane"],
            n_transformer_layers = cfg["n_transformer_layers"],
            n_heads              = cfg["n_attn_heads"],
            dropout              = cfg["transformer_dropout"],
            pretrained_backbone  = cfg.get("pretrained_backbone", True),
            backbone_chunk_size  = cfg["backbone_chunk_size"],
            axial_only           = cfg.get("axial_only", False),
            mean_pool             = cfg.get("mean_pool", False),
        )

        self.model_variant = cfg.get("model_variant", "gated")
        uses_metadata = self.model_variant not in {"mri_only", "no_metadata"}
        if uses_metadata:
            self.tab_branch = TabularMLPBranch(
                in_dim      = cfg["clinical_dim"],
                hidden_dims = cfg["tab_hidden_dims"],
                dropout     = cfg["mlp_dropout"],
            )
        else:
            self.tab_branch = None

        if self.model_variant in {"gated", "weighted_ce", "axial_only", "mean_pool"}:
            self.fusion = GatedResidualFusion(
                mri_dim = cfg["mri_feat_dim"],
                tab_dim = cfg["tab_out_dim"],
            )
            classifier_dim = cfg["mri_feat_dim"]
        elif self.model_variant == "concat":
            self.fusion = None
            classifier_dim = cfg["mri_feat_dim"] + cfg["tab_out_dim"]
        else:
            self.fusion = None
            classifier_dim = cfg["mri_feat_dim"]

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Dropout(cfg["mlp_dropout"]),
            nn.Linear(classifier_dim, cfg["num_classes"]),
        )

        self._init_weights()

    def _init_weights(self):
        """Khởi tạo trọng số cho các lớp linear mới (không phải pretrained)."""
        modules = [self.tab_branch, self.fusion, self.classifier]
        for m in [module for module in modules if module is not None]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        if self.fusion is not None:
            nn.init.constant_(self.fusion.gate_net[0].bias, -2.0)

    def forward(
        self,
        mri_slices: torch.Tensor,
        clinical: torch.Tensor,
    ) -> torch.Tensor:
        """
        mri_slices : (B, 27, 3, 224, 224)
        clinical   : (B, 7)
        → logits   : (B, 3)
        """
        z_mri   = self.mri_branch(mri_slices)    # (B, 768)
        if self.model_variant in {"mri_only", "no_metadata"}:
            representation = z_mri
        else:
            z_tab = self.tab_branch(clinical)     # (B, 128)
            if self.model_variant in {"gated", "weighted_ce", "axial_only", "mean_pool"}:
                representation = self.fusion(z_mri, z_tab)
            else:
                representation = torch.cat([z_mri, z_tab], dim=-1)
        logits = self.classifier(representation)
        return logits

    def get_feature_vectors(
        self,
        mri_slices: torch.Tensor,
        clinical: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Trả về intermediate vectors để phân tích / visualization."""
        z_mri   = self.mri_branch(mri_slices)
        if self.model_variant == "mri_only":
            return {"z_mri": z_mri, "z_fused": z_mri}
        z_tab = self.tab_branch(clinical)
        z_fused = (
            self.fusion(z_mri, z_tab)
            if self.model_variant in {"gated", "weighted_ce", "axial_only", "mean_pool"}
            else torch.cat([z_mri, z_tab], dim=-1)
        )
        return {"z_mri": z_mri, "z_tab": z_tab, "z_fused": z_fused}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING UTILS
# ─────────────────────────────────────────────────────────────────────────────

def load_manifest() -> pd.DataFrame:
    """Tải manifest.csv đã được tạo bởi 01_preprocess_and_split.py."""
    if not MANIFEST_CSV.exists():
        raise FileNotFoundError(
            f"Không tìm thấy manifest: {MANIFEST_CSV}\n"
            "Hãy chạy 01_preprocess_and_split.py trước."
        )
    df = pd.read_csv(MANIFEST_CSV)
    df.columns = df.columns.str.lower().str.strip()
    return df


def load_kfold_splits() -> dict:
    """Tải kfold_splits.json đã được tạo bởi 01_preprocess_and_split.py."""
    if not KFOLD_JSON.exists():
        raise FileNotFoundError(
            f"Không tìm thấy: {KFOLD_JSON}\n"
            "Hãy chạy 01_preprocess_and_split.py trước."
        )
    with open(KFOLD_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_fold_data(
    df: pd.DataFrame,
    splits: dict,
    fold_idx: int,
    seed: int,
    logger: logging.Logger,
) -> Tuple[TriFuseDataset, TriFuseDataset, np.ndarray]:
    """
    Chuẩn bị dữ liệu cho một fold cụ thể.

    Anti-Leakage: StandardScaler chỉ fit trên train fold.

    Returns:
        train_ds       : TriFuseDataset (train)
        val_ds         : TriFuseDataset (val)
        class_weights  : np.ndarray (3,) – dùng cho loss function
    """
    fold_key  = f"fold_{fold_idx}"
    fold_info = splits["folds"][fold_key]

    train_subjects = set(fold_info["train"]["subjects"])
    val_subjects   = set(fold_info["val"]["subjects"])

    # Kiểm tra anti-leakage
    overlap = train_subjects & val_subjects
    if overlap:
        raise RuntimeError(f"Rò rỉ dữ liệu fold {fold_idx}: {overlap}")

    train_df = df[df["subject_id"].isin(train_subjects)].reset_index(drop=True)
    val_df   = df[df["subject_id"].isin(val_subjects)  ].reset_index(drop=True)

    logger.info(f"    Fold {fold_idx}: Train={len(train_df)}, Val={len(val_df)}")

    # ── Impute NaN (ses) – fit trên train ──────────────────────────────────
    imputer = SimpleImputer(strategy="median")
    X_train_raw = train_df[CLINICAL_FEATURES].values
    X_val_raw   = val_df  [CLINICAL_FEATURES].values

    X_train_imp = imputer.fit_transform(X_train_raw)
    X_val_imp   = imputer.transform(X_val_raw)

    # ── StandardScaler – fit trên train ────────────────────────────────────
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_imp).astype(np.float32)
    X_val_s   = scaler.transform(X_val_imp    ).astype(np.float32)

    y_train = train_df["label"].values.astype(np.int64)
    y_val   = val_df  ["label"].values.astype(np.int64)

    # ── Class weights (từ train fold) ──────────────────────────────────────
    classes       = np.array([0, 1, 2])
    cw            = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weights = cw.astype(np.float32)
    logger.info(
        f"    Class weights: CN={cw[0]:.3f}, VMD={cw[1]:.3f}, AD={cw[2]:.3f}"
    )

    # ── Datasets ─────────────────────────────────────────────────────────
    train_ds = TriFuseDataset(
        subject_ids    = train_df["subject_id"].tolist(),
        labels         = y_train.tolist(),
        clinical_features = X_train_s,
        triplanar_dir  = TRIPLANAR_DIR,
        augment        = True,
    )
    val_ds = TriFuseDataset(
        subject_ids    = val_df["subject_id"].tolist(),
        labels         = y_val.tolist(),
        clinical_features = X_val_s,
        triplanar_dir  = TRIPLANAR_DIR,
        augment        = False,
    )

    return train_ds, val_ds, class_weights


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict:
    """
    Tính đầy đủ metrics cho bài toán 3 lớp:
      accuracy, balanced_accuracy, macro_f1, weighted_f1,
      per-class precision/recall/f1, AUC (OvR macro),
      sensitivity (recall per class), specificity per class,
      confusion matrix.
    """
    acc     = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_w     = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    p_cls, r_cls, f1_cls, supp_cls = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # Specificity = TN / (TN + FP) per class
    specificity = []
    for c in range(3):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity.append(float(spec))

    roc_auc = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"
            ))
        except Exception:
            roc_auc = None

    metrics = {
        "accuracy"          : acc,
        "balanced_accuracy" : bal_acc,
        "f1_macro"          : f1_macro,
        "f1_weighted"       : f1_w,
        "roc_auc_macro"     : roc_auc,
        "confusion_matrix"  : cm.tolist(),
    }

    for i, name in enumerate(CLASS_NAMES):
        metrics[f"precision_{name}"] = float(p_cls[i])
        metrics[f"recall_{name}"]    = float(r_cls[i])   # sensitivity
        metrics[f"f1_{name}"]        = float(f1_cls[i])
        metrics[f"support_{name}"]   = int(supp_cls[i])
        metrics[f"specificity_{name}"] = specificity[i]

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class ClassBalancedFocalLoss(nn.Module):
    """Class-balanced focal loss with soft targets for label smoothing."""

    def __init__(
        self,
        class_counts: np.ndarray,
        gamma: float = 2.0,
        beta: float = 0.999,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        effective = 1.0 - torch.pow(torch.tensor(beta), counts)
        weights = (1.0 - beta) / effective.clamp_min(1e-8)
        weights = weights / weights.mean()
        self.register_buffer("class_weights", weights)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        n_classes = logits.shape[1]
        smoothing = self.label_smoothing
        with torch.no_grad():
            target_dist = torch.full_like(
                logits, smoothing / max(1, n_classes - 1)
            )
            target_dist.scatter_(
                1, targets.unsqueeze(1), 1.0 - smoothing
            )
        focal = (1.0 - probs).pow(self.gamma)
        class_weights = self.class_weights.to(logits.dtype).unsqueeze(0)
        loss = -target_dist * focal * class_weights * log_probs
        return loss.sum(dim=1).mean()


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted CE dùng cho ablation theo bài báo."""

    def __init__(self, class_counts: np.ndarray, label_smoothing: float = 0.05):
        super().__init__()
        counts = torch.as_tensor(class_counts, dtype=torch.float32)
        weights = counts.sum() / (len(counts) * counts.clamp_min(1.0))
        weights = weights / weights.mean()
        self.register_buffer("class_weights", weights)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights.to(logits.dtype),
            label_smoothing=self.label_smoothing,
        )


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[optim.Optimizer],
    device: torch.device,
    scaler_amp: Optional[GradScaler],
    use_fp16: bool,
    is_train: bool,
    accumulation_steps: int = 1,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Chạy một epoch (train hoặc val).

    Returns:
        avg_loss : float
        y_true   : ndarray
        y_pred   : ndarray
        y_prob   : ndarray (N, 3)
    """
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_true, all_pred, all_prob = [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    with ctx:
        for batch_idx, (mri, clinical, labels) in enumerate(loader):
            mri      = mri.to(device, non_blocking=True)
            clinical = clinical.to(device, non_blocking=True)
            labels   = labels.to(device, non_blocking=True)

            with build_autocast_context(device, use_fp16):
                logits = model(mri, clinical)
                loss   = criterion(logits, labels)

            if is_train:
                scaled_loss = loss / accumulation_steps
                if use_fp16 and scaler_amp is not None:
                    scaler_amp.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                is_accumulation_boundary = (
                    (batch_idx + 1) % accumulation_steps == 0
                    or batch_idx + 1 == len(loader)
                )
                if is_accumulation_boundary:
                    if use_fp16 and scaler_amp is not None:
                        scaler_amp.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    if use_fp16 and scaler_amp is not None:
                        scaler_amp.step(optimizer)
                        scaler_amp.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()

            probs = torch.softmax(logits.detach(), dim=1).cpu().float().numpy()
            preds = probs.argmax(axis=1)

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds)
            all_prob.extend(probs)

    n = len(loader)
    avg_loss = total_loss / n if n > 0 else 0.0

    return (
        avg_loss,
        np.array(all_true),
        np.array(all_pred),
        np.array(all_prob),
    )


def train_one_fold(
    fold_idx: int,
    seed: int,
    train_ds: TriFuseDataset,
    val_ds: TriFuseDataset,
    class_weights: np.ndarray,
    device: torch.device,
    logger: logging.Logger,
    cfg: dict = CFG,
) -> Tuple[Dict, Dict[str, list]]:
    """
    Huấn luyện TriFuse-AD trên một fold.

    Returns:
        best_val_metrics : dict  – metrics tại checkpoint tốt nhất
        history          : dict  – train_loss, val_loss, train_f1, val_f1 qua epochs
    """
    torch.manual_seed(seed + fold_idx)
    np.random.seed(seed + fold_idx)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_labels = np.asarray(train_ds.labels, dtype=np.int64)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_labels),
        replacement=True,
        generator=torch.Generator().manual_seed(seed + fold_idx),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["batch_size"],
        sampler     = sampler,
        num_workers = cfg["num_workers"],
        pin_memory  = cfg["pin_memory"] and device.type == "cuda",
        drop_last   = False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["batch_size"] * 2,
        shuffle     = False,
        num_workers = cfg["num_workers"],
        pin_memory  = cfg["pin_memory"] and device.type == "cuda",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TriFuseAD(cfg).to(device)
    ckpt_path = CKPT_DIR / f"trifuse_seed{seed}_fold{fold_idx}.pt"
    latest_ckpt_path = CKPT_DIR / f"trifuse_seed{seed}_fold{fold_idx}_latest.pt"
    resume_epoch = 0
    resumed_checkpoint = None

    if RESUME_FROM == (seed, fold_idx) and latest_ckpt_path.exists():
        resumed_checkpoint = torch.load(latest_ckpt_path, map_location=device)
        incompatible_keys = model.load_state_dict(
            resumed_checkpoint["model_state_dict"],
            strict=False,
        )
        if incompatible_keys.missing_keys:
            logger.warning(
                f"    Checkpoint thiếu keys hiện tại: "
                f"{incompatible_keys.missing_keys}"
            )
        if incompatible_keys.unexpected_keys:
            logger.info(
                f"    Bỏ qua keys cũ không còn dùng: "
                f"{incompatible_keys.unexpected_keys}"
            )
        resume_epoch = int(resumed_checkpoint.get("epoch", 0))
        logger.info(
            f"    ↻ Resume checkpoint: {latest_ckpt_path.name} "
            f"(epoch={resume_epoch}, "
            f"val Macro-F1={resumed_checkpoint.get('val_f1_macro', float('nan')):.4f})"
        )

    # ── Loss ──────────────────────────────────────────────────────────────────
    class_counts = np.bincount(
        train_labels, minlength=cfg["num_classes"]
    ).astype(np.int64)
    if cfg.get("weighted_ce", False):
        criterion = WeightedCrossEntropyLoss(
            class_counts=class_counts,
            label_smoothing=cfg["label_smoothing"],
        ).to(device)
    else:
        criterion = ClassBalancedFocalLoss(
            class_counts=class_counts,
            gamma=cfg["focal_gamma"],
            beta=cfg["class_balanced_beta"],
            label_smoothing=cfg["label_smoothing"],
        ).to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Phân biệt backbone (lr nhỏ hơn) và head mới (lr đầy đủ)
    backbone_params = list(model.mri_branch.backbone.parameters())
    def set_backbone_trainable(trainable: bool) -> None:
        for parameter in backbone_params:
            parameter.requires_grad = trainable

    set_backbone_trainable(False)
    other_params    = [p for p in model.parameters()
                       if not any(p is bp for bp in backbone_params)]

    optimizer = optim.AdamW([
        {"params": backbone_params, "lr": cfg["lr"] * 0.1},
        {"params": other_params,    "lr": cfg["lr"]},
    ], weight_decay=cfg["weight_decay"])

    # ── Scheduler: Warmup + CosineAnnealing ───────────────────────────────────
    def lr_lambda(epoch):
        if epoch < cfg["warmup_epochs"]:
            return (epoch + 1) / cfg["warmup_epochs"]
        progress = (epoch - cfg["warmup_epochs"]) / \
                   max(1, cfg["epochs"] - cfg["warmup_epochs"])
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ── AMP Scaler ────────────────────────────────────────────────────────────
    use_fp16   = cfg["fp16"] and device.type == "cuda"
    amp_scaler = build_grad_scaler(device, use_fp16)

    if resumed_checkpoint is not None:
        if resumed_checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(resumed_checkpoint["optimizer_state_dict"])
        if resumed_checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(resumed_checkpoint["scheduler_state_dict"])
        if (
            resumed_checkpoint.get("scaler_state_dict") is not None
            and amp_scaler is not None
        ):
            amp_scaler.load_state_dict(resumed_checkpoint["scaler_state_dict"])

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_val_f1     = -1.0
    best_metrics    = {}
    best_weights    = None
    patience_count  = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_f1"  : [], "val_f1"  : [],
        "lr"        : [],
    }

    if resumed_checkpoint is not None:
        set_backbone_trainable(
            resume_epoch >= cfg["backbone_freeze_epochs"]
        )
        best_val_f1 = float(resumed_checkpoint.get("best_val_f1", -1.0))
        best_metrics = resumed_checkpoint.get("best_metrics", {})
        best_weights = resumed_checkpoint.get(
            "best_model_state_dict",
            resumed_checkpoint["model_state_dict"],
        )
        history = resumed_checkpoint.get("history", history)
        patience_count = int(resumed_checkpoint.get("patience_count", 0))

    for epoch in range(resume_epoch, cfg["epochs"]):
        t0 = time.time()

        if epoch == cfg["backbone_freeze_epochs"]:
            set_backbone_trainable(True)
            logger.info(
                f"    Backbone unfrozen at epoch {epoch + 1} for fine-tuning"
            )

        # Train
        tr_loss, tr_true, tr_pred, tr_prob = run_one_epoch(
            model, train_loader, criterion, optimizer,
            device, amp_scaler, use_fp16, is_train=True,
            accumulation_steps=cfg["gradient_accumulation_steps"],
        )
        scheduler.step()

        # Validate
        vl_loss, vl_true, vl_pred, vl_prob = run_one_epoch(
            model, val_loader, criterion, None,
            device, None, False, is_train=False
        )

        tr_f1 = float(f1_score(tr_true, tr_pred, average="macro", zero_division=0))
        vl_f1 = float(f1_score(vl_true, vl_pred, average="macro", zero_division=0))

        current_lr = optimizer.param_groups[1]["lr"]
        elapsed    = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"  ].append(vl_loss)
        history["train_f1"  ].append(tr_f1)
        history["val_f1"    ].append(vl_f1)
        history["lr"        ].append(current_lr)

        logger.info(
            f"    Epoch [{epoch+1:3d}/{cfg['epochs']}] "
            f"Tr Loss={tr_loss:.4f} F1={tr_f1:.4f} | "
            f"Vl Loss={vl_loss:.4f} F1={vl_f1:.4f} | "
            f"LR={current_lr:.2e} | {elapsed:.1f}s"
        )

        # ── Checkpoint nếu val Macro-F1 tốt nhất ──────────────────────────
        stop_training = False
        if vl_f1 > best_val_f1:
            best_val_f1  = vl_f1
            best_weights = copy.deepcopy(model.state_dict())
            best_metrics = compute_metrics(vl_true, vl_pred, vl_prob)
            best_metrics["epoch"] = epoch + 1
            best_metrics["fold"]  = fold_idx
            best_metrics["seed"]  = seed
            patience_count = 0

            best_checkpoint = {
                "model_state_dict"  : best_weights,
                "epoch"             : epoch + 1,
                "val_f1_macro"      : best_val_f1,
                "best_val_f1"       : best_val_f1,
                "best_metrics"      : best_metrics,
                "fold"              : fold_idx,
                "seed"              : seed,
                "cfg"               : cfg,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": (
                    amp_scaler.state_dict()
                    if amp_scaler is not None else None
                ),
            }
            torch.save(best_checkpoint, ckpt_path)

            logger.info(f"    ✓ Checkpoint saved → val Macro-F1={best_val_f1:.4f}")
        else:
            patience_count += 1
            if patience_count >= cfg["early_stop_patience"]:
                logger.info(
                    f"    ⏹ Early stopping tại epoch {epoch+1} "
                    f"(patience={cfg['early_stop_patience']})"
                )
                stop_training = True

        latest_checkpoint = {
            "model_state_dict": model.state_dict(),
            "best_model_state_dict": best_weights,
            "epoch": epoch + 1,
            "val_f1_macro": vl_f1,
            "best_val_f1": best_val_f1,
            "best_metrics": best_metrics,
            "patience_count": patience_count,
            "history": history,
            "fold": fold_idx,
            "seed": seed,
            "cfg": cfg,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": (
                amp_scaler.state_dict() if amp_scaler is not None else None
            ),
        }
        torch.save(latest_checkpoint, latest_ckpt_path)
        if stop_training:
            break

    logger.info(
        f"  Fold {fold_idx} hoàn thành. "
        f"Best Val Macro-F1={best_val_f1:.4f} (epoch {best_metrics.get('epoch','?')})"
    )

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return best_metrics, history


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curve(
    history: Dict[str, list],
    fold_idx: int,
    seed: int,
    save_dir: Path,
) -> None:
    """Vẽ và lưu biểu đồ Loss + Macro-F1 qua các epoch."""
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"TriFuse-AD Training — Seed {seed}, Fold {fold_idx}",
        fontsize=13, fontweight="bold",
    )

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss",
            color="#2196F3", linewidth=1.5)
    ax.plot(epochs, history["val_loss"],   label="Val Loss",
            color="#F44336", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Cross-Entropy Loss"); ax.legend(); ax.grid(alpha=0.3)

    # Macro-F1
    ax = axes[1]
    ax.plot(epochs, history["train_f1"], label="Train Macro-F1",
            color="#4CAF50", linewidth=1.5)
    ax.plot(epochs, history["val_f1"],   label="Val Macro-F1",
            color="#FF9800", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Macro-F1")
    ax.set_title("Macro-F1 Score"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    save_path = save_dir / f"curve_seed{seed}_fold{fold_idx}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_aggregated_results(all_metrics: List[Dict], save_dir: Path) -> None:
    """Vẽ boxplot Macro-F1 và Balanced Accuracy qua 15 lượt chạy."""
    df = pd.DataFrame(all_metrics)

    key_metrics = ["f1_macro", "balanced_accuracy", "accuracy", "roc_auc_macro"]
    key_labels  = ["Macro-F1", "Balanced Accuracy", "Accuracy", "AUC (OvR)"]

    vals = []
    for k in key_metrics:
        if k in df.columns:
            vals.append(df[k].dropna().values * 100)
        else:
            vals.append(np.array([]))

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(
        vals,
        patch_artist=True,
        notch=False,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "black", "markersize": 5},
    )

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xticks(range(1, len(key_labels) + 1))
    ax.set_xticklabels(key_labels)
    ax.set_ylabel("Score (%)")
    ax.set_title("TriFuse-AD – Aggregated Performance (5-Fold × 3 Seeds = 15 Runs)")
    ax.grid(axis="y", alpha=0.4)
    ax.set_ylim(0, 105)

    # Annotate mean ± std
    for i, (k, label) in enumerate(zip(key_metrics, key_labels)):
        if k in df.columns:
            m = df[k].mean() * 100
            s = df[k].std() * 100
            ax.text(
                i + 1, 102, f"{m:.1f}±{s:.1f}",
                ha="center", va="center", fontsize=8, color="black",
            )

    plt.tight_layout()
    plt.savefig(save_dir / "trifuse_aggregated_results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix_avg(all_metrics: List[Dict], save_dir: Path) -> None:
    """Vẽ ma trận nhầm lẫn trung bình qua 15 lượt chạy."""
    cms = [np.array(m["confusion_matrix"]) for m in all_metrics
           if "confusion_matrix" in m]
    if not cms:
        return

    avg_cm = np.stack(cms).mean(axis=0)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(avg_cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Avg Confusion Matrix (15 Runs)")

    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{avg_cm[i, j]:.1f}",
                    ha="center", va="center",
                    color="white" if avg_cm[i, j] > avg_cm.max() * 0.5 else "black",
                    fontsize=12)

    plt.tight_layout()
    plt.savefig(save_dir / "trifuse_confusion_matrix_avg.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(all_metrics: List[Dict], logger: logging.Logger) -> Dict:
    """In bảng thống kê Mean ± Std và trả về dict để lưu JSON."""
    df = pd.DataFrame(all_metrics)

    stat_keys = [
        ("accuracy",          "Accuracy"),
        ("balanced_accuracy", "Balanced Acc"),
        ("f1_macro",          "Macro F1"),
        ("f1_weighted",       "Weighted F1"),
        ("roc_auc_macro",     "AUC (OvR)"),
        ("recall_CN",         "Sensitivity CN"),
        ("recall_VMD",        "Sensitivity VMD"),
        ("recall_AD",         "Sensitivity AD"),
        ("specificity_CN",    "Specificity CN"),
        ("specificity_VMD",   "Specificity VMD"),
        ("specificity_AD",    "Specificity AD"),
    ]

    logger.info("\n" + "=" * 65)
    logger.info("  TriFuse-AD  —  KẾT QUẢ TỔNG HỢP (5-Fold × 3 Seeds = 15 Runs)")
    logger.info("=" * 65)
    logger.info(f"  {'Metric':<22}  {'Mean (%)':>10}  {'Std (%)':>9}")
    logger.info("  " + "-" * 43)

    aggregated = {}
    for key, label in stat_keys:
        if key not in df.columns:
            continue
        vals = df[key].dropna().values * 100
        mean_v, std_v = vals.mean(), vals.std()
        logger.info(f"  {label:<22}  {mean_v:>9.2f}%  ±{std_v:>7.2f}%")
        aggregated[key] = {"mean": float(mean_v), "std": float(std_v)}

    logger.info("=" * 65)
    return aggregated


def load_resumed_results(
    resume_from: Optional[Tuple[int, int]],
    cfg: dict,
    logger: logging.Logger,
) -> Tuple[List[Dict], List[Dict]]:
    """Load completed runs skipped by RESUME_FROM for correct aggregation."""
    if resume_from is None:
        return [], []

    metrics: List[Dict] = []
    histories: List[Dict] = []
    resume_seed, resume_fold = resume_from

    for seed in cfg["seeds"]:
        for fold_idx in range(cfg["n_folds"]):
            if (seed, fold_idx) >= (resume_seed, resume_fold):
                return metrics, histories

            latest_path = CKPT_DIR / (
                f"trifuse_seed{seed}_fold{fold_idx}_latest.pt"
            )
            if not latest_path.exists():
                raise FileNotFoundError(
                    f"Thiếu checkpoint của run đã bỏ qua: {latest_path}. "
                    "Không thể tổng hợp đủ 15 runs."
                )

            checkpoint_data = torch.load(latest_path, map_location="cpu")
            run_metrics = checkpoint_data.get("best_metrics")
            if not run_metrics:
                raise RuntimeError(
                    f"Checkpoint không có best_metrics: {latest_path}"
                )

            run_metrics = dict(run_metrics)
            run_metrics["seed"] = seed
            run_metrics["fold"] = fold_idx
            metrics.append(run_metrics)

            history = checkpoint_data.get("history")
            if history is not None:
                histories.append({
                    "seed": seed,
                    "fold": fold_idx,
                    **history,
                })
            logger.info(
                f"  ✓ Nạp kết quả đã hoàn tất: Seed={seed}, Fold={fold_idx}"
            )

    return metrics, histories


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    for directory in (CKPT_DIR, LOG_DIR, PLOT_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    # ── Logger ────────────────────────────────────────────────────────────────
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file   = LOG_DIR / f"trifuse_{ts}.log"
    logger     = setup_logger("trifuse", log_file)

    logger.info("=" * 70)
    logger.info("  TriFuse-AD Official Training  —  OASIS-1 Alzheimer 3-Class")
    logger.info(
        f"  Architecture: SlicePlaneTransformer + {MODEL_VARIANT} fusion"
    )
    logger.info(f"  Protocol : {CFG['n_folds']} Folds × {len(CFG['seeds'])} Seeds "
                f"= {CFG['n_folds'] * len(CFG['seeds'])} Runs")
    logger.info("=" * 70)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device : {device}")
    if device.type == "cuda":
        logger.info(f"  GPU    : {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"  FP16   : {CFG['fp16']}")

    # ── Load Data ─────────────────────────────────────────────────────────────
    try:
        df     = load_manifest()
        splits = load_kfold_splits()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"\n  Manifest  : {len(df)} subjects")
    logger.info(f"  Triplanar : {TRIPLANAR_DIR}")

    # Label distribution
    for lbl, name in enumerate(CLASS_NAMES):
        cnt = (df["label"] == lbl).sum()
        logger.info(f"    {name}: {cnt} subjects")

    # ── Cross-Validation ──────────────────────────────────────────────────────
    all_metrics    = []
    all_histories  = []
    run_count      = 0
    total_runs     = CFG["n_folds"] * len(CFG["seeds"])

    resumed_metrics, resumed_histories = load_resumed_results(
        RESUME_FROM, CFG, logger
    )
    all_metrics.extend(resumed_metrics)
    all_histories.extend(resumed_histories)

    for seed in CFG["seeds"]:
        logger.info(f"\n{'─'*60}")
        logger.info(f"  SEED = {seed}")
        logger.info(f"{'─'*60}")

        for fold_idx in range(CFG["n_folds"]):
            run_count += 1

            if RESUME_FROM is not None:
                resume_seed, resume_fold = RESUME_FROM
                if (seed, fold_idx) < (resume_seed, resume_fold):
                    logger.info(
                        f"  ⏭ Bỏ qua Seed={seed}, Fold={fold_idx} "
                        "đã hoàn tất trước đó"
                    )
                    continue

            logger.info(
                f"\n  ▶ Run {run_count}/{total_runs} — "
                f"Seed={seed}, Fold={fold_idx}"
            )

            try:
                train_ds, val_ds, cw = prepare_fold_data(
                    df, splits, fold_idx, seed, logger
                )
            except Exception as e:
                if isinstance(e, (FileNotFoundError, MemoryError)):
                    raise
                logger.exception(f"  Lỗi prepare_fold_data: {e}")
                continue

            try:
                fold_metrics, history = train_one_fold(
                    fold_idx    = fold_idx,
                    seed        = seed,
                    train_ds    = train_ds,
                    val_ds      = val_ds,
                    class_weights = cw,
                    device      = device,
                    logger      = logger,
                    cfg         = CFG,
                )
            except Exception as e:
                if isinstance(e, (FileNotFoundError, MemoryError)) or (
                    device.type == "cuda" and "CUDA" in str(e).upper()
                ):
                    raise
                logger.exception(f"  Lỗi train_one_fold: {e}")
                continue

            all_metrics.append(fold_metrics)
            all_histories.append({"seed": seed, "fold": fold_idx, **history})

            # Vẽ training curve
            try:
                plot_training_curve(history, fold_idx, seed, PLOT_DIR)
            except Exception as e:
                logger.warning(f"  Không thể vẽ training curve: {e}")

            # In kết quả fold
            logger.info(
                f"  ✓ Fold {fold_idx} kết quả: "
                f"Macro-F1={fold_metrics.get('f1_macro', 0):.4f} | "
                f"Bal-Acc={fold_metrics.get('balanced_accuracy', 0):.4f} | "
                f"AUC={fold_metrics.get('roc_auc_macro', 0) or 0:.4f}"
            )

            # Giải phóng VRAM
            if device.type == "cuda":
                torch.cuda.empty_cache()
            del train_ds, val_ds, cw, fold_metrics, history
            gc.collect()

    if not all_metrics:
        logger.error("Không có kết quả nào. Kiểm tra lại dữ liệu và cấu hình.")
        sys.exit(1)

    # ── Tổng hợp kết quả ──────────────────────────────────────────────────────
    aggregated = print_summary_table(all_metrics, logger)

    # ── Plots tổng hợp ────────────────────────────────────────────────────────
    try:
        plot_aggregated_results(all_metrics, PLOT_DIR)
        plot_confusion_matrix_avg(all_metrics, PLOT_DIR)
        logger.info(f"\n  Plots lưu tại: {PLOT_DIR}")
    except Exception as e:
        logger.warning(f"  Lỗi khi vẽ biểu đồ tổng hợp: {e}")

    # ── Lưu JSON ──────────────────────────────────────────────────────────────
    export = {
        "model"            : (
            "TriFuse-AD" if MODEL_VARIANT == "gated"
            else f"TriFuse-AD-{MODEL_VARIANT}"
        ),
        "timestamp"        : ts,
        "protocol"         : {
            "n_folds"  : CFG["n_folds"],
            "seeds"    : CFG["seeds"],
            "total_runs": total_runs,
            "epochs"   : CFG["epochs"],
            "lr"       : CFG["lr"],
            "batch_size": CFG["batch_size"],
            "gradient_accumulation_steps": CFG["gradient_accumulation_steps"],
            "effective_batch_size": (
                CFG["batch_size"] * CFG["gradient_accumulation_steps"]
            ),
            "loss": (
                "Weighted Cross-Entropy"
                if CFG.get("weighted_ce", False)
                else "Class-Balanced Focal Loss"
            ),
            "focal_gamma": CFG["focal_gamma"],
            "class_balanced_beta": CFG["class_balanced_beta"],
            "label_smoothing": CFG["label_smoothing"],
            "sampler": "WeightedRandomSampler",
            "backbone_freeze_epochs": CFG["backbone_freeze_epochs"],
            "model_variant": MODEL_VARIANT,
            "augmentation": (
                "rotation +/-7 degrees and contrast +/-10%; no flips"
            ),
        },
        "aggregated_stats" : aggregated,
        "per_run_metrics"  : all_metrics,
        "training_histories": [
            {
                "seed": h["seed"], "fold": h["fold"],
                "train_loss": h["train_loss"], "val_loss": h["val_loss"],
                "train_f1"  : h["train_f1"],   "val_f1"  : h["val_f1"],
            }
            for h in all_histories
        ],
    }

    # Merge với kết quả cũ nếu đã có
    existing = {}
    if RESULT_JSON.exists():
        try:
            with open(RESULT_JSON, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass

    result_key = (
        "TriFuse-AD" if MODEL_VARIANT == "gated"
        else f"TriFuse-AD-{MODEL_VARIANT}"
    )
    existing[result_key] = export

    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, cls=NpEncoder, ensure_ascii=False)

    logger.info(f"\n  📄 Kết quả lưu tại : {RESULT_JSON}")
    logger.info(f"  📝 Log file         : {log_file}")
    logger.info(f"  📁 Checkpoints      : {CKPT_DIR}")
    logger.info(f"  📊 Plots            : {PLOT_DIR}")
    logger.info("\n  ✅ TriFuse-AD training hoàn thành!")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    raise SystemExit(
        "Use: python Main/train_locked_holdout.py --output outputs_holdout_safe "
        "(fresh train/validation folds with a locked test set)."
    )
