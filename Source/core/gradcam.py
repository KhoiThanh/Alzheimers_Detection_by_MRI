"""Grad-CAM cho TriPlanarViTClinicalFusion (can torch + timm). Hook vao norm1 cua khoi Transformer cuoi.
Ly do: voi ViT, token patch o DAU RA khoi cuoi khong nhan gradient (chi token CLS duoc dung), nen dung dau vao
attention cua khoi cuoi — cach lam chuan cua Grad-CAM cho ViT."""
import os
import numpy as np
import torch
import torch.nn.functional as F
from .model import TriPlanarViTClinicalFusion, TriPlanarViTImageOnly, ViT25DTriPlanar


def _official_model(sd, weights_path):
    """Load the official ConvNeXt + Transformer TriFuse checkpoint."""
    import importlib.util
    from pathlib import Path
    env_source = os.environ.get("TRIFUSE_TRAINING_SCRIPT", "").strip()
    candidates = []
    if env_source:
        candidates.append(Path(env_source))
    # Project layout: .../KLTN/Main/03_train_official_trifuse.py
    candidates.append(Path(__file__).resolve().parents[4] / "Main" / "03_train_official_trifuse.py")
    # Standalone release layout: package/Main/03_train_official_trifuse.py
    candidates.append(Path(__file__).resolve().parents[1] / "Main" / "03_train_official_trifuse.py")
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        raise ImportError(
            "Không tìm thấy 03_train_official_trifuse.py. "
            "Đặt file trong thư mục Main cạnh gói demo hoặc đặt biến "
            "TRIFUSE_TRAINING_SCRIPT."
        )
    spec = importlib.util.spec_from_file_location("trifuse_official_training", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Không tìm thấy định nghĩa TriFuse-AD: {source}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = dict(mod.CFG)
    if isinstance(weights_path, str):
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("cfg"), dict):
            cfg.update(checkpoint["cfg"])
    cfg["pretrained_backbone"] = False
    cfg["model_variant"] = "gated"
    return mod.TriFuseAD(cfg)


def _extract_state(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "net"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def load_model(weights_path=None, arch="vit25d_clin_gated", n_tab=7, device=None):
    """arch: vit25d_clin_gated | vit25d_clin_concat (v2/v1) | vit25d_triplanar (Baseline 3, ViT-Base).
    weights_path=None -> trong so NGAU NHIEN (chi de kiem thu ky thuat, vo nghia ve y khoa)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sd = None
    if weights_path is None:
        torch.manual_seed(0)                 # trong so ngau nhien tai lap duoc
    if weights_path is not None:
        try:
            obj = torch.load(weights_path, map_location="cpu")
        except Exception:
            obj = torch.load(weights_path, map_location="cpu", weights_only=False)  # file cua chinh ban
        sd = {k.replace("module.", "", 1): v for k, v in _extract_state(obj).items()}
    if arch == "trifuse_official":
        model = _official_model(sd, weights_path)
    elif arch == "vit25d_triplanar":
        if sd is not None:                      # tu suy ra co nhanh tab_mlp hay khong tu checkpoint
            n_tab = int(sd["tab_mlp.0.weight"].shape[1]) if "tab_mlp.0.weight" in sd else 0
        model = ViT25DTriPlanar(n_tab_features=n_tab, pretrained=False)
    elif arch == "vit25d_small_imgonly":
        torch.manual_seed(0) if sd is None else None
        model = TriPlanarViTImageOnly(n_tab_features=n_tab, pretrained=False, fusion_mode="gated")
    elif arch in ("vit25d_clin_gated", "vit25d_clin_concat"):
        model = TriPlanarViTClinicalFusion(n_tab_features=n_tab, pretrained=False,
                                           fusion_mode="gated" if arch.endswith("gated") else "concat")
    else:
        raise ValueError(f"Kiến trúc chưa hỗ trợ: {arch}")
    info = dict(random_weights=sd is None, missing=0, unexpected=0, n_tab=n_tab)
    if sd is not None:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        info.update(missing=len(missing), unexpected=len(unexpected))
        if len(missing) > 0.2 * len(model.state_dict()):
            raise ValueError(f"Trọng số không khớp kiến trúc {arch}: thiếu {len(missing)} tham số (ví dụ {missing[:3]}).")
    for p in model.parameters():
        p.requires_grad_(True)            # can de gradient chay qua hook
    return model.to(device).eval(), info


def to_batch(x, tab=None):
    """x: ndarray (T,3,H,W) tu core.preproc.build_slices; tab: ndarray (F,) | None -> tensor co chieu batch."""
    xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0)
    tt = None if tab is None else torch.from_numpy(np.asarray(tab, dtype=np.float32)).unsqueeze(0)
    return xt, tt


def predict(model, x, tab=None):
    """Predict without building an autograd graph or generating Grad-CAM."""
    dev = next(model.parameters()).device
    # A cached model may have gradients left over from a previous CAM request.
    model.zero_grad(set_to_none=True)
    with torch.inference_mode():
        xt = x.detach().to(dev)
        tt = None if tab is None else tab.detach().to(dev)
        probs = model(xt, tt).softmax(-1)[0].cpu().numpy()
        pred = int(probs.argmax())
        gate = getattr(model, "last_gate", None)
        gate_mean = None if gate is None else float(gate.mean())
    return dict(cams=None, probs=probs, pred=pred, target=pred, gate=gate_mean)


def _vit(model):
    if hasattr(model, "mri_branch"):
        return model.mri_branch.transformer
    return model.encoder if hasattr(model, "encoder") else model.backbone      # v1/v2: encoder; Baseline 3: backbone


def _official_conv_gradcam(model, x, tab, target=None):
    """Grad-CAM trên feature map ConvNeXt cuối của TriFuse-AD chính thức."""
    dev = next(model.parameters()).device
    x = x.to(dev).requires_grad_(True)
    tab = None if tab is None else tab.to(dev)
    backbone = model.mri_branch.backbone.backbone
    convs = [m for m in backbone.modules() if isinstance(m, torch.nn.Conv2d)]
    if not convs:
        raise RuntimeError("Không tìm thấy Conv2d trong backbone.")
    acts, grads = [], {}
    def fwd(_, __, out):
        i = len(acts)
        acts.append(out)
        out.register_hook(lambda g, i=i: grads.__setitem__(i, g))
    handle = convs[-1].register_forward_hook(fwd)
    try:
        with torch.enable_grad():
            logits = model(x, tab)
            probs = logits.softmax(-1)[0].detach().cpu().numpy()
            pred = int(probs.argmax())
            target_class = pred if target is None else int(target)
            model.zero_grad(set_to_none=True)
            logits[0, target_class].backward()
    finally:
        handle.remove()
    cams = []
    for i, feat in enumerate(acts):
        grad = grads[i]
        if feat.ndim == 4 and feat.shape[1] != grad.shape[1]:
            feat, grad = feat.permute(0, 3, 1, 2), grad.permute(0, 3, 1, 2)
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * feat).sum(dim=1, keepdim=True))
        cams.append(F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0].detach())
    cam = torch.cat(cams, dim=0)
    cam = (cam / (cam.max() + 1e-8)).cpu().numpy()
    gate = getattr(model, "last_gate", None)
    return dict(cams=cam, probs=probs, pred=pred, target=target_class,
                gate=None if gate is None else float(gate.mean()))


def gradcam(model, x, tab=None, target=None):
    if hasattr(model, "mri_branch"):
        return _official_conv_gradcam(model, x, tab, target)
    """Tra ve dict: cams (T,H,W) trong [0,1] (chuan hoa chung ca khoi), probs (3,), pred, target, gate (float|None).
    Hook goi nhieu lan/forward (Baseline 3 chia 27 lat thanh cac chunk 9) nen gom theo thu tu goi."""
    dev = next(model.parameters()).device
    x = x.to(dev).requires_grad_(True)
    tab = None if tab is None else tab.to(dev)
    acts, grads = [], {}

    def fwd(mod, inp, out):
        i = len(acts); acts.append(out)
        out.register_hook(lambda g, i=i: grads.__setitem__(i, g))

    vit = _vit(model)
    block = vit.layers[-1] if hasattr(vit, "layers") else vit.blocks[-1]
    h = block.norm1.register_forward_hook(fwd)
    try:
        with torch.enable_grad():
            logits = model(x, tab)
            probs = logits.softmax(-1)[0].detach().cpu().numpy()
            pred = int(probs.argmax())
            t = pred if target is None else int(target)
            model.zero_grad(set_to_none=True)
            logits[0, t].backward()
    finally:
        h.remove()
    a = torch.cat([t_.detach() for t_ in acts], 0)[:, 1:, :]                    # bo token CLS: (T, N, D)
    g = torch.cat([grads[i] for i in range(len(acts))], 0)[:, 1:, :]
    w = g.mean(dim=1, keepdim=True)                                              # trong so kenh: (T,1,D)
    token_scores = F.relu((w * a).sum(-1))
    if hasattr(model, "mri_branch"):
        # TriFuse-AD co 27 token, moi token ung voi mot lat cat.
        cam = token_scores[0, :, None, None].expand(-1, x.shape[-2], x.shape[-1])
    else:
        n = int(round(a.shape[1] ** 0.5))
        cam = token_scores.reshape(-1, 1, n, n)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    cam = (cam / (cam.max() + 1e-8)).cpu().numpy()
    gate = getattr(model, "last_gate", None)
    return dict(cams=cam, probs=probs, pred=pred, target=t, gate=None if gate is None else float(gate.mean()))
