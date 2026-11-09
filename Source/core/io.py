"""Nap ket qua OOF cua cac mo hinh. Muon them mo hinh (v1, Baseline 1...): tha <khoa>_oof.csv vao data/."""
import json
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
CLASSES = ["CN", "VMD", "AD"]
CLASS_FULL = {"CN": "Cognitively Normal", "VMD": "Very Mild Dementia", "AD": "Alzheimer's Disease"}
PCOLS = ["prob_0", "prob_1", "prob_2"]

# Khoa <khoa>_oof.csv theo KE HOACH: b1..b4, v1, v2 (+ vitbase = phu luc, cau hinh cu cua Baseline 3: ViT-Base + lam sang concat).
# Khoa khac tu dong nhan ten theo ten file. Ten file KHONG dung khoa se bi tu choi neu summary.json cho thay sai mo hinh (xem registry_issues).
REGISTRY = {
    "b1": dict(name="Baseline 1 (chỉ lâm sàng)", color="#b9a26b", uses_clinical=True),
    "b2": dict(name="Baseline 2 (ConvNeXt-Tiny, chỉ ảnh)", color="#7aa6c2", uses_clinical=False),
    "b3": dict(name="Baseline 3 (ViT-Small, chỉ ảnh)", color="#8d99a6", uses_clinical=False),
    "b4": dict(name="Baseline 4 (ConvNeXt-Tiny + lâm sàng, concat)", color="#4c9a6a", uses_clinical=True),
    "v1": dict(name="v1 (ViT-Small + lâm sàng, concat)", color="#e07b39", uses_clinical=True),
    "v2": dict(name="v2 (ViT-Small + lâm sàng, gated)", color="#1f6fb2", uses_clinical=True),
    "trifuse": dict(name="TriFuse-AD (Chính thức - Gated)", color="#9c27b0", uses_clinical=True),
    "vitbase": dict(name="Phụ lục: ViT-Base + lâm sàng (concat)", color="#b0b7c0", uses_clinical=True),
}
EXPECTED_MODEL = {"b1": "tabular_mlp", "b2": "cnn25d_imgonly", "b3": "vit25d_small_imgonly", "b4": "cnn25d_clin_concat",
                  "v1": "vit25d_clin_concat", "v2": "vit25d_clin_gated", "trifuse": "trifuse_official", "vitbase": "vit25d_triplanar"}
EXTRA_COLORS = ["#a05cb5", "#c9a227", "#2a9d8f", "#d1495b", "#e91e63", "#009688"]

# Goi demo co the mang ten hien thi cu; doi chieu mo hinh <-> file OOF theo KIEN TRUC (arch), khong theo ten.
ARCH_KEY = {"vit25d_clin_gated": "v2", "vit25d_clin_concat": "v1", "vit25d_triplanar": "vitbase",
            "vit25d_small_imgonly": "b3", "cnn25d_imgonly": "b2", "cnn25d_clin_concat": "b4",
            "trifuse_official": "trifuse"}


def registry_issues():
    """Phat hien file dat sai khoa: summary.json ghi mo hinh khac voi mo hinh ma khoa ky vong."""
    out = []
    for p in sorted(DATA.glob("*_summary.json")):
        key = p.name[: -len("_summary.json")]
        try: actual = json.loads(p.read_text(encoding="utf-8")).get("model")
        except Exception: continue
        exp = EXPECTED_MODEL.get(key)
        if exp and actual and actual != exp:
            out.append(f"`{p.name}` ghi mô hình `{actual}` nhưng khóa `{key}` dành cho `{exp}`. "
                       f"File này bị BỎ QUA. Đổi tên cặp file `{key}_*` thành khóa đúng (ví dụ `vitbase_*` cho cấu hình ViT-Base cũ).")
    return out


def model_by_key(models, key):
    return next((n for n, m in models.items() if m["key"] == key), None)


def registry_bad_keys():
    return {m.split("`")[1][: -len("_summary.json")] for m in registry_issues()}


def load_models():
    """Tra ve dict ten_hien_thi -> dict(oof=DataFrame, summary=dict|None, color=str, key=str)."""
    out, extra = {}, 0
    for p in sorted(DATA.glob("*_oof.csv")):
        key = p.name[: -len("_oof.csv")]
        if key in registry_bad_keys():
            continue
        meta = REGISTRY.get(key)
        if meta is None:
            meta = dict(name=key, color=EXTRA_COLORS[extra % len(EXTRA_COLORS)], uses_clinical=None)
            extra += 1
        s = DATA / f"{key}_summary.json"
        out[meta["name"]] = dict(key=key, oof=pd.read_csv(p), color=meta["color"],
                                 summary=json.loads(s.read_text()) if s.exists() else None,
                                 uses_clinical=meta["uses_clinical"])
    # thu tu: b3 truoc, v2 sau, con lai theo ten
    order = {v["name"]: i for i, v in enumerate(REGISTRY.values())}
    return dict(sorted(out.items(), key=lambda kv: order.get(kv[0], 99)))
