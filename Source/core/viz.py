"""Ve hinh bang matplotlib (khong phu thuoc torch) — dung chung cho app va bao cao."""
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .io import CLASSES

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def fig_bytes(fig, dpi=170):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return buf.getvalue()


def cm_figure(cm, title=""):
    cm = np.asarray(cm); n = cm / cm.sum(1, keepdims=True).clip(1)
    fig, ax = plt.subplots(figsize=(4.2, 3.8)); ax.imshow(n, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i, j]}\n({n[i, j]:.0%})", ha="center", va="center", fontsize=9, color="white" if n[i, j] > .5 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES); ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Dự đoán"); ax.set_ylabel("Thực tế"); ax.set_title(title, fontsize=10); fig.tight_layout()
    return fig


def roc_figure(curves):
    """curves: {ten_mo_hinh: (mau, {lop: (fpr, tpr, auc)})}"""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, c in zip(axes, CLASSES):
        for name, (color, d) in curves.items():
            fpr, tpr, auc = d[c]; ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=.5); ax.set_title(f"{c} so với còn lại"); ax.set_xlabel("1 − Specificity")
        ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=.25)
    axes[0].set_ylabel("Sensitivity"); fig.tight_layout()
    return fig


def case_prob_figure(detail, true_label=None):
    """Cot xac suat trung binh (3 seed) theo tung mo hinh cho 1 ca."""
    ms = detail["models"]; x = np.arange(3); w = 0.8 / max(len(ms), 1)
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    for k, (name, m) in enumerate(ms.items()):
        bars = ax.bar(x + (k - (len(ms) - 1) / 2) * w, m["mean_prob"] * 100, w, color=m["color"], label=name)
        for b, v in zip(bars, m["mean_prob"] * 100): ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    if true_label is not None:
        ax.axvspan(true_label - 0.45, true_label + 0.45, color="#2e7d32", alpha=.10, lw=0)
        ax.text(true_label, 103, "nhãn thật", ha="center", fontsize=8, color="#2e7d32")
    ax.set_xticks(x); ax.set_xticklabels(CLASSES); ax.set_ylim(0, 112); ax.set_ylabel("Xác suất TB 3 seed (%)")
    ax.legend(fontsize=8, loc="upper right"); fig.tight_layout()
    return fig


def overlay(gray, cam, alpha=0.45):
    """gray: (H,W) bat ky thang do; cam: (H,W) trong [0,1]. Tra ve anh RGB uint8."""
    g = np.asarray(gray, dtype=float); g = (g - g.min()) / (np.ptp(g) + 1e-8)
    heat = plt.get_cmap("jet")(np.clip(cam, 0, 1))[..., :3]
    out = (1 - alpha) * np.repeat(g[..., None], 3, -1) + alpha * heat * (cam[..., None] > 0.05)
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def cam_grid(volume, cams, plane_names, per_plane=9):
    """volume, cams: (T,H,W). Luoi len(plane_names) hang x per_plane cot; tra ve Figure."""
    P = len(plane_names)
    fig, axes = plt.subplots(P, per_plane, figsize=(1.6 * per_plane, 1.7 * P))
    axes = np.atleast_2d(axes)
    for p in range(P):
        for j in range(per_plane):
            t = p * per_plane + j; ax = axes[p, j]
            ax.imshow(overlay(volume[t], cams[t])); ax.axis("off")
            if j == 0: ax.set_title(plane_names[p], fontsize=8, loc="left")
    fig.tight_layout(pad=0.3)
    return fig


def cam_compare(raw, cams_a, cams_b, label_a, label_b, plane=0, per_plane=9):
    """2 hang (a: mo hinh da huan luyen, b: trong so ngau nhien) x per_plane lat cua mot mat phang."""
    fig, axes = plt.subplots(2, per_plane, figsize=(1.6 * per_plane, 3.5))
    for r, (cams, lab) in enumerate(((cams_a, label_a), (cams_b, label_b))):
        for j in range(per_plane):
            t = plane * per_plane + j; ax = axes[r, j]
            ax.imshow(overlay(raw[t], cams[t])); ax.axis("off")
            if j == 0: ax.set_title(lab, fontsize=8, loc="left")
    fig.tight_layout(pad=0.3)
    return fig


def risk_overview_figure(probs, risk_score, risk_info):
    """Vẽ thanh đo nguy cơ tổng thể kết hợp phân bố xác suất 3 lớp."""
    fig, (ax_bar, ax_gauge) = plt.subplots(1, 2, figsize=(9.5, 3.2), gridspec_kw={'width_ratios': [1.2, 1]})
    
    # 1. Biểu đồ cột phân bố xác suất
    classes = ["CN (Bình thường)", "VMD (Suy giảm nhẹ)", "AD (Alzheimer)"]
    colors = ["#2e7d32", "#f57f17", "#c62828"]
    p_pcts = [probs[0] * 100, probs[1] * 100, probs[2] * 100]
    bars = ax_bar.bar(classes, p_pcts, color=colors, width=0.55, edgecolor="none")
    for b, p in zip(bars, p_pcts):
        ax_bar.text(b.get_x() + b.get_width() / 2, p + 1.5, f"{p:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax_bar.set_ylim(0, 115)
    ax_bar.set_ylabel("Xác suất dự đoán (%)", fontsize=9)
    ax_bar.set_title("Phân bố xác suất 3 lớp nhận thức", fontsize=10, fontweight="bold")
    ax_bar.grid(axis="y", linestyle="--", alpha=0.3)
    ax_bar.tick_params(axis='x', labelsize=8.5)

    # 2. Thang đo nguy cơ tổng hợp
    ax_gauge.set_xlim(0, 100)
    ax_gauge.set_ylim(-0.5, 1.2)
    # Các vùng màu
    ax_gauge.barh(0.4, 30, height=0.35, left=0, color="#a5d6a7", alpha=0.9, label="Thấp (<30%)")
    ax_gauge.barh(0.4, 30, height=0.35, left=30, color="#ffe082", alpha=0.9, label="Trung bình (30-60%)")
    ax_gauge.barh(0.4, 40, height=0.35, left=60, color="#ef9a9a", alpha=0.9, label="Cao (>60%)")
    
    # Kim chỉ mức nguy cơ hiện tại
    ax_gauge.scatter([risk_score], [0.4], color=risk_info["badge_color"], s=180, zorder=5, edgecolor="white", linewidth=1.5)
    ax_gauge.axvline(risk_score, color=risk_info["badge_color"], linestyle="--", linewidth=2.0)
    ax_gauge.text(risk_score, 0.75, f"{risk_score:.1f}%", color=risk_info["badge_color"],
                  fontweight="bold", fontsize=11, ha="center")
    
    ax_gauge.set_yticks([])
    ax_gauge.set_xlabel("Chỉ số nguy cơ sa sút trí tuệ (0 - 100%)", fontsize=9)
    ax_gauge.set_title(f"Mức độ: {risk_info['level'].split('(')[0].strip()}", fontsize=10, fontweight="bold", color=risk_info["badge_color"])
    ax_gauge.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3, fontsize=7.5, frameon=False)
    
    fig.tight_layout()
    return fig


def triplanar_summary_figure(raw_slices):
    """
    Hiển thị 3 lát cắt trung tâm đặc trưng đại diện cho 3 mặt phẳng giải phẫu não:
      Axial (lát 4), Coronal (lát 13), Sagittal (lát 22).
    """
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
    planes = [
        ("Mặt phẳng Axial (Ngang)", raw_slices[4]),
        ("Mặt phẳng Coronal (Đứng ngang)", raw_slices[13]),
        ("Mặt phẳng Sagittal (Đứng dọc)", raw_slices[22]),
    ]
    for ax, (title, sl) in zip(axes, planes):
        ax.imshow(sl, cmap="bone")
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.axis("off")
    fig.tight_layout(pad=0.5)
    return fig
