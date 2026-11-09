"""Tien xu ly dau vao GIONG HET trifuse/data/datasets.py (SliceDataset + _encode_tab), chi dung numpy/pandas.
Slice luu san o <root>/<subject_id>/<plane>/slices.npy, shape (9,224,224) float16 (da z-score theo nao, nen = 0)."""
from pathlib import Path
import numpy as np
import pandas as pd

PLANES = ("axial", "coronal", "sagittal")
TAB_FEATURES = ("age", "sex", "education", "ses", "etiv", "nwbv", "asf")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_plane(root, sid, plane, n_slices=9):
    arr = np.load(Path(root) / sid / plane / "slices.npy").astype(np.float32)
    if n_slices < arr.shape[0]:
        start = (arr.shape[0] - n_slices) // 2
        arr = arr[start:start + n_slices]
    return arr


def to_rgb(sl):
    x = np.stack([sl, sl, sl], axis=0)
    return ((x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]).astype(np.float32)


def build_slices(root, sid, planes=PLANES, n_slices=9):
    """Tra ve (x, raw): x (T,3,H,W) dua vao mo hinh; raw (T,H,W) anh z-score de hien thi."""
    raw = np.concatenate([load_plane(root, sid, p, n_slices) for p in planes], axis=0)
    x = np.stack([to_rgb(s) for s in raw])
    return x, raw


def encode_tab(row, features, stats):
    """row: dict/Series; stats: {feature: {median, mean, std}} fit tren tap TRAIN cua fold."""
    out = np.zeros(len(features), dtype=np.float32)
    for i, f in enumerate(features):
        v = row.get(f) if isinstance(row, dict) else row[f]
        s = stats[f]
        if pd.isna(v) or v is None:
            v = s["median"]
        out[i] = (float(v) - s["mean"]) / s["std"]
    return out


# =============================================================================
# HÀM MỞ RỘNG: XỬ LÝ DỮ LIỆU TẢI LÊN (MRI UPLOAD & LÂM SÀNG TÙY CHỌN)
# =============================================================================

FEATURE_LABELS_VI = {
    "age": "Tuổi",
    "sex": "Giới tính",
    "education": "Trình độ học vấn",
    "ses": "Chỉ số kinh tế - xã hội (SES)",
    "etiv": "Thể tích nội sọ ước tính (eTIV, mm³)",
    "nwbv": "Thể tích não toàn bộ chuẩn hóa (nWBV)",
    "asf": "Hệ số tỷ lệ Atlas (ASF)",
}

# Các hồ sơ lâm sàng mẫu đại diện cho 3 nhóm nhận thức
CLINICAL_PRESETS = {
    "CN": {
        "name": "Bình thường (CN) - Lão hóa khỏe mạnh",
        "values": {"age": 68.0, "sex": 0.0, "education": 4.0, "ses": 2.0, "etiv": 1520.0, "nwbv": 0.795, "asf": 1.15},
        "description": "Tuổi 68, nữ, học vấn đại học, thể tích não bảo tồn tốt (nWBV cao 79.5%)."
    },
    "VMD": {
        "name": "Suy giảm nhận thức rất nhẹ (VMD) - Giai đoạn tiền đình",
        "values": {"age": 76.0, "sex": 1.0, "education": 3.0, "ses": 3.0, "etiv": 1445.0, "nwbv": 0.730, "asf": 1.22},
        "description": "Tuổi 76, nam, học vấn phổ thông, teo não ranh giới (nWBV 73.0%)."
    },
    "AD": {
        "name": "Sa sút trí tuệ / Alzheimer (AD) - Giai đoạn tiến triển",
        "values": {"age": 82.0, "sex": 0.0, "education": 2.0, "ses": 3.5, "etiv": 1360.0, "nwbv": 0.675, "asf": 1.29},
        "description": "Tuổi 82, nữ, học vấn cơ bản, teo nhu mô não rõ rệt (nWBV giảm sâu 67.5%)."
    }
}


def resize_slice(slice_2d: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """Resize lát cắt 2D về target_size (mặc định 224x224)."""
    try:
        from scipy.ndimage import zoom
        h, w = slice_2d.shape
        if (h, w) == target_size:
            return slice_2d.astype(np.float32)
        zoom_h = target_size[0] / h
        zoom_w = target_size[1] / w
        return zoom(slice_2d, (zoom_h, zoom_w), order=1).astype(np.float32)
    except ImportError:
        h_orig, w_orig = slice_2d.shape
        h_new, w_new = target_size
        row_idx = (np.arange(h_new) * h_orig / h_new).astype(int)
        col_idx = (np.arange(w_new) * w_orig / w_new).astype(int)
        return slice_2d[np.ix_(row_idx, col_idx)].astype(np.float32)


def percentile_clip_normalize(volume: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5) -> np.ndarray:
    """
    Chuẩn hóa cường độ voxel 2 bước chuẩn OASIS/ADNI:
      1. Clip theo bách phân vị [0.5%, 99.5%] để khử outlier/nhiễu
      2. Z-score (mean=0, std=1)
    """
    vol = volume.astype(np.float32)
    p_low = np.percentile(vol, low_pct)
    p_high = np.percentile(vol, high_pct)
    vol = np.clip(vol, p_low, p_high)
    mean = vol.mean()
    std = vol.std()
    return ((vol - mean) / (std if std > 1e-8 else 1.0)).astype(np.float32)


def extract_triplanar_from_volume(volume: np.ndarray, n_slices: int = 9,
                                  start_pct: float = 0.30, end_pct: float = 0.70,
                                  target_size: tuple = (224, 224)) -> tuple:
    """
    Trích xuất 27 lát cắt Tri-planar (9 Axial, 9 Coronal, 9 Sagittal) từ thể tích não 3D.
    Trả về:
      - x: (27, 3, 224, 224) - tensor sẵn sàng cho ViT/ConvNeXt
      - raw: (27, 224, 224) - lát cắt ảnh chuẩn hóa để hiển thị & tính Grad-CAM
    """
    if volume.ndim == 4:
        volume = volume.squeeze()
        if volume.ndim == 4:
            volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Khối dữ liệu MRI 3D cần có 3 chiều (D, H, W). Nhận được shape: {volume.shape}")

    # Chuẩn hóa volume
    norm_vol = percentile_clip_normalize(volume)
    D, H, W = norm_vol.shape

    def _sample_indices(dim_size: int) -> np.ndarray:
        return np.clip(
            np.linspace(int(dim_size * start_pct), int(dim_size * end_pct), n_slices, dtype=int),
            0, dim_size - 1
        )

    axial_slices = [resize_slice(norm_vol[idx, :, :], target_size) for idx in _sample_indices(D)]
    coronal_slices = [resize_slice(norm_vol[:, idx, :], target_size) for idx in _sample_indices(H)]
    sagittal_slices = [resize_slice(norm_vol[:, :, idx], target_size) for idx in _sample_indices(W)]

    raw = np.stack(axial_slices + coronal_slices + sagittal_slices, axis=0)  # (27, 224, 224)
    x = np.stack([to_rgb(s) for s in raw], axis=0)  # (27, 3, 224, 224)
    return x, raw


def process_uploaded_mri(file_bytes: bytes, filename: str, n_slices: int = 9,
                         target_size: tuple = (224, 224)) -> tuple:
    """
    Đọc và tiền xử lý file MRI người dùng tải lên đa định dạng:
      - NIfTI: .nii, .nii.gz
      - NumPy: .npy, .npz
      - ZIP archive: chứa file slices.npy hoặc các file ảnh
      - 2D Image: .png, .jpg, .jpeg
    Trả về (x, raw, info_dict).
    """
    import tempfile
    import os
    import io
    import zipfile

    fname_lower = filename.lower()
    info = {"filename": filename, "format": "", "orig_shape": None}

    # 1. Định dạng NIfTI
    if fname_lower.endswith(".nii") or fname_lower.endswith(".nii.gz"):
        info["format"] = "NIfTI (3D Volume)"
        import nibabel as nib
        suffix = ".nii.gz" if fname_lower.endswith(".nii.gz") else ".nii"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            # Disable memory mapping so Windows can remove the temporary file.
            img = nib.load(tmp_path, mmap=False, keep_file_open=False)
            vol = img.get_fdata(dtype=np.float32)
            info["orig_shape"] = vol.shape
            x, raw = extract_triplanar_from_volume(vol, n_slices=n_slices, target_size=target_size)
            return x, raw, info
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # 2. Định dạng NumPy (.npy, .npz)
    elif fname_lower.endswith(".npy") or fname_lower.endswith(".npz"):
        info["format"] = "NumPy Array"
        bio = io.BytesIO(file_bytes)
        if fname_lower.endswith(".npz"):
            npz = np.load(bio)
            key = list(npz.keys())[0]
            arr = npz[key].astype(np.float32)
        else:
            arr = np.load(bio).astype(np.float32)
        info["orig_shape"] = arr.shape

        # Trường hợp 1: Mảng 3D volume (D, H, W)
        if arr.ndim == 3 and arr.shape[0] != 27:
            x, raw = extract_triplanar_from_volume(arr, n_slices=n_slices, target_size=target_size)
            return x, raw, info

        # Trường hợp 2: Đã là 27 lát cắt (27, H, W)
        if arr.ndim == 3 and arr.shape[0] == 27:
            if arr.shape[1:] != target_size:
                arr = np.stack([resize_slice(s, target_size) for s in arr], axis=0)
            raw = arr
            x = np.stack([to_rgb(s) for s in raw], axis=0)
            return x, raw, info

        # Trường hợp 3: (3 planes, 9 slices, H, W)
        if arr.ndim == 4 and arr.shape[0] == 3 and arr.shape[1] == 9:
            arr = arr.reshape(27, arr.shape[2], arr.shape[3])
            if arr.shape[1:] != target_size:
                arr = np.stack([resize_slice(s, target_size) for s in arr], axis=0)
            raw = arr
            x = np.stack([to_rgb(s) for s in raw], axis=0)
            return x, raw, info

        # Trường hợp 4: 4D volume (1, D, H, W) hoặc (D, H, W, 1)
        if arr.ndim == 4:
            arr = arr.squeeze()
            if arr.ndim == 3:
                x, raw = extract_triplanar_from_volume(arr, n_slices=n_slices, target_size=target_size)
                return x, raw, info

        raise ValueError(f"Shape của file numpy không phù hợp: {arr.shape}. Cần (D, H, W) hoặc (27, 224, 224).")

    # 3. File nén ZIP
    elif fname_lower.endswith(".zip"):
        info["format"] = "ZIP Archive"
        bio = io.BytesIO(file_bytes)
        with zipfile.ZipFile(bio, "r") as zf:
            # Tìm xem có file .npy nào bên trong không
            npy_files = [f for f in zf.namelist() if f.endswith(".npy")]
            if npy_files:
                npy_bytes = zf.read(npy_files[0])
                return process_uploaded_mri(npy_bytes, npy_files[0], n_slices=n_slices, target_size=target_size)
            # Tìm các file ảnh .png, .jpg
            img_files = sorted([f for f in zf.namelist() if f.lower().endswith((".png", ".jpg", ".jpeg")) and not f.startswith("__MACOSX")])
            if len(img_files) >= 27:
                from PIL import Image
                slices = []
                for f in img_files[:27]:
                    im = Image.open(io.BytesIO(zf.read(f))).convert("L").resize(target_size)
                    slices.append(np.array(im, dtype=np.float32) / 255.0)
                raw = np.stack(slices, axis=0)
                # Chuẩn hóa z-score
                raw = (raw - raw.mean()) / (raw.std() + 1e-8)
                x = np.stack([to_rgb(s) for s in raw], axis=0)
                return x, raw, info
        raise ValueError("File ZIP không chứa file lát cắt .npy hoặc đủ 27 ảnh MRI hợp lệ.")

    # 4. File ảnh đơn lẻ 2D (PNG, JPG) - dành cho chế độ thử nghiệm nhanh
    elif fname_lower.endswith((".png", ".jpg", ".jpeg")):
        info["format"] = "2D Single Image (Chế độ mô phỏng Tri-planar)"
        from PIL import Image
        im = Image.open(io.BytesIO(file_bytes)).convert("L").resize(target_size)
        base_slice = np.array(im, dtype=np.float32) / 255.0
        base_slice = (base_slice - base_slice.mean()) / (base_slice.std() + 1e-8)

        # Mô phỏng 27 lát cắt xung quanh ảnh đại diện với độ suy biến nhẹ theo độ sâu
        slices = []
        for i in range(27):
            scale = 1.0 - 0.015 * abs(i % 9 - 4)
            slices.append(base_slice * scale)
        raw = np.stack(slices, axis=0).astype(np.float32)
        x = np.stack([to_rgb(s) for s in raw], axis=0)
        return x, raw, info

    else:
        raise ValueError(f"Định dạng tệp không được hỗ trợ: {filename}. Vui lòng tải lên file .nii, .nii.gz, .npy, .zip hoặc .png/.jpg.")


def calculate_risk_score(probs: np.ndarray) -> dict:
    """
    Tính điểm nguy cơ sa sút trí tuệ tổng hợp dựa trên phân bố xác suất:
      Risk Score = P(VMD) * 50% + P(AD) * 100%
    Thang điểm: 0% -> 100%.
    """
    p_cn, p_vmd, p_ad = float(probs[0]), float(probs[1]), float(probs[2])
    score = (p_vmd * 50.0) + (p_ad * 100.0)
    score = max(0.0, min(100.0, score))

    if score < 30.0:
        level = "Thấp (Nguy cơ bình thường / Lão hóa khỏe mạnh)"
        badge_color = "#2e7d32"
        badge_bg = "#e8f5e9"
        icon = "🟢"
        recommendation = (
            "Dấu hiệu chức năng nhận thức trong giới hạn bình thường của lứa tuổi. "
            "Khuyến nghị duy trì lối sống lành mạnh, rèn luyện trí não và tái khám tầm soát định kỳ sau 12-24 tháng."
        )
    elif score < 60.0:
        level = "Trung bình (Nguy cơ suy giảm nhận thức nhẹ - VMD)"
        badge_color = "#f57f17"
        badge_bg = "#fffde7"
        icon = "🟡"
        recommendation = (
            "Phát hiện dấu hiệu teo não ranh giới hoặc suy giảm nhận thức giai đoạn sớm. "
            "Khuyến nghị thực hiện thêm bài kiểm tra nhận thức chuyên sâu (MMSE, MoCA, CDR) "
            "và chụp MRI kiểm tra đối chiếu sau 6-12 tháng."
        )
    else:
        level = "Cao (Nguy cơ cao mắc Bệnh Alzheimer / Sa sút trí tuệ tiến triển)"
        badge_color = "#c62828"
        badge_bg = "#ffebee"
        icon = "🔴"
        recommendation = (
            "Xác suất cao có biểu hiện tổn thương/teo vùng vỏ não thùy thái dương và hải mã phù hợp với bệnh lý Alzheimer. "
            "Khuyến nghị khẩn trương hội chẩn chuyên khoa Thần kinh học, đánh giá các dấu ấn sinh học bổ sung và lập phác đồ can thiệp."
        )

    return {
        "score": score,
        "level": level,
        "badge_color": badge_color,
        "badge_bg": badge_bg,
        "icon": icon,
        "recommendation": recommendation,
        "p_cn": p_cn * 100,
        "p_vmd": p_vmd * 100,
        "p_ad": p_ad * 100,
    }
