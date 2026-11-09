"""T1 brain extraction, tissue segmentation, and affine MNI registration."""
import io
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile

import nibabel as nib
import numpy as np


FSL_SCRIPT = r"""
set -eo pipefail
cd "$1"
export FSLDIR="${2:-${FSLDIR:-$HOME/fsl}}"
if [ ! -f "$FSLDIR/etc/fslconf/fsl.sh" ]; then
    echo "FSL not found at $FSLDIR. Set TRIFUSE_FSLDIR to its Linux location." >&2
    exit 2
fi
. "$FSLDIR/etc/fslconf/fsl.sh"
export FSLOUTPUTTYPE=NIFTI_GZ
export PATH="$FSLDIR/share/fsl/bin:$FSLDIR/bin:$PATH"
atlas="$FSLDIR/data/standard/MNI152_T1_2mm_brain.nii.gz"
test -f "$atlas" || { echo 'MNI152 brain template missing.' >&2; exit 2; }
bet t1.nii.gz brain.nii.gz -R -m
fast -t 1 -n 3 -o tissue brain.nii.gz
flirt -in brain.nii.gz -ref "$atlas" -dof 12 -cost corratio -out brain_mni.nii.gz -omat native_to_mni.mat
flirt -in tissue_seg.nii.gz -ref "$atlas" -applyxfm -init native_to_mni.mat -interp nearestneighbour -out tissue_mni.nii.gz
cp "$atlas" atlas_mni.nii.gz
"""


def _run(command, timeout):
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Xử lý MRI vượt thời gian cho phép.") from exc
    except OSError as exc:
        raise RuntimeError("Không mở được Ubuntu WSL hoặc Bash.") from exc
    if result.returncode:
        log = result.stderr or result.stdout
        encoding = "utf-16-le" if b"\x00" in log else "utf-8"
        raise RuntimeError(log.decode(encoding, errors="replace").strip()[-1000:] or "FSL dừng với lỗi.")
    encoding = "utf-16-le" if b"\x00" in result.stdout else "utf-8"
    return result.stdout.decode(encoding, errors="replace").lstrip("\ufeff").strip()


def _validate_input(image):
    if len(image.shape) != 3 or min(image.shape) < 16:
        raise ValueError("Cần ảnh T1 NIfTI 3D toàn đầu, ví dụ .nii.gz.")
    if max(image.shape) > 512 or np.prod(image.shape) > 64_000_000:
        raise ValueError("Kích thước ảnh vượt giới hạn xử lý cục bộ.")
    if not (image.header["qform_code"] or image.header["sform_code"]):
        raise ValueError("NIfTI thiếu thông tin hướng ảnh qform/sform.")
    affine = image.affine
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-8:
        raise ValueError("Ma trận tọa độ NIfTI không hợp lệ.")
    data = image.get_fdata(dtype=np.float32, caching="unchanged")
    if not np.isfinite(data).all() or float(data.max()) <= float(data.min()):
        raise ValueError("Ảnh có giá trị không hữu hạn hoặc cường độ đồng nhất.")


def process_t1(payload, filename):
    name = filename.lower()
    if not name.endswith((".nii", ".nii.gz")):
        raise ValueError("Hãy chọn ảnh T1 NIfTI 3D (.nii hoặc .nii.gz).")
    with tempfile.TemporaryDirectory(prefix="trifuse_mri3d_") as directory:
        folder = Path(directory)
        input_path = folder / ("t1.nii.gz" if name.endswith(".gz") else "t1.nii")
        input_path.write_bytes(payload)
        try:
            image = nib.load(input_path)
        except nib.filebasedimages.ImageFileError as exc:
            raise ValueError("Tệp không phải ảnh NIfTI đọc được.") from exc
        _validate_input(image)
        if input_path.suffix == ".nii":
            nib.save(image, folder / "t1.nii.gz")

        fsl_dir = os.environ.get("TRIFUSE_FSLDIR", "")
        if os.name == "nt":
            distro = os.environ.get("TRIFUSE_WSL_DISTRO")
            if not distro:
                installed = _run(["wsl.exe", "--list", "--quiet"], 15).splitlines()
                distro = "TriFuse-FSL" if "TriFuse-FSL" in installed else "Ubuntu"
            if distro == "TriFuse-FSL" and not fsl_dir:
                fsl_dir = "/opt/trifuse-fsl"
            runner = ["wsl.exe", "-d", distro, "--exec"]
            linux_dir = _run([*runner, "wslpath", "-a", str(folder)], 30)
        else:
            runner = []
            linux_dir = str(folder)
        # Keep shell text fixed and pass the working directory as a positional argument.
        _run([*runner, "bash", "-c", FSL_SCRIPT, "trifuse-fsl", linux_dir, fsl_dir], 1800)
        return _collect(folder)


def _collect(folder):
    template = nib.as_closest_canonical(nib.load(folder / "atlas_mni.nii.gz"))
    atlas = template.get_fdata(dtype=np.float32)
    previews = {"atlas": [], "brain": [], "tissues": []}
    for key, filename in (("atlas", "atlas_mni.nii.gz"), ("brain", "brain_mni.nii.gz"),
                          ("tissues", "tissue_mni.nii.gz")):
        image = nib.as_closest_canonical(nib.load(folder / filename))
        data = image.get_fdata(dtype=np.float32)
        if data.shape != atlas.shape or not np.allclose(image.affine, template.affine, atol=1e-4):
            raise ValueError("Ảnh đầu ra lệch lưới atlas.")
        if not np.isfinite(data).all() or not np.any(data):
            raise ValueError("Ảnh đầu ra trống hoặc chứa giá trị không hợp lệ.")
        if key == "tissues" and not np.array_equal(np.unique(data), [0, 1, 2, 3]):
            raise ValueError("Bản phân đoạn thiếu lớp mô hoặc có nhãn lạ.")
        for axis in (2, 1, 0):
            previews[key].append(np.rot90(np.take(data, data.shape[axis] // 2, axis=axis)))

    files = ["brain.nii.gz", "brain_mask.nii.gz", "tissue_seg.nii.gz",
             "tissue_pve_0.nii.gz", "tissue_pve_1.nii.gz", "tissue_pve_2.nii.gz",
             "brain_mni.nii.gz", "tissue_mni.nii.gz", "native_to_mni.mat", "atlas_mni.nii.gz"]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for filename in files:
            archive.write(folder / filename, arcname=filename)
    return previews, buffer.getvalue()
