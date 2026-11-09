"""Quan ly luu tru lich su chan doan benh nhan xuong o dia (persistent storage)."""
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import uuid
import numpy as np

HISTORY_DIR = Path(__file__).resolve().parents[2] / "Runtime" / "saved_predictions"
HISTORY_CSV = HISTORY_DIR / "patient_history.csv"


def _json_safe(value):
    if isinstance(value, dict):return {k:_json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):return _json_safe(value.tolist())
    if isinstance(value, np.generic):return _json_safe(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def save_prediction_snapshot(result, filename):
    """Save each visit separately; metadata is committed after MRI data."""
    record_id=uuid.uuid4().hex
    folder=HISTORY_DIR/'records'/record_id
    folder.mkdir(parents=True,exist_ok=False)
    raw=np.asarray(result['raw_slices'],dtype=np.float32)
    if raw.shape!=(27,224,224) or not np.isfinite(raw).all():
        raise ValueError('Ảnh lưu hồ sơ phải có 27 lát 224 × 224 hợp lệ.')
    arrays={'raw_slices':raw}
    if result.get('cams') is not None:arrays['cams']=np.asarray(result['cams'],dtype=np.float32)
    np.savez_compressed(folder/'images.npz',**arrays)
    metadata={k:v for k,v in result.items() if k not in {'raw_slices','cams'}}
    metadata.update(record_id=record_id,saved_at=datetime.now().isoformat(timespec='seconds'),filename=filename,has_cam='cams' in arrays)
    metadata=_json_safe(metadata)
    temporary=folder/'metadata.tmp'
    temporary.write_text(json.dumps(metadata,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
    temporary.replace(folder/'metadata.json')
    return record_id


def list_prediction_snapshots():
    records=[]
    for path in (HISTORY_DIR/'records').glob('*/metadata.json'):
        try:
            info=json.loads(path.read_text(encoding='utf-8'))
            if info.get('record_id')==path.parent.name:records.append(info)
        except (OSError,ValueError):continue
    return sorted(records,key=lambda r:r['saved_at'],reverse=True)


def load_prediction_snapshot(record_id):
    if len(record_id)!=32 or any(c not in '0123456789abcdef' for c in record_id):
        raise ValueError('Mã hồ sơ không hợp lệ.')
    folder=HISTORY_DIR/'records'/record_id
    info=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    with np.load(folder/'images.npz',allow_pickle=False) as images:
        info['raw_slices']=images['raw_slices'].copy()
        info['cams']=images['cams'].copy() if 'cams' in images else None
    info['probs']=np.asarray(info['probs'],dtype=np.float32)
    return info


def ensure_history_dir():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def save_patient_record(patient_id: str, mri_name: str, clinical_dict: dict,
                        model_name: str, pred_class: str, probs: list,
                        risk_info: dict, gate_val: float = None, notes: str = "") -> pd.DataFrame:
    """Luu thong tin dinh danh va ket qua chan doan benh nhan xuong file CSV tren o dia."""
    ensure_history_dir()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    row = {
        "ThoiGian": now_str,
        "MaBenhNhan": str(patient_id).strip() or "BN-ANON",
        "TepMRI": str(mri_name),
        "Tuoi": clinical_dict.get("age", ""),
        "GioiTinh": "Nam" if clinical_dict.get("sex") == 1 else "Nữ",
        "HocVan": clinical_dict.get("education", ""),
        "SES": clinical_dict.get("ses", ""),
        "eTIV": clinical_dict.get("etiv", ""),
        "nWBV": clinical_dict.get("nwbv", ""),
        "ASF": clinical_dict.get("asf", ""),
        "MoHinh": model_name,
        "GiaiDoanDuDoan": pred_class,
        "XacSuat_CN_%": round(float(probs[0]) * 100, 2),
        "XacSuat_VMD_%": round(float(probs[1]) * 100, 2),
        "XacSuat_AD_%": round(float(probs[2]) * 100, 2),
        "ChiSoNguyCo_%": round(float(risk_info.get("score", 0)), 2),
        "MucDoNguyCo": risk_info.get("level", "").split("(")[0].strip(),
        "Gate": round(float(gate_val), 3) if gate_val is not None else "N/A",
        "GhiChu": notes
    }

    df_new = pd.DataFrame([row])
    if HISTORY_CSV.exists():
        try:
            df_old = pd.read_csv(HISTORY_CSV, encoding="utf-8-sig")
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            df_all = df_new
    else:
        df_all = df_new

    df_all.to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")
    return df_all


def load_patient_history() -> pd.DataFrame:
    """Doc toan bo danh sach lich su cac ca benh nhan da duoc chan doan."""
    if not HISTORY_CSV.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(HISTORY_CSV, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()


def clear_patient_history() -> bool:
    """Xoa toan bo lich su chan doan neu nguoi dung muon don dep."""
    if HISTORY_CSV.exists():
        try:
            HISTORY_CSV.unlink()
            return True
        except Exception:
            return False
    return True
