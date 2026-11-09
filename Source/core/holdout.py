"""Frozen 15-checkpoint inference; never fits preprocessing on a new case."""
import hashlib
import importlib.util
import json
import os
import threading
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import torch

FEATURES=['age','sex','education','ses','etiv','nwbv','asf']
CLASSES=['CN','VMD','AD']
ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT.parent / 'Runtime'
OUTPUT = Path(os.environ.get('TRIFUSE_HOLDOUT_ROOT') or RUNTIME / 'holdout')
DATA = RUNTIME / 'samples'
MODEL_NAME='TriFuse-AD — ensemble hold-out 15 checkpoint'

@lru_cache(maxsize=1)
def get_default_engine():
    return HoldoutEnsemble()

def normalize_slices(raw):
    a=np.asarray(raw,dtype=np.float32)
    if a.shape!=(27,224,224) or not np.isfinite(a).all():
        raise ValueError('Cần 27 lát cắt hữu hạn, kích thước 224 × 224.')
    # This is the same Torch min-max operation as TriFuseDataset._load_triplanar.
    t=torch.from_numpy(np.ascontiguousarray(a))
    lo=t.amin((1,2),keepdim=True);hi=t.amax((1,2),keepdim=True)
    t=(t-lo)/(hi-lo).clamp_min(1e-8)
    mean=torch.tensor([.485,.456,.406],dtype=torch.float32).view(1,3,1,1)
    std=torch.tensor([.229,.224,.225],dtype=torch.float32).view(1,3,1,1)
    return ((t.unsqueeze(1).expand(-1,3,-1,-1)-mean)/std).unsqueeze(0)

def fingerprint(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def load_training():
    path=ROOT/'Main/03_train_official_trifuse.py'
    spec=importlib.util.spec_from_file_location('holdout_app_training',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

class HoldoutEnsemble:
    def __init__(self,output=OUTPUT,device=None):
        self.output=Path(output);self.device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.lock=threading.RLock()
        audit=json.loads((self.output/'split_audit.json').read_text(encoding='utf-8'))
        if not audit.get('training_complete'):raise ValueError('Huấn luyện chưa hoàn tất.')
        self.protocol_id=audit['protocol_id'];self.members=[];self.preprocessing={}
        self.training=load_training();self.model=None
        for fold in range(5):
            prep=joblib.load(self.output/f'fold_{fold}_preprocessing.joblib')
            if prep['features']!=FEATURES or prep['protocol_id']!=self.protocol_id:
                raise ValueError('Sai bộ tiền xử lý hoặc thứ tự đặc trưng.')
            self.preprocessing[fold]=prep
        expected=[(s,f) for s in [42,123,2024] for f in range(5)]
        if audit['seeds']!=[42,123,2024]:raise ValueError('Sai danh sách seed.')
        for seed,fold in expected:
            path=self.output/'checkpoints/trifuse'/f'trifuse_seed{seed}_fold{fold}.pt'
            digest=fingerprint(path)
            if digest!=audit['checkpoint_sha256'].get(path.name):raise ValueError(f'Trọng số thay đổi: {path.name}')
            self.members.append(dict(seed=seed,fold=fold,path=path,sha256=digest))
        if len(self.members)!=15:raise ValueError('Cần đủ 15 checkpoint.')

    def encode_tab(self,clinical,fold):
        vals=np.array([[clinical.get(f,np.nan) for f in FEATURES]],dtype=np.float64)
        if np.isinf(vals).any():raise ValueError('Đặc trưng chứa giá trị vô hạn.')
        prep=self.preprocessing[fold]
        vals=prep['scaler'].transform(prep['imputer'].transform(vals)).astype(np.float32)
        return torch.from_numpy(vals)

    def _load_member(self,member):
        obj=torch.load(str(member['path']),map_location='cpu',weights_only=False,mmap=True)
        if (obj.get('seed'),obj.get('fold'))!=(member['seed'],member['fold']):
            raise ValueError('Sai seed/fold trong checkpoint.')
        cfg=dict(obj['cfg']);cfg['pretrained_backbone']=False
        if self.model is None:self.model=self.training.TriFuseAD(cfg)
        self.model.load_state_dict(obj['model_state_dict'],strict=True)
        del obj
        return self.model.to(self.device).eval()

    def predict_many(self,raws,clinical_rows,progress=None):
        """Load one checkpoint at a time; bounded memory for a local laptop."""
        from . import gradcam as G
        if len(raws)!=len(clinical_rows) or not len(raws):raise ValueError('Danh sách ca không hợp lệ.')
        with self.lock:
            probabilities=[];gates=[]
            for member in self.members:
                model=self._load_member(member);run=[];run_gates=[]
                try:
                    for raw,clinical in zip(raws,clinical_rows):
                        xb=normalize_slices(raw).to(self.device)
                        out=G.predict(model,xb,self.encode_tab(clinical,member['fold']).to(self.device))
                        run.append(out['probs'].copy());run_gates.append(out['gate'])
                        del xb
                finally:model.zero_grad(set_to_none=True);model.to('cpu')
                probabilities.append(run);gates.append(run_gates)
                if progress is not None:progress(member)
            return np.stack(probabilities),gates

    def predict(self,raw,clinical,with_cam=False,target=None,progress=None):
        from . import gradcam as G
        with self.lock:
            runs,gate_rows=self.predict_many([raw],[clinical],progress=progress)
            probs=runs[:,0,:];gates=[r[0] for r in gate_rows if r[0] is not None]
            mean=np.mean(probs,axis=0);pred=int(mean.argmax())
            cams=None;cam_source=None
            if with_cam:
                member=self.members[0];model=self._load_member(member)
                try:
                    xb=normalize_slices(raw).to(self.device)
                    cam=G.gradcam(model,xb,self.encode_tab(clinical,member['fold']).to(self.device),target=pred if target is None else target)
                    cams=cam['cams'];cam_source='Checkpoint seed 42, fold 0; lớp mục tiêu theo ensemble'
                finally:model.zero_grad(set_to_none=True);model.to('cpu')
            return dict(probs=mean,pred=pred,cams=cams,cam_source=cam_source,
                        target=pred if target is None else target,gate=None if not gates else float(np.mean(gates)),
                        member_probabilities=probs,models_averaged=15,protocol_id=self.protocol_id)
