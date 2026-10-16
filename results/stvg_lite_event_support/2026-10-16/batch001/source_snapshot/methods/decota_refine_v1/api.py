"""Frozen trajectory correction and matched baselines; no optimizer or GT."""
import importlib.util
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from torchvision.ops import box_convert
from methods.decota_s_v1.api import interpolate_corrections, valid_boxes

ROOT=Path(__file__).resolve().parents[2]
MODES=('direct','absolute','residual','gsi_official','gsi_matched')


@lru_cache(maxsize=1)
def official_gsi():
    """Load the unchanged, separately retained GPL-3.0 upstream source."""
    path=ROOT/'external/StrongSORT/GSI.py'
    spec=importlib.util.spec_from_file_location('decota_strongsort_gsi',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class LegacyShapeGPR(module.GPR):
        # sklearn 1.9 squeezes a single target; upstream expects [N,1].
        # Preserve values, kernel, noise and all fitting defaults unchanged.
        def predict(self,X,*args,**kwargs):
            result=super().predict(X,*args,**kwargs)
            return result[:,None] if isinstance(result,np.ndarray) and result.ndim==1 else result
    module.GPR=LegacyShapeGPR
    return module


def _apply(base,positions,values):
    out=base.clone();invalid=[]
    values=torch.as_tensor(values,dtype=base.dtype)
    values=box_convert(box_convert(values,'cxcywh','xyxy').clamp(0,1),'xyxy','cxcywh')
    for i,pos in enumerate(positions):
        if valid_boxes(values[i]):out[int(pos)]=values[i]
        else:invalid.append(int(pos))
    return out,invalid


@torch.no_grad()
def refine(base,pseudo,frame_ids,mode='residual'):
    """Default is EXACTLY the previously sealed cxcywh correction interpolator.

    All baselines use identical accepted anchors and fixed output time.
    Absolute and matched-GSI share the anchor hull. Official-GSI uses upstream
    interval=20 raw frames and tau=10, so its fill coverage can be smaller.
    Matched-GSI removes this gap cutoff by smoothing the already dense absolute
    interpolation on the same sampled hull; it is explicitly an adaptation.
    """
    if mode not in MODES:raise ValueError(mode)
    # Native TubeDETR boxes may be BF16. Widen the cached values exactly,
    # then use the same FP32 correction arithmetic as the sealed TA path.
    base=base.detach().cpu().float();ids=np.asarray(frame_ids)
    assert len(base)==len(ids) and np.all(np.diff(ids)>0)
    if mode=='residual':return interpolate_corrections(base,pseudo,frame_ids)
    if not pseudo:return base.clone(),dict(mode=mode,changed_support=0,fallback='empty_supervision')
    pp=sorted(pseudo,key=lambda p:p['position']);pos=np.array([p['position'] for p in pp])
    target=np.asarray([p['box'] for p in pp],dtype=np.float32)
    assert len(set(pos.tolist()))==len(pos) and pos.min()>=0 and pos.max()<len(base)
    if not valid_boxes(torch.from_numpy(target)):
        return base.clone(),dict(mode=mode,changed_support=0,fallback='invalid_pseudo_boxes')
    if mode=='direct' or len(pos)==1:
        # Exact direct replacement, no second rounding/clipping of accepted boxes.
        out=base.clone();out[pos]=torch.from_numpy(target)
        return out,dict(mode=mode,changed_support=len(pos),single_anchor=len(pos)==1)
    hull=np.arange(pos[0],pos[-1]+1)
    absolute=np.stack([np.interp(ids[hull],ids[pos],target[:,k]) for k in range(4)],axis=1)
    if mode=='absolute':
        out,invalid=_apply(base,hull,absolute)
        return out,dict(mode=mode,changed_support=len(hull),invalid=invalid)
    gsi=official_gsi()
    if mode=='gsi_official':
        xywh=box_convert(torch.from_numpy(target),'cxcywh','xywh').numpy()
        data=np.column_stack([ids[pos],np.ones(len(pos)),xywh,np.ones(len(pos)),np.full((len(pos),3),-1.)])
        smooth=np.asarray(gsi.GaussianSmooth(gsi.LinearInterpolation(data,interval=20),tau=10))
    else:
        xywh=box_convert(torch.tensor(absolute),'cxcywh','xywh').numpy()
        data=np.column_stack([ids[hull],np.ones(len(hull)),xywh,np.ones(len(hull)),np.full((len(hull),3),-1.)])
        smooth=np.asarray(gsi.GaussianSmooth(data,tau=10))
    lookup={int(row[0]):row[2:6] for row in smooth}
    supported=[int(i) for i in hull if int(ids[i]) in lookup]
    values=box_convert(torch.tensor(np.array([lookup[int(ids[i])] for i in supported])),'xywh','cxcywh')
    out,invalid=_apply(base,supported,values)
    return out,dict(mode=mode,changed_support=len(supported),hull_size=len(hull),invalid=invalid,
        tau=10,interval=20 if mode=='gsi_official' else None,
        upstream='StrongSORT ee995076da5083e28d0da1f885297df62705ebd7',
        coordinate_adaptation='normalized TLWH, physical original frame IDs',
        matched_support_adaptation=mode=='gsi_matched')


def spatial_variants(base,pseudo,frame_ids):
    return {name:refine(base,pseudo,frame_ids,name) for name in MODES}
