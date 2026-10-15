#!/usr/bin/env python3
"""Small synthetic CPU contract for the LR-x2 M0matched/M1/M2 fit APIs."""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
BASE=ROOT/'artifacts/stvg_soft_ep_opd_v1/lr2'
def sha(p):
    import hashlib
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def main():
    import torch
    from scripts import stvg_target_teacher_student_v1 as core
    from scripts.stvg_reference_recovery_fit_v1 import locked_config
    from scripts.stvg_weightfree_coordinate_fit_v1 import fit_validated
    from scripts.stvg_soft_ep_fit_v1 import fit
    out=BASE/'CPU_SYNTHETIC_READBACK.json'
    if out.exists(): raise FileExistsError('refusing to overwrite synthetic contract receipt')
    base=core._synthetic_student_base(8); cfg=locked_config('hc2','current5'); cfg['lr']*=2.0
    boxes=base.zero['boxes']
    anchors=[{'position':1,'physical_frame_id':1,'box':boxes[1]+torch.tensor([.01,0.,0.,0.]),'weight':1.0},
        {'position':3,'physical_frame_id':3,'box':boxes[3]+torch.tensor([-.01,0.,0.,0.]),'weight':1.0}]
    ids=list(range(8)); key='caption_query_fixed_space|hc2_synthetic'; ref='a'*64
    f0=fit_validated(base,base.initial,anchors,ids,key,cfg,validate=True,reference_sha256=ref)
    settings={'mu':.25,'cg_max_iter':16,'cg_rtol':.001,'response_ratio':.75,'response_abs_tolerance':1e-7,'backtracking':[1.,.5,.25,.125]}
    results={'M0matched':f0};
    for mode in ('M1','M2'): results[mode]=fit(base,base.initial,anchors,ids,key,cfg,mode,settings,qualification=False)
    for name,result in results.items():
        assert result.get('GT_read') is False and result['config']==cfg and len(result['path'])==5
        assert torch.isfinite(result['final_cached_decoder_boxes']).all()
        if name=='M0matched': assert result['schema']=='stvg-instance-loss-weighted-fit-v1' and result['algorithm']=='validated_loss_weighted_iou'
        else: assert result['schema']=='stvg-soft-ep-fit-v1' and result['mode']==name
    assert f0['proposed_optimizer_steps']==f0['optimizer_steps']+f0['rejected_optimizer_steps']
    assert not torch.cuda.is_initialized()
    payload=dict(schema='stvg-soft-ep-lr2-synthetic-contract-v1',status='pass',CPU_only=True,model_loaded=False,GPU_initialized=False,GT_read=False,
        cases=1,synthetic_grid_frames=8,synthetic_anchor_count=2,fit_arms=['M0matched','M1','M2'],steps_per_fit=5,
        accepted_steps={name:int(result['optimizer_steps']) for name,result in results.items()},
        proposed_steps={name:int(result['proposed_optimizer_steps']) for name,result in results.items()},
        config=cfg,settings=settings,bindings={name:sha(ROOT/'scripts'/name) for name in (
            'stvg_soft_ep_fit_v1.py','stvg_soft_ep_response_v1.py','stvg_weightfree_coordinate_fit_v1.py','stvg_reference_recovery_fit_v1.py')},time=time.time())
    tmp=out.with_name(out.name+'.tmp')
    tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8'); os.replace(tmp,out)
    print(json.dumps(payload,indent=2,allow_nan=False))
if __name__=='__main__': main()
