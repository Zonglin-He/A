"""Finite dev calibration of existing native TENT/SAR ports, separate evidence."""
import copy
import hashlib
import math
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,status,sha,load as original_load
from scripts.stvg_opd_design_ablation_common_v1 import activate_old
BASE=ROOT/'artifacts/stvg_opd_supplement_v1/baselines'
PUB=ROOT/'results/stvg_opd_supplement/2026-10-11/baselines'
PAPER=ROOT/'artifacts/decota_paper_experiments_v1'
PYTHON=ROOT/'.conda/tubedetr/bin/python'
ALLOCATOR='expandable_segments:True'

def save(path,value):
    """Lossless complete evidence, including failed preacceptance payloads."""
    import gzip
    import io
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    buffer=io.BytesIO();torch.save(value,buffer)
    temporary=path.with_suffix(path.suffix+'.writing')
    with temporary.open('wb') as f:f.write(gzip.compress(buffer.getvalue(),compresslevel=1,mtime=0))
    temporary.replace(path)

def load(path):
    """The old immutable raw torch files remain supported byte for byte."""
    import gzip
    import io
    import torch
    path=Path(path)
    with path.open('rb') as f:magic=f.read(2)
    if magic==b'\x1f\x8b':
        return torch.load(io.BytesIO(gzip.decompress(path.read_bytes())),map_location='cpu',weights_only=False)
    return original_load(path)

def candidate_configs():
    from vg_tta.decota_paper_baselines_20261005_v1 import CONFIGS
    result={}
    for lr in [.0001,.001,.01]:
        result[f'TENT_lr{lr:g}']=dict(method='TENT',config=dict(CONFIGS['TENT'],lr=lr))
    for lr in [.0001,.001]:
        for margin in [.4,.6,.8]:
            result[f'SAR_lr{lr:g}_m{margin:g}']=dict(method='SAR',config=dict(CONFIGS['SAR'],lr=lr,margin_fraction=margin))
    return result

def design(): return read(BASE/'DESIGN_LOCK.json')
def active_runtime_path():
    p=BASE/'recovery/qualification_optimizer_isolation005/REVISION_RUNTIME.json'
    if p.exists():return p
    p=BASE/'preGPU_lossless_storage_engineering004/REVISION_RUNTIME.json'
    if p.exists():return p
    p=BASE/'preGPU_default_state_scope_engineering003/REVISION_RUNTIME.json'
    if p.exists(): return p
    p=BASE/'preGPU_controller_readback_engineering002/REVISION_RUNTIME.json'
    if p.exists(): return p
    p=BASE/'preGPU_input_export_engineering001/REVISION_RUNTIME.json'
    return p if p.exists() else BASE/'RUNTIME_LOCK.json'

def runtime_sha(): return sha(active_runtime_path())

def verify():
    r=read(active_runtime_path()); assert sha(BASE/'DESIGN_LOCK.json')==r['design_sha256']
    for p,h in {**r['pins'],**r['protected']}.items(): assert sha(ROOT/p)==h,p
    return r

def prepare():
    if (BASE/'RUNTIME_LOCK.json').exists(): return verify()
    old=read(ROOT/'artifacts/decota_spatial_opd_tuning_v1/DESIGN_LOCK.json')['datasets']; ds={}
    for name in ['hc2','vidstg']:
        z=old[name]; p=read(PAPER/name/'PLAN.json'); seq=z['refine_orders']
        assert all(len(v)==32 and len(set(v))==32 for v in seq.values()) and len(seq)==2
        assert set(seq['order1'])==set(seq['order2']) and len({p['rows'][q]['source'] for q in seq['order1']})==32
        ds[name]=dict(source=z['source'],development_orders=seq,development_parents=z['development_parents'],
            excluded_primary_parent_sources=sorted({p['rows'][q]['source'] for q in seq['order1']}),
            formal_plan_sha256=sha(PAPER/name/'PLAN.json'),formal_queries=p['queries'],formal_arrivals=p['queries']*3)
    d=dict(version='stvg_opd_baseline_dev_calibration_v1',datasets=ds,candidates=candidate_configs(),
        dev_arrivals=1152,qualification_arrivals=36,qualification_actual_arrive_calls=108,
        selection='max dev parent-macro final vIoU across two orders; ties1e-12 harm>20pp, smaller LR, smaller margin',
        all_dev_predictions_before_dev_GT=True,all_selected_formal_predictions_before_formal_GT=True,
        formal_primary_excludes_development_parents=True,formal_max_new_arrivals=82710,
        native_entropy='H(start)+H(end)+mean binary actionness; offset average; nats',
        forced_updates=False,main_method_parameter_selection=False,historical_dev_exposure=True,
        source_reset_per_order=True,persistent_baseline_optimizer_within_stream=True,
        baseline_scope='live ground_decoder LayerNorm affine, eval dropout, post-update native readout',
        TubeDETR=False,EATA=False,production_promotion=False,time=time.time())
    write(BASE/'DESIGN_LOCK.json',d)
    files=['protocols/stvg_opd_supplement_v1.md','scripts/stvg_opd_baseline_supplement_common_v1.py','scripts/run_stvg_opd_baseline_supplement_v1.py','scripts/score_stvg_opd_baseline_supplement_v1.py','scripts/continue_stvg_opd_baseline_supplement_v1.py',
        'vg_tta/decota_paper_baselines_20261005_v1.py','vg_tta/decota_paper_baseline_math_v1.py','vg_tta/native_baselines_paper_v1.py','vg_tta/native_probability_interface_v1.py','vg_tta/tastvg_baseline_expansion.py','scripts/run_decota_paper_main_v1.py','scripts/score_stvg_opd_p1_v1.py','vg_tta/tastvg_oracle_event5_v1.py']
    protected=['methods/CURRENT_METHOD.json','methods/decota_spatial_opd_v1/configs.json','artifacts/stvg_opd_paper_hc2_revision_v2/FINAL_COMPLETION.json',
        'artifacts/decota_spatial_opd_tuning_v1/DESIGN_LOCK.json','artifacts/decota_paper_experiments_v1/baselines/RUNTIME_LOCK.json']+[str((PAPER/n/'PLAN.json').relative_to(ROOT)) for n in ds]
    for m in ['TENT','SAR']:
        for n in ds: protected.append(str((PAPER/'baselines'/f'{m}_{n}'/'PREDICTION_BARRIER.json').relative_to(ROOT)))
    write(BASE/'RUNTIME_LOCK.json',dict(status='locked_before_dev_GPU_and_GT',design_sha256=sha(BASE/'DESIGN_LOCK.json'),pins={p:sha(ROOT/p) for p in files},protected={p:sha(ROOT/p) for p in protected},GT_read=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='prepared_waiting_single_GPU_after_order_experiment',dev_arrivals_total=1152,GT_read=False,time=time.time()))
    return verify()

def audit_baseline(z):
    """Independent entropy scale/gating and NumPy live-gradient optimizer arithmetic."""
    import numpy as np
    import torch
    from vg_tta.decota_paper_baseline_math_v1 import audit_updates
    check=audit_updates(z['audit']); ent=[]; maximum=0.
    for c in z['closure_evidence']:
        parts=[]; ceilings=[]
        for o in c:
            logits=o['sted'].numpy().astype(float); a=o['actionness_logits'].numpy().astype(float).reshape(-1)
            mask=o['valid'].numpy(); logits=logits[mask]; a=a[mask]
            p=np.exp(logits-logits.max(0)); p/=p.sum(0)
            categorical=-np.sum(p*np.log(np.maximum(p,np.finfo(float).tiny)))
            # Stable Bernoulli entropy from logits, no extra sigmoid or normalization.
            ap=1/(1+np.exp(-np.clip(a,-709,709)))
            binary=np.mean(np.logaddexp(0,a)-a*ap)
            parts.append(float(categorical+binary)); ceilings.append(2*math.log(len(a))+math.log(2))
        ent.append((float(np.mean(parts)),float(np.mean(ceilings))))
    k=0; counts=z['audit']['counts']; cfg=z['config']; updates=0; skips=0
    for t in z['audit']['trace']:
        h,ceiling=ent[k]; k+=1; maximum=max(maximum,abs(t['entropy']-h)); assert abs(t['entropy']-h)<1e-5
        if z['method']=='SAR':
            assert abs(t['margin']-cfg['margin_fraction']*ceiling)<1e-12
            if t.get('skip')=='entropy': assert t['entropy']>=t['margin']; skips+=1; continue
            assert t['entropy']<t['margin']
            second,_=ent[k]; k+=1; assert abs(t['second_entropy']-second)<1e-5
            maximum=max(maximum,abs(t['second_entropy']-second))
            if t.get('skip')=='second_entropy': assert t['second_entropy']>=t['margin']; skips+=1; continue
            assert t['second_entropy']<t['margin']
        updates+=int(t['updated'])
    assert k==len(ent)==counts['forward_closures'] and updates==counts['optimizer_steps']
    assert z['state_after']['arrivals']==z['state_before']['arrivals']+1==z['audit']['arrival']
    assert z['state_after']['config']==z['state_before']['config']==cfg
    assert z['state_before']['names']==z['state_after']['names']
    if updates==0:
        assert all(torch.equal(a,b) for a,b in zip(z['state_before']['parameters'],z['state_after']['parameters']))
    return dict(status='pass',optimizer=check,native_entropy_closures=k,max_entropy_formula_error=maximum,actual_updates=updates,actual_skips=skips,GT_read=False,independent_full_decoder_Jacobian=False)

if __name__=='__main__': prepare(); print('BASELINE_SUPPLEMENT_LOCKED',sha(BASE/'RUNTIME_LOCK.json'),flush=True)
