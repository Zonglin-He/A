"""Limited human-authorized post-seal CPU scoring; never imports a model."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
import hashlib
import json
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PAPER = ROOT/'artifacts/decota_paper_experiments_v1'
BASE = ROOT/'artifacts/decota_paper_baseline_scoring_v1'
PUB = ROOT/'results/decota_paper_baseline_scoring/2026-10-08'
METHODS = ['Source Only','DINO-Refine','TENT-STVG','SAR-STVG','Target-trained reference']
SEED = 20261008

def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def write(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def status(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)

def groups():
    out=[]
    for ds,n,job in [('vidstg',10303,'t1_ours_vid'),('hc2',3482,'t1_ours_hc2')]:
        out.append(dict(dataset=ds,kind='stateless',methods=METHODS[:2],queries=n,
                        barrier=f'table1_stateless/{job}/PREDICTION_BARRIER.json',
                        paths=[f'table1_stateless/{job}/{i:05}.npz' for i in range(n)]))
        out.append(dict(dataset=ds,kind='reference',methods=[METHODS[4]],queries=n,
                        barrier=f'table1_target_reference/{ds}/PREDICTION_BARRIER.json',
                        paths=[f'table1_target_reference/{ds}/{i:05}.npz' for i in range(n)]))
        for method in ['TENT','SAR']+(['EATA'] if ds=='hc2' else []):
            out.append(dict(dataset=ds,kind=method,methods=[method+'-STVG'],queries=n,
                            barrier=f'baselines/{method}_{ds}/PREDICTION_BARRIER.json',
                            paths=[f'baselines/{method}_{ds}/order{o}/{i:05}.npz' for o in (1,2,3) for i in range(n)]))
    return out

def seal():
    from scripts.audit_decota_paper_stage_bytes_v1 import check_barrier,check_payload
    from concurrent.futures import ThreadPoolExecutor
    BASE.mkdir(parents=True,exist_ok=True)
    if (BASE/'EVALUATION_BARRIER.json').exists(): return verify()
    assert read(PAPER/'EATA_USER_HOLD.json')['status']=='user_eata_paused'
    rh=sha(PAPER/'RUNTIME_LOCK.json'); proofs=[]
    for g in groups():
        status(BASE/'STATUS.json',dict(status='verifying_selected_prediction_seals',dataset=g['dataset'],kind=g['kind'],GT_read=False,time=time.time()))
        b,r=check_barrier(PAPER,g['barrier'],g['paths'],rh)
        assert b.get('dataset',g['dataset'])==g['dataset']
        assert b.get('method',g['kind'])==g['kind'] or g['kind'] in ('stateless','reference')
        proofs.append({k:v for k,v in g.items() if k!='paths'}|r|dict(seal_time=b['time']))
        print('SELECTED_BARRIER_VERIFIED',g['dataset'],g['kind'],len(g['paths']),flush=True)
    shared={}
    for ds,n,job in [('vidstg',10303,'t1_ours_vid'),('hc2',3482,'t1_ours_hc2')]:
        paths=[f'{job}/inputs/clean/{i:05}.npz' for i in range(n)]
        hashes={p:read((PAPER/p).with_suffix('.json'))['sha256'] for p in paths}
        seal_time=read(PAPER/f'table1_stateless/{job}/PREDICTION_BARRIER.json')['time']
        with ThreadPoolExecutor(max_workers=4) as pool:
            sizes=list(pool.map(lambda p:check_payload(PAPER,p,hashes[p],rh,seal_time),paths))
        shared[ds]=dict(files=hashes,verified_bytes=sum(sizes),GT_read=False,before_control_seal=seal_time)
    write(BASE/'SHARED_INPUT_BYTE_PROOF.json',shared)
    gt=['artifacts/stvg_fullscale_diagnostics_v1/labels_diagnostic_only.json',
        'external/VidSTG-Dataset/annotations/test_annotations.json','downloads/vidor/validation-annotation.zip',
        'data/hcstvg2_official_metadata/val_v2.json']
    inputs=['methods/CURRENT_METHOD.json','protocols/decota_paper_baseline_scoring_v1.md',
            'artifacts/decota_paper_experiments_v1/DESIGN_LOCK.json',
            'artifacts/decota_paper_experiments_v1/RUNTIME_LOCK.json',
            'artifacts/decota_paper_experiments_v1/baselines/RUNTIME_LOCK.json',
            'artifacts/decota_paper_experiments_v1/table1_stateless/RUNTIME_LOCK.json',
            'artifacts/decota_paper_experiments_v1/table1_target_reference/RUNTIME_LOCK.json']
    inputs += [str((PAPER/ds/'PLAN.json').relative_to(ROOT)) for ds in ['vidstg','hc2']]
    inputs += [str((PAPER/g['barrier']).relative_to(ROOT)) for g in groups()]
    inputs.append(str((BASE/'SHARED_INPUT_BYTE_PROOF.json').relative_to(ROOT)))
    for ds,job in [('vidstg','t1_ours_vid'),('hc2','t1_ours_hc2')]:
        for method in ['TENT','SAR']+(['EATA'] if ds=='hc2' else []):
            inputs.append(str((PAPER/f'baselines/{method}_{ds}/PARAMETER_SCOPE.json').relative_to(ROOT)))
            inputs += [str((PAPER/f'baselines/smoke/{job}/{method}/{i:05}.pt').relative_to(ROOT)) for i in [0,1]]
    code=['scripts/decota_paper_baseline_scoring_common_v1.py','scripts/score_decota_paper_baseline_scoring_v1.py',
          'scripts/run_decota_paper_baseline_scoring_v1.py',
          'scripts/audit_decota_paper_baseline_scoring_v1.py','scripts/report_decota_paper_baseline_scoring_v1.py',
          'scripts/test_decota_paper_baseline_scoring_v1.py','scripts/audit_decota_paper_stage_bytes_v1.py','scripts/diagnose_tastvg_pipeline_cpu_v1.py',
          'scripts/prepare_vidstg_wrong_domain_support.py','vg_tta/tastvg_oracle_event5_v1.py',
          'vg_tta/tastvg_paper48_metrics_v1.py','vg_tta/tastvg_paper48_hc2_metrics_v1.py',
          'vg_tta/decota_paper_controls_v1.py','vg_tta/decota_paper_baseline_math_v1.py',
          'external/TA-STVG/utils/box_utils.py','external/TA-STVG/engine/evaluate.py',
          'external/TA-STVG/datasets/evaluation/vidstg_eval.py','external/TA-STVG/datasets/evaluation/hcstvg_eval.py']
    assert sum(p['prediction_files_verified'] for p in proofs)==120726
    logical=sum(g['queries']*3*len(g['methods']) for g in groups());assert logical==217221
    write(BASE/'RUNTIME_LOCK.json',dict(pins={f:sha(ROOT/f) for f in code},inputs={f:sha(ROOT/f) for f in inputs},
                                     GT_files={f:sha(ROOT/f) for f in gt},original_science_unchanged=True,time=time.time()))
    write(BASE/'EVALUATION_BARRIER.json',dict(status='sealed',human_authorized_after_EATA_pause=True,
          methods_both_directions=METHODS,EATA_HC2_only='sealed_one_direction_supplement',
          EATA_VidSTG='paused_unavailable',OPD='not_evaluated_on_this_full_roster',old_Ours='excluded_user_deleted_payloads',
          original_barriers_unchanged=True,groups=proofs,logical_rows=logical,physical_files=120726,
          runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),GT_read=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='selected_scope_sealed_ready_for_authorized_GT_score',GT_read=False,time=time.time()))
    return verify()

def verify():
    r=read(BASE/'RUNTIME_LOCK.json');b=read(BASE/'EVALUATION_BARRIER.json')
    assert b['status']=='sealed' and b['runtime_lock_sha256']==sha(BASE/'RUNTIME_LOCK.json')
    for p,h in {**r['pins'],**r['inputs'],**r['GT_files']}.items(): assert sha(ROOT/p)==h,p
    return r,b

def load_prediction(relative):
    import numpy as np
    p=PAPER/relative;rc=read(p.with_suffix('.json'))
    assert rc['sha256']==sha(p) and rc['bytes']==p.stat().st_size and not rc['GT_read']
    with np.load(p,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files if k!='metadata'},json.loads(z['metadata'].tobytes()),rc

def truth(ds,plan):
    if ds=='vidstg':
        from scripts.diagnose_tastvg_pipeline_cpu_v1 import truth as old
        return old('P1',plan)
    ap=ROOT/'data/hcstvg2_official_metadata/val_v2.json';ann=read(ap);dense={};spans={}
    for i,r in enumerate(plan['rows']):
        v=ann[r['annotation_key']];assert v['English'].lower()==r['input']['caption']
        s=int(v['st_frame'])-1;spans[i]=[s,s+len(v['bbox'])-1]
        dense[i]={s+j:[x,y,min(x+w,r['input']['width']),min(y+h,r['input']['height'])] for j,(x,y,w,h) in enumerate(v['bbox'])}
    assert all(dense.values())
    return dense,spans,{str(ap.relative_to(ROOT)):sha(ap)}

if __name__=='__main__': seal()
