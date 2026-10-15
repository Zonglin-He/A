#!/usr/bin/env python3
"""Prepare the fixed 32-source, mu=.25, LR-x2 Soft EP comparison (metadata only)."""
from __future__ import annotations
import copy, hashlib, json, os, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/stvg_soft_ep_opd_v1/lr2'
LR1=ROOT/'artifacts/stvg_soft_ep_opd_v1/mu025'
PARENT=ROOT/'artifacts/stvg_evidence_constrained_v1/round2_prompt009'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write_once(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(f'refusing to overwrite {path}')
    tmp=path.with_name(path.name+'.tmp')
    if tmp.exists(): raise FileExistsError(f'stale temporary file: {tmp}')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)

def validate_inputs():
    rt=read(LR1/'RUNTIME.json')
    if rt.get('status')!='locked' or rt.get('GT_read') is not False: raise RuntimeError('mu=.25 LR1 runtime is not locked and GT-free')
    for rel,digest in {**rt['pins'],**rt['protected']}.items():
        if sha(ROOT/rel)!=digest: raise RuntimeError(f'LR1 runtime pin mismatch: {rel}')
    design=read(LR1/'DESIGN_LOCK.json'); roster=read(LR1/'ROSTER_LOCK.json'); refs=read(LR1/'REFERENCE_LOCK.json')
    if design.get('status')!='locked' or design.get('GT_read') is not False: raise RuntimeError('LR1 design lock invalid')
    if roster.get('status')!='locked' or roster.get('count')!=32 or roster.get('GT_read') is not False: raise RuntimeError('LR1 roster lock invalid')
    if refs.get('status')!='locked' or refs.get('count')!=32 or refs.get('GT_read') is not False: raise RuntimeError('LR1 reference lock invalid')
    if len(roster.get('records',[]))!=32 or len(refs.get('records',[]))!=32: raise RuntimeError('LR1 record count mismatch')
    rid=[r['probe_alias'] for r in roster['records']]; refids=[r['case_id'] for r in refs['records']]
    if len(set(rid))!=32 or set(rid)!=set(refids): raise RuntimeError('LR1 roster/reference membership mismatch')
    if sum(len(r['anchors']) for r in refs['records'])!=int(design['reference_anchor_count']): raise RuntimeError('LR1 reference anchor count mismatch')
    # Require the existing LR1 prediction and independent CPU readback to be sealed.
    for name,expected_status in [('STUDENT_GLOBAL_BARRIER.json','complete'),('ROOT_READBACK.json','pass'),('GLOBAL_PREDICTION_BARRIER.json','complete')]:
        d=read(LR1/name)
        if d.get('status')!=expected_status or d.get('GT_read') is not False or d.get('count')!=32:
            raise RuntimeError(f'LR1 {name} gate mismatch')
    # M0 is the exact pre-existing episodic baseline used by both trials.
    for name,expected_status in [('STUDENT_GLOBAL_BARRIER.json','complete'),('GLOBAL_PREDICTION_BARRIER.json','complete')]:
        d=read(PARENT/name)
        if d.get('status')!=expected_status or d.get('GT_read') is not False or d.get('count')!=32:
            raise RuntimeError(f'parent {name} gate mismatch')
    for name in ('ROOT_INPUT_READBACK.json','ROOT_TEMPORAL_READBACK.json','ROOT_SPATIAL_READBACK.json'):
        d=read(PARENT/name)
        if d.get('status')!='pass' or d.get('GT_read') is not False or d.get('case_count')!=32:
            raise RuntimeError(f'parent {name} gate mismatch')
    return design,roster,refs

def derive_design(source):
    d=copy.deepcopy(source)
    old=source['configs']; new={}
    for ds in ('hc2','vidstg'):
        cfg=copy.deepcopy(old[ds]); cfg['lr']=float(cfg['lr'])*2.0; new[ds]=cfg
    d['scope']='Fixed 32-source Soft EP-OPD LR x2 comparison at mu=0.25; no new references or temporal changes.'
    d['configs']=new
    d['response_settings']=copy.deepcopy(source['response_settings'])
    d['response_settings']['mu']=0.25
    d['lr2_trial']={
        'schema':'stvg-soft-ep-lr2-trial-v1',
        'baseline_scope':'artifacts/stvg_soft_ep_opd_v1/mu025',
        'lr_multiplier':2.0,
        'fixed_parameters':['sigma','tau','steps','writeback','samples','response_settings','references','temporal_intervals'],
        'new_fit_arms':['M1','M2','M0matched'],
        'exact_baseline_arm_source':'artifacts/stvg_evidence_constrained_v1/round2_prompt009',
        'GT_read':False,
    }
    return d

def contracts(design,old_design,roster,refs):
    assert design['response_settings']['mu']==0.25
    assert len(roster['records'])==len(refs['records'])==32
    for ds in ('hc2','vidstg'):
        before=old_design['configs'][ds]; after=design['configs'][ds]
        assert after['lr']==before['lr']*2.0
        assert {k:v for k,v in before.items() if k!='lr'}=={k:v for k,v in after.items() if k!='lr'}
        assert after['steps']==5 and after['samples']==32
    assert set(design['lr2_trial']['new_fit_arms'])=={'M1','M2','M0matched'}
    nonempty=sum(bool(r['anchors']) for r in refs['records'])
    assert nonempty==23 and sum(len(r['anchors']) for r in refs['records'])==49
    arms={'F','T','S_M0','TS_M0','S_M1','TS_M1','S_M2','TS_M2','S_M0matched','TS_M0matched'}
    assert len(arms)==10
    return dict(status='pass',schema='stvg-soft-ep-lr2-cpu-contracts-v1',scope='metadata/config and arm cardinality only',source_count=32,domain_counts={'hc2':16,'vidstg':16},reference_anchor_count=49,nonempty_reference_sources=23,empty_reference_sources=9,expected_new_fit_invocations=69,arms=sorted(arms),lr_multiplier=2.0,mu=0.25,GT_read=False,CPU_only=True,model_loaded=False,GPU_used=False)

def main():
    if BASE.exists() and any(BASE.iterdir()): raise FileExistsError(f'new LR2 scope already contains files: {BASE}')
    old_design,roster,refs=validate_inputs(); design=derive_design(old_design); result=contracts(design,old_design,roster,refs)
    BASE.mkdir(parents=True,exist_ok=True)
    design['lr1_design_sha256']=sha(LR1/'DESIGN_LOCK.json')
    design['lr1_runtime_sha256']=sha(LR1/'RUNTIME.json')
    design['status']='locked'; design['GT_read']=False; design['time']=time.time()
    # The roster and reference lock bytes are copied exactly; no reselection or transformation occurs.
    for source,target in [(LR1/'ROSTER_LOCK.json',BASE/'ROSTER_LOCK.json'),(LR1/'REFERENCE_LOCK.json',BASE/'REFERENCE_LOCK.json')]:
        if target.exists(): raise FileExistsError(target)
        target.write_bytes(source.read_bytes())
    write_once(BASE/'DESIGN_LOCK.json',design)
    result.update(bindings={'baseline_design_sha256':sha(LR1/'DESIGN_LOCK.json'),'baseline_runtime_sha256':sha(LR1/'RUNTIME.json'),
        'roster_sha256':sha(BASE/'ROSTER_LOCK.json'),'reference_lock_sha256':sha(BASE/'REFERENCE_LOCK.json'),
        'parent_prediction_barrier_sha256':sha(PARENT/'GLOBAL_PREDICTION_BARRIER.json')},time=time.time())
    write_once(BASE/'CPU_CONTRACTS.json',result)
    write_once(BASE/'PREPARATION_READBACK.json',dict(schema='stvg-soft-ep-lr2-preparation-readback-v1',status='pass',count=32,
        baseline_scope=str(LR1.relative_to(ROOT)),design_lock_sha256=sha(BASE/'DESIGN_LOCK.json'),roster_sha256=sha(BASE/'ROSTER_LOCK.json'),
        reference_lock_sha256=sha(BASE/'REFERENCE_LOCK.json'),cpu_contracts_sha256=sha(BASE/'CPU_CONTRACTS.json'),GT_read=False,CPU_only=True,time=time.time()))
    print(json.dumps({'status':'pass','base':str(BASE.relative_to(ROOT)),'new_fits':69,'arms':result['arms'],'GT_read':False},indent=2))

if __name__=='__main__': main()
