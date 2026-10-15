#!/usr/bin/env python3
"""Create the immutable execution runtime for the LR-x2 trial."""
from __future__ import annotations
import importlib.metadata, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
from scripts.lock_stvg_ecld_runtime002 import closure
from scripts.stvg_reference_recovery_fit_v1 import source_hashes
BASE=ROOT/'artifacts/stvg_soft_ep_opd_v1/lr2'
LR1=ROOT/'artifacts/stvg_soft_ep_opd_v1/mu025'
PARENT=ROOT/'artifacts/stvg_evidence_constrained_v1/round2_prompt009'

def add(pins,path):
    path=Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    pins[str(path.relative_to(ROOT))]=sha(path)

def main():
    rtpath=BASE/'RUNTIME.json'
    if rtpath.exists(): raise FileExistsError('LR2 runtime already exists; refusing relock')
    prep=read(BASE/'PREPARATION_READBACK.json'); assert prep.get('status')=='pass' and prep.get('GT_read') is False
    assert prep['design_lock_sha256']==sha(BASE/'DESIGN_LOCK.json')
    assert prep['roster_sha256']==sha(BASE/'ROSTER_LOCK.json') and prep['reference_lock_sha256']==sha(BASE/'REFERENCE_LOCK.json')
    oldrt=read(LR1/'RUNTIME.json'); assert oldrt.get('status')=='locked' and oldrt.get('GT_read') is False
    for rel,digest in {**oldrt['pins'],**oldrt['protected']}.items(): assert sha(ROOT/rel)==digest,rel
    seeds=[ROOT/'scripts'/name for name in (
        'prepare_stvg_soft_ep_lr2_v1.py','contract_stvg_soft_ep_lr2_v1.py','run_stvg_soft_ep_lr2_v1.py','review_stvg_soft_ep_lr2_v1.py',
        'measure_stvg_soft_ep_lr2_v1.py','launch_stvg_soft_ep_lr2_v1.py','lock_stvg_soft_ep_lr2_v1.py')]
    sources=closure(seeds); pins={str(p.relative_to(ROOT)):sha(p) for p in sorted(sources)}
    pins.update(source_hashes())
    # Preserve the complete already-locked LR1 scientific/source dependency set transitively.
    for rel,digest in {**oldrt['pins'],**oldrt['protected']}.items():
        if rel in pins and pins[rel]!=digest: raise RuntimeError(f'conflicting LR1 dependency pin: {rel}')
        pins[rel]=digest
    for p in [BASE/'DESIGN_LOCK.json',BASE/'ROSTER_LOCK.json',BASE/'REFERENCE_LOCK.json',BASE/'CPU_CONTRACTS.json',BASE/'PREPARATION_READBACK.json',BASE/'CPU_SYNTHETIC_READBACK.json']:
        add(pins,p)
    for scope,names in [
        (LR1,('DESIGN_LOCK.json','ROSTER_LOCK.json','REFERENCE_LOCK.json','RUNTIME.json','STUDENT_GLOBAL_BARRIER.json','ROOT_READBACK.json','GLOBAL_PREDICTION_BARRIER.json','POSTSEAL_SCORE_ROWS.json')),
        (PARENT,('RUNTIME006.json','STUDENT_GLOBAL_BARRIER.json','ROOT_READBACK.json','GLOBAL_PREDICTION_BARRIER.json','REFERENCE_LOCK.json','TEACHER_GLOBAL_BARRIER.json'))]:
        for name in names:
            p=scope/name
            if p.exists(): add(pins,p)
    parent_runtime=PARENT/'RUNTIME006.json'
    if parent_runtime.exists():
        parent_rt=read(parent_runtime)
        for rel,digest in {**parent_rt.get('pins',{}),**parent_rt.get('protected',{})}.items():
            if sha(ROOT/rel)!=digest: raise RuntimeError(f'parent dependency pin mismatch: {rel}')
            if rel in pins and pins[rel]!=digest: raise RuntimeError(f'conflicting parent dependency pin: {rel}')
            pins[rel]=digest
    for rec in read(BASE/'ROSTER_LOCK.json')['records']:
        for key in ('plan_path','subject_path'):
            if rec.get(key): add(pins,ROOT/rec[key])
    protected={}
    for rel in ('methods/CURRENT_METHOD.json','artifacts/stvg_evidence_constrained_v1/FROZEN_METHOD_LOCK.json'):
        protected[rel]=sha(ROOT/rel)
    versions={}
    for package in ('torch','numpy','transformers','safetensors'):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]=None
    write(rtpath,dict(schema='stvg-soft-ep-lr2-runtime-v1',status='locked',scope='32-source mu=.25 LR-x2 M1/M2 plus same-LR validated OPD M0matched; exact M0 control; no GT before seal',
        GT_read=False,pins=pins,protected=protected,local_import_closure_count=len(sources),environment_versions=versions,
        bindings={'preparation_readback_sha256':sha(BASE/'PREPARATION_READBACK.json'),'lr1_runtime_sha256':sha(LR1/'RUNTIME.json'),
            'lr1_student_barrier_sha256':sha(LR1/'STUDENT_GLOBAL_BARRIER.json'),'lr1_score_rows_sha256':sha(LR1/'POSTSEAL_SCORE_ROWS.json'),
            'parent_prediction_barrier_sha256':sha(PARENT/'GLOBAL_PREDICTION_BARRIER.json')},
        bootstrap_executed=False,full_trial_started=False,time=time.time()))
    print('locked',len(sources),'local sources and',len(pins),'pins')
if __name__=='__main__': main()
