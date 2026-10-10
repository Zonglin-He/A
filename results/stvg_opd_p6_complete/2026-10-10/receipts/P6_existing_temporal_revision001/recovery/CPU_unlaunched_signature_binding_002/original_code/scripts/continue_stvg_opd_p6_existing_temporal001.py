"""Finite P6 controller: serial real qualification, deployment seal, then CPU."""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_p6_existing_common001 import *


def child(script,args,label):
    env=dict(os.environ);env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True';env['PYTHONUNBUFFERED']='1'
    log=NS/'logs'/(label+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as out:
        p=subprocess.Popen([str(PYTHON),'-B',str(ROOT/'scripts'/script),*args],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,env=env)
        status(NS/'STATUS.json',dict(status='running',stage=label,controller=os.getpid(),worker=p.pid,CPU_only=label.startswith('root_') or label=='postseal_CPU',GT_read=label=='postseal_CPU',time=time.time()))
        status(BASE/'STATUS.json',dict(status='running',stage=label,controller=os.getpid(),worker=p.pid,original_fixed_P6=True,paper_suite_complete=False,time=time.time()))
        status(BASE/'STAGE.json',dict(status='running',stage=label,controller=os.getpid(),worker=p.pid,time=time.time()))
        status(BASE/'COMPONENT_STAGE.json',dict(status='running',stage=label,controller=os.getpid(),worker=p.pid,time=time.time()))
        rc=p.wait()
    if rc:raise RuntimeError(f'{label} child exited {rc}; preserve original log and payloads')


def run():
    runtime=verify();assert read(NS/'CPU_CONTRACTS.json')['status']=='pass'
    root_records=[]
    for ds in ['hc2','vidstg']:
        p=NS/'qualification'/ds/'GPU_QUALIFICATION.json'
        if not p.exists():child('run_stvg_opd_p6_existing_temporal001.py',['--mode','qualification','--dataset',ds],'P6_qualification_'+ds)
        q=read(p);assert q['status']=='pass' and q['arrivals']==2 and q['actual_complete_GPU_fits']==4 and not q['GT_read']
        for r in q['records']:
            for k in ['original_fit','recorded_fit']:assert sha(BASE/r[k]['path'])==r[k]['sha256']
            z=load(BASE/r['recorded_fit']['path']);math=audit(z['fit'],z['trajectory'],z['capture']['records'],config(ds));assert math==z['math_audit']
            root_records.append(dict(dataset=ds,query_ordinal=r['query_ordinal'],status='pass',math_audit=math,complete_scientific_fit_bitwise=r['complete_scientific_fit_bitwise']))
    if not (BASE/'P6_QUALIFICATION.json').exists():
        write(BASE/'P6_QUALIFICATION.json',dict(status='pass',scope='all four existing P6 first-arrival real head comparisons and CPU root algebra',
            actual_arrivals=4,actual_complete_GPU_fits=8,complete_CPU_math_readbacks=4,records=root_records,
            formal_predictions_accepted=0,new_DINO_calls=0,GT_read=False,runtime_sha256=sha(NS/'REVISION_RUNTIME.json'),time=time.time()))
    for ds in ['hc2','vidstg']:
        if not (NS/'formal'/ds/'PREDICTION_BARRIER.json').exists():child('run_stvg_opd_p6_existing_temporal001.py',['--mode','formal','--dataset',ds],'P6_'+ds)
    if not (BASE/'P6_PREDICTION_BARRIER.json').exists():
        records=[];source_barriers={}
        for ds in ['hc2','vidstg']:
            p=NS/'formal'/ds/'PREDICTION_BARRIER.json';b=read(p);assert b['status']=='sealed' and not b['GT_read'] and len(b['records'])==32
            source_barriers[ds]=sha(p)
            for r in b['records']:
                f=BASE/r['path'];assert sha(f)==r['sha256'] and f.stat().st_size==r['bytes'] and not r['GT_read'];records.append(r)
        assert len(records)==64 and sum(r['outputs'] for r in records)==256
        write(BASE/'P6_PREDICTION_BARRIER.json',dict(status='sealed',scope='all original P6 deployment arms, both targets and complete fixed cohorts',
            queries=64,parent_sources=64,logical_outputs=256,arms=ARMS,records=records,source_barriers=source_barriers,
            GT_read=False,new_DINO_calls=0,runtime_sha256=sha(NS/'REVISION_RUNTIME.json'),time=time.time()))
        write(BASE/'P6_GPU_COMPLETION.json',dict(status='complete_deployment_global_seal',queries=64,logical_outputs=256,
            global_prediction_barrier_sha256=sha(BASE/'P6_PREDICTION_BARRIER.json'),GT_read=False,new_DINO_calls=0,time=time.time()))
    if not (BASE/'P6_CPU_COMPLETION.json').exists():child('score_stvg_opd_p6_existing_temporal001.py',[],'postseal_CPU')
    write(NS/'COMPLETION.json',dict(status='pending_actual_root_visual_and_publication',scope='actual P6 complete deployment seal and postseal offline oracle/CPU readback',
        queries=64,logical_deployment_outputs=256,CPU_completion_sha256=sha(BASE/'P6_CPU_COMPLETION.json'),P6_phase_complete=False,paper_suite_complete=False,time=time.time()))
    status(NS/'STATUS.json',dict(status='pending_actual_root_visual_and_publication',controller=os.getpid(),P6_phase_complete=False,paper_suite_complete=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='P6_CPU_complete_pending_actual_root_visual_and_publication',P6_phase_complete=False,paper_suite_complete=False,time=time.time()))
    status(BASE/'STAGE.json',dict(status='pending_actual_root',stage='P6',paper_suite_complete=False,time=time.time()))
    status(BASE/'LATER_CPU_STAGE.json',dict(status='pending_actual_root',stage='P6',paper_suite_complete=False,time=time.time()))


if __name__=='__main__':
    try:run()
    except BaseException:
        status(NS/'CONTROLLER_FAILURE.json',dict(status='failed',process=os.getpid(),traceback=traceback.format_exc(),GT_read=False,time=time.time()));raise
