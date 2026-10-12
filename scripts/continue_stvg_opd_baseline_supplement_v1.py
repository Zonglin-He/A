"""Finite serial baseline calibration, then selected full-cohort deployment."""
import collections
import fcntl
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_baseline_supplement_common_v1 import *

def reuse_default(ds,candidate):
    """Byte/state/config/roster readback, not a new gradient or inference claim."""
    import json
    import numpy as np
    from scripts.decota_paper_common_v1 import digest
    from vg_tta.decota_paper_baselines_20261005_v1 import CONFIGS
    d=design();cfg=d['candidates'][candidate];method=cfg['method'];assert cfg['config']==CONFIGS[method]
    old=PAPER/'baselines'/f'{method}_{ds}';b=read(old/'PREDICTION_BARRIER.json');plan=read(PAPER/ds/'PLAN.json')
    assert read(PAPER/'baselines/RUNTIME_LOCK.json')['configs'][method]==cfg['config']
    import torch
    qualified=load(BASE/'qualification'/ds/candidate/'order1/00000.pt')['state_before']
    qualified_source=np.concatenate([p.numpy().reshape(-1) for p in qualified['parameters']]).astype(np.float32)
    scope=read(old/'PARAMETER_SCOPE.json');assert scope['names']==qualified['names']
    dest=BASE/'formal'/ds/candidate
    if (dest/'PREDICTION_BARRIER.json').exists(): return
    states=count=coords=snapshots=0;source_state=None;metadata={};bytes_total=0;snapshot_files={}
    for order,seq in plan['orders'].items():
        previous=None;before=None
        for at,q in enumerate(seq):
            f=old/order/f'{at:05}.npz';rel=str(f.relative_to(PAPER));rc=read(f.with_suffix('.json'))
            assert sha(f)==rc['sha256']==b['files'][rel] and not rc['GT_read'];bytes_total+=f.stat().st_size
            with np.load(f,allow_pickle=False) as a:
                md=json.loads(a['metadata'].tobytes());sb=a['state_before'];sa=a['state_after']
                assert md['method']==method and md['dataset']==ds and md['order']==order and md['arrival']==at and md['parent']==q
                assert md['source_checkpoint']==d['datasets'][ds]['source']
                assert md['frame_ids']==plan['rows'][q]['frame_ids'] and md['input_metadata_sha256']==digest(plan['rows'][q]['input'])
                assert md['port_lock_sha256']==sha(PAPER/'baselines/RUNTIME_LOCK.json') and md['previous_payload_sha256']==previous and not md['GT_read']
                if before is None:
                    if source_state is None:source_state=sb.copy()
                    assert np.array_equal(source_state,sb) and np.array_equal(qualified_source,sb)
                else:assert np.array_equal(before,sb);states+=1
                assert digest(sb.tolist())==md['parameter_before_sha256'] and digest(sa.tolist())==md['parameter_after_sha256']
                assert md['audit']['arrival']==at+1 and not md['audit']['GT_read']
                if md.get('optimizer_sample'):
                    binding=md['optimizer_sample'];sf=PAPER/binding['path'];assert sha(sf)==binding['sha256'];sample=load(sf)
                    assert sample['method']==method and sample['config']==cfg['config'] and sample['names']==qualified['names'] and sample['arrivals']==at+1
                    flat=np.concatenate([p.numpy().reshape(-1) for p in sample['parameters']]).astype(np.float32);assert np.array_equal(flat,sa)
                    assert len(sample['optimizer']['param_groups'])==1 and sample['optimizer']['param_groups'][0]['lr']==cfg['config']['lr']
                    for value in sample['optimizer']['state'].values():
                        assert all(not torch.is_tensor(v) or torch.isfinite(v).all() for v in value.values())
                    snapshots+=1;snapshot_files[str(sf.relative_to(ROOT))]=sha(sf)
                before=sa.copy();previous=rc['sha256'];coords+=sb.size+sa.size;count+=1
                metadata[(order,q)]=dict(audit=md['audit'],compute=md['compute'],frame_ids=md['frame_ids'])
    assert count==plan['queries']*3==b['cells']
    save(dest/'REUSED_METADATA.pt',metadata)
    write(dest/'PREDICTION_BARRIER.json',dict(status='sealed',reuse_identical_default=True,original_barrier_sha256=sha(old/'PREDICTION_BARRIER.json'),dataset=ds,candidate=candidate,method=method,config=cfg['config'],arrivals=count,
        exact_parameter_chain_checks=states,parameter_coordinate_checks=coords,bytes_read=bytes_total,new_inference_arrivals=0,GT_read=False,
        metadata_sha256=sha(dest/'REUSED_METADATA.pt'),optimizer_snapshots_checked=snapshots,optimizer_snapshot_files=snapshot_files,source_parameter_state_matches_current_GPU_qualification=True,
        full_optimizer_state_available_every_query=False,scope='all original predictions and per-query parameter chains plus every referenced optimizer snapshot; old optimizer snapshots are sparse and per-query GPU gradients were not saved',time=time.time()))

def convert_default_rows(ds,candidate):
    # This function is CPU-only and is called only AFTER selected formal global seal.
    assert read(BASE/'FORMAL_GLOBAL_BARRIER.json')['status']=='sealed'
    import json,numpy as np
    d=design();cfg=d['candidates'][candidate];method=cfg['method'];plan=read(PAPER/ds/'PLAN.json');dest=BASE/'formal'/ds/candidate
    receipt=read(dest/'PREDICTION_BARRIER.json');assert sha(dest/'REUSED_METADATA.pt')==receipt['metadata_sha256']
    metadata=load(dest/'REUSED_METADATA.pt');count=receipt['arrivals'];states=receipt['exact_parameter_chain_checks']
    from scripts.score_stvg_opd_baseline_supplement_v1 import rows
    oldrows=ROOT/'results/decota_paper_baseline_scoring/2026-10-08'/ds/(method.lower()+'_stvg.jsonl.gz')
    part=rows(oldrows);assert len(part)==count
    sourceids={x:i for i,x in enumerate(sorted({r['source'] for r in plan['rows']}))};converted=[]
    for r in part:
        q=r['query_id'];m=metadata[(r['order'],q)];a=m['audit'];ids=m['frame_ids'];n0=(len(ids)+1)//2;n1=len(ids)//2
        ceiling=(2*np.log(n0)+np.log(2)+2*np.log(n1)+np.log(2))/2
        z=dict(dataset=ds,method=method,candidate=candidate,config=cfg['config'],source_id=r['source_id'],query_ordinal=q,order=r['order'],arrival=r['arrival'],condition='clean',
            is_development_parent=plan['rows'][q]['source'] in d['datasets'][ds]['excluded_primary_parent_sources'],updates=a['counts']['optimizer_steps'],backwards=a['counts']['backward_steps'],recoveries=a['counts']['recoveries'],skips=sum('skip' in t for t in a['trace']),entropy_ratios=[t['entropy']/ceiling for t in a['trace']],compute=m['compute'],reused_exact_original_default=True)
        assert r['source_id']==sourceids[plan['rows'][q]['source']]
        for label in ['Before','After']:
            for k in ['v','s','t']: z[label+'_'+k]=r[label+'_'+k]
            for k in ['R30','R50']: z[label+'_'+k]=r[label+'_R'+('0.3' if k=='R30' else '0.5')]
        z['delta_v']=z['After_v']-z['Before_v'];converted.append(z)
    import gzip
    out=PUB/'formal'/ds/candidate/'ROWS.jsonl.gz';out.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(out,'wt') as f:
        for z in converted:f.write(json.dumps(z,allow_nan=False)+'\n')
    write(dest/'CPU_COMPLETION.json',dict(status='exact_default_reuse_previously_scored_state_chain',rows_sha256=sha(out),counts=dict(arrivals=count,optimizer_scalar_checks=0,native_entropy_closures=0,state_chain_pairs=states,dense_scalar_checks=0),scope=receipt['scope'],GT_read=True,time=time.time()))

def run():
    verify()
    with (BASE/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        order=ROOT/'artifacts/stvg_opd_supplement_v1/orders'
        while not (order/'GLOBAL_PREDICTION_BARRIER.json').exists():
            state=read(order/'STATUS.json');assert 'fail' not in state['status'], 'Order dependency actual failure requires root'
            status(BASE/'STATUS.json',dict(status='CPU_waiting_for_actual_order_GPU_seal',controller_pid=os.getpid(),GT_read=False,time=time.time()));time.sleep(15)
        def child(job,ds=None,candidate=None,CPU=False):
            script='scripts/score_stvg_opd_baseline_supplement_v1.py' if job in ['score_development','score_formal'] else 'scripts/run_stvg_opd_baseline_supplement_v1.py'
            args=[job.split('_',1)[1]] if job.startswith('score_') else [job]+([ds,candidate] if ds else [])
            cmd=[str(PYTHON),'-B',script]+args;log=BASE/'logs'/(job+('_'+ds+'_'+candidate if ds else '')+'.log');log.parent.mkdir(parents=True,exist_ok=True)
            env={**os.environ,'PYTORCH_CUDA_ALLOC_CONF':ALLOCATOR,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
            if CPU: env['CUDA_VISIBLE_DEVICES']=''
            with log.open('a') as out:
                p=subprocess.Popen(cmd,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,env=env)
                status(BASE/'STATUS.json',dict(status='running',job=job,dataset=ds,candidate=candidate,CPU_only=CPU,controller_pid=os.getpid(),worker_pid=p.pid,log=str(log.relative_to(ROOT)),GT_read=job.startswith('score_'),time=time.time()))
                code=p.wait();assert code==0, f'Preserved actual baseline failure, {job}, exit {code}'
        if not (BASE/'ROOT_QUALIFICATION_READBACK.json').exists():
            for ds in design()['datasets']:
                for c in design()['candidates']:child('qualification',ds,c)
            child('root_qualification',CPU=True)
        if not (BASE/'DEVELOPMENT_GLOBAL_BARRIER.json').exists():
            barriers={};count=0
            for ds in design()['datasets']:
                for c in design()['candidates']:
                    f=BASE/'development'/ds/c/'PREDICTION_BARRIER.json'
                    if not f.exists():child('development',ds,c)
                    b=read(f);assert b['status']=='sealed' and b['arrivals']==64 and not b['GT_read'];barriers[str(f.relative_to(ROOT))]=sha(f);count+=b['arrivals']
            assert count==1152;write(BASE/'DEVELOPMENT_GLOBAL_BARRIER.json',dict(status='sealed',arrivals=count,barriers=barriers,GT_read=False,time=time.time()))
        if not (BASE/'DEVELOPMENT_CPU_COMPLETION.json').exists():child('score_development',CPU=True)
        if not (BASE/'FORMAL_GLOBAL_BARRIER.json').exists():
            chosen=read(BASE/'SELECTED_CONFIGS.json')['selected'];barriers={};count=0
            for ds,z in chosen.items():
                for method,s in z.items():
                    c=s['candidate'];f=BASE/'formal'/ds/c/'PREDICTION_BARRIER.json'
                    if not f.exists():
                        if s['reuse_identical_default']:reuse_default(ds,c)
                        else:
                            child('formal_qualification',ds,c);child('formal',ds,c)
                    b=read(f);assert b['status']=='sealed' and b['arrivals']==design()['datasets'][ds]['formal_arrivals'];count+=b['arrivals'];barriers[str(f.relative_to(ROOT))]=sha(f)
            assert count==82710;write(BASE/'FORMAL_GLOBAL_BARRIER.json',dict(status='sealed',logical_arrivals=count,barriers=barriers,selected_configs_sha256=sha(BASE/'SELECTED_CONFIGS.json'),GT_read=False,time=time.time()))
        if not (BASE/'FORMAL_CPU_COMPLETION.json').exists():child('score_formal',CPU=True)
        status(BASE/'STATUS.json',dict(status='pending_actual_root_comparison_figures_publication',experiment_complete=False,controller_pid=os.getpid(),time=time.time()))

if __name__=='__main__':
    try: run()
    except BaseException:
        p=BASE/'controller_failures'/str(time.time_ns());p.mkdir(parents=True);(p/'traceback.txt').write_text(traceback.format_exc());status(BASE/'STATUS.json',dict(status='failed_preserved_pending_actual_root',evidence=str(p.relative_to(ROOT)),experiment_complete=False,time=time.time()));raise
