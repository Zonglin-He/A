"""One complete label-blind development trial, or independent-process smoke."""
import sys,os,time,collections,gc,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_opd_tuning_common_v1 import *

def run(ds,trial,qualify=False):
    verify();bridge()
    import torch
    from scripts.run_decota_paper_main_v1 import gpu,model_for,read_row
    import scripts.run_decota_spatial_opd_v1 as capture_module
    from methods.decota_final_simplified_v1.observations import SpatialExpert
    from methods.decota_final_simplified_v1.config import EXPERT_SNAPSHOT
    from methods.decota_final_simplified_v1.tensors import detached,state_hash
    from vg_tta.spatial_online_state_v1 import arrival
    from vg_tta.decota_spatial_opd_tunable_v1 import fit
    from vg_tta.decota_spatial_opd_tunable_audit_v1 import audit
    capture_module.BASE=BASE
    t=read(BASE/'trials'/ds/trial/'CONFIG.json');cfg=t['config'];dest=BASE/'trials'/ds/trial
    if (dest/'PREDICTION_BARRIER.json').exists():return
    d=read(BASE/'DESIGN_LOCK.json')['datasets'][ds]
    stage=dict(dataset=ds,source=d['source'],split='tuning',parents=t['parents'],orders=t['orders'])
    if not qualify:assert read(BASE/'QUALIFICATION.json')['status']=='pass'
    lease=gpu();model=model_for(d['source']);mh=state_hash(model.state_dict())
    expert=SpatialExpert(ROOT/EXPERT_SNAPSHOT);cache=collections.OrderedDict();files={};count=0;checks=[];tick=time.time()
    try:
        for order,seq in t['orders'].items():
            previous=None;prevhash=None
            for at,parent in enumerate(seq):
                budget();p=dest/order/f'{at:05}.pt'
                if p.exists():
                    assert sha(p)==read(p.with_suffix('.json'))['sha256'];z=load(p)
                    assert z['parent']==parent and z['previous_payload_sha256']==prevhash
                    previous=z['committed'];prevhash=sha(p);files[str(p.relative_to(BASE))]=prevhash;count+=1;continue
                row=read_row(ds,parent);begin=time.perf_counter()
                base,native,ex,inputrc=capture_module.capture(model,expert,stage,row,'clean',cache)
                captureseconds=time.perf_counter()-begin;initial=arrival(base.initial,previous,'O-split')
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
                result=fit(base,initial,ex,row['frame_ids'],row['key'],'on_policy',config=cfg)
                torch.cuda.synchronize();seconds=time.perf_counter()-begin
                begin=time.perf_counter();mathcheck=audit(result,ex);mathseconds=time.perf_counter()-begin
                if qualify and cfg==DEFAULT:
                    from vg_tta.decota_spatial_opd_v1 import fit as original_fit
                    original=original_fit(base,initial,ex,row['frame_ids'],row['key'],'on_policy')
                    assert torch.equal(original['final'],result['final'])
                    assert all(torch.equal(original['state'][n],result['state'][n]) for n in result['state'])
                    checks.append(dict(default_original_exact_output_and_state=True,parent=parent))
                committed={n:(torch.zeros_like(v.cpu()) if n=='spatial.query_residual' else
                             v.cpu()+(result['state'][n]-v.cpu())*cfg['writeback']) for n,v in initial.items()}
                z=dict(dataset=ds,source=d['source'],trial=trial,parent=parent,order=order,arrival=at,
                       fit=result,committed=committed,interval=native['physical_interval'],input=inputrc,
                       previous_payload_sha256=prevhash,math_audit=mathcheck,config=cfg,GT_read=False,
                       compute=dict(capture_seconds=captureseconds,fit_GPU_seconds=seconds,CPU_math_seconds=mathseconds,
                                    new_DINO_calls=inputrc['new_DINO_calls'],CUDA_peak_allocated=torch.cuda.max_memory_allocated()))
                save(p,z);h=sha(p);write(p.with_suffix('.json'),dict(sha256=h,GT_read=False,time=time.time()))
                files[str(p.relative_to(BASE))]=h;previous=committed;prevhash=h;count+=1
                status(dest/'STATUS.json',dict(status='running',pid=os.getpid(),dataset=ds,trial=trial,done=count,
                       total=sum(map(len,t['orders'].values())),time=time.time(),GT_read=False))
                print('OPD_TUNE_PROGRESS',ds,trial,count,sum(map(len,t['orders'].values())),round(time.time()-tick,2),flush=True)
                base.restore(base.initial);del base,result,z,initial;gc.collect();torch.cuda.empty_cache()
        assert state_hash(model.state_dict())==mh
        expected=sum(map(len,t['orders'].values()));assert count==expected
        write(dest/'PREDICTION_BARRIER.json',dict(status='sealed',dataset=ds,trial=trial,cells=count,files=files,
              config_sha256=sha(dest/'CONFIG.json'),GT_read=False,time=time.time()))
        if qualify:
            informative=sum(load(BASE/p)['fit']['gradient_calls'] for p in files)
            write(dest/'QUALIFICATION.json',dict(status='pass',dataset=ds,queries=count,checks=checks,
                  informative_updates=informative,default_parity=cfg==DEFAULT,config=cfg,time=time.time(),GT_read=False))
        status(dest/'STATUS.json',dict(status='sealed_pending_cpu',done=count,total=expected,time=time.time(),GT_read=False))
    finally:lease.close()

if __name__=='__main__':
    ds,trial=sys.argv[1:3]
    try:run(ds,trial,trial.startswith('qual_'))
    except BaseException:
        dest=BASE/'trials'/ds/trial/'failure';dest.mkdir(parents=True,exist_ok=True)
        (dest/'traceback.txt').write_text(traceback.format_exc())
        status(dest.parent/'STATUS.json',dict(status='failed_preserved',pid=os.getpid(),time=time.time()))
        raise
