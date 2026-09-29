"""Minimal Frozen/Rerank/Hard/OPD, no GT in inference or adaptation."""
import sys,time,gc,traceback,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
from scripts.run_tastvg_fourarm_critic_v1 import OUT,OLD,D,CONDS


def prepare():
    capture=read(OLD/'CAPTURE_BARRIER.json');critic=read(OUT/'C2_BARRIER.json')
    paths=['protocols/tastvg_temporal_fourarm_v1.md','scripts/run_tastvg_temporal_fourarm_v1.py','vg_tta/tastvg_temporal_fourarm_v1.py',
           'vg_tta/tastvg_evidence_capture_v1.py','vg_tta/tastvg_causal_round2_v1.py','methods/CURRENT_METHOD.json']
    write(OUT/'LOCK.json',dict(rows=read(OLD/'LOCK.json')['rows'][:16],conditions=CONDS,capture_files=capture['files'],critic_files=critic['files'],pins={f:sha(ROOT/f) for f in paths},steps=1,step_norm=.004,GT_read=False,time=time.time()))


def verify():
    p=read(OUT/'LOCK.json')
    for f,h in p['pins'].items():assert sha(ROOT/f)==h,f
    return p


def run(limit=0):
    start=time.monotonic();p=verify();done=0;state='failed';failure=None;lease=None
    prior=sum(read(f)['seconds'] for f in (OUT/'adapt_allocations').glob('*.json'))
    try:
        import numpy as np,torch
        from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
        from scripts.run_tastvg_evidence_vulnerability_v1 import device_tree
        from scripts.run_final_simplification_v1 import lease as gpu_lease
        from methods.decota_final_simplified_v1.tensors import state_hash
        from vg_tta.exact_frame_decode_audit_v2 import decode
        from vg_tta.tastvg_evidence_capture_v1 import capture
        from vg_tta.tastvg_temporal_fourarm_v1 import step,reinsert
        from vg_tta.tastvg_deployment_corruption_v2 import apply_burst
        from scripts.c1_controlled_corruption_v1 import pixelhash
        procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);assert all('/opt/todesk/' in l for l in procs.splitlines() if l.strip()),procs
        lease=gpu_lease();install_clean_loader()
        from scripts.run_spatial_regression_alignment_v1 import model_load
        torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        model=model_load('hcstvg1_test');model.eval().requires_grad_(False);mh=state_hash(model.state_dict());assert mh==read(OLD/'CAPTURE_BARRIER.json')['model_state_sha256']
        for r in p['rows']:
            frames=None
            for cond in CONDS:
                f=OUT/'predictions'/cond/f"{r['ordinal']:03}.pt"
                if f.with_suffix('.json').exists():assert sha(f)==read(f.with_suffix('.json'))['sha256'];done+=1;continue
                if limit and done>=limit:break
                assert prior+time.monotonic()-start<3580 and shutil.disk_usage(ROOT).free>8*2**30
                if frames is None:frames,ids=decode(r['input'])
                rel=f"capture/{cond}/{r['ordinal']:03}.pt";assert sha(OLD/rel)==p['capture_files'][rel];x=load(OLD/rel)
                rel=f"c2/{cond}/{r['ordinal']:03}.pt";assert sha(OUT/rel)==p['critic_files'][rel];expert=load(OUT/rel)
                if cond=='clean':shifted=frames;d_obs=0.
                else:
                    c05=load(D/'c05'/cond/f.name);shifted=apply_burst(frames,r['input'],c05['generation'],cond.rsplit('_',1)[0],r['source']);d_obs=len(c05['actual_changed_positions'])/len(ids)
                assert pixelhash(shifted)==x['pixel_sha']==expert['pixel_sha256']
                assert expert['candidate_intervals']==[c['physical_interval'] for c in x['candidates']['temporal']]
                tick=time.monotonic();native=x['native'];req=dict(input=r['input'],frame_ids=ids,subject=r['parses']['subject'],native_boxes=native['boxes'],native_logits=native['logits'])
                data=capture(model,shifted,req);data['frame_ids']=ids;data=device_tree(data,'cuda')
                selected=x['candidates']['temporal'][expert['selected']]
                arms=dict(Frozen=dict(prediction=native),Rerank=dict(prediction={**native,'indices':selected['indices'],'physical_interval':selected['physical_interval']}))
                for arm in ['Hard','OPD']:
                    result,fields,evidence=step(model,data,expert,arm)
                    if r['ordinal']==p['rows'][0]['ordinal'] and cond in ['clean','frame_drop_1']:result['reinsertion']=reinsert(model,shifted,r,fields,evidence)
                    arms[arm]=result;del fields,evidence;gc.collect();torch.cuda.empty_cache()
                save(f,dict(parent=r['ordinal'],condition=cond,pixel_sha256=x['pixel_sha'],d_obs=d_obs,arms=arms,selected=expert['selected'],GT_read=False,exact_historical_native=True,model_state_sha256=mh,seconds=time.monotonic()-tick))
                write(f.with_suffix('.json'),dict(sha256=sha(f)));done+=1;status(OUT/'STATUS.json',dict(status='running',done=done,total=256));print('FOURARM',done,256,cond,'seconds',round(time.monotonic()-tick,2),flush=True)
                del data,x,expert,arms,shifted;gc.collect();torch.cuda.empty_cache()
            if limit and done>=limit:break
        assert state_hash(model.state_dict())==mh;verify();state='completed' if done==256 else 'smoke_complete'
        if done==256:write(OUT/'PREDICTION_BARRIER.json',dict(files={str(f.relative_to(OUT)):sha(f) for f in (OUT/'predictions').rglob('*.pt')},cells=256,arms=4,model_state_sha256=mh,GT_read=False,time=time.time()))
    except BaseException as e:failure=dict(error=repr(e),traceback=traceback.format_exc());raise
    finally:
        receipt=dict(status=state,done=done,seconds=time.monotonic()-start,prior_seconds=prior,failure=failure,time=time.time());write(OUT/'adapt_allocations'/f'{time.time_ns()}.json',receipt);status(OUT/'STATUS.json',receipt)
        if lease:lease.close()

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','run']);a.add_argument('--limit',type=int,default=0);x=a.parse_args();prepare() if x.stage=='prepare' else run(x.limit)
