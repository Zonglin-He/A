"""Unchanged student candidate generator on the existing transient seed0 panel."""
import sys,time,gc,traceback,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
from vg_tta.tastvg_deployment_corruption_v2 import FAMILIES,apply_burst
OUT=ROOT/'artifacts/tastvg_transient_c25_c3_c06_v1'
D=ROOT/'artifacts/tastvg_deployment_c05_c2t_v2';B=ROOT/'artifacts/tastvg_corruption_c0c1_v1'
CONDS=['clean']+[f'{f}_{s}' for f in FAMILIES for s in [1,5,10]]

def prepare():
    parent=read(D/'LOCK.json');bar=read(D/'C05_BARRIER.json');files={}
    for rel,h in bar['files'].items():assert sha(D/rel)==h;files[str(D/rel)]=h
    old=read(B/'PREDICTION_BARRIER.json')
    for r in parent['rows'][:16]:
        f=B/'capture/clean'/f"{r['ordinal']:03}.pt";assert sha(f)==old['files'][str(f.relative_to(B))]
        save(OUT/'capture/clean'/f.name,load(f));write((OUT/'capture/clean'/f.name).with_suffix('.json'),dict(sha256=sha(OUT/'capture/clean'/f.name),reused=str(f)))
    pins={f:sha(ROOT/f) for f in ['protocols/tastvg_transient_c25_c3_c06_v1.md','scripts/run_tastvg_transient_capture_v1.py','vg_tta/tastvg_corruption_c0c1_v1.py','vg_tta/tastvg_deployment_corruption_v2.py','vg_tta/tastvg_temporal_qualification_v1.py','methods/CURRENT_METHOD.json']}
    write(OUT/'LOCK.json',dict(rows=parent['rows'],pins=pins,baseline_files=files,model=parent['model'],conditions=CONDS,margin_terciles=read(D/'C2_BARRIER.json')['margin_terciles'],GT_read=False,time=time.time()))

def verify():
    p=read(OUT/'LOCK.json')
    for f,h in p['pins'].items():assert sha(ROOT/f)==h,f
    return p

def run(limit=0):
    start=time.monotonic();p=verify();prior=sum(read(f)['seconds'] for f in (OUT/'capture_allocations').glob('*.json'));done=0;state='failed';failure=None;lease=None
    try:
        import numpy as np,torch
        from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
        from scripts.run_final_simplification_v1 import lease as gpu_lease
        from methods.decota_final_simplified_v1.tensors import state_hash
        from vg_tta.exact_frame_decode_audit_v2 import decode
        from vg_tta.tastvg_corruption_c0c1_v1 import capture
        from scripts.c1_controlled_corruption_v1 import pixelhash
        procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);assert all('/opt/todesk/' in l for l in procs.splitlines() if l.strip()),procs
        lease=gpu_lease();install_clean_loader()
        from scripts.run_spatial_regression_alignment_v1 import model_load
        torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        model=model_load('hcstvg1_test');mh=state_hash(model.state_dict())
        for r in p['rows'][:16]:
            frames=None
            for cond in CONDS:
                f=OUT/'capture'/cond/f"{r['ordinal']:03}.pt"
                if f.with_suffix('.json').exists():assert sha(f)==read(f.with_suffix('.json'))['sha256'];done+=1;continue
                if limit and done>=limit:break
                assert prior+time.monotonic()-start<3580 and shutil.disk_usage(ROOT).free>8*2**30
                if frames is None:frames,ids=decode(r['input'])
                ref=D/'c05'/cond/f.name;assert sha(ref)==p['baseline_files'][str(ref)];old=load(ref);family=cond.rsplit('_',1)[0]
                shifted=apply_burst(frames,r['input'],old['generation'],family,r['source']);assert pixelhash(shifted)==old['pixel_sha256']
                old['batch_pixels_sha']=old['batch_sha256'];x=capture(model,shifted,r,old,True)
                x.update(ordinal=r['ordinal'],pixel_sha=old['pixel_sha256'],condition=cond,actual_changed_positions=old['actual_changed_positions'],d_obs=len(old['actual_changed_positions'])/len(ids),model_state_sha256=mh)
                save(f,x);write(f.with_suffix('.json'),dict(sha256=sha(f)));done+=1;status(OUT/'CAPTURE_STATUS.json',dict(done=done,total=256,status='running'));print('CAPTURE',done,256,cond,flush=True)
                del shifted,x,old;gc.collect();torch.cuda.empty_cache()
            if limit and done>=limit:break
        assert state_hash(model.state_dict())==mh;verify();state='completed' if done==256 else 'smoke_complete'
        if done==256:write(OUT/'CAPTURE_BARRIER.json',dict(files={str(f.relative_to(OUT)):sha(f) for f in (OUT/'capture').rglob('*.pt')},cells=256,clean_reused=16,model_state_sha256=mh,time=time.time()))
    except BaseException as e:failure=dict(error=repr(e),traceback=traceback.format_exc());raise
    finally:
        sec=time.monotonic()-start;z=dict(status=state,done=done,seconds=sec,failure=failure,time=time.time());write(OUT/'capture_allocations'/f'{time.time_ns()}.json',z);status(OUT/'CAPTURE_STATUS.json',z)
        if lease:lease.close()

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','run']);a.add_argument('--limit',type=int,default=0);x=a.parse_args();prepare() if x.stage=='prepare' else run(x.limit)
