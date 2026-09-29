"""Input-only O1 roster and frozen hidden/candidate capture."""
import sys,time,traceback,subprocess,gc,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
from vg_tta.tastvg_sparse_online_v1 import digest,native_scores,features
from scripts.run_tastvg_transient_capture_v1 import CONDS
OUT=ROOT/'artifacts/tastvg_sparse_online_o1_v1';D=ROOT/'artifacts/tastvg_deployment_c05_c2t_v2'


def prepare():
    p=read(D/'LOCK.json');rows=sorted(p['rows'],key=lambda r:(digest('O1-order-20260929|'+r['source']),r['source']))
    assert len(rows)==len({r['source'] for r in rows})==32
    for i,r in enumerate(rows):r.update(position=i+1,condition=CONDS[1:][int(digest('O1-condition-20260929|'+r['source']),16)%15],expert=(i%4==0))
    files={}
    for r in rows:
        rel=f"c05/{r['condition']}/{r['ordinal']:03}.pt";h=read(D/'C05_BARRIER.json')['files'][rel];assert sha(D/rel)==h;files[rel]=h
    paths=['protocols/tastvg_sparse_online_o1_v1.md','vg_tta/tastvg_sparse_online_v1.py','scripts/run_tastvg_sparse_online_capture_v1.py','vg_tta/tastvg_corruption_c0c1_v1.py','vg_tta/spatial_online_state_v1.py','methods/CURRENT_METHOD.json']
    write(OUT/'LOCK.json',dict(rows=rows,baseline_files=files,pins={f:sha(ROOT/f) for f in paths},lr=.001,steps=1,expert_positions=[r['position'] for r in rows if r['expert']],GT_read=False,created=time.time()))


def verify():
    p=read(OUT/'LOCK.json')
    for f,h in p['pins'].items():assert sha(ROOT/f)==h,f
    return p


def run(limit=0):
    tick=time.monotonic();p=verify();done=0;state='failed';failure=None;lease=None
    prior=sum(read(f)['seconds'] for f in (OUT/'gpu_allocations').glob('*.json'))
    try:
        import numpy as np,torch
        from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
        from scripts.run_final_simplification_v1 import lease as gpu_lease
        from methods.decota_final_simplified_v1.tensors import state_hash
        from vg_tta.exact_frame_decode_audit_v2 import decode
        from vg_tta.tastvg_corruption_c0c1_v1 import capture
        from vg_tta.tastvg_deployment_corruption_v2 import apply_burst
        from scripts.c1_controlled_corruption_v1 import pixelhash
        procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);assert all('/opt/todesk/' in l for l in procs.splitlines() if l.strip()),procs
        lease=gpu_lease();install_clean_loader()
        from scripts.run_spatial_regression_alignment_v1 import model_load
        torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        model=model_load('hcstvg1_test');mh=state_hash(model.state_dict())
        for r in p['rows']:
            f=OUT/'capture'/f"{r['position']:03}.pt"
            if f.with_suffix('.json').exists():assert sha(f)==read(f.with_suffix('.json'))['sha256'];done+=1;continue
            if limit and done>=limit:break
            assert prior+time.monotonic()-tick<1780 and shutil.disk_usage(ROOT).free>8*2**30
            frames,ids=decode(r['input']);rel=f"c05/{r['condition']}/{r['ordinal']:03}.pt";assert sha(D/rel)==p['baseline_files'][rel];old=load(D/rel)
            shifted=apply_burst(frames,r['input'],old['generation'],r['condition'].rsplit('_',1)[0],r['source']);assert pixelhash(shifted)==old['pixel_sha256']
            hidden=[]
            def hook(mod,args):hidden.append(args[0][-1].detach().cpu().clone())
            handle=model.temp_embed.register_forward_pre_hook(hook)
            try:old['batch_pixels_sha']=old['batch_sha256'];x=capture(model,shifted,r,old,True)
            finally:handle.remove()
            assert len(hidden)==2 and all(h.shape==(1,len(rec['frame_ids']),256) for h,rec in zip(hidden,x['records']))
            hh=torch.stack([hidden[i%2][0,i//2] for i in range(len(ids))]);cc=x['candidates']['temporal'];ell=native_scores(x['native']['logits'],x['records'],[c['physical_interval'] for c in cc])
            assert int(ell.argmax())==0,('zero_state_native',r['position'],ell)
            phi=features(hh,[c['indices'] for c in cc])
            save(f,dict(position=r['position'],parent=r['ordinal'],condition=r['condition'],expert=r['expert'],native=x['native'],candidates=cc,records=x['records'],hidden=hh,phi=phi,base_score=ell,pixel_sha256=old['pixel_sha256'],d_obs=len(old['actual_changed_positions'])/len(ids),model_state_sha256=mh,zero_state_native=True,GT_read=False))
            write(f.with_suffix('.json'),dict(sha256=sha(f)));done+=1;status(OUT/'CAPTURE_STATUS.json',dict(status='running',done=done,total=32));print('CAPTURE',done,32,flush=True)
            del x,hh,phi,frames,shifted,old;gc.collect();torch.cuda.empty_cache()
        assert state_hash(model.state_dict())==mh;verify();state='completed' if done==32 else 'smoke_complete'
        if done==32:write(OUT/'CAPTURE_BARRIER.json',dict(files={str(f.relative_to(OUT)):sha(f) for f in (OUT/'capture').glob('*.pt')},model_state_sha256=mh,cells=32,GT_read=False))
    except BaseException as e:failure=dict(error=repr(e),traceback=traceback.format_exc());raise
    finally:
        z=dict(stage='capture',status=state,done=done,seconds=time.monotonic()-tick,failure=failure,time=time.time());write(OUT/'gpu_allocations'/f'{time.time_ns()}.json',z);status(OUT/'CAPTURE_STATUS.json',z)
        if lease:lease.close()

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','run']);a.add_argument('--limit',type=int,default=0);x=a.parse_args();prepare() if x.stage=='prepare' else run(x.limit)
