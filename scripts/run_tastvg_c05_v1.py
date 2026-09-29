"""Bounded C0.5 generation/inference, isolated from metric computation."""
import sys,time,gc,traceback,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
from vg_tta.tastvg_temporal_qualification_v1 import FAMILIES,LEVELS,positions,corrupt_local
OUT=ROOT/'artifacts/tastvg_c05_c2t_v1';OLD=ROOT/'artifacts/tastvg_corruption_c0c1_v1'

def prepare():
    p=read(OLD/'LOCK.json');gt=read(OLD/'GT_SUBSET.json');rows=p['rows'];panel={}
    for r in rows:
        d={}
        for pct in LEVELS:
            ids,n=positions(r['frame_ids'],gt[r['key']]['interval'],pct)
            d[str(pct)]=dict(positions=ids,event_observations=n,actual_event_fraction=len(ids)/n,actual_video_fraction=len(ids)/len(r['frame_ids']))
        panel[str(r['ordinal'])]=d
    paths=['protocols/tastvg_c05_c2t_v1.md','vg_tta/tastvg_temporal_qualification_v1.py','scripts/run_tastvg_c05_v1.py','methods/CURRENT_METHOD.json']
    write(OUT/'LOCK.json',dict(rows=rows,pins={f:sha(ROOT/f) for f in paths},parent_lock_sha256=sha(OLD/'LOCK.json'),parent_barrier_sha256=sha(OLD/'PREDICTION_BARRIER.json'),model=p['model'],families=FAMILIES,levels=LEVELS,cap_seconds=3600,created=time.time()))
    write(OUT/'GENERATION.json',dict(panel=panel,interval_GT_for_generation=True,boxes_used=False,GT_subset_sha256=sha(OLD/'GT_SUBSET.json'),historically_exposed=True,created=time.time()))

def verify():
    p=read(OUT/'LOCK.json')
    for f,h in p['pins'].items():assert sha(ROOT/f)==h,f
    return p

def native(model,frames,row):
    import torch
    from methods.decota_final_simplified_v1.backbone import make_batch,query_subject,inserted_state,offset_batch
    from methods.decota_final_simplified_v1.tensors import detached
    from methods.decota_final_simplified_v1.objectives import prediction
    from scripts.c1_controlled_corruption_v1 import pixelhash
    batch=make_batch(frames,row['frame_ids'],row['input'],model);outs=[];records=[]
    with torch.inference_mode(),query_subject(model,batch,row['parses']['subject']),inserted_state(model,{}):
        for offset in (0,1):
            b=offset_batch(batch,offset)
            with torch.autocast('cuda',dtype=torch.float16):z=model(b['videos'],b['texts'],b['targets'],iteration_rate=-1)
            outs.append(detached(z,'cpu'));records.append(dict(offset=offset,flip=False,frame_ids=b['targets'][0]['frame_ids']))
    boxes=torch.stack([outs[i%2]['pred_boxes'][i//2] for i in range(len(frames))])
    return prediction([o['pred_sted'] for o in outs],boxes,records,row['frame_ids']),pixelhash(batch['videos'].tensors.cpu().numpy())

def run(limit=0):
    start=time.monotonic();p=verify();prior=sum(read(f)['seconds'] for f in (OUT/'c05_allocations').glob('*.json'));done=0;state='failed';failure=None;lease=None
    try:
        import torch,numpy as np
        from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
        from scripts.run_final_simplification_v1 import lease as gpu_lease
        from methods.decota_final_simplified_v1.tensors import state_hash
        from vg_tta.exact_frame_decode_audit_v2 import decode
        from scripts.c1_controlled_corruption_v1 import pixelhash
        procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);assert all('/opt/todesk/' in l for l in procs.splitlines() if l.strip()),procs
        lease=gpu_lease();install_clean_loader()
        from scripts.run_spatial_regression_alignment_v1 import model_load
        torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        model=model_load('hcstvg1_test');mh=state_hash(model.state_dict());gen=read(OUT/'GENERATION.json')['panel']
        for r in p['rows']:
            frames=None;memo={}
            for fam in FAMILIES:
                for pct in LEVELS:
                    cond=f'{fam}_{pct}';f=OUT/'c05'/cond/f"{r['ordinal']:03}.pt"
                    if f.with_suffix('.json').exists():assert sha(f)==read(f.with_suffix('.json'))['sha256'];done+=1;continue
                    if limit and done>=limit:break
                    assert prior+time.monotonic()-start<p['cap_seconds']-20
                    assert shutil.disk_usage(ROOT).free>8*2**30
                    assert sum(x.stat().st_size for x in OUT.rglob('*') if x.is_file())<8*2**30
                    if frames is None:
                        frames,ids=decode(r['input']);assert ids==r['frame_ids']
                        old=load(OLD/'capture/clean'/f"{r['ordinal']:03}.pt");assert pixelhash(frames)==old['pixel_sha']
                        if r['ordinal']==0 and not (OUT/'CLEAN_PARITY.json').exists():
                            check,bh=native(model,frames,r);assert bh==old['batch_pixels_sha'];assert torch.equal(check['boxes'],old['native']['boxes']);assert all(torch.equal(a,b) for a,b in zip(check['logits'],old['native']['logits']))
                            write(OUT/'CLEAN_PARITY.json',dict(status='pass',model_state_sha256=mh,boxes_exact=True,logits_exact=True))
                    generation=gen[str(r['ordinal'])][str(pct)];pos=generation['positions'];memo_key=(fam,tuple(pos));reused=memo.get(memo_key)
                    if reused:z=load(reused);z={**z,'condition':cond,'generation':generation,'reused':str(reused.relative_to(OUT))}
                    else:
                        shifted=corrupt_local(frames,pos,fam,r['source']);mask=np.ones(len(frames),bool);mask[pos]=False;assert np.array_equal(shifted[mask],frames[mask]);tick=time.monotonic()
                        pred,bh=native(model,shifted,r)
                        changed=np.flatnonzero(np.any(shifted!=frames,axis=(1,2,3))).tolist();assert set(changed)<=set(pos)
                        z=dict(ordinal=r['ordinal'],condition=cond,native=pred,pixel_sha256=pixelhash(shifted),batch_sha256=bh,generation=generation,actual_changed_positions=changed,model_state_sha256=mh,GT_in_model=False,seconds=time.monotonic()-tick,reused=None)
                        del shifted
                    save(f,z);write(f.with_suffix('.json'),dict(sha256=sha(f)));memo[memo_key]=f;done+=1
                    status(OUT/'C05_STATUS.json',dict(status='running',done=done,total=480));print('C05',done,480,cond,'reuse',bool(reused),flush=True)
                    del z;gc.collect();torch.cuda.empty_cache()
                if limit and done>=limit:break
            if limit and done>=limit:break
        assert state_hash(model.state_dict())==mh;verify();state='completed' if done==480 else 'smoke_complete'
        if done==480:write(OUT/'C05_BARRIER.json',dict(files={str(f.relative_to(OUT)):sha(f) for f in (OUT/'c05').rglob('*.pt')},cells=480,clean_reused=32,model_state_sha256=mh,created=time.time()))
    except BaseException as e:failure=dict(error=repr(e),traceback=traceback.format_exc());raise
    finally:
        sec=time.monotonic()-start;receipt=dict(status=state,done=done,seconds=sec,prior_seconds=prior,failure=failure,time=time.time());write(OUT/'c05_allocations'/f'{time.time_ns()}.json',receipt);status(OUT/'C05_STATUS.json',receipt)
        if lease:lease.close()

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','run']);a.add_argument('--limit',type=int,default=0);x=a.parse_args()
    prepare() if x.stage=='prepare' else run(x.limit)
