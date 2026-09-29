"""Bounded frozen-only C0/C1 with historical exact parity and label isolation."""
import argparse,gc,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
from scripts.c1_controlled_corruption_v1 import CONDITIONS,corrupt,pixelhash,configuration
BASE=ROOT/'artifacts/c1_controlled_corruption_v1'
OUT=ROOT/'artifacts/tastvg_corruption_c0c1_v1'
MEDIUM=['clean','noise_medium','defocus_medium','jpeg_medium']
OWN=['protocols/tastvg_corruption_c0c1_v1.md','vg_tta/tastvg_corruption_c0c1_v1.py','scripts/run_tastvg_corruption_c0c1_v1.py','tests/test_tastvg_corruption_c0c1_v1.py']


def prepare():
    rows=read(BASE/'INPUTS.json');parent=read(BASE/'LOCK.json');assert sha(BASE/'INPUTS.json')==parent['inputs_sha'];assert len(rows)==len({r['source'] for r in rows})==32
    assert [r['ordinal'] for r in rows]==list(range(32));bar=read(BASE/'homogeneous/BARRIER.json');files={}
    for r in rows:
        for cond in CONDITIONS:
            f=BASE/'homogeneous/vid_source'/cond/f"{r['ordinal']:03}.pt";assert sha(f)==bar['files'][str(f)];z=load(f)
            assert z['key']==r['key'] and z['source']==r['source'] and z['frame_ids']==r['frame_ids'];assert z['model']=='vid_source' and z['condition']==cond
            target=OUT/'baseline'/cond/f"{r['ordinal']:03}.pt"
            save(target,{k:z[k] for k in ['key','source','ordinal','condition','native','frame_ids','pixel_sha','clean_pixel_sha','batch_pixels_sha']})
            files[str(target.relative_to(OUT))]=dict(sha256=sha(target),original=str(f),original_sha256=sha(f))
    cfg=configuration('vid_source').to_dict();assert cfg['checkpoint_sha256']==parent['configs']['vid_source']['checkpoint_sha256']
    pins={f:sha(ROOT/f) for f in OWN+['scripts/c1_controlled_corruption_v1.py','scripts/run_tastvg_evidence_vulnerability_v2.py','scripts/run_spatial_regression_alignment_v1.py','methods/decota_final_simplified_v1/backbone.py','methods/decota_final_simplified_v1/objectives.py','methods/decota_final_simplified_v1/_tastvg_load.py','external/TA-STVG/models/pipeline.py','external/TA-STVG/models/grounding_model/query_decoder.py','methods/CURRENT_METHOD.json','methods/CURRENT_WORKING_METHOD.json']}
    write(OUT/'LOCK.json',dict(rows=rows,conditions=CONDITIONS,candidate_conditions=MEDIUM,candidate_ordinals=list(range(16)),model=cfg,pins=pins,baseline_files=files,parent_lock_sha256=sha(BASE/'LOCK.json'),inputs_sha256=sha(BASE/'INPUTS.json'),labels_path=parent['labels'],labels_sha256=parent['labels_sha'],cap_seconds=3600,max_new_bytes=8*2**30,minimum_free_bytes=8*2**30,GT_this_round_read=False,experts_invoked=0,backwards=0,created=time.time()))
    print('REGISTERED224 frozen +64 candidate sets; no GT read')


def verify():
    p=read(OUT/'LOCK.json');pins=dict(p['pins'])
    for rev in sorted(OUT.glob('ENGINEERING_REVISION_*.json')):
        for f,z in read(rev)['files'].items():assert pins[f]==z['old_sha256'];pins[f]=z['new_sha256']
    for f,h in pins.items():assert sha(ROOT/f)==h,('pin',f)
    assert sha(BASE/'INPUTS.json')==p['inputs_sha256']
    return p


def run(limit=0):
    start=time.monotonic();p=verify();prior=sum(read(f)['seconds'] for f in (OUT/'allocations').glob('*.json'));done=0;failure=None;state='failed';lease=None
    try:
        import numpy as np,torch
        from scripts.run_tastvg_evidence_vulnerability_v2 import install_clean_loader
        from scripts.run_final_simplification_v1 import lease as gpu_lease
        from methods.decota_final_simplified_v1.tensors import state_hash
        from vg_tta.exact_frame_decode_audit_v2 import decode
        from vg_tta.tastvg_corruption_c0c1_v1 import capture
        install_clean_loader()
        from scripts.run_spatial_regression_alignment_v1 import model_load
        procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);assert all('/opt/todesk/' in l for l in procs.splitlines() if l.strip()),procs
        lease=gpu_lease();torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        model=model_load('hcstvg1_test');mh=state_hash(model.state_dict());assert all(not q.requires_grad for q in model.parameters())
        for r in p['rows']:
            frames=None
            for cond in CONDITIONS:
                f=OUT/'capture'/cond/f"{r['ordinal']:03}.pt"
                if f.with_suffix('.json').exists():assert sha(f)==read(f.with_suffix('.json'))['sha256'];done+=1;continue
                assert not f.exists()
                if limit and done>=limit:break
                assert prior+time.monotonic()-start<p['cap_seconds']-20,'process time cap'
                assert shutil.disk_usage(ROOT).free>p['minimum_free_bytes'],'disk floor'
                assert sum(x.stat().st_size for x in OUT.rglob('*') if x.is_file())<p['max_new_bytes'],'artifact cap'
                if frames is None:
                    assert sha(r['input']['video_path'])==r['input']['video_sha256'];frames,ids=decode(r['input']);assert ids==r['frame_ids']
                b=OUT/'baseline'/cond/f"{r['ordinal']:03}.pt";assert sha(b)==p['baseline_files'][str(b.relative_to(OUT))]['sha256'];old=load(b)
                assert pixelhash(frames)==old['clean_pixel_sha'];shifted=corrupt(frames,r['frame_ids'],r['source'],cond);assert pixelhash(shifted)==old['pixel_sha']
                tick=time.monotonic();want=r['ordinal']<16 and cond in MEDIUM
                result=capture(model,shifted,r,old,want)
                result.update(key=r['key'],source=r['source'],ordinal=r['ordinal'],condition=cond,pixel_sha=old['pixel_sha'],model_state_sha256=mh,GT_read=False,annotation_loader=model.taev_loader_provenance,seconds=time.monotonic()-tick)
                save(f,result);write(f.with_suffix('.json'),dict(sha256=sha(f),lock_sha256=sha(OUT/'LOCK.json'),time=time.time()));done+=1
                status(OUT/'STATUS.json',dict(status='running',done=done,total=224,ordinal=r['ordinal'],condition=cond,pid=os.getpid()));print('CAPTURE',done,224,r['key'],cond,'candidates',want,'seconds',round(result['seconds'],3),flush=True)
                del shifted,result,old;gc.collect();torch.cuda.empty_cache()
            del frames
            if limit and done>=limit:break
        assert state_hash(model.state_dict())==mh;verify();state='completed' if done==224 else 'smoke_complete'
        if done==224:
            files={str(f.relative_to(OUT)):sha(f) for f in (OUT/'capture').rglob('*.pt')};assert len(files)==224
            write(OUT/'PREDICTION_BARRIER.json',dict(files=files,frozen_cells=224,candidate_cells=64,model_state_sha256=mh,GT_read=False,created=time.time()))
    except BaseException as e:failure=dict(error=repr(e),traceback=traceback.format_exc());raise
    finally:
        sec=time.monotonic()-start;receipt=dict(status=state,done=done,seconds=sec,prior_seconds=prior,total_seconds=prior+sec,failure=failure,pid=os.getpid(),time=time.time())
        write(OUT/'allocations'/f'{time.time_ns()}.json',receipt);status(OUT/'STATUS.json',receipt)
        if lease:lease.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','capture']);parser.add_argument('--limit',type=int,default=0);a=parser.parse_args()
    if a.stage=='prepare':prepare()
    else:run(a.limit)
