"""Cached critic for the minimal four-arm experiment, no separate gate or GT access."""
import os,sys,time,gc,traceback,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
from vg_tta.tastvg_temporal_qualification_v1 import critic_scores
OUT=ROOT/'artifacts/tastvg_temporal_fourarm_v1';OLD=ROOT/'artifacts/tastvg_transient_c25_c3_c06_v1'
D=ROOT/'artifacts/tastvg_deployment_c05_c2t_v2'
from scripts.run_tastvg_transient_capture_v1 import CONDS
from vg_tta.tastvg_deployment_corruption_v2 import apply_burst

def prepare():
    assert (OLD/'CAPTURE_BARRIER.json').exists()
    from huggingface_hub import hf_hub_download
    pe=Path(hf_hub_download('facebook/PE-Core-L14-336','PE-Core-L14-336.pt',local_files_only=True))
    paths=[Path(__file__),ROOT/'vg_tta/tastvg_temporal_qualification_v1.py',ROOT/'protocols/tastvg_temporal_fourarm_v1.md',ROOT/'checkpoints/universalvtg/models/best.pth',ROOT/'checkpoints/universalvtg/opt.yaml',pe,ROOT/'external/UniversalVTG/universal_vtg_inference.py']
    p=read(OLD/'LOCK.json');bar=read(OLD/'CAPTURE_BARRIER.json')
    rows=p['rows'][:16];files={}
    for r in rows:
        for cond in CONDS:
            rel=f"capture/{cond}/{r['ordinal']:03}.pt";assert sha(OLD/rel)==bar['files'][rel];files[rel]=bar['files'][rel]
    previous=read(D/'C2_BARRIER.json')
    for r in rows:
        rel=f"c2/clean/{r['ordinal']:03}.pt";assert sha(D/rel)==previous['files'][rel]
        e=load(D/rel);x=load(OLD/'capture/clean'/f"{r['ordinal']:03}.pt")
        assert e['pixel_sha256']==x['pixel_sha'] and e['candidate_intervals']==[c['physical_interval'] for c in x['candidates']['temporal']]
        save(OUT/rel,e);write((OUT/rel).with_suffix('.json'),dict(sha256=sha(OUT/rel),reused=str(D/rel)))
    write(OUT/'C2_LOCK.json',dict(rows=rows,candidate_files=files,pins={str(f):sha(f) for f in paths},cap_seconds=3600,GT_in_worker=False,created=time.time()))

def verify():
    p=read(OUT/'C2_LOCK.json')
    for f,h in p['pins'].items():assert sha(f)==h,f
    for f,h in p['candidate_files'].items():assert sha(OLD/f)==h
    return p

def run(limit=0):
    start=time.monotonic();p=verify();prior=sum(read(f)['seconds'] for f in (OUT/'c2_allocations').glob('*.json'));done=0;state='failed';failure=None;lease=None
    try:
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import numpy as np,torch
        from PIL import Image
        from vg_tta.exact_frame_decode_audit_v2 import decode
        from scripts.c1_controlled_corruption_v1 import pixelhash
        from scripts.run_final_simplification_v1 import lease as gpu_lease
        from methods.decota_final_simplified_v1.tensors import state_hash
        procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);assert all('/opt/todesk/' in l for l in procs.splitlines() if l.strip()),procs
        lease=gpu_lease();repo=ROOT/'external/UniversalVTG';sys.path[:0]=[str(repo),str(repo/'perception_models')]
        from universal_vtg_inference import UniversalVTG
        torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False
        expert=UniversalVTG(experiment_name=str(ROOT/'checkpoints/universalvtg'),device='cuda',enable_query_unifier=False)
        expert._ensure_video_encoder();expert._ensure_text_encoder()
        models=[expert.model,expert._video_extractor,expert._text_model]
        for m in models:m.eval().requires_grad_(False)
        hashes=[state_hash(m.state_dict()) for m in models]
        for r in p['rows']:
            frames=None;text=None
            for cond in CONDS:
                f=OUT/'c2'/cond/f"{r['ordinal']:03}.pt"
                if f.with_suffix('.json').exists():assert sha(f)==read(f.with_suffix('.json'))['sha256'];done+=1;continue
                if limit and done>=limit:break
                assert prior+time.monotonic()-start<p['cap_seconds']-20
                assert shutil.disk_usage(ROOT).free>8*2**30
                assert sum(x.stat().st_size for x in OUT.rglob('*') if x.is_file())<8*2**30
                if frames is None:
                    frames,ids=decode(r['input']);text=expert.encode_text(r['input']['caption'])
                x=load(OLD/'capture'/cond/f"{r['ordinal']:03}.pt")
                c05=load(D/'c05'/cond/f"{r['ordinal']:03}.pt")
                shifted=apply_burst(frames,r['input'],c05['generation'],cond.rsplit('_',1)[0],r['source']);assert pixelhash(shifted)==x['pixel_sha']
                duration=(ids[-1]-ids[0]+1)/r['input']['fps'];slots=np.arange(max(1,int(duration*2)))/2
                physical=ids[0]+slots*r['input']['fps'];pick=np.abs(np.asarray(ids)[None,:]-physical[:,None]).argmin(1)
                unique,inverse=np.unique(pick,return_inverse=True);features=[];tick=time.monotonic()
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                    for at in range(0,len(unique),16):
                        pixels=torch.stack([expert._video_preprocess(Image.fromarray(shifted[i])) for i in unique[at:at+16]]).cuda().half()
                        features.append(expert._video_extractor(pixels).float().cpu())
                video=torch.cat(features)[inverse].T.contiguous()
                with torch.inference_mode():raw=expert.predict(video,text,return_raw=True,fps=None,feature_fps=2.,duration=duration,use_unifier=False)
                seg=raw['raw_segments'];scores=raw['raw_scores']
                if isinstance(seg,list):seg=seg[0];scores=scores[0]
                sec=expert._convert_segments_to_seconds(seg,fps=None,feature_fps=2.).clamp(0,duration).float().cpu().numpy();confidence=scores.float().cpu().numpy()
                assert np.isfinite(sec).all() and np.isfinite(confidence).all()
                valid=sec[:,1]>sec[:,0];sec=sec[valid];confidence=confidence[valid]
                proposals=sec*r['input']['fps']+ids[0];cand=x['candidates']['temporal'];intervals=[c['physical_interval'] for c in cand]
                ranking=critic_scores(intervals,proposals,confidence);selected=int(np.argmax(ranking))
                save(f,dict(parent=r['ordinal'],condition=cond,candidate_intervals=intervals,candidate_origins=[c['origin'] for c in cand],scores=ranking.tolist(),selected=selected,
                    proposals=proposals.tolist(),proposal_confidence=confidence.tolist(),video_features=video,text_features=text.cpu(),pixel_sha256=x['pixel_sha'],slots_seconds=slots.tolist(),picked_observations=pick.tolist(),duration=duration,
                    GT_read=False,teacher_interval_as_output=False,seconds=time.monotonic()-tick))
                write(f.with_suffix('.json'),dict(sha256=sha(f)));done+=1;status(OUT/'C2_STATUS.json',dict(status='running',done=done,total=256));print('C2',done,256,cond,'selected',selected,'proposals',len(sec),'seconds',round(time.monotonic()-tick,2),flush=True)
                del shifted,x,raw,pixels,video,features;gc.collect();torch.cuda.empty_cache()
            if limit and done>=limit:break
        assert hashes==[state_hash(m.state_dict()) for m in models];verify();state='completed' if done==256 else 'smoke_complete'
        if done==256:
            gaps=[]
            for f in (OUT/'c2').rglob('*.pt'):
                z=load(f);v=z['scores'];gaps.extend(abs(v[i]-v[j]) for i in range(len(v)) for j in range(i+1,len(v)))
            write(OUT/'C2_BARRIER.json',dict(files={str(f.relative_to(OUT)):sha(f) for f in (OUT/'c2').rglob('*.pt')},cells=256,margin_terciles=np.quantile(gaps,[1/3,2/3]).tolist(),model_hashes=hashes,GT_read=False,created=time.time()))
    except BaseException as e:failure=dict(error=repr(e),traceback=traceback.format_exc());raise
    finally:
        sec=time.monotonic()-start;receipt=dict(status=state,done=done,seconds=sec,prior_seconds=prior,failure=failure,time=time.time());write(OUT/'c2_allocations'/f'{time.time_ns()}.json',receipt);status(OUT/'C2_STATUS.json',receipt)
        if lease:lease.close()

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','run']);a.add_argument('--limit',type=int,default=0);x=a.parse_args()
    prepare() if x.stage=='prepare' else run(x.limit)
