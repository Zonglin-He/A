"""Eight online expert reads first; full-budget reference only after online seal."""
import os,sys,time,gc,traceback,subprocess,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,save,load,sha,status
from scripts.run_tastvg_sparse_online_capture_v1 import OUT,D,verify
PREV=ROOT/'artifacts/tastvg_temporal_fourarm_v1'


def prepare():
    p=verify();assert (OUT/'CAPTURE_BARRIER.json').exists()
    previous=read(PREV/'C2_LOCK.json')
    pins={k:v for k,v in previous['pins'].items() if k.endswith(('.pth','.pt','opt.yaml','universal_vtg_inference.py'))}
    pins[str(Path(__file__))]=sha(__file__)
    for f,h in pins.items():assert sha(f)==h
    write(OUT/'TEACHER_LOCK.json',dict(pins=pins,capture_barrier_sha256=sha(OUT/'CAPTURE_BARRIER.json'),GT_read=False))


def run(phase):
    p=verify();lock=read(OUT/'TEACHER_LOCK.json')
    for f,h in lock['pins'].items():assert sha(f)==h
    assert sha(OUT/'CAPTURE_BARRIER.json')==lock['capture_barrier_sha256']
    assert phase in ['online','full']
    if phase=='full':assert (OUT/'ONLINE_BARRIER.json').exists()
    rows=[r for r in p['rows'] if r['expert']==(phase=='online')];assert len(rows)==(8 if phase=='online' else 24)
    reused=0;done=0;computed=0;models=None;lease=None;failure=None;state='failed';tick=time.monotonic()
    prior=sum(read(f)['seconds'] for f in (OUT/'gpu_allocations').glob('*.json'))
    try:
        import numpy as np,torch
        from PIL import Image
        from vg_tta.exact_frame_decode_audit_v2 import decode
        from vg_tta.tastvg_deployment_corruption_v2 import apply_burst
        from vg_tta.tastvg_temporal_qualification_v1 import critic_scores
        from scripts.c1_controlled_corruption_v1 import pixelhash
        from scripts.run_final_simplification_v1 import lease as gpu_lease
        from methods.decota_final_simplified_v1.tensors import state_hash
        bar=read(OUT/'CAPTURE_BARRIER.json');oldbar=read(PREV/'C2_BARRIER.json')
        torch.set_num_threads(4);torch.manual_seed(20260929);np.random.seed(20260929);torch.backends.cudnn.benchmark=False
        for r in rows:
            f=OUT/'teacher'/phase/f"{r['position']:03}.pt";rel=f"capture/{r['position']:03}.pt";assert sha(OUT/rel)==bar['files'][rel];x=load(OUT/rel)
            assert not f.exists();oldrel=f"c2/{r['condition']}/{r['ordinal']:03}.pt"
            if oldrel in oldbar['files']:
                assert sha(PREV/oldrel)==oldbar['files'][oldrel];z=load(PREV/oldrel)
                assert z['pixel_sha256']==x['pixel_sha256'] and z['candidate_intervals']==[c['physical_interval'] for c in x['candidates']]
                save(f,dict(position=r['position'],phase=phase,scores=z['scores'],selected=z['selected'],candidate_intervals=z['candidate_intervals'],proposals=z['proposals'],proposal_confidence=z['proposal_confidence'],pixel_sha256=z['pixel_sha256'],reused=True,reused_sha256=oldbar['files'][oldrel],GT_read=False));reused+=1
            else:
                if models is None:
                    procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True);assert all('/opt/todesk/' in l for l in procs.splitlines() if l.strip()),procs
                    lease=gpu_lease();os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';repo=ROOT/'external/UniversalVTG';sys.path[:0]=[str(repo),str(repo/'perception_models')]
                    from universal_vtg_inference import UniversalVTG
                    expert=UniversalVTG(experiment_name=str(ROOT/'checkpoints/universalvtg'),device='cuda',enable_query_unifier=False);expert._ensure_video_encoder();expert._ensure_text_encoder()
                    models=[expert.model,expert._video_extractor,expert._text_model]
                    for m in models:m.eval().requires_grad_(False)
                    hashes=[state_hash(m.state_dict()) for m in models];assert hashes==read(PREV/'C2_BARRIER.json')['model_hashes']
                assert prior+time.monotonic()-tick<1780 and shutil.disk_usage(ROOT).free>8*2**30
                frames,ids=decode(r['input']);old=load(D/'c05'/r['condition']/f"{r['ordinal']:03}.pt");shifted=apply_burst(frames,r['input'],old['generation'],r['condition'].rsplit('_',1)[0],r['source']);assert pixelhash(shifted)==x['pixel_sha256']
                text=expert.encode_text(r['input']['caption']);duration=(ids[-1]-ids[0]+1)/r['input']['fps'];slots=np.arange(max(1,int(duration*2)))/2
                pick=np.abs(np.asarray(ids)[None,:]-(ids[0]+slots*r['input']['fps'])[:,None]).argmin(1);unique,inverse=np.unique(pick,return_inverse=True);ff=[]
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                    for at in range(0,len(unique),16):
                        pixels=torch.stack([expert._video_preprocess(Image.fromarray(shifted[i])) for i in unique[at:at+16]]).cuda().half();ff.append(expert._video_extractor(pixels).float().cpu())
                video=torch.cat(ff)[inverse].T.contiguous()
                with torch.inference_mode():raw=expert.predict(video,text,return_raw=True,fps=None,feature_fps=2.,duration=duration,use_unifier=False)
                seg=raw['raw_segments'];conf=raw['raw_scores']
                if isinstance(seg,list):seg=seg[0];conf=conf[0]
                sec=expert._convert_segments_to_seconds(seg,fps=None,feature_fps=2.).clamp(0,duration).float().cpu().numpy();conf=conf.float().cpu().numpy();valid=sec[:,1]>sec[:,0];proposals=sec[valid]*r['input']['fps']+ids[0];conf=conf[valid]
                intervals=[c['physical_interval'] for c in x['candidates']];scores=critic_scores(intervals,proposals,conf)
                save(f,dict(position=r['position'],phase=phase,scores=scores.tolist(),selected=int(np.argmax(scores)),candidate_intervals=intervals,proposals=proposals.tolist(),proposal_confidence=conf.tolist(),pixel_sha256=x['pixel_sha256'],reused=False,GT_read=False));computed+=1
                del frames,shifted,raw,ff,video,text,pixels;gc.collect();torch.cuda.empty_cache()
            write(f.with_suffix('.json'),dict(sha256=sha(f)));done+=1;print('TEACHER',phase,done,len(rows),flush=True)
        if models is not None:assert hashes==[state_hash(m.state_dict()) for m in models]
        for f,h in lock['pins'].items():assert sha(f)==h
        write(OUT/f'TEACHER_{phase.upper()}_BARRIER.json',dict(phase=phase,files={str(f.relative_to(OUT)):sha(f) for f in (OUT/'teacher'/phase).glob('*.pt')},logical_calls=len(rows),new_evidence=computed,reused=reused,online_barrier_sha256=sha(OUT/'ONLINE_BARRIER.json') if phase=='full' else None,time=time.time()));state='completed'
    except BaseException as e:failure=dict(error=repr(e),traceback=traceback.format_exc());raise
    finally:
        z=dict(stage='teacher_'+phase,status=state,done=done,new_evidence=computed,reused=reused,seconds=time.monotonic()-tick,failure=failure,time=time.time());write(OUT/'gpu_allocations'/f'{time.time_ns()}.json',z);status(OUT/f'TEACHER_{phase.upper()}_STATUS.json',z)
        if lease:lease.close()

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','online','full']);x=a.parse_args();prepare() if x.stage=='prepare' else run(x.stage)
