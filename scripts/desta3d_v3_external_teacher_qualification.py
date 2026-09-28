"""Source-only external teacher inference; raw outputs before offline scoring."""
import argparse,fcntl,hashlib,json,os,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
OUT=ROOT/'artifacts/desta3d_v3/external_privileged_opd_v1'
OFFICIAL=ROOT/'external/LLaVA-ST';CK=ROOT/'checkpoints/LLaVA-ST-Qwen2-7B'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')

def register(name,count):
    out=OUT/name;assert not out.exists();assert count in (1,16)
    rows=read(ROOT/'artifacts/desta3d_v2/tta_v2/source_task_control_v2/INPUTS.json')
    assert len(rows)==16 and len({r['source'] for r in rows})==16
    completed=read(OUT/'MODEL_DOWNLOAD_COMPLETE.json')
    assert completed['all_lfs_hashes_match']
    cfg=dict(name=name,queries=count,parents=count,stage='engineering_smoke' if count==1 else 'source_teacher_qualification',
        optimizer_steps=0,target_input=False,target_GT=False,source_GT_in_worker=False,seed=20260928,
        revision=completed['revision'],max_new_tokens=1024,decode='greedy',precision='FP16',attention='sdpa',
        frames='100 nearest repetitions of the exact existing PTD observations; physical mapping stored',
        phase_seconds=900 if count==1 else 3600,free_disk_floor=8*2**30,cumulative_cap=None,
        prior_GPU_seconds=39977.51307785203+sum(read(p)['seconds'] for p in OUT.glob('*/RECEIPT.json')),
        expected_model_bytes=sum(f['size'] for f in completed['files']))
    write(out/'CONFIG.json',cfg);write(out/'INPUTS.json',rows[:count])
    paths=[Path(__file__),ROOT/'vg_tta/llava_st_teacher.py',ROOT/'vg_tta/external_privileged_views.py',
        ROOT/'protocols/desta3d_v3_external_privileged_opd_v1.md',OUT/'CPU_PREFLIGHT.json',out/'CONFIG.json',out/'INPUTS.json',
        OUT/'MODEL_DOWNLOAD_COMPLETE.json',ROOT/'vg_tta/exact_frame_decode_audit_v2.py']
    paths+=list((OFFICIAL/'llava').rglob('*.py'))+list((OFFICIAL/'inference').rglob('*.py'))
    write(out/'LOCK.json',dict(pins={str(p):sha(p) for p in paths},checkpoint={f['path']:f['actual_sha256'] for f in completed['files']},
        official_commit='bacf6d61e1de27a78fc0025083aa445d3d41b0b3'))
    write(out/'REGISTRATION.json',dict(time=time.time(),status='registered_before_GPU',source_training_panel=True,teacher_training_overlap_possible=True,GT_selection=False))
    print('REGISTERED',out,flush=True)

def run(name):
    out=OUT/name;cfg=read(out/'CONFIG.json');lock=read(out/'LOCK.json');assert not (out/'STARTED.json').exists()
    for p,h in lock['pins'].items():assert sha(p)==h,p
    for p,h in lock['checkpoint'].items():assert sha(CK/p)==h,p
    lease=(ROOT/'artifacts/spatial_tta_research_v2/gpu.lock').open('a');fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.monotonic();status='running'
    try:
        write(out/'STARTED.json',dict(time=time.time(),pid=os.getpid(),prior_GPU_seconds=cfg['prior_GPU_seconds']))
        import torch,numpy as np
        from vg_tta.llava_st_teacher import load,predict
        from vg_tta.external_privileged_views import repeated_frame_indices,parse_teacher_text
        from vg_tta.exact_frame_decode_audit_v2 import decode
        def guard():
            assert shutil.disk_usage(ROOT).free>=cfg['free_disk_floor'],'8GiB disk reserve'
            assert time.monotonic()-start<cfg['phase_seconds'],'phase limit'
        guard();torch.set_num_threads(4);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        torch.cuda.reset_peak_memory_stats()
        tokenizer,model,processor,info=load(CK)
        write(out/'MODEL_LOADING.json',dict(info=info,model_dtype=str(model.dtype),all_parameters_frozen=all(not p.requires_grad for p in model.parameters()),vision_tensors=421))
        for i,row in enumerate(read(out/'INPUTS.json')):
            guard();frames,ids=decode(row['input']);positions,teacher_ids=repeated_frame_indices(ids)
            ep=out/'episodes'/f'{i:03d}'
            write(ep/'INPUT.json',dict(key=row['key'],source=row['source'],frame_ids=ids,frame_shape=list(frames.shape),
                pixel_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),original_video_sha256=row['input']['video_sha256'],
                repeated_observation_indices=positions.tolist(),teacher_physical_frame_ids=teacher_ids.tolist(),no_unseen_pixels=True))
            pred,pixels=predict(tokenizer,model,processor,frames[positions],row['input']['caption'],OFFICIAL)
            # Always preserve raw output before parsing, including format failures.
            pred['processed_pixel_sha256']=hashlib.sha256(pixels.numpy().tobytes()).hexdigest()
            write(ep/'RAW_PREDICTION.json',pred)
            evidence=parse_teacher_text(pred['raw_text'],teacher_ids)
            write(ep/'EVIDENCE.json',evidence)
            assert all(p.grad is None and not p.requires_grad for p in model.parameters())
            write(ep/'COMPLETE.json',dict(key=row['key'],optimizer_steps=0,parameter_grad_none=True,format_errors=evidence['errors']))
            print('EPISODE',i+1,'/',cfg['queries'],'errors',evidence['errors'],flush=True)
        files=sorted((out/'episodes').rglob('*.json'))
        write(out/'PREDICTIONS_SEAL.json',dict(files={str(p.relative_to(out)):sha(p) for p in files},predictions=cfg['queries'],source_GT_read=False,target_read=False))
        write(out/'COMPLETE.json',dict(status='inference_complete_not_scored',queries=cfg['queries'],optimizer_steps=0,
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),source_GT_read=False,target_read=False))
        status='completed'
    except BaseException as e:
        status='failed';write(out/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
    finally:
        elapsed=time.monotonic()-start
        write(out/'RECEIPT.json',dict(status=status,seconds=elapsed,prior_seconds=cfg['prior_GPU_seconds'],
            cumulative_seconds=cfg['prior_GPU_seconds']+elapsed,cap=None,includes='imports model loading and inference; pre-GPU CPU hash checks excluded'))
        fcntl.flock(lease,fcntl.LOCK_UN)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['register','run']);p.add_argument('--name',required=True);p.add_argument('--count',type=int,default=1)
    a=p.parse_args();register(a.name,a.count) if a.action=='register' else run(a.name)
