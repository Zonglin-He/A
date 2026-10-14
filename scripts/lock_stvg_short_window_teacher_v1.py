"""Root lock for 64 paired frozen-R1 requests after actual input review."""
import sys,json,hashlib,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R));B=R/'artifacts/stvg_short_event_tracklets_v1';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
def save(p,d):assert not p.exists();p.write_text(json.dumps(d,indent=2)+'\n')
def main():
 import numpy as np
 jobs=read(B/'TEACHER_JOB_LOCK.json')['records'];assert len(jobs)==64;count=0
 for j in jobs:
  assert j['dataset']==j['case_id'].rsplit('_',1)[0];assert sha(B/'CANDIDATE_INPUT_BARRIER.json')==j['teacher_input_barrier_sha256'];assert len(j['videos'])==2
  for idx,v in enumerate(j['videos']):
   p=R/v['path'];assert sha(p)==v['sha256'];a=np.load(p,allow_pickle=False);assert a.shape==((4 if idx==0 else 5),3,448,448)and a.dtype==np.uint8;assert hashlib.sha256(a.tobytes()).hexdigest()==v['tensor_sha256']
  assert hashlib.sha256(j['prompt'].encode()).hexdigest()==j['prompt_sha256'];count+=1
 old=read(R/'artifacts/stvg_adjacent_motion_opd_v1/RUNTIME.json');paths={p for p in old['pins']if p.startswith(('scripts/','vg_tta/','methods/'))};paths|={'scripts/launch_stvg_short_window_v1.py','scripts/run_stvg_short_window_teacher_v1.py','scripts/prepare_stvg_short_window_teacher_v1.py','scripts/lock_stvg_short_window_teacher_v1.py','protocols/stvg_short_event_tracklets_execution_v1.md'};paths|={str(p.relative_to(R))for p in [B/'GLOBAL_TRACKLET_BARRIER.json',B/'ROOT_TRACKLET_READBACK.json',B/'FEATURE_CAPTURE_BARRIER.json',B/'ROOT_FEATURE_READBACK.json',B/'CANDIDATE_INPUT_BARRIER.json',B/'TEACHER_JOB_LOCK.json',B/'TEACHER_PREP_CPU_CONTRACTS.json']};pins={p:sha(R/p)for p in sorted(paths)}
 save(B/'ROOT_INPUT_VISUAL_REVIEW.json',dict(status='pass',actual_viewed_private_contacts=8,window_count=32,base_rotation_pair_count=32,context_frames=32,local_rendered_frame_slots=320,scope='all fixed inputs actual pixel view; missing tracklet frames preserved; no geometric quality claim',images=[dict(path=str(p.relative_to(R)),sha256=sha(p))for p in sorted((B/'private_views').glob('*.png'))],GT_read=False,time=time.time()))
 save(B/'RUNTIME.json',dict(status='locked',scope='64 fixed paired window R1 requests',GT_read=False,job_lock_sha256=sha(B/'TEACHER_JOB_LOCK.json'),pins=pins,protected=old['protected'],teacher_max_input_tokens=6144,max_new_tokens=96,teacher_dtype='BF16',quantization=False,time=time.time()));print('64 teacher jobs/root SHA/shape preflight pass')
if __name__=='__main__':main()
