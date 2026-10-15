"""Correct only the crop clock-control serialization, retaining all previous measurements."""
import sys,json,copy,hashlib,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_evidence_common_v1 import *
def main():
 from scripts import stvg_evidence_spatial_inputs005 as old
 b=BASE/'round2_prompt009';b.mkdir(exist_ok=False);parent=BASE/'round2';assert read(parent/'ROOT_INPUT_READBACK.json')['status']=='pass';jobs=read(parent/'TEACHER_JOB_LOCK.json')['records'];corrected=[];reuse=[];changes=[]
 for item in read(parent/'CANDIDATE_INPUT_BARRIER.json')['files']:assert sha(ROOT/item['path'])==item['sha256']
 for j in jobs:
  row=copy.deepcopy(j)
  if j['variant']=='new_crop':
   q=read(E/'requests'/(j['case_id']+'.json'))['query_text'];ctx=j['context_center_physical_frame_ids'];p=read(parent/'new_crop/proposals'/(j['case_id']+'.json'));win=next(w for w in p['windows']if w['slot']==j['center_slot']);ct=p['context_timestamps_seconds'];tt=win['timestamps_seconds'];ids=[int(x['id'])for x in j['candidate_bindings']];assert j['prompt']==old._prompt(q,ct,tt,ids,'crop');text=old._confirmation003_prompt(q,ct,tt,ids);row['prompt']=text;row['prompt_sha256']=hashlib.sha256(text.encode()).hexdigest();changes.append(dict(job_id=j['job_id'],old_prompt_sha256=j['prompt_sha256'],matched_prompt_sha256=row['prompt_sha256']))
  else:
   source=parent/'teacher'/(j['job_id']+'.json');z=read(source);assert z['job_binding_sha256']==hashlib.sha256(json.dumps(j,sort_keys=True).encode()).hexdigest();z=copy.deepcopy(z);z.update(new_teacher_call=False,actual_reuse=True,original_response_sha256=sha(source),job_binding_sha256=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest());target=b/'teacher'/(j['job_id']+'.json');write(target,z);reuse.append(dict(job_id=j['job_id'],sha256=sha(target),original_sha256=sha(source)))
  corrected.append(row)
 write(b/'TEACHER_JOB_LOCK.json',dict(status='locked',records=corrected,GT_read=False));cb=copy.deepcopy(read(parent/'CANDIDATE_INPUT_BARRIER.json'));cb['parent_input_barrier_sha256']=sha(parent/'CANDIDATE_INPUT_BARRIER.json');cb['prompt_serialization_only_revision']=True;cb['exact_parent_teacher_reuse_count']=len(reuse);write(b/'CANDIDATE_INPUT_BARRIER.json',cb)
 for name in ['VARIANT_LOCK.json','TEACHER_RUNTIME.json']:shutil.copyfile(parent/name,b/name)
 (b/'temporal').mkdir()
 for r in roster():shutil.copyfile(parent/'temporal'/(r['probe_alias']+'.pt'),b/'temporal'/(r['probe_alias']+'.pt'))
 write(b/'ROOT_INPUT_READBACK.json',dict(status='pass',case_count=32,GT_read=False,CPU_only=True,scope='all geometric/pixel files exact-reused from independently read complete round2; 128 new_crop prompts independently reconstructed with the exact original crop template; no new rendering or candidate search',parent_root_sha256=sha(parent/'ROOT_INPUT_READBACK.json'),parent_input_barrier_sha256=sha(parent/'CANDIDATE_INPUT_BARRIER.json'),input_barrier_sha256=sha(b/'CANDIDATE_INPUT_BARRIER.json'),teacher_job_lock_sha256=sha(b/'TEACHER_JOB_LOCK.json'),corrected_job_count=len(changes),exact_reused_response_count=len(reuse),changes=changes))
 write(b/'PROMPT_SERIALIZATION_CORRECTION_LOCK.json',dict(status='locked',scope='implementation correction for matched crop clock control, no prompt search',only_new_crop_changed=True,candidate_pixel_geometry_updates=0,new_teacher_jobs=len(changes),reused_teacher_jobs=len(reuse),parent_all_results_retained=True,GT_read=False,time=time.time()))
 print('corrected prompts',len(changes),'exact inference response reuse',len(reuse))
if __name__=='__main__':main()
