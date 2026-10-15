"""Build the four matched spatial input cells, then bind exact R1 reuse."""
import sys,copy,time,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_evidence_common_v1 import *
def main():
 import torch,numpy as np
 from scripts import stvg_evidence_spatial_inputs005 as buildmod
 from scripts.run_stvg_weightfree_expand_v1 import selected_rgb
 from scripts import run_stvg_table1_source_v1 as source
 verify_runtime();phase=sys.argv[1];cohort='confirmation'if phase=='confirmation'else'development';b=BASE/phase;start=time.perf_counter();jobs=[];files=[];reuses=[];case_provenance=[]
 for rec in roster(cohort):
  cid=rec['probe_alias'];c,cp=load_case(rec);temp=torch.load(b/'temporal'/(cid+'.pt'),map_location='cpu',weights_only=False);req_path=(E/'requests'if phase=='round2'else b/'requests')/(cid+'.json');request=read(req_path);assert sha(Path(request['video_path']))==request['canonical_video_sha256'];assert temp['case_id']==cid and temp['dataset']==rec['dataset'] and not temp['GT_read'];assert temp['temporal_selection_sha256']==sha(BASE/'TEMPORAL_SELECTION_LOCK.json')
  if phase=='round2':
   r1p=BASE/'round1/student'/(cid+'.pt');r1bar=read(BASE/'round1/GLOBAL_PREDICTION_BARRIER.json');locked=next(x for x in r1bar['files']if ROOT/x['path']==r1p);assert temp['source_round1_sha256']==sha(r1p)==locked['sha256'];r1=torch.load(r1p,map_location='cpu',weights_only=False);winner=read(BASE/'TEMPORAL_SELECTION_LOCK.json')['variant'];assert temp['selected_interval']==r1['arms']['T_'+winner]['interval'];old=torch.load(E/'temporal'/(cid+'.pt'),map_location='cpu',weights_only=False);assert temp['old_interval']==old['temporal']['final_interval']
  _,inputrow=source._validate_roster_record(rec);query=inputrow['input']['caption'];wanted=set()
  for iv in [temp['old_interval'],temp['selected_interval']]:
   fm={int(x['physical_frame_id'])for x in request['frame_map']};eligible=[int(f)for f in c['frame_ids']if iv[0]<=int(f)<iv[1]and all(int(f)+k in fm for k in range(-2,3))];centers=[]
   if len(eligible)>=4:
    for i in range(4):centers.append(min((f for f in eligible if f not in centers),key=lambda f:(abs(f-(iv[0]+(i+.5)*(iv[1]-iv[0])/4)),f)))
   wanted.update(f+k for f in centers for k in range(-2,3))
  cache=buildmod.load_confirmation003_old_cache(cid,c['source_case_sha256'])if phase=='round2'else None
  if cache:
   for w in cache['proposal']['windows']:wanted.update(w['physical_frame_ids'])
  rgb=selected_rgb(request,wanted)if wanted else{}
  result=buildmod.build(b,c,request,temp['old_interval'],temp['selected_interval'],cid,rgb_by_physical_id=rgb,old_cache=cache,source_case_sha256=c['source_case_sha256'],request_sha256=sha(req_path),query_text=query)
  for rel,data in result['file_bytes'].items():
   path=ROOT/rel;path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();path.write_bytes(data);files.append(dict(path=rel,sha256=sha(path),bytes=len(data)))
  case_jobs=result['jobs'];jobs.extend(case_jobs)
  if phase=='round2':
   old_jobs={x['center_slot']:x for x in read(E/'TEACHER_JOB_LOCK.json')['records']if x['case_id']==cid}
   for job in case_jobs:
    if job['variant']!='old_crop':continue
    oldjob=old_jobs[job['center_slot']];assert job['prompt_sha256']==oldjob['prompt_sha256']and job['candidate_bindings']==oldjob['candidate_bindings']and job['visible_id_to_candidate']==oldjob['visible_id_to_candidate']
    for a,z in zip(job['videos'],oldjob['videos']):assert a['sha256']==z['sha256']and a['tensor_sha256']==z['tensor_sha256']and a['fps']==z['fps']
    op=E/'teacher'/(oldjob['job_id']+'.json');old=read(op);assert old['prompt_sha256']==job['prompt_sha256']and old['source_case_sha256']==c['source_case_sha256'];d=copy.deepcopy(old);d.update(job_id=job['job_id'],variant=job['variant'],input_videos=job['videos'],job_binding_sha256=hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest(),new_teacher_call=False,original_response_sha256=sha(op),actual_reuse=True)
    dest=b/'teacher'/(job['job_id']+'.json');write(dest,d);reuses.append(dict(job_id=job['job_id'],original_response_sha256=sha(op),path=str(dest.relative_to(ROOT)),sha256=sha(dest)))
  safe={k:v for k,v in result.items()if k not in ['file_bytes','jobs','conditions']};safe['file_count']=len(result['file_bytes']);write(b/'input_case_records'/(cid+'.json'),safe);case_provenance.append(dict(case_id=cid,job_count=len(case_jobs),path=str((b/'input_case_records'/(cid+'.json')).relative_to(ROOT)),sha256=sha(b/'input_case_records'/(cid+'.json'))));write(b/'CANDIDATE_STATUS.json',dict(status='running',done=len(case_provenance),total=32,GT_read=False),mutable=True);print('inputs',cid,len(case_jobs),flush=True)
 assert len({j['job_id']for j in jobs})==len(jobs)
 write(b/'TEACHER_JOB_LOCK.json',dict(status='locked',records=jobs,GT_read=False));write(b/'CANDIDATE_INPUT_BARRIER.json',dict(status='complete',count=32,teacher_job_count=len(jobs),files=files,case_records=case_provenance,exact_old_teacher_reuse_count=len(reuses),teacher_reuse=reuses,wall_seconds=time.perf_counter()-start,GT_read=False))
 # Per-case builder replays old pixels and records fresh new pixel hashes; the separate reader independently checks the resulting arrays and masks.
if __name__=='__main__':main()
