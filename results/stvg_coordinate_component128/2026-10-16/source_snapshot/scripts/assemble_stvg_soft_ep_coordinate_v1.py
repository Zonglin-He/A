"""Assemble 128 exact reference/Temporal identities after the frozen-input stages."""
from pathlib import Path
import sys,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_soft_ep_coordinate_common_v1 import *
def main():
 import torch
 verify_runtime();bar=read(BASE/'SOURCE_CAPTURE_BARRIER.json');assert bar['count']==128 and not bar['GT_read'];sourcefiles={r['case_id']:r for r in bar['files']}
 freshbar=read(BASE/'confirmation/TEACHER_GLOBAL_BARRIER.json');assert freshbar['status']=='complete'and not freshbar['GT_read']
 for x in freshbar['files']:assert sha(ROOT/x['path'])==x['sha256']
 rows=[];times=[];sources=[];jobs_total=0
 for rec in roster():
  cid=rec['probe_alias'];case,sp=load_case(rec);assert sourcefiles[cid]['sha256']==sha(sp)and sourcefiles[cid]['source_case_sha256']==case['source_case_sha256']
  parent=ROOT/rec.get('cached_parent',str((BASE/'confirmation').relative_to(ROOT)))
  ref=next(x for x in read(parent/'REFERENCE_LOCK.json')['records']if x['case_id']==cid);assert ref['source_case_sha256']==case['source_case_sha256']and not ref['GT_read']
  anchors=copy.deepcopy(ref['new_crop']['anchors']);jobs={j['job_id']:j for j in read(parent/'TEACHER_JOB_LOCK.json')['records']if j['case_id']==cid and j['variant']=='new_crop'}
  for a in anchors:
   assert case['frame_ids'][a['position']]==a['physical_frame_id'] and 0<a['weight']<=1
   matches=[]
   for jid,j in jobs.items():
    p=parent/'teacher'/(jid+'.json')
    if sha(p)==a['teacher_output_sha256']:matches.append((j,read(p)))
   assert len(matches)==1
   job,z=matches[0];chosen=z['parsed']['selected_candidate_id'];assert chosen not in [None,1]and chosen==a['candidate_id']
   cb=next(x for x in job['candidate_bindings']if int(x['id'])==chosen);l,t,r,b=cb['center_box'];assert a['box_cxcywh']==[(l+r)/2,(t+b)/2,r-l,b-t]and a['weight']==cb['presence']
  tp=ROOT/rec['cached_temporal_path']if'cached_temporal_path'in rec else parent/'temporal'/(cid+'.pt');temp=torch.load(tp,map_location='cpu',weights_only=False);assert temp['case_id']==cid and not temp['GT_read'];assert temp['source_case_sha256']==case['source_case_sha256']
  rows.append(dict(case_id=cid,dataset=rec['dataset'],source_case_sha256=case['source_case_sha256'],anchors=anchors,GT_read=False))
  times.append(dict(case_id=cid,selected_interval=temp['selected_interval'],source_temporal_sha256=sha(tp),GT_read=False))
  sources.append(dict(case_id=cid,source_file_sha256=sha(sp),pixel_sha256=case['pixel_sha256'],reference_document_sha256=sha(parent/'REFERENCE_LOCK.json'),temporal_file_sha256=sha(tp)));jobs_total+=len(jobs)
 write(BASE/'REFERENCE_LOCK.json',dict(status='locked',count=128,anchor_count=sum(len(x['anchors'])for x in rows),records=rows,GT_read=False))
 write(BASE/'TEMPORAL_LOCK.json',dict(status='locked',count=128,records=times,GT_read=False))
 write(BASE/'ROOT_INPUT_ASSEMBLY.json',dict(status='pass',count=128,source_bindings=sources,teacher_job_count=jobs_total,reference_sources=sum(bool(r['anchors'])for r in rows),anchor_count=sum(len(r['anchors'])for r in rows),old_fresh_split=dict(exact_saved_source_temporal_reference=64,new_source_temporal_reference=64),no_GT_selection=True,GT_read=False,CPU_only=True))
 print('assembled128',sum(len(r['anchors'])for r in rows),'anchors')
if __name__=='__main__':main()
