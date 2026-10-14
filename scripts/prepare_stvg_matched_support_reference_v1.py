"""Construct explicitly GT-assisted, matched-support references from sealed candidates."""
import copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
B=R/'artifacts/stvg_matched_support_reference_v1';OLD=R/'artifacts/stvg_short_event_tracklets_v1'
ROLES=('source','r1','pool_oracle','gt_oracle')
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
 import torch
 from scripts.score_stvg_short_window_postseal_v1 import _preflight_no_gt,_validate_math_readback
 from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal
 from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
 from vg_tta.tastvg_oracle_event5_v1 import box_iou
 gate=_preflight_no_gt();_validate_math_readback(gate)
 assert read(OLD/'FINAL_COMPLETION.json')['status']=='complete'
 assert read(OLD/'ROOT_POSTSEAL_INDEPENDENT_READBACK.json')['status']=='pass'
 manifest=read(OLD/'PREPARED_MANIFEST.json');oldrefs=read(OLD/'METHOD_REFERENCE_LOCK.json')
 byid={r['case_id']:r for r in manifest['cases']};cases={cid:load_source_case(R/r['source_case_path'])for cid,r in byid.items()}
 # New user explicitly authorizes label-assisted Oracle references. No new
 # candidate is constructed from GT and no GT geometry enters the method pool.
 truth,gtbindings=_truths_after_seal(cases)
 olddiag=read(OLD/'ROOT_POSTSEAL_REPORT.json');wins={(w['case_id'],w['slot']):w for w in olddiag['window_pool_teacher']}
 records=[];supports=[]
 for oldrow in oldrefs['records']:
  cid=oldrow['case_id'];c=cases[cid];mc=byid[cid];track=gate['trackdoc_by_id'][cid]
  proposals={v['tracklet_id']:v for v in track['tracklets']};refs={role:dict(anchors=[],offline_diagnostic=True,no_update=True,eligible_for_update=False,reason='empty_matched_support')for role in ROLES}
  for center in track['centers']:
   slot=int(center['slot']);fid=int(center['center_physical_frame_id']);mcenter=mc['centers'][slot];pos=int(mcenter['source_grid_position']);assert int(c['frame_ids'][pos])==fid
   job=gate['jobs_by_id'][f'{cid}_window{slot:02d}_base'];response=gate['responses_by_job'][job['job_id']];selected=response['parsed']['selected_candidate_id'];display=job['candidate_order'];boxes={};ids={}
   for tid in center['tracklet_ids']:
    t=proposals[tid];a=next(a for a in t['frames']if int(a['physical_frame_id'])==fid)
    if t['candidate_id']in display and a['valid'] and a['box_xyxy_norm']is not None:boxes[t['candidate_id']]=np.array(a['box_xyxy_norm'],dtype=np.float64);ids[t['candidate_id']]=tid
   known=fid in truth[cid]['truth'];valid_pool=bool(display)and set(display)<=set(boxes)and 1 in boxes
   valid_r1=response['parsed']['status']=='valid'and not response.get('output_truncated_at_cap')and selected in display
   use=known and valid_pool and valid_r1
   supports.append(dict(case_id=cid,dataset=c['dataset'],slot=slot,position=pos,physical_frame_id=fid,GT_known=known,all_displayed_candidates_valid=valid_pool,R1_valid_nonabstain=valid_r1,included=use,selection_uses_IoU_magnitude=False))
   assert use==(wins[(cid,slot)]['teacher_IoU']is not None)
   if not use:continue
   gt=np.array(truth[cid]['truth'][fid],dtype=np.float64)/[c['width'],c['height'],c['width'],c['height']]
   ious={k:float(box_iou(v[None],gt[None])[0])for k,v in boxes.items()};best=min(display,key=lambda k:(-ious[k],k))
   assert abs(ious[best]-wins[(cid,slot)]['oracle_IoU'])<1e-12
   for role,k in [('source',1),('r1',selected),('pool_oracle',best),('gt_oracle',None)]:
    box=gt if k is None else boxes[k];x1,y1,x2,y2=box;assert x2>x1 and y2>y1
    a=dict(position=pos,physical_frame_id=fid,box_cxcywh=[float((x1+x2)/2),float((y1+y2)/2),float(x2-x1),float(y2-y1)],selected_candidate_id=k,candidate_id=k,is_Source=k==1,slot=slot,reference_role=role,offline_label_assisted=role in ('pool_oracle','gt_oracle'),teacher_job_id=job['job_id'],teacher_output_sha256=next(x['sha256']for x in gate['teacher_file_rows']if x['path'].endswith('/'+job['job_id']+'.json')))
    if k is not None:a['tracklet_id']=ids[k]
    refs[role]['anchors'].append(a)
  for role in ROLES:
   n=len(refs[role]['anchors']);refs[role].update(no_update=not n,eligible_for_update=bool(n),reason='matched_support_reference_diagnostic'if n else'empty_matched_support')
  row=copy.deepcopy(oldrow);row.update(references=refs,GT_read=True,scope='offline_diagnostic_only');records.append(row)
 support_count=sum(r['included']for r in supports);fit_count=sum(bool(r['references'][role]['anchors'])for r in records for role in ROLES)
 lock=dict(roster_lock_sha256=sha(OLD/'ROSTER_LOCK.json'),manifest_sha256=sha(OLD/'PREPARED_MANIFEST.json'),tracklet_barrier_sha256=sha(OLD/'GLOBAL_TRACKLET_BARRIER.json'),old_student_barrier_sha256=sha(OLD/'STUDENT_GLOBAL_BARRIER.json'),schema='stvg-matched-support-reference-lock-v1',status='locked',scope='offline_diagnostic_only',GT_read=True,GT_used_for_oracle_references=True,GT_used_for_hyperparameters=False,count=8,roles=list(ROLES),records=records,support_count=support_count,expected_fit_count=fit_count,source_bindings={str(p.relative_to(R)):sha(p)for p in [OLD/'PREPARED_MANIFEST.json',OLD/'GLOBAL_TRACKLET_BARRIER.json',OLD/'TEACHER_GLOBAL_BARRIER.json',OLD/'TEACHER_JOB_LOCK.json',OLD/'ROOT_POSTSEAL_REPORT.json',OLD/'ROOT_POSTSEAL_INDEPENDENT_READBACK.json',R/'protocols/stvg_matched_support_reference_v1.md']},time=time.time())
 save(B/'REFERENCE_LOCK.json',lock);save(B/'SUPPORT_READBACK.json',dict(status='pass',records=supports,count=32,included=support_count,excluded=32-support_count,GT_read=True,GT_binding_sha256=hashlib.sha256(json.dumps(gtbindings,sort_keys=True).encode()).hexdigest(),GT_used_for_support_availability_only=True))
 save(B/'ROSTER_LOCK.json',read(OLD/'ROSTER_LOCK.json'))
 print(json.dumps(dict(status='locked',support_count=support_count,expected_fit_count=fit_count,per_case={r['case_id']:len(r['references']['source']['anchors'])for r in records},GPU_used=False)))
if __name__=='__main__':main()
