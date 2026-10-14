"""Independent saved endpoint checks and pre-GT global A/B recipe selection."""
import sys,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_cross_frame_common_v1 import *

def main():
 verify();sys.addaudithook(guard)
 import torch,numpy as np
 from scripts.review_stvg_instance_step_diagnostic_v1 import iou_cxcywh
 from scripts import stvg_target_teacher_student_v1 as core
 seal=read(BASE/'STUDENT_GLOBAL_BARRIER.json')
 if seal['status']!='complete' or seal['count']!=8 or seal['GT_read'] is not False:raise RuntimeError('full Student seal required')
 refs=read(BASE/'REFERENCE_LOCK.json');rows=[];checks=0;maxerr=0
 def quality(boxes,anchors,xyxy=False,weighted=False):
  if not anchors:return None
  p=np.asarray(boxes,dtype=np.float64)
  if xyxy:p=np.concatenate(((p[:,:2]+p[:,2:])/2,p[:,2:]-p[:,:2]),axis=1)
  values=iou_cxcywh(p[[a['position']for a in anchors]],np.asarray([a['box_cxcywh']for a in anchors]))
  w=np.asarray([a['reliability']['presence_fraction']for a in anchors]) if weighted else np.ones(len(anchors))
  return float(np.dot(w,values)/w.sum())
 for r,item in zip(refs['records'],seal['files']):
  cid=r['case_id'];rp=ROOT/item['path']
  if sha(rp)!=item['sha256']:raise RuntimeError('receipt not in seal')
  receipt=read(rp);p=ROOT/receipt['path']
  if sha(p)!=receipt['sha256']:raise RuntimeError('student bytes mismatch')
  d=torch.load(p,map_location='cpu',weights_only=False)
  if d['case_id']!=cid or d['reference_record']!=r or d['GT_read'] or not d['source_restored']:raise RuntimeError('student record identity/state mismatch')
  fit=d['fit'];train=r['training_anchors'];vals=r['validation_anchors']
  if fit:
   if not torch.equal(d['readout']['boxes'].cpu(),fit['final_cached_decoder_boxes']):raise RuntimeError('full readout mismatch')
   rejected=False
   for s in fit['path']:
    if s.get('skipped'):continue
    if rejected:raise RuntimeError('update after rejection')
    q0=quality(s['boxes'],train,weighted=True);q1=quality(s['proposed_boxes'],train,weighted=True)
    v0=quality(s['boxes'],vals);v1=quality(s['proposed_boxes'],vals)
    # Independent float64 geometry. Accept near rounding boundaries only if
    # the recorded FP32 score has the same sign within the disclosed tolerance.
    if s['accepted'] and (q1<q0-2e-6 or (v0 is not None and v1<v0-2e-6)):raise RuntimeError('accepted reference regression')
    if s['proposed'] and not s['accepted']:
     if not core._tree_equal(s['optimizer_after'],s['optimizer_before']) or not core._tree_equal(s['update_state'],s['state']):raise RuntimeError('rollback mismatch')
     if q1>=q0+2e-6 and (v0 is None or v1>=v0+2e-6):raise RuntimeError('unexplained rejection')
     rejected=True
    checks+=1
   if quality(fit['final_cached_decoder_boxes'],train,weighted=True)<quality(d['arms']['F']['boxes'],train,xyxy=True,weighted=True)-2e-6:raise RuntimeError('final train regression')
  a=quality(d['arms']['validated_loss_weighted_iou']['boxes'],vals,xyxy=True)
  b=quality(d['arms']['cross_frame']['boxes'],vals,xyxy=True)
  rows.append(dict(case_id=cid,dataset=r['dataset'],validation_count=len(vals),Qval_A=a,Qval_B=b,delta_B_minus_A=None if a is None else b-a,accepted_steps=0 if fit is None else fit['optimizer_steps'],proposed_steps=0 if fit is None else fit['proposed_optimizer_steps'],fallback=not vals,full_readout_pass=True))
 criteria=[]
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];deltas=[r['delta_B_minus_A']for r in rr if r['delta_B_minus_A'] is not None]
  mean=math.fsum(deltas)/len(deltas) if deltas else None
  ok=bool(deltas) and mean>=-1e-7 and sum(r['accepted_steps']for r in rr)>0
  criteria.append(dict(dataset=ds,evaluable_count=len(deltas),mean_validation_delta=mean,informative_updates=sum(r['accepted_steps']for r in rr),pass_criterion=ok))
 selected='B_cross_frame' if all(c['pass_criterion']for c in criteria) else 'A_train_only'
 write(BASE/'ROOT_ACTUAL_GATE_READBACK.json',dict(status='pass',scope='independent endpoint geometry, recorded parameter/Adam rollback and complete model re-decode; not independent GPU backward',count=8,step_checks=checks,records=rows,GT_read=False,CPU_only=True,time=time.time()))
 write(BASE/'METHOD_SELECTION_LOCK.json',dict(status='locked',scope='one global spatial rule chosen before new GT scores',selected_recipe=selected,criteria=criteria,selection_rule_protocol_sha256=sha(ROOT/'protocols/stvg_cross_frame_validation_v1.md'),source_runtime_sha256=sha(BASE/'STUDENT_RUNTIME.json'),reference_sha256=sha(BASE/'REFERENCE_LOCK.json'),root_readback_sha256=sha(BASE/'ROOT_ACTUAL_GATE_READBACK.json'),GT_read=False,no_per_dataset_choice=True,no_more_algorithm_search=True,time=time.time()))
 print('Selected',selected,'using only fixed teacher references')
if __name__=='__main__':main()
