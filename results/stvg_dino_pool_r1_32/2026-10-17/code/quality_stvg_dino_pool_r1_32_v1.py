"""Post-seal reference geometry at actually annotated positions; no oracle fitting."""
from pathlib import Path
import math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_dino_pool_r1_32_v3 import BASE,read,sha,put,xyxy,iou
def main():
 import numpy as np
 from scripts.stvg_reference_recovery_common_v1 import load_case
 from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal
 from scripts.score_stvg_native_instance_nativeclock005 import _truth_map
 assert read(BASE/'CPU_COMPLETION.json')['status']=='complete' and read(BASE/'ROOT_READBACK.json')['status']=='pass'
 rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['case_id']:load_case(r)[0] for r in rr};targets,_=_truths_after_seal(cases);refs=read(BASE/'R1_REFERENCE_LOCK.json');dec={r['job_id']:r for r in refs['decisions']};pools=read(BASE/'POOL_LOCK.json')['records'];rows=[]
 for p in pools:
  cid=p['case_id'];case=cases[cid];gt=_truth_map(targets[cid]);fid=p['physical_frame_id'];chosen=dec[p['job_id']]['selected_id'];r=dict(case_id=cid,dataset=case['dataset'],position=p['position'],selected_id=chosen,gt_observed=fid in gt,Top1_iou=None,R1_iou=None,R1_minus_Top1_pp=None)
  if fid in gt:
   scale=np.array([case['width'],case['height']]*2);a=np.asarray(xyxy(p['top1_box_cxcywh']))*scale;b=np.asarray(gt[fid],float);r['Top1_iou']=float(iou(a,b))
   if chosen is not None:
    a=np.asarray(xyxy(p['candidates'][int(chosen)-1]['box_cxcywh']))*scale;r['R1_iou']=float(iou(a,b));r['R1_minus_Top1_pp']=100*(r['R1_iou']-r['Top1_iou'])
  rows.append(r)
 summary=[]
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];known=[r for r in rr if r['gt_observed']];common=[r for r in known if r['R1_iou'] is not None];d=[r['R1_minus_Top1_pp'] for r in common];summary.append(dict(dataset=ds,all_positions=len(rr),GT_known=len(known),GT_unknown=len(rr)-len(known),R1_selected_known=len(common),R1_abstained_or_invalid_known=len(known)-len(common),Top1_all_known_mean=math.fsum(r['Top1_iou'] for r in known)/len(known) if known else None,Top1_common_selected_mean=math.fsum(r['Top1_iou'] for r in common)/len(common) if common else None,R1_common_selected_mean=math.fsum(r['R1_iou'] for r in common)/len(common) if common else None,mean_R1_minus_Top1_common_pp=math.fsum(d)/len(d) if d else None,positive=sum(x>1e-6 for x in d),negative=sum(x< -1e-6 for x in d),neutral=sum(abs(x)<=1e-6 for x in d)))
 previous=read(ROOT/'artifacts/stvg_reference_recovery_v1/Reference_swap/POSTSEAL_REFERENCE_QUALITY.json')['summary']
 for r in summary:
  old=next(x for x in previous if x['dataset']==r['dataset']);assert r['GT_known']==old['observed_count'] and abs(r['Top1_all_known_mean']-old['dino_iou'])<1e-12
 put(BASE/'POSTSEAL_REFERENCE_QUALITY.json',dict(status='complete',count=32,rows=rows,summary=summary,GT_read=True,scope='descriptive sparse observed-reference IoU; abstention and unannotated positions remain separate; not identity/role labels or full tube quality',no_GT_reference_selection_or_fitting=True))
if __name__=='__main__':main()
