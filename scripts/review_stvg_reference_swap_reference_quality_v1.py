"""Post-seal same-observed-frame reference quality; never builds training references."""
from pathlib import Path
import sys,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
BASE=ROOT/'artifacts/stvg_reference_recovery_v1/Reference_swap'
def main():
 from scripts.measure_stvg_reference_swap_v1 import _verify_all_seals
 locked=_verify_all_seals(BASE)
 assert read(BASE/'ROOT_POSTSEAL_READBACK.json')['status']=='complete'
 from scripts.stvg_evidence_common_v1 import load_case
 from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal
 from scripts.score_stvg_native_instance_nativeclock005 import _truth_map
 import numpy as np
 cases={cid:load_case(r)[0]for cid,r in locked['roster'].items()}
 targets,_=_truths_after_seal(cases)
 def iou(a,b):
  a,b=np.asarray(a,float),np.asarray(b,float);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
  return float(inter/max(1e-12,np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-inter))
 rows=[]
 for cid,case in cases.items():
  gt=_truth_map(targets[cid]);ref=locked['references'][cid]
  for wf,di in zip(ref['common_wf_anchors'],ref['common_dino_anchors']):
   assert (wf['position'],wf['physical_frame_id'],wf['weight'])==(di['position'],di['physical_frame_id'],di['weight'])
   pos,fid=wf['position'],wf['physical_frame_id'];row=dict(case_id=cid,dataset=case['dataset'],position=pos,gt_observed=fid in gt,source_iou=None,wf_iou=None,dino_iou=None,dino_minus_wf_pp=None)
   if fid in gt:
    scale=np.array([case['input']['width'],case['input']['height']]*2,float)
    row['source_iou']=iou(np.array(case['native_boxes_xyxy_normalized'][pos])*scale,gt[fid])
    for key,a in [('wf_iou',wf),('dino_iou',di)]:
     b=np.asarray(a['box_cxcywh'],float);row[key]=iou(np.r_[b[:2]-b[2:]/2,b[:2]+b[2:]/2]*scale,gt[fid])
    row['dino_minus_wf_pp']=100*(row['dino_iou']-row['wf_iou'])
   rows.append(row)
 summary=[]
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];obs=[r for r in rr if r['gt_observed']];d=[r['dino_minus_wf_pp']for r in obs]
  summary.append(dict(dataset=ds,count=len(rr),observed_count=len(obs),unknown_count=len(rr)-len(obs),**{k:math.fsum(r[k]for r in obs)/len(obs)if obs else None for k in ['source_iou','wf_iou','dino_iou']},mean_dino_minus_wf_pp=math.fsum(d)/len(d)if d else None,positive=sum(x>1e-10 for x in d),negative=sum(x< -1e-10 for x in d),neutral=sum(abs(x)<=1e-10 for x in d)))
 assert len(rows)==38
 out=dict(status='complete',rows=rows,summary=summary,GT_read=True,CPU_only=True,bootstrap_executed=False,scope='postseal descriptive IoU only at shared actually annotated observation positions; not full-tube vIoU, no GT reference construction or oracle fitting',bindings=dict(prediction_seal_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),reference_lock_sha256=sha(BASE/'REFERENCE_LOCK.json')))
 write(BASE/'POSTSEAL_REFERENCE_QUALITY.json',out);print(summary)
if __name__=='__main__':main()
