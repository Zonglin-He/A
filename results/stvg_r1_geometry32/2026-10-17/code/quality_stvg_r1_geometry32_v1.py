"""Four reference sources and Source geometry at identical post-seal planned anchors."""
from pathlib import Path
import math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_r1_geometry32_v2 import BASE,PAPER,read,sha,put
def main():
 import numpy as np
 from scripts.stvg_soft_ep_coordinate_common_v1 import load_case
 from scripts import review_stvg_sparse_r1_v2 as truth,score_stvg_native_instance_nativeclock005 as metric
 assert read(BASE/'CPU_COMPLETION.json')['status']=='complete';rr=read(BASE/'ROSTER_LOCK.json')['records'];cases={r['probe_alias']:load_case(r)[0] for r in rr};targets,_=truth._truths_after_seal(cases);maps={a:{r['case_id']:{x['position']:x for x in r['anchors']} for r in read(BASE/(a+'_REFERENCE_LOCK.json'))['records']} for a in ['DIRECT','HEADS','DINO']};maps['Current_WF']={r['case_id']:{x['position']:x for x in r['anchors']} for r in read(PAPER/'arms/B/trials/G32/REFERENCE_LOCK.json')['records']};rows=[];old={(r['case_id'],r['slot']):r for r in read(BASE/'POSTSEAL_RESULTS.json')['reference_quality']}
 def iou(a,b):
  v=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod();return float(v/max(1e-12,np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-v))
 for j in read(BASE/'INPUT_JOB_LOCK.json')['records']:
  cid=j['case_id'];case=cases[cid];gt=metric._truth_map(targets[cid]);fid,pos=j['physical_frame_id'],j['position'];scale=np.array([case['input']['width'],case['input']['height']]*2);row=dict(case_id=cid,dataset=j['dataset'],slot=j['slot'],GT_known=fid in gt,arms={})
  for a in ['Source','Current_WF','DIRECT','HEADS','DINO']:
   ref=maps[a][cid].get(pos) if a!='Source' else None;available=a=='Source' or ref is not None;value=None
   if available and fid in gt:
    if a=='Source':box=np.asarray(case['native_boxes_xyxy_normalized'][pos])*scale
    else:
     z=np.asarray(ref['box_cxcywh']);box=np.r_[z[:2]-z[2:]/2,z[:2]+z[2:]/2]*scale
    value=iou(box,np.asarray(gt[fid],float))
   row['arms'][a]=dict(available=available,IoU=value,success_at_0p5=None if fid not in gt else bool(available and value>=.5))
   if a in ['DIRECT','HEADS','DINO']:assert row['arms'][a]==dict(old[cid,j['slot']]['reference'][a],success_at_0p5=None if fid not in gt else bool(available and value>=.5))
  rows.append(row)
 summary=[]
 for ds in ['hc2','vidstg']:
  for a in ['Source','Current_WF','DIRECT','HEADS','DINO']:
   rr=[r for r in rows if r['dataset']==ds];known=[r for r in rr if r['GT_known']];valid=[r['arms'][a]['IoU'] for r in known if r['arms'][a]['available']];summary.append(dict(dataset=ds,arm=a,planned=len(rr),GT_known=len(known),GT_unknown=len(rr)-len(known),available=sum(r['arms'][a]['available'] for r in rr),available_known=len(valid),failure_or_abstention_known=len(known)-len(valid),reference_IoU_mean_available_known=math.fsum(valid)/len(valid) if valid else None,success_at_0p5_all_known=sum(bool(r['arms'][a]['success_at_0p5']) for r in known)/len(known) if known else None))
 put(BASE/'POSTSEAL_REFERENCE_QUALITY.json',dict(status='complete',count=32,rows=rows,summary=summary,GT_read=True,known_invalid_counts_as_localization_failure_but_IoU_mean_is_explicitly_conditional=True,no_identity_role_labels_inferred_from_IoU=True))
if __name__=='__main__':main()
