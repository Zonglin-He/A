"""Root independent final-output metric and anchor arithmetic readback; CPU post-seal."""
import sys,json,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch,numpy as np
from scripts.stvg_temporal_caption_query_v1 import read,write,sha,catalog
from scripts.review_stvg_sparse_r1_v2 import _truths_after_seal,_metric_functions,score_tube_xyxy
B=ROOT/'artifacts/stvg_motion_guided_instance_opd_v1';torch.set_num_threads(2)
def main():
 for phase,count in [('METHOD',16),('REFERENCE_CONTROL',1)]:
  b=read(B/(phase+'_STUDENT_GLOBAL_BARRIER.json'));assert b['status']=='complete'and b['count']==count
  assert len(b['files'])==count and len({r['path']for r in b['files']})==count
  for r in b['files']:assert sha(ROOT/r['path'])==r['sha256']
 roster=read(B/'ROSTER_LOCK.json')['records'];cases={r['case_id']:catalog(r)[0]for r in roster};gt,_=_truths_after_seal(cases);off,ind,_=_metric_functions();z=read(B/'ROOT_POSTSEAL_REPORT.json');n=0;outputs={}
 for row in z['metric_rows']:
  cid=row['case_id'];key=(row['phase'],cid)
  if key not in outputs:outputs[key]=torch.load(B/row['phase']/'student'/f'{cid}.pt',map_location='cpu',weights_only=False)
  o=outputs[key];a=o['arms'][row['arm']];c=cases[cid];v=score_tube_xyxy(a['boxes'],o['frame_ids'],c['width'],c['height'],a['interval'],gt[cid]['truth'],gt[cid]['span'],c['dataset'],official=off,independent=ind)
  for k,val in v['metrics'].items():assert math.isclose(val,row['metrics'][k],rel_tol=0,abs_tol=1e-13),(cid,row['arm'],k)
  n+=1
 def iou(a,b):
  a=np.asarray(a);b=np.asarray(b);s=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]));inter=s.prod();return float(inter/((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter))
 inputs={r['case_id']:r for r in read(B/'CANDIDATE_INPUT_BARRIER.json')['records']};count=0;slots=0
 for row in z['candidate_present_anchor_rows']:
  assert row['phase']=='method';cid=row['case_id'];c=cases[cid];cand=next(a for a in inputs[cid]['candidates']if a['candidate_id']==row['candidate_id']);truth=gt[cid]['truth'];known=present=0;values=[]
  for k,a in enumerate(row['per_original4_anchor']):
   f=inputs[cid]['physical_frame_ids'][k];box=cand['boxes_xyxy'][k];hasgt=f in truth;known+=hasgt;assert (a['GT_support']=='known')==hasgt
   assert (a['candidate_support']=='present_valid')==(box is not None)
   if box is not None and hasgt:
    val=iou(box,np.array(truth[f])/[c['width'],c['height'],c['width'],c['height']]);assert abs(val-a['candidate_IoU'])<1e-13;values.append(val);present+=1
   else:assert a['candidate_IoU']is None
   slots+=1
  assert known==row['GT_known_original4_anchor_count']and present==row['candidate_present_and_GT_known_anchor_count'];expect=sum(values)/known if known else None;assert expect==row['common_denominator_original4_GT_known_missing_as_zero_candidate_IoU'];count+=1
 control=read(B/'REFERENCE_CONTROL_REFERENCE_LOCK.json')['records'][0]['references'];cr=z['reference_control_locked_anchor_rows'];assert len(cr)==3
 for r in cr:
  role=r.get('reference_role',r.get('role'));assert role in control
  anchors=control[role]['anchors'];c=cases['vidstg_002'];truth=gt['vidstg_002']['truth'];vals=[]
  for a in anchors:
   f=a['physical_frame_id']
   if f in truth:
    b=np.array(a['box_cxcywh']);vals.append(iou(np.r_[b[:2]-b[2:]/2,b[:2]+b[2:]/2],np.array(truth[f])/[c['width'],c['height'],c['width'],c['height']]))
  assert len(vals)==1
  target=r['present_anchor_GT_IoU_mean_known_only'];assert abs(vals[0]-target)<1e-13
 write(B/'ROOT_INDEPENDENT_METRIC_READBACK.json',dict(status='pass',metric_rows=n,method_candidate_rows=count,method_candidate_slots=slots,V03_locked_references=3,explicit_Missing_separate_from_GT_unknown=True,GT_read=True,new_GPU_or_model_calls=0,scorer_sha256=sha(ROOT/'scripts/score_stvg_motion_guided_postseal_v1.py'),report_sha256=sha(B/'ROOT_POSTSEAL_REPORT.json'),reader_sha256=sha(Path(__file__)),CURRENT_sha256=sha(ROOT/'methods/CURRENT_METHOD.json')))
 print('root metric and anchor readback pass',n,count,slots)
if __name__=='__main__':main()
