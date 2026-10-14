"""Post-seal evaluation, after immutable label-free A/B selection."""
import sys,math,time,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_cross_frame_common_v1 import *
def main():
 import torch,numpy as np
 from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
 from scripts import score_stvg_native_instance_nativeclock005 as metric
 from scripts import review_stvg_sparse_r1_v2 as scorer
 verify();selection=read(BASE/'METHOD_SELECTION_LOCK.json');seal=read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
 if selection['GT_read'] is not False or selection['status']!='locked' or seal['status']!='complete' or seal['GT_read'] is not False:raise RuntimeError('pre-score seal and selection required')
 if seal['scorer_sha256']!=sha(__file__) or seal['method_selection_sha256']!=sha(BASE/'METHOD_SELECTION_LOCK.json'):raise RuntimeError('scorer/selection mismatch')
 for f in seal['files']:
  if sha(ROOT/f['path'])!=f['sha256']:raise RuntimeError('sealed input changed')
 roster=read(OLD/'ROSTER_LOCK.json')['records'];refs={r['case_id']:r for r in read(BASE/'REFERENCE_LOCK.json')['records']};geos={r['case_id']:r for r in read(PRIOR/'TEACHER_GEOMETRY_ONLY_READOUT.json')['records']}
 cases={};outputs={};aliases={'F':'F','T':'T','raw_iou':'S_original','T_raw_iou':'TS_original','validated_loss_weighted_iou':'S_A','T_validated_loss_weighted_iou':'TS_A','cross_frame':'S_B','T_cross_frame':'TS_B'}
 for r in roster:
  cid=r['case_id'];d=torch.load(BASE/'student'/(cid+'.pt'),map_location='cpu',weights_only=False);fast=torch.load(BASE/'temporal_fast'/(cid+'.pt'),map_location='cpu',weights_only=False);case=load_source_case(ROOT/r['source_case_path'])
  if d['case_id']!=cid or fast['case_id']!=cid or d['GT_read'] or fast['GT_read'] or d['reference_record']!=refs[cid]:raise RuntimeError('record custody mismatch')
  arms={new:copy.deepcopy(d['arms'][old])for old,new in aliases.items()};chosen='S_B'if selection['selected_recipe']=='B_cross_frame'else'S_A';iv=fast['fast_fit']['final_interval']
  arms['T_fast']=dict(boxes=arms['F']['boxes'],interval=iv);arms['TS_fast']=dict(boxes=arms[chosen]['boxes'],interval=iv)
  g=geos[cid]['raw_decoupled'];arms['T_geometry']=dict(boxes=g['boxes_xyxy_normalized'],interval=arms['T']['interval']);arms['T_fast_geometry']=dict(boxes=g['boxes_xyxy_normalized'],interval=iv)
  if len(arms)!=12 or list(case['frame_ids'])!=d['frame_ids']:raise RuntimeError('arm/grid count mismatch')
  for a in arms.values():
   if np.asarray(a['boxes']).shape!=(len(d['frame_ids']),4) or not np.isfinite(a['boxes']).all():raise RuntimeError('prediction shape/finite error')
  cases[cid]=case;outputs[cid]=(r['dataset'],arms)
 if len(cases)!=8:raise RuntimeError('all8 required')
 write(BASE/'GT_EXPOSURE.json',dict(status='authorized_after_all_predictions_and_AB_selection_sealed',GT_read=True,barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),time=time.time()))
 targets,provenance=scorer._truths_after_seal(cases);official,independent,contract=scorer._metric_functions();rows=[]
 for cid,(ds,arms)in outputs.items():
  support={a['position']for a in refs[cid]['training_anchors']}
  for name,a in arms.items():
   m,audit=metric._score_student_arm(a,cases[cid],targets[cid],support,official,independent)
   rows.append(dict(case_id=cid,dataset=ds,arm=name,metrics=m,support=audit['support'],metric_audit=audit['metric_audit']))
 summaries=[];contrasts=[];names=list(next(iter(outputs.values()))[1]);chosen='TS_B'if selection['selected_recipe']=='B_cross_frame'else'TS_A'
 for ds in ['hc2','vidstg']:
  subset=[r for r in rows if r['dataset']==ds];lookup={(r['case_id'],r['arm']):r for r in subset}
  for name in names:
   rr=[r for r in subset if r['arm']==name]
   for metricname in rr[0]['metrics']:
    values=[r['metrics'][metricname]for r in rr if r['metrics'][metricname]is not None]
    summaries.append(dict(dataset=ds,arm=name,metric=metricname,mean=math.fsum(values)/len(values)if values else None,count=len(values),unknown_count=4-len(values)))
  for left,right in [('TS_A','T'),('TS_B','T'),('TS_B','TS_A'),('T_fast','T'),('TS_fast',chosen),(chosen,'T_geometry')]:
   for k in subset[0]['metrics']:
    vals=[]
    for cid in cases:
     if (cid,left)not in lookup:continue
     a=lookup[cid,left]['metrics'][k];b=lookup[cid,right]['metrics'][k]
     if a is not None and b is not None:vals.append(a-b)
    contrasts.append(dict(dataset=ds,left=left,right=right,metric=k,mean_delta_pp=100*math.fsum(vals)/len(vals)if vals else None,count=len(vals),positive=sum(v>1e-12 for v in vals),negative=sum(v< -1e-12 for v in vals),neutral=sum(abs(v)<=1e-12 for v in vals),paired_bootstrap_95_pp=metric._paired_bootstrap(vals,ds,left+'-'+right,k)))
 write(BASE/'POSTSEAL_ROWS.json',dict(status='complete',GT_read=True,rows=rows))
 write(BASE/'POSTSEAL_SUMMARY.json',dict(status='complete',GT_read=True,historically_exposed_development=True,summary=summaries,contrasts=contrasts,selection_before_new_GT=selection['selected_recipe']))
 write(BASE/'CPU_COMPLETION.json',dict(status='pending_root_public_archive',count=8,row_count=len(rows),GT_read=True,time=time.time()))
 print('Scored',len(rows),'sealed arms')
if __name__=='__main__':main()
