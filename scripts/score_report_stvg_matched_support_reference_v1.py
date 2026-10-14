"""Postseal matched-reference metrics, all signed/unknown diagnostics and figures."""
import json,hashlib,sys,statistics,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
B=R/'artifacts/stvg_matched_support_reference_v1';OLD=R/'artifacts/stvg_short_event_tracklets_v1'
ROLES=('source','r1','pool_oracle','gt_oracle');read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def main():
 import torch,numpy as np
 from scripts.review_stvg_sparse_r1_v2 import _metric_functions,score_tube_xyxy,_truths_after_seal
 from scripts.stvg_sparse_r1_prompt_v1 import load_source_case
 from scripts.score_stvg_motion_guided_postseal_v1 import _per_frame_iou,_subset_delta
 from vg_tta.tastvg_oracle_event5_v1 import box_iou
 rt=read(B/'STUDENT_RUNTIME.json');assert rt['status']=='locked'
 for p,h in {**rt['pins'],**rt['protected']}.items():assert sha(R/p)==h,p
 bar=read(B/'STUDENT_GLOBAL_BARRIER.json');assert bar['status']=='complete'and bar['count']==8
 root=read(B/'ROOT_STUDENT_MATH_READBACK.json');assert root['status']=='pass'
 assert root['student_barrier_sha256']==sha(B/'STUDENT_GLOBAL_BARRIER.json')
 refs=read(B/'REFERENCE_LOCK.json');assert rt['reference_lock_sha256']==sha(B/'REFERENCE_LOCK.json')
 roster=read(B/'ROSTER_LOCK.json')['records'];cases={r['case_id']:load_source_case(R/r['source_case_path'])for r in roster};outputs={}
 for f in bar['files']:
  p=R/f['path'];assert sha(p)==f['sha256']and p.stat().st_size==f['bytes'];z=torch.load(p,map_location='cpu',weights_only=False);assert z['case_id']not in outputs;outputs[z['case_id']]=z
 assert set(cases)==set(outputs)and len(outputs)==8
 truth,gtbindings=_truths_after_seal(cases);official,independent,_=_metric_functions();rows=[];quality=[];movements=[];refmap={r['case_id']:r for r in refs['records']}
 for cid,c in cases.items():
  z=outputs[cid];fid=list(map(int,c['frame_ids']));tr=truth[cid];assert z['frame_ids']==fid
  expected={'F','T',*ROLES,*['T_'+k for k in ROLES]};assert set(z['arms'])==expected
  vals={k:score_tube_xyxy(a['boxes'],fid,c['width'],c['height'],a['interval'],tr['truth'],tr['span'],c['dataset'],official=official,independent=independent)for k,a in z['arms'].items()}
  denseF=_per_frame_iou(z['arms']['F'],fid,c['width'],c['height'],c['dataset'],tr['truth'],tr['span'])
  for arm,q in vals.items():
   baseline='T'if arm=='T'or arm.startswith('T_')else'F';role=arm.removeprefix('T_');sup={a['physical_frame_id']for a in refmap[cid]['references'].get(role,{}).get('anchors',[])}
   d=_per_frame_iou(z['arms'][arm],fid,c['width'],c['height'],c['dataset'],tr['truth'],tr['span'])
   rows.append(dict(case_id=cid,dataset=c['dataset'],arm=arm,metrics=q['metrics'],baseline=baseline,delta_vIoU_pp=100*(q['metrics']['m_vIoU']-vals[baseline]['metrics']['m_vIoU']),official_independent_max_abs_error=q['official_independent_max_abs_error'],support_count=len(sup),supervised_geometry=_subset_delta(denseF,d,sup,selection_status='same_support_GT_known'),unsupervised_geometry=_subset_delta(denseF,d,set(map(int,denseF.fids))-sup,selection_status='other_GT_known_frames')))
  for role in ROLES:
   rr=refmap[cid]['references'][role]
   for a in rr['anchors']:
    f=a['physical_frame_id'];assert f in tr['truth'];cx,cy,w,h=a['box_cxcywh'];box=np.array([cx-w/2,cy-h/2,cx+w/2,cy+h/2]);gt=np.array(tr['truth'][f],float)/[c['width'],c['height'],c['width'],c['height']]
    quality.append(dict(case_id=cid,dataset=c['dataset'],role=role,slot=a['slot'],candidate_id=a['candidate_id'],reference_IoU=float(box_iou(box[None],gt[None])[0]),offline_label_assisted=role in ('pool_oracle','gt_oracle')))
   if role in z['fits']:
    fit=z['fits'][role];before_ref=[];after_ref=[]
    for a in rr['anchors']:
     cx,cy,w,h=a['box_cxcywh'];ref=np.array([cx-w/2,cy-h/2,cx+w/2,cy+h/2]);pos=a['position']
     before_ref.append(float(box_iou(np.asarray(z['arms']['F']['boxes'])[pos:pos+1],ref[None])[0]))
     after_ref.append(float(box_iou(np.asarray(z['arms'][role]['boxes'])[pos:pos+1],ref[None])[0]))
    movements.append(dict(case_id=cid,role=role,optimizer_steps=fit['optimizer_steps'],parameter_delta_norm=float(torch.cat([(fit['final_state'][k]-v).flatten().double()for k,v in fit['initial_state'].items()]).norm()),mean_reference_IoU_before=statistics.mean(before_ref),mean_reference_IoU_after=statistics.mean(after_ref),reference_IoU_delta_pp=100*(statistics.mean(after_ref)-statistics.mean(before_ref))))
 groups=[]
 for dataset in ('hc2','vidstg'):
  for arm in ['F','T',*ROLES,*['T_'+x for x in ROLES]]:
   a=[x for x in rows if x['dataset']==dataset and x['arm']==arm];v=[x['delta_vIoU_pp']for x in a];groups.append(dict(dataset=dataset,arm=arm,count=len(a),mean_vIoU=statistics.mean(x['metrics']['m_vIoU']for x in a),mean_delta_pp=statistics.mean(v),positive=sum(x>1e-8 for x in v),negative=sum(x< -1e-8 for x in v),neutral=sum(abs(x)<=1e-8 for x in v)))
 out=dict(status='pass',scope='same-support offline reference diagnosis on eight historical sources',rows=rows,groups=groups,reference_quality=quality,fit_movement=movements,metric_count=len(rows),support_count=refs['support_count'],actual_fit_count=bar['actual_fit_count'],GT_read=True,GT_used_for_oracle_reference=True,GT_used_for_method_or_hyperparameter_selection=False,student_barrier_sha256=sha(B/'STUDENT_GLOBAL_BARRIER.json'),reference_lock_sha256=sha(B/'REFERENCE_LOCK.json'),root_math_sha256=sha(B/'ROOT_STUDENT_MATH_READBACK.json'),GT_bindings_sha256=hashlib.sha256(json.dumps(gtbindings,sort_keys=True).encode()).hexdigest(),time=time.time())
 save(B/'ROOT_POSTSEAL_REPORT.json',out)
 print(json.dumps(dict(status='pass',metrics=len(rows),groups=groups)))
if __name__=='__main__':main()
