"""Sealed 32-source temporal decomposition, no model/teacher calls or oracle fitting."""
import sys,copy,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_native_instance_common_v1 import read,write,sha
from scripts.run_stvg_weightfree_expand_v1 import BASE as E,roster,load_case,save_pt
BASE=ROOT/'artifacts/stvg_weightfree_coordinate_v1/temporal_diagnosis'
def main():
 import torch,numpy as np
 from scripts import stvg_table1_temporal_adapter_v1 as ta
 from scripts import stvg_target_teacher_student_v1 as core
 from scripts import review_stvg_sparse_r1_v2 as scorer
 from scripts import score_stvg_native_instance_nativeclock005 as metric
 torch.set_num_threads(2);cases={};outputs={};files=[];replays=0
 for rec in roster():
  cid=rec['probe_alias'];case,p=load_case(rec);t=torch.load(E/'temporal'/(cid+'.pt'),map_location='cpu',weights_only=False)['temporal'];cases[cid]=case
  intervals={'Native':case['Native_interval'],'Projection':case['Projection_interval'],'Temporal':t['final_interval']};feedback={}
  if 'temporal_fit'in t:
   f=ta.fit_query(case,case['temporal_candidates'],t['teacher_responses']);assert core._tree_equal(f,t['temporal_fit']);replays+=1
   for c,r in zip(case['temporal_candidates'],t['teacher_responses']):intervals[c['canonical_id']]=c['interval_frames'];feedback[c['canonical_id']]=r['reward']
   for step in f['path']:intervals['Adam_step'+str(step['step'])]=step['physical_interval']
  d=dict(case_id=cid,dataset=rec['dataset'],intervals=intervals,feedback=feedback,source_sha256=sha(p),GT_read=False)
  files.append(save_pt(BASE/'predictions'/(cid+'.pt'),d));outputs[cid]=d
 write(BASE/'GLOBAL_READOUT_BARRIER.json',dict(status='complete',count=32,files=files,GT_read=False,exact_temporal_replays=replays))
 truth,prov=scorer._truths_after_seal(cases);official,independent,contract=scorer._metric_functions();rows=[]
 for cid,d in outputs.items():
  for arm,iv in d['intervals'].items():
   a=dict(boxes=np.asarray(cases[cid]['native_boxes_xyxy_normalized']),interval=iv)
   m,audit=metric._score_student_arm(a,cases[cid],truth[cid],set(),official,independent)
   rows.append(dict(case_id=cid,dataset=d['dataset'],arm=arm,metrics=m,reward=d['feedback'].get(arm)))
 write(BASE/'POSTSEAL_ROWS.json',dict(status='complete',rows=rows,GT_read=True,development_diagnosis_only=True))
 out=[]
 for ds in ['hc2','vidstg']:
  rr=[x for x in rows if x['dataset']==ds];lookup={(x['case_id'],x['arm']):x for x in rr};ids=sorted({x['case_id']for x in rr})
  for l,r in [('Projection','Native'),('Temporal','Projection'),('Temporal','Native')]:
   vals=[lookup[c,l]['metrics']['m_vIoU']-lookup[c,r]['metrics']['m_vIoU'] for c in ids]
   out.append(dict(dataset=ds,left=l,right=r,mean_delta_pp=100*math.fsum(vals)/len(vals),negative=sum(x< -1e-12 for x in vals)))
 write(BASE/'ROOT_DIAGNOSIS.json',dict(status='pass',exact_replays=replays,metrics_official_independent=True,comparisons=out,GT_read=True,not_new_full_backbone_inference=True))
 print(out)
 for cid in ['HC01','HC06','HC14','HC16','V02','V11']:
  print(cid,[(r['arm'],round(r['metrics']['m_vIoU']*100,4),r['reward'])for r in rows if r['case_id']==cid])
if __name__=='__main__':main()
