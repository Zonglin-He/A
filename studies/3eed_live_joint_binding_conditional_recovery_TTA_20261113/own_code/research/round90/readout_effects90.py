"""Saved scores only: distinguish active numerical effects from task decisions."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round90';P=R/'crossplatform/round90/prediction336_attempt001';read=lambda p:json.loads(p.read_text());tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');pairs=[('candidate_joint_frozen',a) for a in ['query_joint_frozen','no_ref_frozen','soft_pool_frozen','wrong_space_frozen']]+[('candidate_joint_tta1',a) for a in ['query_joint_tta1','no_ref_tta1','soft_pool_tta1','wrong_space_tta1','candidate_joint_bypass1','candidate_joint_physical1','candidate_joint_shuffle1']];d={p:[] for p in pairs};own=[]
for t in tasks:
 with np.load(P/'samples_PRIVATE'/(t['id_PRIVATE']+'.npz'),allow_pickle=False) as a:s=dict(zip(a['arm_names'],a['scores']))
 for p in pairs:
  x,y=s[p[0]],s[p[1]];d[p].append((float(np.max(abs(x-y))),float(np.sum(abs(x-y))),int(x.argmax()!=y.argmax()),int(abs(t['IoU'][p[0]]-t['IoU'][p[1]])>1e-10),float(np.sort(x)[-1]-np.sort(x)[-2])))
 with np.load(P/'conditional_PRIVATE'/(t['id_PRIVATE']+'_initial.npz'),allow_pickle=False) as a:
  w=a['subject_weights']
  if w.shape[1]:own.append({'row':t['row'],'platform':t['platform'],'own_weight_selected_candidate':float(w[int(s['candidate_joint_frozen'].argmax()),0]),'own_weight_all_candidates_mean':float(w[:,0].mean()),'candidate_hypotheses_not_instance_labels':True})
out=[]
for p,v in d.items():
 v=np.array(v);out.append({'arm':p[0],'control':p[1],'n':len(v),'probability_Linf_mean':float(v[:,0].mean()),'probability_Linf_max':float(v[:,0].max()),'probability_L1_mean':float(v[:,1].mean()),'top1_query_changes':int(v[:,2].sum()),'IoU_changes':int(v[:,3].sum()),'arm_top1_top2_margin_median':float(np.median(v[:,4]))})
for name,a in [('READOUT_NUMERICAL_EFFECT_ROWS',out),('SUBJECT_ROUTE_DESCRIPTIVE_ROWS',own)]: (H/(name+'.json')).write_text(json.dumps(a,indent=2)+'\n')
print(out)
