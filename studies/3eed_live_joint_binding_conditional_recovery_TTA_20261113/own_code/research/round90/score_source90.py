"""Postseal source validation task diagnostics, not model selection."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round90';B=R/'crossplatform/round90';read=lambda p:json.loads(p.read_text());assert read(H/'independent_source_aux_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_');P=B/'source_aux128_attempt001';se=read(P/'SEALED.json');rows=read(R/'research/round76/input/SOURCE_VAL128_PRIVATE.json');rr=[]
for i,r in enumerate(rows):
 a=np.load(P/'samples_PRIVATE'/(r['id']+'.npz'),allow_pickle=False);y=np.load(R/'research/round25/experiment/source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'),allow_pickle=False)['ious'];rr.append({'row':'SOURCEVAL'+str(i+1).zfill(4),**{k:float(y[int(q)]) for k,q in zip(a['arm_names'],a['queries'])}})
summary=[{'arm':a,'mIoU_percent':float(np.mean([r[a] for r in rr])*100),'Acc25_percent':float(np.mean([r[a]>.25 for r in rr])*100),'Acc50_percent':float(np.mean([r[a]>.5 for r in rr])*100)} for a in se['arm_names']];base='candidate_joint_frozen';changes=[]
for a in se['arm_names']:
 d=np.array([r[a]-r[base] for r in rr]);changes.append({'arm':a,'reference':base,'delta_pp':float(d.mean()*100),'better_n':int((d>1e-10).sum()),'worse_n':int((d< -1e-10).sum()),'unchanged_n':int((abs(d)<=1e-10).sum())})
for name,x in [('SOURCE_AUX_ALL_TASK_ROWS',rr),('SOURCE_AUX_TASK_SUMMARY_ROWS',summary),('SOURCE_AUX_TASK_CHANGE_ROWS',changes)]: (H/(name+'.json')).write_text(json.dumps(x,indent=2)+'\n')
print(summary)
