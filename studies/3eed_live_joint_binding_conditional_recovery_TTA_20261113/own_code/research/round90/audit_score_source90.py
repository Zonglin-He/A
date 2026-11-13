"""Saved-only source scalar/provenance recomputation independent of inner optimizer."""
from pathlib import Path
import json,hashlib,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round90';P=R/'crossplatform/round90/source_aux128_attempt001';read=lambda p:json.loads(p.read_text());se=read(P/'SEALED.json');tasks=read(H/'SOURCE_AUX_ALL_TASK_ROWS.json');rows=read(R/'research/round76/input/SOURCE_VAL128_PRIVATE.json');summary=read(H/'SOURCE_AUX_TASK_SUMMARY_ROWS.json');changes=read(H/'SOURCE_AUX_TASK_CHANGE_ROWS.json');assert len(tasks)==len(rows)==128
for t,r in zip(tasks,rows):
 f=P/'samples_PRIVATE'/(r['id']+'.npz');assert hashlib.sha256(f.read_bytes()).hexdigest()==se['files'][f.relative_to(P).as_posix()];a=np.load(f,allow_pickle=False);y=np.load(R/'research/round25/experiment/source_labels_attempt002/samples_PRIVATE'/(r['id']+'.npz'),allow_pickle=False)['ious']
 for arm,q,p in zip(a['arm_names'],a['queries'],a['scores']):assert int(q)==int(p.argmax()) and t[str(arm)]==float(y[int(q)])
for s in summary:
 v=np.array([t[s['arm']] for t in tasks]);assert s['mIoU_percent']==float(v.mean()*100) and s['Acc25_percent']==float((v>.25).mean()*100) and s['Acc50_percent']==float((v>.5).mean()*100)
for c in changes:
 v=np.array([t[c['arm']]-t[c['reference']] for t in tasks]);assert c['delta_pp']==float(v.mean()*100) and c['better_n']==int((v>1e-10).sum()) and c['worse_n']==int((v< -1e-10).sum())
out={'state':'PASS_R90_SOURCE_AUXILIARY_TASK_SCALARS_AND_ALL128_DENOMINATORS','n':128,'arms':len(se['arm_names']),'task_values':128*len(se['arm_names']),'used_for_model_selection':False,'target_task_labels_read':False};(H/'SOURCE_SCORE_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
