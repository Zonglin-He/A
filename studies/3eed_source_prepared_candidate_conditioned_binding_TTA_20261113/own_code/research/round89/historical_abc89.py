"""Retain historical Frozen ABC on exactly the same rows, no new model/scoring."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round89'
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
assert read(H/'independent_native_audit_attempt001/AUDIT_RESULT.json')['state'].startswith('PASS_');old={t['id_PRIVATE']:t for t in read(R/'research/round88/score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json')};tt=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');arms=list(tt[0]['IoU']);rr=[];cc=[]
for t in tt:
 o=old[t['id_PRIVATE']];assert o['platform']==t['platform'] and o['scene_PRIVATE']==t['scene_PRIVATE'];v=o['IoU']['frozen_ABC'];rr.append({'row':t['row'],'platform':t['platform'],'frozen_ABC_IoU':v,'candidate_tta1_IoU':t['IoU']['candidate_tta1'],'candidate_tta1_minus_ABC_pp':100*(t['IoU']['candidate_tta1']-v)})
 for a in arms:cc.append({'row':t['row'],'platform':t['platform'],'arm':a,'reference':'historical_frozen_ABC','delta_pp':100*(t['IoU'][a]-v)})
su=[]
for pl in ['balanced','drone','quad']:
 rows=rr if pl=='balanced' else [r for r in rr if r['platform']==pl];v=np.array([r['frozen_ABC_IoU'] for r in rows]);dv=np.array([r['candidate_tta1_minus_ABC_pp'] for r in rows]);su.append({'platform':pl,'n':len(rows),'frozen_ABC_mIoU_percent':float(v.mean()*100),'frozen_ABC_Acc25_percent':float((v>.25).mean()*100),'frozen_ABC_Acc50_percent':float((v>.5).mean()*100),'candidate_tta1_gain_pp':float(dv.mean()),'candidate_tta1_better_n':int((dv>1e-8).sum()),'candidate_tta1_worse_n':int((dv< -1e-8).sum())})
dump(H/'HISTORICAL_ABC_TASK_ROWS.json',rr);dump(H/'HISTORICAL_ABC_ALL_COMPARISONS.json',cc);dump(H/'HISTORICAL_ABC_SUMMARY_ROWS.json',su);print(su)
