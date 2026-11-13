from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round89'
read=lambda p:json.loads(p.read_text());tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');old={t['id_PRIVATE']:t for t in read(R/'research/round88/score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json')};rr=read(H/'HISTORICAL_ABC_TASK_ROWS.json');cc=read(H/'HISTORICAL_ABC_ALL_COMPARISONS.json');su=read(H/'HISTORICAL_ABC_SUMMARY_ROWS.json');text=(H/'REPORT.md').read_text();assert len(rr)==336 and len(cc)==336*29
for t,r in zip(tasks,rr):
 v=old[t['id_PRIVATE']]['IoU']['frozen_ABC'];assert r['row']==t['row'] and r['frozen_ABC_IoU']==v and r['candidate_tta1_minus_ABC_pp']==100*(t['IoU']['candidate_tta1']-v)
for row in cc:
 t=tasks[int(row['row'][3:])-1];v=old[t['id_PRIVATE']]['IoU']['frozen_ABC'];assert row['delta_pp']==100*(t['IoU'][row['arm']]-v)
for s in su:
 rows=rr if s['platform']=='balanced' else [r for r in rr if r['platform']==s['platform']];assert s['frozen_ABC_mIoU_percent']==np.mean([r['frozen_ABC_IoU'] for r in rows])*100;assert f'{s["frozen_ABC_mIoU_percent"]:.6f}' in text
out={'state':'PASS_R89_CACHED_HISTORICAL_ABC_ALL_ROWS_AND_ALL29_ARM_COMPARISONS','n':336,'comparisons':len(cc),'negative_comparisons':sum(x['delta_pp']< -1e-8 for x in cc),'new_model_calls':0,'new_GT_scores':0};(H/'CONTEXT_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
