"""Aggregate already computed diagnostics, no mask/model/GT reads."""
import os,json,numpy as np
from pathlib import Path
ROOT=Path(os.environ.get('STVG_PROJECT_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve();R=ROOT/'outputs/layout_reward_dev/research';x=json.loads((R/'mask_completeness.json').read_text());qs=x['queries'];out={}
for pool in ['prediction_pool','prediction_pool_distinct']:
 d=[]
 for q in qs:
  if not q['anchors']:continue
  p=np.nanmean([a['rankings'][pool]['projection_Dice']for a in q['anchors']]);g=np.nanmean([a['rankings'][pool]['geometry_reference']for a in q['anchors']])
  if np.isfinite(p)and np.isfinite(g):d.append(p-g)
 d=np.asarray(d);rng=np.random.default_rng(20261021);idx=rng.integers(0,len(d),(10000,len(d)));out[pool]=dict(query_n=len(d),projection_minus_geometry_rho=float(d.mean()),CI95=np.quantile(d[idx].mean(-1),[.025,.975]).tolist(),queries_projection_better=int((d>1e-10).sum()),queries_geometry_better=int((d < -1e-10).sum()),candidate_pool_warning='Native other object-query boxes plus fixed Teacher/ERef scales and fullframe; not the actual cached OPD32 pool; GT-generated scales excluded')
out['teacher_quality_groups']={}
for label,lo,hi in [('low',0,.3),('mid',.3,.7),('high',.7,1.00001)]:
 rows=[]
 for q in qs:
  aa=[a for a in q['anchors']if lo<=a['teacher_GT_IoU']<hi]
  if not aa:continue
  d={}
  for score in ['mask_mean','mask_coverage','pixel_Dice','projection_Dice']:
   val=[]
   for a in aa:
    teacher=next(c for c in a['candidates']if c['base']=='Teacher'and c['scale']==1);gt=next(c for c in a['candidates']if c['base']=='GT_posthoc'and c['scale']==1);val.append(float(teacher[score]>gt[score]+1e-10))
   d[score]=float(np.mean(val))
  rows.append(d)
 out['teacher_quality_groups'][label]=dict(queries=len(rows),teacher_score_higher_than_GT_reference_fraction_query_macro={k:float(np.mean([r[k]for r in rows]))for k in rows[0]})
(R/'mask_posthoc_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
