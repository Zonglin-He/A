"""Aggregate sealed HC results; no new model computation."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__import__('os').environ['STVG_PROJECT_ROOT']).resolve(); T=ROOT/'outputs/layout_reward_dev'; S=T/'scope_hc2'; O=ROOT/'results/layout_reward';O.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
x=read(S/'metrics.json'); rows=x['rows'];assert len(rows)==128 and x['queries']==128
assert read(S/'completion.json')['metrics_sha256']==sha(S/'metrics.json')
rename={'WithoutOPD':'ERefStrong','GeometryOPD':'GeometryOPD','Full':'LayoutOPD','LayoutDirect':'LayoutDirect','Source':'SourceStrong'}
means=x['datasets']['hc2']['query_main_table'];rng=np.random.default_rng(20261021);idx=rng.integers(0,128,(10000,128));pairs={}
for a,b in [('GeometryOPD','WithoutOPD'),('Full','WithoutOPD'),('LayoutDirect','WithoutOPD'),('Full','GeometryOPD'),('Full','LayoutDirect'),('LayoutDirect','GeometryOPD')]:
 fields={}
 for k in ['v','unobserved_actual_sIoU']:
  d=np.asarray([r['values'][a][k]-r['values'][b][k]for r in rows])*100
  fields[k]=dict(mean_pp=float(d.mean()),CI95_pp=np.quantile(d[idx].mean(1),[.025,.975]).tolist(),improved=int((d>1e-8).sum()),declined=int((d< -1e-8).sum()),unchanged=int((abs(d)<=1e-8).sum()),declined_over5pp=int((d< -5).sum()),positive_mean_contribution_pp=float(d[d>0].sum()/128),negative_mean_contribution_pp=float(d[d<0].sum()/128))
 pairs[rename[a]+'-'+rename[b]]=fields
anon=[];private=[]
for n,r in enumerate(rows,1):
 a=dict(query='H%03d'%n,values={rename[k]:v for k,v in r['values'].items()});anon.append(a);private.append(dict(public_id=a['query'],private_id=r['id'],values=a['values']))
receipts=[read(T/'predictions'/(r['id']+'.json'))for r in rows]
costs={arm:dict(mean_adaptation_seconds=float(np.mean([r['costs'][arm]['seconds']for r in receipts])),mean_actual_updates=float(np.mean([r['costs'][arm]['updates']for r in receipts])))for arm in ['LayoutOPD','LayoutDirect']}
out=dict(status='complete_HC128_exposed_development',direction='VidSTG_to_HC2',queries=128,parents=128,seed=41,all_arms_same_Strong_interval=True,old_geometry_reused_after_exact_source_ERef_positions_checks=True,means={rename[k]:v for k,v in means.items()},paired=pairs,queries_anonymous=anon,costs=costs,proof=dict(seal_sha256=sha(S/'sealed.json'),metrics_sha256=sha(S/'metrics.json'),inference_fingerprint=read(S/'scope.json')['inference_fingerprint'],official_dense_max_absolute_error=x['official_dense_max_absolute_error']),config=x['config'])
(O/'hc128.json').write_text(json.dumps(out,indent=2)+'\n');(S/'private_case_mapping.json').write_text(json.dumps(private,indent=2)+'\n')
lines=['# HC128 layout reward comparison','','Completed exposed development panel: VidSTG to HC2, 128 parent videos, one query each, seed 41. All arms use the identical saved Strong interval. Geometry OPD reuses sealed predictions after exact Source, ERef and observation-position checks. No per-query method or stopping-step selection.','','| Method | Full vIoU (%) | Unobserved-node sIoU (%) | Observed-node sIoU (%) |','|---|---:|---:|---:|']
for a in ['WithoutOPD','GeometryOPD','Full','LayoutDirect']:
 z=means[a];lines.append('| '+rename[a]+' | %.4f | %.4f | %.4f |'%(z['m_vIoU']*100,z['m_sIoU_unobserved_actual']*100,z['m_sIoU_observed_actual']*100))
lines+=['','Unobserved sIoU uses actual sampled frames with GT after excluding all eight expert observation positions, independently of the predicted temporal interval. It does not treat interpolated frames as observations.','','| Comparison | vIoU delta pp [95% CI] | Improved / declined / unchanged | vIoU drops >5 pp | Unobserved sIoU delta pp [95% CI] | Unobserved drops >5 pp |','|---|---:|---:|---:|---:|---:|']
for name,z in pairs.items():
 v=z['v'];s=z['unobserved_actual_sIoU'];lines.append('| %s | %+.4f [%.4f, %.4f] | %d / %d / %d | %d | %+.4f [%.4f, %.4f] | %d |'%(name,v['mean_pp'],*v['CI95_pp'],v['improved'],v['declined'],v['unchanged'],v['declined_over5pp'],s['mean_pp'],*s['CI95_pp'],s['declined_over5pp']))
lines+=['','Intervals use 10,000 paired parent-video bootstrap draws. This is an exposed development comparison, not independent confirmation. All 128 positive and negative anonymous query results are retained in hc128.json.','','## Declining queries','','All declines are included in the accompanying JSON. The following table shows the ten largest full-vIoU declines for each adaptation relative to ERef+Strong; values are percentage points.','','| Query | Geometry OPD | Layout OPD | Layout Direct |','|---|---:|---:|---:|']
worst=set()
for a in ['GeometryOPD','LayoutOPD','LayoutDirect']:
 ordered=sorted(anon,key=lambda z:z['values'][a]['v']-z['values']['ERefStrong']['v']);worst.update(z['query']for z in ordered[:10]if z['values'][a]['v']<z['values']['ERefStrong']['v'])
for z in anon:
 if z['query']in worst:lines.append('| %s | %+.4f | %+.4f | %+.4f |'%(z['query'],*[(z['values'][a]['v']-z['values']['ERefStrong']['v'])*100 for a in ['GeometryOPD','LayoutOPD','LayoutDirect']]))
lines+=['','## Execution and limits','','Layout temperature 0.1 was prespecified; HC learning rate 0.014 and 40 updates were retained. No temperature, learning-rate, step or feature-layer search was performed. Direct differentiates the same layout reward through ROI coordinates. Only the per-query temporary content state changes; network weights and feature maps remain frozen.','','Adaptation time only: Layout OPD %.4f seconds/query, Layout Direct %.4f seconds/query. These are observed trial step costs, not end-to-end standalone deployment latency; Source and ReferDINO feature export were shared and geometry predictions reused.'%(costs['LayoutOPD']['mean_adaptation_seconds'],costs['LayoutDirect']['mean_adaptation_seconds']),'','The original formal VidSTG evaluation remains paused at 8405/10303. No Vid128 layout experiment was run. Old methods/results are preserved.']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(means=out['means'],paired=pairs,costs=costs),indent=2))
