"""Independent R10 sealed scalar reductions, no newGT/IoU/model calls."""
from pathlib import Path
import argparse,collections,hashlib,json,math
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def mean(x):return math.fsum(x)/len(x) if x else None
def near(x,y):
 if x is None or y is None:assert x is None and y is None,(x,y)
 else:assert np.allclose(x,y,atol=1e-12,rtol=0),(x,y)
def eligible(r,pop,arm=None):
 a=r['arms'];f=a['full_rgb'];n=a['no_rgb_coordinate'];p=a['rgb_gated_pointmask_ablation']
 if pop=='all':return True
 if pop=='coordinate_pair_union_changed':return f['actual_changed'] or n['actual_changed']
 if pop=='coordinate_pair_common_eligible':return f['accepted_any'] and n['accepted_any']
 if pop=='pointmask_pair_union_changed':return f['actual_changed'] or p['actual_changed']
 if pop=='pointmask_pair_common_eligible':return f['accepted_any'] and p['accepted_any']
 if pop=='own_changed_descriptive':return a[arm]['actual_changed']
 raise ValueError(pop)
def pairscene(rows,v):
 g=collections.defaultdict(list)
 for r,x in zip(rows,v):g[r['scene']].append(float(x))
 gm=np.array([mean(g[k]) for k in sorted(g)]);ci=np.quantile(np.random.default_rng(1001).choice(gm,(2000,len(gm))).mean(1),[.025,.975]).tolist() if len(gm)>1 else None
 return len(g),mean(gm.tolist()),ci

def main(a):
 s=read(a.scored/'SEALED.json');assert sha(a.predictions/'SEALED.json')==s['prediction_seal_sha256']
 for root,seal in [(a.scored,s),(a.predictions,read(a.predictions/'SEALED.json'))]:
  for rel,h in seal['files'].items():assert sha(root/rel)==h
 d=read(a.scored/'RESULTS_PRIVATE.json');rows=d['rows_PRIVATE'];assert len(rows)==d['n'];coordinate_fields=0
 for r in rows:
  for arm,v in r['arms'].items():
   e=np.array(v['coordinate_errors']['abs_error6_m']);near(v['coordinate_errors']['center_L2_m'],float(np.linalg.norm(e[:3])));near(v['coordinate_errors']['size_L2_m'],float(np.linalg.norm(e[3:])));coordinate_fields+=2
   assert v['accepted_any']==any(v['accepted_axes']) and v['actual_changed']==any(v['actual_changed_axes'])
  assert r['arms']['full_rgb']['accepted_axes']==r['arms']['no_rgb_coordinate']['accepted_axes']==r['arms']['rgb_gated_pointmask_ablation']['accepted_axes']
 cells=0
 for x in d['summary']:
  pp=[r for r in rows if r['platform']==x['platform'] and eligible(r,x['population'],x['arm'])];v=[r['arms'][x['arm']] for r in pp];n=len(pp);assert x['n']==n and x['scene_clusters']==len({r['scene'] for r in pp});near(x['mean_iou'],mean([z['iou'] for z in v]));near(x['Acc25'],sum(z['iou']>.25 for z in v)/n if n else None);near(x['Acc50'],sum(z['iou']>.5 for z in v)/n if n else None)
  for field,key in [('center_L2_m_mean','center_L2_m'),('size_L2_m_mean','size_L2_m')]:near(x[field],mean([z['coordinate_errors'][key] for z in v]))
  for j in range(6):near(x['mean_abs_error_by_coordinate_m'][j],mean([z['coordinate_errors']['abs_error6_m'][j] for z in v]))
  assert x['accepted_any_n']==sum(z['accepted_any'] for z in v) and x['actual_changed_n']==sum(z['actual_changed'] for z in v) and x['status_histogram']==dict(collections.Counter(z['status'] for z in v));cells+=14
 for x in d['paired_contrasts']:
  pp=[r for r in rows if r['platform']==x['platform'] and eligible(r,x['population'])];arm=x['arm'];ref=x['reference'];assert x['n']==len(pp)
  vals={'IoU_gain':[r['arms'][arm]['iou']-r['arms'][ref]['iou'] for r in pp],'center_error_reduction_m':[r['arms'][ref]['coordinate_errors']['center_L2_m']-r['arms'][arm]['coordinate_errors']['center_L2_m'] for r in pp],'size_error_reduction_m':[r['arms'][ref]['coordinate_errors']['size_L2_m']-r['arms'][arm]['coordinate_errors']['size_L2_m'] for r in pp]}
  for key,v in vals.items():
   s=x[key];tol=1e-8 if key=='IoU_gain' else 1e-6;positive=[y for y in v if y>tol];negative=[y for y in v if y< -tol];assert s['n']==len(v) and s['positive_count']==len(positive) and s['negative_count']==len(negative) and s['ties_within_tol']==sum(abs(y)<=tol for y in v);near(s['mean'],mean(v));near(s['sum'],math.fsum(v));near(s['positive_sum'],math.fsum(positive));near(s['negative_sum'],math.fsum(negative));assert s['gain_sum_exceeds_abs_harm_sum']==(math.fsum(positive)>abs(math.fsum(negative)))
   scene=x['IoU_scene' if key=='IoU_gain' else 'center_scene' if key=='center_error_reduction_m' else 'size_scene'];ns,macro,ci=pairscene(pp,v);assert scene['n']==len(v) and scene['scene_clusters']==ns;near(scene['micro_mean'],mean(v));near(scene['equal_scene_mean'],macro);near(scene['descriptive_scene_CI95'],ci);cells+=13
 out={'state':'independent_R10_geometry4_scored_reduction_audit_pass','auditor_sha256':sha(Path(__file__)),'n':d['n'],'results_sha256':sha(a.scored/'RESULTS_PRIVATE.json'),'scored_seal_sha256':sha(a.scored/'SEALED.json'),'prediction_seal_sha256':sha(a.predictions/'SEALED.json'),'summary_and_paired_fields_checked':cells,'coordinate_norm_fields_reduced':coordinate_fields,'source_full_noRGB_pointmask_common_denominators_verified':True,'all_strict25_50_signed_counts_and_scene_bootstrap_reduced':True,'scope':['Only sealed score rows/results/prediction hash tables; no newIoU, GTmetadata, model, GPU orGcall.','Coordinate norm consistency checked from storedabs6, not an independent rawGT transform audit.','Ownchanged descriptive masks differ; pairedunion/commoneligible use identical cases for allarms.','2Drone/3Quad scene descriptive only; inheritedRGB gating maskablation is notpurePC.']};assert not a.output.exists();a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('scored','predictions','output'):p.add_argument('--'+n,type=Path,required=True)
 main(p.parse_args())
