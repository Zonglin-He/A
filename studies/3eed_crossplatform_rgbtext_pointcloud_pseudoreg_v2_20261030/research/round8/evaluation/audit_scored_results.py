"""Independent reduction audit on sealed R8 score rows; no GT/meta/model access.
Geometry independence is the scorer's general3D-vs-batch comparison, not a claim
that this reduction auditor recalculates every source256 candidate in general3D.
"""
from pathlib import Path
import argparse,collections,json,math,time
import numpy as np
import score_crossdomain as C

def mean(values):return math.fsum(values)/len(values) if values else None
def near(x,y):return x is None and y is None or x is not None and y is not None and math.isclose(x,y,rel_tol=0,abs_tol=1e-12)
def eligible(r,mode,arm,pop):
 if pop=='all':return True
 if pop=='q_association_changed':return r['G'][mode]['q_changed']
 if pop=='accepted_any_axis':return r['G'][mode]['accepted_axis']
 if pop=='actual_pseudo_nonzero':return r['G'][mode]['actual_float32_pseudo_delta_nonzero']
 if pop=='selected_changed_vs_q1':return r['arms'][arm]['selected_changed_vs_q1_anchor']
 if pop=='common_eligible_all_modes':return all(r['G'][m]['accepted_axis'] for m in C.MODES)
 raise ValueError(pop)

def review(result):
 rows=result['rows_PRIVATE'];assert len(rows)==result['n'];comparisons=0;maxerr=0.;negative=collections.Counter()
 for r in rows:
  assert set(r['arms'])==set(C.ARMS) and r['arms']['source']['query_index']==r['q_source']
  for mi,mode in enumerate(C.MODES):
   q1=r['q_association'][mi];assert r['G'][mode]['q_changed']==(q1!=r['q_source']);assoc=r['arms'][mode+'__association_only']['iou'];direct=r['arms'][mode+'__direct_G']['iou']
   for kind in C.KINDS:
    name=mode+'__'+kind;a=r['arms'][name];assert a['query_index']==q1
    for key,expected in [('delta_vs_q0',a['iou']-r['source_q0_iou']),('delta_vs_association',a['iou']-assoc),('delta_vs_directG',a['iou']-direct)]:assert a[key]==expected;negative[mode+'|'+kind+'|'+key]+=int(expected<0)
 for s in result['summary']:
  pp=[r for r in rows if r['platform']==s['platform'] and eligible(r,s['mode'],s['arm'],s['population'])];vals=[r['arms'][s['arm']]['iou'] for r in pp];assert len(pp)==s['n'];checks={'mean_iou':mean(vals),'Acc25':sum(v>.25 for v in vals)/len(vals) if vals else None,'Acc50':sum(v>.5 for v in vals)/len(vals) if vals else None};groups=collections.defaultdict(list)
  for r in pp:groups[r['scene']].append(r['arms'][s['arm']]['iou'])
  checks['equal_scene_mean_iou']=mean([mean(v) for v in groups.values()]);assert len(groups)==s['scene_clusters']
  for key,val in checks.items():assert near(val,s[key]),(key,val,s[key]);maxerr=max(maxerr,abs(val-s[key]) if val is not None else 0);comparisons+=1
  for key,refarm in [('delta_vs_q0','source'),('delta_vs_association',s['mode']+'__association_only'),('delta_vs_directG',s['mode']+'__direct_G')]:
   ds=[r['arms'][s['arm']]['iou']-r['arms'][refarm]['iou'] for r in pp];by=collections.defaultdict(list)
   for r,d in zip(pp,ds):by[r['scene']].append(d)
   signed=s['paired'][key];assert signed['n']==len(ds) and signed['scene_clusters']==len(by);assert near(signed['micro_delta'],mean(ds)) and near(signed['equal_scene_delta'],mean([mean(v) for v in by.values()]));comparisons+=2
 assert result['strict_threshold_disagreements']==0 and result['max_selected_3D_vs_batch_abs_error']<1e-8
 for cover in result['candidate_and_group_coverage']:
  pp=[r for r in rows if r['platform']==cover['platform']];assert cover['n']==len(pp)
  for k,v in cover['source_pool_topk'].items():
   assert v['n']==len(pp);assert near(v['mean_best_iou_GTassisted_if_k_gt1'],mean([r['source_coverage'][k]['best_iou'] for r in pp]));assert v['Acc25_successes']==sum(r['source_coverage'][k]['acc25'] for r in pp) and v['Acc50_successes']==sum(r['source_coverage'][k]['acc50'] for r in pp);comparisons+=3
 return {'aggregate_cells_checked':comparisons,'max_aggregate_float_error':maxerr,'signed_negative_counts_by_mode_arm_contrast':dict(negative),'all19_fixed_query_and_delta_arithmetic_verified':True,'source256_GTassistedcoverage_aggregation_verified':True,'geometry_scope':'Selected independent3D-vs-batch error/threshold receipts checked; no new rawGT or fullpool3D scoring.'}

def main(a):
 assert C.sha(a.results)==a.expected_results_sha256;seal=C.read(a.results.parent/'SEALED.json');assert C.sha(a.results)==seal['files'][a.results.name];result=C.read(a.results);checked=review(result);record={'state':'independent_R8_score_reduction_audit_pass','auditor_sha256':C.sha(Path(__file__)),'results_sha256':a.expected_results_sha256,'source_scorer_sha256':C.sha(Path(C.__file__)),'n':result['n'],**checked,'no_model_GPU_targetmetadata_GT_or_new_prediction_scoring':True};assert not a.output.exists();C.dump(a.output,record);print(json.dumps({k:v for k,v in record.items() if k!='signed_negative_counts_by_mode_arm_contrast'}),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,required=True);ap.add_argument('--expected-results-sha256',required=True);ap.add_argument('--output',type=Path,required=True);main(ap.parse_args())
