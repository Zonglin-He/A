"""Independent R13 sealed scalar reductions. No GT/index/IoU/model API."""
from pathlib import Path
import argparse,collections,hashlib,json,math
import numpy as np
B=Path(__file__).resolve().parents[3]
ARMS=('source','A_full_consensus','A_full_order0','A_noun_consensus','A_deranged_consensus','B_inequality_full','B_inequality_noRGB','B_inequality_half','B_old_surface_equality')
PAIRS=[(x,'source') for x in ARMS[1:]]+[('A_full_consensus','A_full_order0'),('A_full_consensus','A_noun_consensus'),('A_full_consensus','A_deranged_consensus'),('B_inequality_full','B_inequality_noRGB'),('B_inequality_full','B_inequality_half'),('B_inequality_full','B_old_surface_equality')]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def avg(v):return math.fsum(map(float,v))/len(v) if len(v) else None
def scene(rr,v):
 d=collections.defaultdict(list)
 for r,x in zip(rr,v):d[r['platform'],r['scene']].append(float(x))
 means=np.array([avg(d[k]) for k in sorted(d)],float)
 ci=np.quantile(np.random.default_rng(1201).choice(means,(2000,len(means))).mean(1),[.025,.975]).tolist() if len(means)>1 else None
 return {'n':len(v),'scene_clusters':len(d),'micro_mean':avg(v),'equal_scene_mean':avg(means),'equal_scene_descriptive_CI95':ci,'bootstrap_seed':1201,'bootstrap_draws':2000,'few_scene_not_significance_or_population_proof':True}
def signed(v,tol):
 p=[x for x in v if x>tol];n=[x for x in v if x< -tol]
 return {'n':len(v),'mean':avg(v),'sum':math.fsum(v),'positive_count':len(p),'negative_count':len(n),'ties_within_tol':sum(abs(x)<=tol for x in v),'positive_sum':math.fsum(p),'negative_sum':math.fsum(n),'gain_sum_exceeds_abs_harm_sum':math.fsum(p)>abs(math.fsum(n)),'min':min(v,default=None),'max':max(v,default=None),'tolerance':tol}
def main(a):
 s=read(a.scored/'SEALED.json')
 for name,h in s['files'].items():p=(a.scored/name).resolve();assert p.is_relative_to(a.scored.resolve()) and sha(p)==h
 result=read(a.scored/'RESULTS_PRIVATE.json');public=read(a.scored/'SCALAR_SUMMARY.json');assert public=={k:v for k,v in result.items() if not k.endswith('_PRIVATE')}
 lock=read(a.scored/'SCORER_LOCK.json');assert lock['prediction_seal_sha256']==s['prediction_seal_sha256'] and lock['release_sha256']==s['release_sha256'];assert s['scorer_sha256']==sha(Path(__file__).with_name('score_direct.py'))
 for rel,h in lock['geometry_dependency_sha256'].items():assert sha(B/rel)==h
 rows=result['rows_PRIVATE'];assert len(rows)==116 and len({r['id'] for r in rows})==116 and result['case_role_counts']=={'new_content108':108,'regression8':8}
 checks=0;maxdiff=0.;failures=[]
 def eq(actual,expected,path):
  nonlocal checks,maxdiff
  if isinstance(expected,dict):
   checks+=1
   if not isinstance(actual,dict) or set(actual)!=set(expected):failures.append(path+' keys');return
   for k,v in expected.items():eq(actual[k],v,path+'/'+k)
  elif isinstance(expected,list):
   checks+=1
   if not isinstance(actual,list) or len(actual)!=len(expected):failures.append(path+' length');return
   for i,v in enumerate(expected):eq(actual[i],v,path+'/'+str(i))
  elif expected is None or isinstance(expected,(str,bool,int)):
   checks+=1
   if actual!=expected:failures.append(path+' exact')
  else:
   checks+=1
   if not isinstance(actual,(int,float)) or not math.isfinite(actual) or abs(actual-expected)>1e-11:failures.append(path+' numeric')
   else:maxdiff=max(maxdiff,abs(actual-expected))
 for r in rows:
  assert tuple(r['arms'])==ARMS
  for name,x in r['arms'].items():
   assert math.isfinite(x['iou']) and 0<=x['iou']<=1
   eq(x['Acc25'],x['iou']>.25,'row/strict25');eq(x['Acc50'],x['iou']>.5,'row/strict50');eq(x['actual_changed'],any(x['actual_changed_axes']),'row/changed')
   e=x['coordinate_errors']['abs_error6_m'];assert len(e)==6 and all(math.isfinite(t) and t>=0 for t in e)
   eq(x['coordinate_errors']['center_L2_m'],math.sqrt(math.fsum(t*t for t in e[:3])),'row/centerL2');eq(x['coordinate_errors']['size_L2_m'],math.sqrt(math.fsum(t*t for t in e[3:])),'row/sizeL2')
   eq(x['query_changed'],x['query_index']!=r['q_source'],'row/query_changed');assert x['center_shift_L2_m']>=0 and x['size_shift_L2_m']>=0
  o=r['offline_source_pool_oracle'];src=r['arms']['source']['iou'];best=o['max_iou_all256'];eq(o['source_q0_iou'],src,'row/oracle_source');eq(o['source_q0_index'],r['q_source'],'row/oracle_q');assert 0<=src<=best<=1 and o['candidate_count']==256
  eq(o['all256_Acc25'],best>.25,'oracle/25');eq(o['all256_Acc50'],best>.5,'oracle/50');eq(o['rank_recoverable_25'],src<=.25<best,'oracle/recover25');eq(o['rank_recoverable_50'],src<=.5<best,'oracle/recover50')
  previous=-1.
  for k in (1,5,10,256):
   z=o['score_retrieval_'+str(k)];v=z['best_iou_GTassisted'];assert previous<=v<=best;eq(z['Acc25'],v>.25,'oracle/k25');eq(z['Acc50'],v>.5,'oracle/k50');eq(z['native_GPU_rank_claimed'],False,'oracle/notNative');previous=v
  eq(o['score_retrieval_1']['best_iou_GTassisted'],src,'oracle/top1');eq(previous,best,'oracle/top256')
  for cutoff in ('25','50'):
   lo,hi=o['first_threshold_candidate_source_score_rank_interval_'+cutoff]
   if lo is None:assert hi is None and not o['all256_Acc'+cutoff]
   else:assert isinstance(lo,int) and isinstance(hi,int) and 1<=lo<=hi<=256 and o['all256_Acc'+cutoff]
  for name in ARMS[1:5]:assert r['arms'][name]['iou']<=best+1e-12
 summaries=[];contrasts=[];oracles=[];tails=[]
 for role in ('new_content108','regression8'):
  for platform in ('drone','quad','all_platforms'):
   rr=[r for r in rows if r['case_role']==role and (platform=='all_platforms' or r['platform']==platform)]
   if not rr:continue
   for name in ARMS:
    aa=[r['arms'][name] for r in rr];iou=[x['iou'] for x in aa];ce=[x['coordinate_errors']['center_L2_m'] for x in aa];se=[x['coordinate_errors']['size_L2_m'] for x in aa];ae=np.array([x['coordinate_errors']['abs_error6_m'] for x in aa])
    summaries.append({'case_role':role,'platform':platform,'arm':name,'population':'all_cases_in_cohort','n':len(rr),'scene_clusters':len({(r['platform'],r['scene']) for r in rr}),'mean_iou':avg(iou),'equal_scene_mean_iou':scene(rr,iou)['equal_scene_mean'],'Acc25':avg([float(v>.25) for v in iou]),'Acc50':avg([float(v>.5) for v in iou]),'equal_scene_Acc25':scene(rr,[float(v>.25) for v in iou])['equal_scene_mean'],'equal_scene_Acc50':scene(rr,[float(v>.5) for v in iou])['equal_scene_mean'],'center_L2_m_mean':avg(ce),'size_L2_m_mean':avg(se),'equal_scene_center_L2_m_mean':scene(rr,ce)['equal_scene_mean'],'equal_scene_size_L2_m_mean':scene(rr,se)['equal_scene_mean'],'abs_error6_m_mean':[avg(ae[:,j]) for j in range(6)],'equal_scene_abs_error6_m_mean':[scene(rr,ae[:,j])['equal_scene_mean'] for j in range(6)],'actual_changed_n':sum(x['actual_changed'] for x in aa),'query_changed_n':sum(x['query_changed'] for x in aa),'actual_changed_axis_counts':[sum(x['actual_changed_axes'][j] for x in aa) for j in range(6)],'status_histogram':dict(collections.Counter(x['status'] for x in aa))})
   for arm,ref in PAIRS:
    delta=[r['arms'][arm]['iou']-r['arms'][ref]['iou'] for r in rr];ce=[r['arms'][ref]['coordinate_errors']['center_L2_m']-r['arms'][arm]['coordinate_errors']['center_L2_m'] for r in rr];se=[r['arms'][ref]['coordinate_errors']['size_L2_m']-r['arms'][arm]['coordinate_errors']['size_L2_m'] for r in rr];ae=np.array([np.array(r['arms'][ref]['coordinate_errors']['abs_error6_m'])-r['arms'][arm]['coordinate_errors']['abs_error6_m'] for r in rr])
    contrasts.append({'case_role':role,'platform':platform,'arm':arm,'reference':ref,'n':len(rr),'IoU_gain':signed(delta,1e-8),'center_error_reduction_m':signed(ce,1e-6),'size_error_reduction_m':signed(se,1e-6),'abs6_error_reduction_m':[signed(ae[:,j].tolist(),1e-6) for j in range(6)],'IoU_scene':scene(rr,delta),'center_scene':scene(rr,ce),'size_scene':scene(rr,se),'positive_means_improvement':True})
    ordered=sorted(zip(rr,delta),key=lambda x:(x[1],x[0]['id']));tails.append({'case_role':role,'platform':platform,'arm':arm,'reference':ref,'most_negative_PRIVATE':[{'id':r['id'],'delta_iou':v} for r,v in ordered[:5]],'most_positive_PRIVATE':[{'id':r['id'],'delta_iou':v} for r,v in ordered[-5:][::-1]],'selection':'fixed smallest/largest signed delta, ID ties; may include zero if no positive/negative'})
   oo=[r['offline_source_pool_oracle'] for r in rr];oracles.append({'case_role':role,'platform':platform,'n':len(rr),'max_all256_iou_mean':avg([o['max_iou_all256'] for o in oo]),'equal_scene_max_all256_iou_mean':scene(rr,[o['max_iou_all256'] for o in oo])['equal_scene_mean'],'all256_Acc25':avg([float(o['all256_Acc25']) for o in oo]),'all256_Acc50':avg([float(o['all256_Acc50']) for o in oo]),'equal_scene_all256_Acc25':scene(rr,[float(o['all256_Acc25']) for o in oo])['equal_scene_mean'],'equal_scene_all256_Acc50':scene(rr,[float(o['all256_Acc50']) for o in oo])['equal_scene_mean'],'rank_recoverable25_n':sum(o['rank_recoverable_25'] for o in oo),'rank_recoverable50_n':sum(o['rank_recoverable_50'] for o in oo),'CPU_stable_score_retrieval_diagnostic':{str(k):{'best_iou_within_k_mean_GTassisted':avg([o['score_retrieval_'+str(k)]['best_iou_GTassisted'] for o in oo]),'Acc25':avg([float(o['score_retrieval_'+str(k)]['Acc25']) for o in oo]),'Acc50':avg([float(o['score_retrieval_'+str(k)]['Acc50']) for o in oo]),'native_GPU_rank_claimed':False} for k in (1,5,10,256)},'first_qualified_source_rank25_defined_n':sum(o['first_threshold_candidate_source_score_rank_interval_25'][0] is not None for o in oo),'first_qualified_source_rank50_defined_n':sum(o['first_threshold_candidate_source_score_rank_interval_50'][0] is not None for o in oo),'scope':'offline GT-assisted source-pool coverage; neither a method nor instance-identity labels'})
 eq(result['summary'],summaries,'summary');eq(result['paired_contrasts'],contrasts,'contrasts');eq(result['offline_source_pool_oracle'],oracles,'oracles');eq(result['signed_tails_PRIVATE'],tails,'tails')
 output={'state':'PASS_R13_INDEPENDENT_SEALED_SCALAR_REDUCTION_AUDIT' if not failures else 'FAIL_R13_SCALAR_REDUCTION_AUDIT','scored_seal_sha256':sha(a.scored/'SEALED.json'),'results_sha256':sha(a.scored/'RESULTS_PRIVATE.json'),'scalar_summary_sha256':sha(a.scored/'SCALAR_SUMMARY.json'),'prediction_seal_sha256':s['prediction_seal_sha256'],'release_sha256':s['release_sha256'],'scorer_sha256':s['scorer_sha256'],'auditor_sha256':sha(Path(__file__)),'n':116,'summary_rows':len(summaries),'paired_contrasts':len(contrasts),'oracle_summary_rows':len(oracles),'tail_groups':len(tails),'checks':checks,'max_numeric_difference':maxdiff,'all_source_pool_oracle_internal_checks_passed':True,'new108_regression8_never_pooled':True,'CI_scope':'Platform-namespaced scene clusters; Drone newcohort1scene CI=None, others descriptive only.','historical_exposure_scope':'R7/R8 alreadyforward/scoredall5912; currentnew108 only relative914recentcontent exclusion.','no_new_GT_index_payload_task_IoU_model_GPU_or_methodchanges':True,'failures':failures}
 assert not a.output.exists();a.output.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps({'state':output['state'],'checks':checks,'maxdiff':maxdiff,'failures':failures[:10],'sha256':sha(a.output)}))
 if failures:raise SystemExit(1)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--scored',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
