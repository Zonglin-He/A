"""Finite R15 scalar projections. No annotation/model/sensor IO or mutation."""
import math,re
STUDY='3eed_unified_fulltext_relational_G_20261031'
ARMS=('source','proposal_assoc','proposal_geo','graph_assoc','graph_geo','graph_geo_noRGB','graph_geo_relation_disabled','graph_geo_relation_flipped_diagnostic')
ROLES=('fresh64_recent_content_relative_only','R14_regression140_exposed')
SUMMARY_KEYS=('case_role','platform','arm','population','n','scene_clusters','mean_iou','equal_scene_mean_iou','Acc25','Acc50','equal_scene_Acc25','equal_scene_Acc50','center_L2_m_mean','size_L2_m_mean','equal_scene_center_L2_m_mean','equal_scene_size_L2_m_mean','abs_error6_m_mean','equal_scene_abs_error6_m_mean','actual_changed_n','query_changed_n','actual_changed_axis_counts','status_histogram')
SIGNED_KEYS=('n','mean','sum','positive_count','negative_count','ties_within_tol','positive_sum','negative_sum','gain_sum_exceeds_abs_harm_sum','min','max','tolerance')
SCENE_KEYS=('n','scene_clusters','micro_mean','equal_scene_mean','equal_scene_descriptive_CI95','bootstrap_seed','bootstrap_draws','few_scene_not_significance_or_population_proof')
ROW_FIELDS=('case_alias','case_role','platform','scene_alias','arm','iou','Acc25','Acc50','center_L2_error_m','size_L2_error_m',*[f'abs_error_{x}_m' for x in ('cx','cy','cz','sx','sy','sz')],'actual_changed','actual_changed_axis_bitmask','query_changed','raw_nonpositive_sizes','status','center_shift_L2_m','size_shift_L2_m')

def numeric(v):
 if v is not None and (not isinstance(v,(bool,int,float)) or isinstance(v,float) and not math.isfinite(v)):raise ValueError('Finite scalar or explicit null required')
 return v
def label(v):
 if not isinstance(v,str) or re.fullmatch(r'[A-Za-z0-9_.: -]{1,200}',v) is None or '/' in v:raise ValueError('Controlled enum required, no free payload/path')
 return v
def normalized_private_key(k):
 n=re.sub('[^a-z0-9]','',str(k).lower())
 return n.endswith(('queryindex','queryindices','maskindex','maskindices')) or any(x in n for x in ('rawanswer','caption','pointpath','imagepath','metapath','groundtruth','target6','target7','boxraw','boxeval','embedding','featurevector','maskbool','physicalrawids'))
def ensure_public(obj):
 if isinstance(obj,dict):
  for k,v in obj.items():
   if normalized_private_key(k):raise ValueError('Private payload field '+str(k))
   ensure_public(v)
 elif isinstance(obj,list):
  for v in obj:ensure_public(v)
 elif obj is not None and not isinstance(obj,(str,bool,int,float)):raise ValueError('Non JSON scalar type')
 elif isinstance(obj,float) and not math.isfinite(obj):raise ValueError('Nonfinite public value')
def scalar_dict(source,keys):
 if set(source)!=set(keys):raise ValueError('Unexpected aggregate schema '+str(set(source)^set(keys)))
 return {k:numeric(source[k]) for k in keys}
def scene_stat(source):
 if set(source)!=set(SCENE_KEYS):raise ValueError('Scene summary schema')
 out={k:numeric(v) for k,v in source.items() if k!='equal_scene_descriptive_CI95'}
 ci=source['equal_scene_descriptive_CI95'];out['equal_scene_descriptive_CI95']=None if ci is None else [numeric(v) for v in ci]
 if ci is not None and len(ci)!=2:raise ValueError('CI pair')
 return out
def summary_row(x):
 if set(x)!=set(SUMMARY_KEYS):raise ValueError('Summary row schema')
 if x['case_role'] not in ROLES or x['arm'] not in ARMS or x['platform'] not in ('drone','quad','all_platforms'):raise ValueError('Unexpected role/domain/arm')
 out={}
 for k,v in x.items():
  if k in ('case_role','platform','arm','population'):out[k]=label(v)
  elif k=='status_histogram':out[k]={label(a):numeric(b) for a,b in v.items()}
  elif k in ('abs_error6_m_mean','equal_scene_abs_error6_m_mean','actual_changed_axis_counts'):
   if len(v)!=6:raise ValueError('Six scalar coordinate error/count roles')
   out[k]=[numeric(z) for z in v]
  else:out[k]=numeric(v)
 return out
def contrast(x):
 expected={'case_role','platform','arm','reference','n','IoU_gain','center_error_reduction_m','size_error_reduction_m','abs6_error_reduction_m','IoU_scene','center_scene','size_scene','positive_means_improvement'}
 if set(x)!=expected:raise ValueError('Contrast schema')
 out={k:label(x[k]) for k in ('case_role','platform','arm','reference')};out['n']=numeric(x['n']);out['positive_means_improvement']=numeric(x['positive_means_improvement'])
 for k in ('IoU_gain','center_error_reduction_m','size_error_reduction_m'):out[k]=scalar_dict(x[k],SIGNED_KEYS)
 if len(x['abs6_error_reduction_m'])!=6:raise ValueError('Six scalar error contrasts')
 out['abs6_error_reduction_m']=[scalar_dict(z,SIGNED_KEYS) for z in x['abs6_error_reduction_m']]
 for k in ('IoU_scene','center_scene','size_scene'):out[k]=scene_stat(x[k])
 return out
def oracle_summary(x):
 scalar=('n','max_all256_iou_mean','equal_scene_max_all256_iou_mean','all256_Acc25','all256_Acc50','equal_scene_all256_Acc25','equal_scene_all256_Acc50','rank_recoverable25_n','rank_recoverable50_n','first_qualified_source_rank25_defined_n','first_qualified_source_rank50_defined_n')
 expected={'case_role','platform',*scalar,'CPU_stable_score_retrieval_diagnostic','scope'}
 if set(x)!=expected:raise ValueError('Explicit oracle aggregate schema')
 out={k:label(x[k]) for k in ('case_role','platform')};out.update({k:numeric(x[k]) for k in scalar})
 if set(x['CPU_stable_score_retrieval_diagnostic'])!={'1','5','10','256'}:raise ValueError('Oracle diagnostic K set')
 out['CPU_stable_score_retrieval_diagnostic']={k:scalar_dict(v,('best_iou_within_k_mean_GTassisted','Acc25','Acc50','native_GPU_rank_claimed')) for k,v in x['CPU_stable_score_retrieval_diagnostic'].items()}
 out['scope']='Offline post-seal GT-assisted source-pool coverage only; no deployed selector, instance identity or native GPU rank claim.'
 return out
def project_result(result,cases,scene_alias):
 """Only explicit scalar cells, no source arrays, candidate ranks or indices."""
 byid={c['id']:c for c in cases};rows=result['rows_PRIVATE'];assert len(rows)==204 and len(byid)==204
 observed=set();out=[]
 for row in rows:
  sid=row['id'];c=byid[sid]
  if sid in observed or row['case_role']!=c['role'] or row['platform']!=c['platform'] or row['scene']!=c['scene_PRIVATE']:raise ValueError('Case join mismatch')
  observed.add(sid);alias=c['case_alias']
  if re.fullmatch(r'(fresh|reg)_(drone|quad)_[0-9]{5}',alias) is None:raise ValueError('Anonymous five-digit case alias')
  if set(row['arms'])!=set(ARMS):raise ValueError('Complete eight arms required')
  for arm in ARMS:
   x=row['arms'][arm];e=x['coordinate_errors'];bits=x['actual_changed_axes']
   if len(bits)!=6 or not all(isinstance(v,bool) for v in bits) or len(e['abs_error6_m'])!=6:raise ValueError('Role scalar dimensions')
   y={'case_alias':alias,'case_role':row['case_role'],'platform':row['platform'],'scene_alias':scene_alias[(row['platform'],row['scene'])],'arm':arm,
    'iou':numeric(x['iou']),'Acc25':numeric(x['Acc25']),'Acc50':numeric(x['Acc50']),
    'center_L2_error_m':numeric(e['center_L2_m']),'size_L2_error_m':numeric(e['size_L2_m']),
    **{f'abs_error_{k}_m':numeric(v) for k,v in zip(('cx','cy','cz','sx','sy','sz'),e['abs_error6_m'])},
    'actual_changed':numeric(x['actual_changed']),'actual_changed_axis_bitmask':sum((1<<i) for i,v in enumerate(bits) if v),
    'query_changed':numeric(x['query_changed']),'raw_nonpositive_sizes':numeric(x['raw_nonpositive_sizes']),'status':label(x['status']),
    'center_shift_L2_m':numeric(x['center_shift_L2_m']),'size_shift_L2_m':numeric(x['size_shift_L2_m'])}
   if tuple(y)!=ROW_FIELDS:raise ValueError('Frozen column order')
   out.append(y)
 if observed!=set(byid):raise ValueError('Missing sample')
 summ={'state':label(result['state']),'n':numeric(result['n']),
  'case_role_counts':{label(k):numeric(v) for k,v in result['case_role_counts'].items()},
  'summary':[summary_row(x) for x in result['summary']],
  'paired_contrasts':[contrast(x) for x in result['paired_contrasts']],
  'offline_source_pool_oracle':[oracle_summary(x) for x in result['offline_source_pool_oracle']],
  'seconds':numeric(result['seconds']),'source_checkpoint_sha256':result['source_checkpoint_sha256'],
  'scope':'All targetVAL historically model/GT exposed. Fresh64 excludes registered recent content only; regression140 separate. No automatic neural TTA. Native yaw0 and strict thresholds; scene intervals descriptive.'}
 tails=[]
 for x in result['signed_tails_PRIVATE']:
  y={k:label(x[k]) for k in ('case_role','platform','arm','reference')}
  for private,public in [('most_negative_PRIVATE','most_negative'),('most_positive_PRIVATE','most_positive')]:
   y[public]=[{'case_alias':byid[z['id']]['case_alias'],'delta_iou':numeric(z['delta_iou'])} for z in x[private]]
  y['selection']='fixed signed order; may include ties if no positive or negative';tails.append(y)
 ensure_public([out,summ,tails]);return out,summ,tails
