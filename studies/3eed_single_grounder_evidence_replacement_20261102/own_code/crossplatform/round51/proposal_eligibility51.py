"""R51 A-only proposal-bag qualification diagnostic, root CPU run only.

Reads sealed raw-safe point ID bags and the unchanged three G factors. No B,
raw media, GT, models, Qwen, threshold sweep, parameter fit or G modification.
Nonzero qualification does not authorize TTA or certify semantic identity.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from proposal_groups51 import build_proposal_groups,pair_geometry,MAXIMUM_PAIR_SHARED_FRACTION

MANIFEST='research/round24/input/ALL336_PRIVATE.json'
FACTORS=[.5,1.,2.]
ARMS=['replace_all_c05','replace_all_c1','replace_all_c2']

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def path(s):
 p=(ROOT/s).resolve() if not Path(s).is_absolute() else Path(s).resolve()
 assert p.is_relative_to(ROOT) and 'round46/final' not in str(p) and 'FINAL384' not in str(p)
 return p
def load(p,fields):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in fields}
def exact(a,b,name):assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),name
def distribution(values):
 a=np.asarray(values,float)
 if not len(a):return {'n':0,'min':None,'median':None,'max':None,'mean':None}
 assert np.isfinite(a).all()
 return {'n':len(a),'min':float(a.min()),'median':float(np.median(a)),'max':float(a.max()),'mean':float(a.mean())}

def summarize(records):
 result={}
 for factor,arm in zip(FACTORS,ARMS):
  rows=[r for r in records if r['arm']==arm];assert len(rows)==336
  scopes={}
  for scope,rr in [('ALL336',rows),('DRONE',[r for r in rows if r['platform']=='drone']),('QUAD',[r for r in rows if r['platform']=='quad'])]:
   scopes[scope]={'n':len(rr),'G_changes_query':sum(r['G_changes_query'] for r in rr),
     'source_unique_proposal_group':sum(r['source']['group_id']>=0 for r in rr),
     'G_unique_proposal_group':sum(r['G']['group_id']>=0 for r in rr),
     'both_unique_distinct_proposal_groups':sum(r['both_unique_distinct_proposal_groups'] for r in rr),
     'A_potential_feedback_upper_bound':sum(r['A_potential_feedback_upper_bound'] for r in rr),
     'A_rejection_counts_multilabel':dict(collections.Counter(z for r in rr for z in r['rejection_reasons'])),
     'original_strict_A_potential_feedback_upper_bound':sum(r['old_strict_A_potential_feedback_upper_bound'] for r in rr),
     'counterfactual_same_new_bags_plus_global_shared_gate':sum(r['counterfactual_global_shared_gate_eligible'] for r in rr),
     'counterfactual_global_veto_additional_rejections':sum(r['A_potential_feedback_upper_bound'] and not r['counterfactual_global_shared_gate_eligible'] for r in rr),
     'endpoint_status_counts':{k:dict(collections.Counter(r[k]['status'] for r in rr)) for k in ('source','G')},
     'support_distributions':{k:{name:distribution([r[k][name] for r in rr]) for name in ('support_points','original_components','global_alternative_shared','cross_group_shared')} for k in ('source','G')},
     'local_pair_max_directional_overlap':distribution([r['pair_geometry']['max_directional_overlap'] for r in rr]),
     'g_preference_difference':distribution([r['g_preference_difference'] for r in rr])}
  result[arm]={'appearance_factor':factor,**scopes}
 return result

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',required=True);ap.add_argument('--authorization',required=True);args=ap.parse_args()
 plan=read(args.plan);auth=read(args.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R51_A_ONLY_PROPOSAL_BAG_ELIGIBILITY_NO_GT' and auth['plan_sha256']==sha(args.plan) and auth['program_sha256']==sha(__file__)
 assert plan['state']=='ROOT_LOCKED_R51_A_ONLY_PROPOSAL_BAG_ELIGIBILITY' and plan['n']==336
 assert plan['manifest']==MANIFEST and plan['appearance_factor_values']==FACTORS
 assert plan['no_GT_B_media_models_Qwen_G_change_or_fit'] is True and 0<plan['max_wall_seconds']<=600
 expected={MANIFEST,plan['prepared_root']+'/SEALED.json',plan['safe_root']+'/SEALED.json',plan['original_eligibility_root']+'/SEALED.json'}
 assert set(plan['inputs_sha256'])==expected
 for group in ('programs_sha256','inputs_sha256'):
  for f,h in plan[group].items():assert sha(path(f))==h,('pin',f)
 for f in ('crossplatform/round51/proposal_groups51.py','crossplatform/round51/proposal_eligibility51.py'):
  assert plan['programs_sha256'][f]==sha(ROOT/f)
 roots={k:path(plan[k]) for k in ('prepared_root','safe_root','original_eligibility_root')}
 seals={k:read(p/'SEALED.json') for k,p in roots.items()}
 assert seals['prepared_root']['state']=='R51_RAW_SAFE_INSTANCE_ATTRIBUTE_PREPARATION_COMPLETE_NO_GT'
 assert seals['safe_root']['state']=='R51_FINAL_SAFE_THREE_FACTOR_PREDICTIONS_NO_GT'
 assert seals['safe_root']['appearance_factor_values']==FACTORS
 assert seals['safe_root']['relation_operator']=='literal_R23_weighted/(1+total)'
 assert seals['original_eligibility_root']['state']=='R51_A_ONLY_PREFERENCE_ELIGIBILITY_COMPLETE_NO_GT'
 assert all(s['n']==336 for s in seals.values())
 rows=read(ROOT/MANIFEST);assert len(rows)==336 and len({r['id'] for r in rows})==336
 out=path(plan['output_root']);assert out.parent==ROOT/'crossplatform/round51' and not out.exists();out.mkdir(parents=True)
 for d in ('groups_PRIVATE','arrays_PRIVATE'):(out/d).mkdir()
 bindings=[]
 def member(key,rel,role):
  p=roots[key]/rel;digest=sha(p);assert digest==seals[key]['files'][rel]
  bindings.append({'role':role,'path_PRIVATE':str(p),'sha256':digest});return p
 old=read(member('original_eligibility_root','ROWS_PRIVATE.json','IMMUTABLE_ORIGINAL_STRICT_ELIGIBILITY'))
 original={(r['id_PRIVATE'],r['arm']):r for r in old};assert len(original)==1008
 start=time.perf_counter();records=[];group_counts=[];done=[]
 try:
  for row in rows:
   if time.perf_counter()-start>plan['max_wall_seconds']:raise TimeoutError('Fixed CPU-only diagnostic wall cap')
   sid=row['id']
   a=load(member('prepared_root','arrays_PRIVATE/'+sid+'.npz','WHOLE_RAW_POINT_SUPPORT_BAGS'),
       ['source_boxes','source_scores','query_support_point_offsets','query_support_point_indices','query_component_counts'])
   s=load(member('safe_root','samples_PRIVATE/'+sid+'.npz','UNCHANGED_THREE_FACTOR_G'),
       ['source_boxes','source_scores','arm_names','scores','queries','components','appearance_factor_values'])
   exact(a['source_boxes'],s['source_boxes'],'source boxes unchanged');exact(a['source_scores'],s['source_scores'],'source scores unchanged')
   assert s['arm_names'].tolist()==['source']+ARMS and s['appearance_factor_values'].tolist()==FACTORS
   info,ga=build_proposal_groups(source_boxes=a['source_boxes'],point_offsets=a['query_support_point_offsets'],
       point_indices=a['query_support_point_indices'],component_counts=a['query_component_counts'])
   dump(out/'groups_PRIVATE'/(sid+'.json'),info);np.savez_compressed(out/'arrays_PRIVATE'/(sid+'.npz'),**ga)
   intersections=ga['pair_intersection_counts'];npoints=ga['query_support_counts'];union=npoints[:,None]+npoints[None]-intersections
   jaccard=np.divide(intersections,union,out=np.zeros_like(intersections,dtype=float),where=union>0)
   coverage=np.divide(intersections,npoints[:,None],out=np.zeros_like(intersections,dtype=float),where=npoints[:,None]>0)
   point_compatible=(jaccard>=.7)&(coverage>=.75)&(coverage.T>=.75)&(npoints[:,None]>=6)&(npoints[None]>=6)
   group_counts.append({'id_PRIVATE':sid,'platform':row['platform'],'groups':len(info['groups']),
     'mapped_queries':int((ga['query_group_ids']>=0).sum()),
     'multiple_group_UNKNOWN':info['query_status'].count('UNKNOWN_multiple_complete_link_proposal_groups'),
     'point_compatible_but_geometry_veto_pair_count':int(np.sum(np.triu(point_compatible&~ga['pair_mutual_centers'],1)))})
   q0=int(s['source_scores'].argmax());ids=ga['query_group_ids']
   for fi,(factor,arm) in enumerate(zip(FACTORS,ARMS)):
    F=s['source_scores'].astype(float)+2*s['components'][fi,0]+2*s['components'][fi,1]+2*s['components'][fi,2]
    exact(F,s['scores'][fi+1],'original G byteexact and original addition order')
    winner=int(F.argmax());assert winner==int(s['queries'][fi+1]);g=F-s['source_scores'].astype(float)
    previous=original[(sid,arm)]
    assert q0==previous['source']['query_PRIVATE'] and winner==previous['G']['query_PRIVATE']
    pg=int(ids[winner]);ng=int(ids[q0]);pair=pair_geometry(info,pg,ng,ga['support_offsets'],ga['support_indices'])
    rejected=list(pair['rejection_reasons'])
    if winner==q0:rejected.append('G_keeps_source_top1')
    if not g[winner]>g[q0]:rejected.append('G_increment_does_not_prefer_replacement')
    endpoints={}
    for key,q in (('source',q0),('G',winner)):
     endpoints[key]={'query_PRIVATE':q,'group_id':int(ids[q]),'status':info['query_status'][q],
       'support_points':int(ga['query_support_counts'][q]),'original_components':int(ga['component_counts'][q]),
       'global_alternative_shared':float(ga['query_global_alternative_shared_fraction'][q]),
       'cross_group_shared':float(ga['query_cross_group_shared_fraction'][q])}
    valid=not rejected
    global_would_accept=valid and max(endpoints[k]['cross_group_shared'] for k in ('source','G'))<=MAXIMUM_PAIR_SHARED_FRACTION
    records.append({'id_PRIVATE':sid,'platform':row['platform'],'arm':arm,'appearance_factor':factor,
      'source':endpoints['source'],'G':endpoints['G'],'G_changes_query':winner!=q0,
      'both_unique_distinct_proposal_groups':pg>=0 and ng>=0 and pg!=ng,
      'g_preference_difference':float(g[winner]-g[q0]),'pair_geometry':pair,
      'rejection_reasons':rejected,'A_potential_feedback_upper_bound':valid,
      'old_strict_A_potential_feedback_upper_bound':bool(previous['A_potential_feedback_upper_bound']),
      'counterfactual_global_shared_gate_eligible':bool(global_would_accept),
      'global_shared_used_for_actual_eligibility':False,
      'B_correspondence_not_checked':True,'semantic_identity_not_certified':True,'task_correctness_not_measured':True})
   done.append(sid);dump(out/'PROGRESS.json',{'completed_n':len(done),'n':336,'seconds':time.perf_counter()-start})
   if len(done)%16==0:print('R51_PROPOSAL_BAG_A_ONLY',len(done),336,flush=True)
  assert len(done)==336 and len(records)==1008
  dump(out/'ROWS_PRIVATE.json',records);dump(out/'GROUP_COUNTS_PRIVATE.json',group_counts)
  dump(out/'INPUT_LEDGER_PRIVATE.json',{'files':bindings,'no_raw_media_or_GT_or_B_read':True})
  dump(out/'RESULT.json',{'state':'COMPLETED_R51_A_ONLY_PROPOSAL_SUPPORT_BAG_ELIGIBILITY_NO_GT',
    'n':336,'factor_count':3,'rows':1008,'summary':summarize(records),
    'group_count_summary':{p:distribution([r['groups'] for r in group_counts if r['platform']==p]) for p in ('drone','quad')},
    'factor_or_threshold_selection_performed':False,'G_Qwen_boxes_and_original_strict_result_unchanged':True,
    'interpretation':'CANDIDATE_PROPOSAL_SUPPORT_ONLY_NOT_SEMANTIC_IDENTITY_OR_ACCEPTED_B_TTA',
    'global_shared_diagnostic_only':True,'GT_read':False,'B_read':False,'media_read':False,
    'model_forwards':0,'optimizer_steps':0,'seconds':time.perf_counter()-start})
  dump(out/'SEALED.json',{'state':'R51_A_ONLY_PROPOSAL_BAG_ELIGIBILITY_COMPLETE_NO_GT','n':336,
    'plan_sha256':sha(args.plan),'files':{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
 except BaseException as error:
  dump(out/'FAILED.json',{'error_type':type(error).__name__,'error':str(error),'completed_n':len(done),'automatic_retry':False,'GT_read':False});raise

if __name__=='__main__':main()
