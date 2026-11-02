"""UNEXECUTED GT-free A-side fixed-rule feedback eligibility upper bound.

Only sealed DEV336 prepared strict query identities/status and repaired safeG
vectors are consumed. No B, raw media, annotations, model or optimizer. A-only
eligibility is a necessary condition, not accepted cross-sampling feedback.
The three existing factors are all reported; counts never select a factor.
"""
from pathlib import Path
import argparse,collections,hashlib,json,time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
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
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
 tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def path(v):
 p=(ROOT/v).resolve();assert p.is_relative_to(ROOT) and 'FINAL' not in str(p) and 'round46/final' not in str(p)
 return p
def exact(a,b,label):
 assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),label
def npz(p,fields):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in fields}
def stats(values):
 v=np.asarray(values,np.float64)
 if not len(v):return {'n':0,'min':None,'q10':None,'q25':None,'median':None,'q75':None,'q90':None,'max':None,'mean':None}
 assert np.isfinite(v).all()
 q=np.quantile(v,[0,.1,.25,.5,.75,.9,1])
 return dict(n=len(v),**dict(zip(['min','q10','q25','median','q75','q90','max'],q.tolist())),mean=float(v.mean()))


def summarize(rows):
 result={}
 for factor,arm in zip(FACTORS,ARMS):
  rr=[r for r in rows if r['arm']==arm];assert len(rr)==336
  metrics={}
  for scope,subset in [('ALL336',rr),('DRONE',[r for r in rr if r['platform']=='drone']),('QUAD',[r for r in rr if r['platform']=='quad'])]:
   reject=collections.Counter(x for r in subset for x in r['A_rule_rejection_reasons'])
   statuses={endpoint:dict(collections.Counter(r[endpoint]['query_status'] for r in subset)) for endpoint in ('source','G')}
   distributions={}
   for group,ss in [('all',subset),('query_changed',[r for r in subset if r['G_changes_query']]),
    ('both_strict_known_distinct',[r for r in subset if r['both_strict_known_distinct']]),
    ('A_potential_feedback_upper_bound',[r for r in subset if r['A_potential_feedback_upper_bound']])]:
    distributions[group]={endpoint:{name:stats([r[endpoint][name] for r in ss]) for name in ('dominant','runner','shared')}
     for endpoint in ('source','G')}
    distributions[group]['g_preference_difference']=stats([r['g_preference_difference'] for r in ss])
   metrics[scope]={'n':len(subset),'G_changes_query':sum(r['G_changes_query'] for r in subset),
    'G_query_change_with_same_known_carrier':sum(r['G_changes_query'] and r['both_strict_known'] and not r['both_strict_known_distinct'] for r in subset),
    'source_strict_known':sum(r['source']['strict_instance_id']>=0 for r in subset),
    'G_strict_known':sum(r['G']['strict_instance_id']>=0 for r in subset),
    'both_strict_known':sum(r['both_strict_known'] for r in subset),
    'both_strict_known_distinct':sum(r['both_strict_known_distinct'] for r in subset),
    'g_increment_supports_G_over_source':sum(r['g_increment_supports_G_over_source'] for r in subset),
    'A_potential_feedback_upper_bound':sum(r['A_potential_feedback_upper_bound'] for r in subset),
    'A_rule_rejection_counts_multilabel':dict(reject),'underlying_endpoint_status_counts':statuses,
    'endpoint_fraction_distributions':distributions,
    'cue_known_endpoint_counts':{endpoint:{cue:sum(r[endpoint]['cue_known'][j] for r in subset)
     for j,cue in enumerate(('V','P','R'))} for endpoint in ('source','G')}}
  result[arm]={'appearance_factor':factor,**metrics}
 return result


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True);parser.add_argument('--authorization',required=True)
 args=parser.parse_args();pp=path(args.plan);plan=read(pp);auth=read(path(args.authorization))
 assert pp.parent==ROOT/'research/round51/experiment'
 assert auth['state']=='ROOT_AUTHORIZED_R51_A_ONLY_PREFERENCE_ELIGIBILITY_NO_GT' and auth['plan_sha256']==sha(pp) and auth['program_sha256']==sha(__file__)
 assert plan['state']=='ROOT_LOCKED_R51_A_ONLY_PREFERENCE_ELIGIBILITY' and plan['n']==336
 assert plan['manifest']==MANIFEST and plan['appearance_factor_values']==FACTORS
 assert plan['no_GT_B_media_external_models_or_method_change'] is True
 expected={MANIFEST,plan['prepared_root']+'/SEALED.json',plan['safe_root']+'/SEALED.json'}
 assert set(plan['inputs_sha256'])==expected
 for group in ('inputs_sha256','programs_sha256'):
  for rel,digest in plan[group].items():assert sha(path(rel))==digest,rel
 assert plan['programs_sha256']['crossplatform/round51/pref_eligibility51.py']==sha(__file__)
 prep=path(plan['prepared_root']);safe=path(plan['safe_root']);out=path(plan['output_root'])
 assert all(p.parent==ROOT/'crossplatform/round51' for p in (prep,safe,out)) and not out.exists()
 ps=read(prep/'SEALED.json');ss=read(safe/'SEALED.json')
 assert ps['state']=='R51_RAW_SAFE_INSTANCE_ATTRIBUTE_PREPARATION_COMPLETE_NO_GT' and ps['n']==336
 assert ss['state']=='R51_FINAL_SAFE_THREE_FACTOR_PREDICTIONS_NO_GT' and ss['n']==336
 assert ss['relation_operator']=='literal_R23_weighted/(1+total)' and ss['appearance_factor_values']==FACTORS
 assert ps['external_vision_outputs_or_mask_derived_cache_read'] is False
 assert ss['external_vision_outputs_or_mask_derived_caches_read'] is False
 manifest=read(ROOT/MANIFEST);assert len(manifest)==336 and len({r['id'] for r in manifest})==336
 out.mkdir();bindings=[];records=[];start=time.perf_counter();configs={}
 def bound(root,seal,rel,role):
  assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
  p=root/rel;actual=sha(p);bindings.append({'path_PRIVATE':str(p),'role':role,'sha256':actual})
  dump(out/'INPUT_LEDGER_PRIVATE.json',{'files':bindings,'no_GT_B_or_media_reads':True})
  assert actual==seal['files'][rel],role
  return p
 try:
  for row in manifest:
   sid=row['id'];pcpath=bound(prep,ps,'cases_PRIVATE/'+sid+'.json','STRICT_QUERY_STATUS_AND_CONFIG');pc=read(pcpath)
   pa=bound(prep,ps,'arrays_PRIVATE/'+sid+'.npz','SOURCE_AND_STRICT_ID_FRACTIONS')
   a=npz(pa,['source_boxes','source_scores','query_instance_ids','query_dominant_runner_shared_fractions'])
   assert sha(pa)==pc['arrays_sha256'] and len(pc['query_status'])==256
   digest=hashlib.sha256(json.dumps(pc['config'],sort_keys=True).encode()).hexdigest();configs[digest]=pc['config']
   sc=read(bound(safe,ss,'cases_PRIVATE/'+sid+'.json','SAFE_G_BINDING'))
   sp=bound(safe,ss,'samples_PRIVATE/'+sid+'.npz','THREE_FIXED_SAFE_G_FACTORS')
   s=npz(sp,['source_boxes','source_scores','arm_names','scores','components','cue_query_known','appearance_factor_values','queries'])
   assert sc['input_prepared_case_sha256']==sha(pcpath) and sc['samples_sha256']==sha(sp)
   exact(a['source_boxes'],s['source_boxes'],'source_boxes_immutable');exact(a['source_scores'],s['source_scores'],'source_nativep_immutable')
   assert s['arm_names'].tolist()==['source']+ARMS and s['appearance_factor_values'].tolist()==FACTORS
   ids=a['query_instance_ids'];fractions=a['query_dominant_runner_shared_fractions'];q0=int(s['source_scores'].argmax())
   assert ids.shape==(256,) and fractions.shape==(256,3)
   for fi,(factor,arm) in enumerate(zip(FACTORS,ARMS)):
    cues=s['components'][fi];F=s['source_scores'].astype(np.float64)+2*cues[0]+2*cues[1]+2*cues[2]
    exact(F,s['scores'][fi+1],'original_G_addition_order')
    winner=int(F.argmax());assert winner==int(s['queries'][fi+1]);g=F-s['source_scores'].astype(np.float64)
    positive=int(ids[winner]);negative=int(ids[q0]);changed=winner!=q0
    known=positive>=0 and negative>=0;distinct=known and positive!=negative;supports=bool(g[winner]>g[q0])
    reject=[]
    if not changed:reject.append('G_keeps_source_top1')
    if not known:reject.append('UNKNOWN_A_point_identity')
    # Match structure51's existing diagnostic condition exactly, including
    # coincident -1 UNKNOWN sentinels. This is not a second identity assertion.
    if positive==negative:reject.append('A_same_point_hypothesis')
    if not supports:reject.append('G_increment_does_not_prefer_replacement')
    endpoints={}
    for label,q in [('source',q0),('G',winner)]:
     endpoints[label]={'query_PRIVATE':q,'strict_instance_id':int(ids[q]),'query_status':pc['query_status'][q],
      'dominant':float(fractions[q,0]),'runner':float(fractions[q,1]),'shared':float(fractions[q,2]),
      'native_probability':float(s['source_scores'][q]),'frozen_G_score':float(F[q]),
      'G_increment':float(g[q]),'cue_known':[bool(x) for x in s['cue_query_known'][fi,:,q]],
      'cue_components':[float(x) for x in cues[:,q]]}
    records.append({'id_PRIVATE':sid,'platform':row['platform'],'arm':arm,'appearance_factor':factor,
     'source':endpoints['source'],'G':endpoints['G'],'G_changes_query':changed,
     'both_strict_known':known,'both_strict_known_distinct':distinct,
     'g_increment_supports_G_over_source':supports,'g_preference_difference':float(g[winner]-g[q0]),
     'A_potential_feedback_upper_bound':not reject,'A_rule_rejection_reasons':reject,
     'B_correspondence_not_checked':True,'task_correctness_not_measured':True})
   if len(records)%96==0:print('R51_PREF_ELIGIBILITY',len(records)//3,336,flush=True)
  assert len(records)==1008 and len(configs)==1
  summary=summarize(records)
  dump(out/'ROWS_PRIVATE.json',records)
  dump(out/'RESULT.json',{'state':'COMPLETED_R51_GT_FREE_A_ONLY_PREFERENCE_ELIGIBILITY_UPPER_BOUND','n':336,
   'factor_count':3,'rows':1008,'summary':summary,'prepared_configs':configs,
   'interpretation':'A_ONLY_NECESSARY_CONDITION_UPPER_BOUND_NOT_ACCEPTED_B_FEEDBACK_OR_TTA_EFFECT',
   'factor_selection_performed':False,'thresholds_changed':False,'GT_read':False,'B_read':False,
   'media_read':False,'model_forwards':0,'optimizer_steps':0,'seconds':time.perf_counter()-start})
  dump(out/'SEALED.json',{'state':'R51_A_ONLY_PREFERENCE_ELIGIBILITY_COMPLETE_NO_GT','n':336,
   'plan_sha256':sha(pp),'files':{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
 except BaseException as error:
  dump(out/'FAILED.json',{'error_type':type(error).__name__,'error':str(error),'completed_rows':len(records),
   'automatic_retry':False,'GT_read':False});raise


if __name__=='__main__':main()
