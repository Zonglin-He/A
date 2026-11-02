"""Independent R51 SAVED-ONLY aggregate audit. No GT/geometry/model imports.

Recompute all summaries, scenes, ordered negatives and calibration from sealed
task scalars. Reused baseline IoUs/queries are exact six-field identity joins.
Supports any score51 arm set (including a separately scored TTA table).
"""
from pathlib import Path
import argparse
import collections
import hashlib
import itertools
import json
import math
import time
ROOT=Path(__file__).resolve().parents[2]
IDENTITY=('imagepath','pointpath','metapath','caption','platform','scene')
PLATFORMS=('drone','quad')
SUMMARY_ABS_TOLERANCE_PP=1e-10

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,v):
 p=Path(p);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def path(s):
 p=Path(s);assert not p.is_absolute() and '..' not in p.parts
 assert 'round46/final' not in s and 'FINAL384' not in s
 return ROOT/p
def ident(r):return tuple(r[k] for k in IDENTITY)
def average(v):
 assert v
 return math.fsum(v)/len(v)

def aggregate(tasks,arms,refs,n):
 scopes=[('ALL'+str(n),tasks)]
 if n==5912:
  scopes.extend((part,[r for r in tasks if r['part']==part]) for part in ('DEV336','NONDEV5576'))
 else:
  scopes.extend((s,[r for r in tasks if r['split']==s]) for s in sorted({r['split'] for r in tasks}) if s!='ALL336')
 summary={};scenes=[]
 for scope,rows in scopes:
  assert rows
  block={}
  for platform in PLATFORMS:
   rr=[r for r in rows if r['platform']==platform];assert rr
   block[platform]={}
   for arm in arms:
    v=[r['IoU'][arm] for r in rr]
    record={'n':len(rr),'mIoU_percent':average(v)*100,
      'Acc25_percent':sum(x>.25 for x in v)/len(v)*100,
      'Acc50_percent':sum(x>.5 for x in v)/len(v)*100,
      'scene_clusters':len({r['scene_PRIVATE'] for r in rr})}
    for ref in refs:
     d=[r['IoU'][arm]-r['IoU'][ref] for r in rr]
     record.update({f'gain_vs_{ref}_pp':average(d)*100,
       f'improved_vs_{ref}':sum(x>1e-10 for x in d),
       f'worsened_vs_{ref}':sum(x< -1e-10 for x in d),
       f'query_changes_vs_{ref}':sum(r['queries'][arm]!=r['queries'][ref] for r in rr)})
    for status in ('unknown','fallback','no_parameter_update'):
     record[status+'_n']=sum(r['statuses'][arm][status] for r in rr)
    block[platform][arm]=record
   for scene in sorted({r['scene_PRIVATE'] for r in rr}):
    sr=[r for r in rr if r['scene_PRIVATE']==scene]
    scenes.append({'scope':scope,'platform':platform,'scene_PRIVATE':scene,'n':len(sr),
      'mIoU_percent':{arm:average([r['IoU'][arm] for r in sr])*100 for arm in arms}})
  block['balanced_mIoU_percent']={a:average([block[p][a]['mIoU_percent'] for p in PLATFORMS]) for a in arms}
  block['micro_mIoU_percent']={a:average([r['IoU'][a] for r in rows])*100 for a in arms}
  summary[scope]=block
 return summary,scenes

def check_tree(expected,actual,tolerance=SUMMARY_ABS_TOLERANCE_PP):
 """Numeric tolerance is for independent summation only, never task scoring."""
 if isinstance(expected,dict):
  assert isinstance(actual,dict) and set(expected)==set(actual)
  return max([check_tree(expected[k],actual[k],tolerance) for k in expected] or [0.])
 if isinstance(expected,list):
  assert isinstance(actual,list) and len(expected)==len(actual)
  return max([check_tree(x,y,tolerance) for x,y in zip(expected,actual)] or [0.])
 if type(expected) is bool or expected is None or isinstance(expected,str):
  assert type(expected)==type(actual) and expected==actual
  return 0.
 if type(expected) is int:
  assert type(actual) is int and expected==actual
  return 0.
 assert isinstance(expected,(int,float)) and isinstance(actual,(int,float))
 assert math.isfinite(expected) and math.isfinite(actual)
 error=abs(expected-actual);assert error<=tolerance,('aggregate_error',error)
 return error

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--plan',required=True);ap.add_argument('--authorization',required=True);args=ap.parse_args()
 plan=read(args.plan);auth=read(args.authorization)
 assert auth['state']=='ROOT_AUTHORIZED_R51_SAVED_SCALAR_AUDIT_NO_GT' and auth['plan_sha256']==sha(args.plan) and auth['program_sha256']==sha(__file__)
 assert plan['state']=='ROOT_LOCKED_R51_SAVED_SCALAR_AUDIT' and plan['n'] in (336,5912)
 assert plan['no_GT_geometry_scoring_media_models_or_prediction_arrays'] is True
 for group in ('programs_sha256','inputs_sha256'):
  for p,h in plan[group].items():assert sha(path(p))==h,('pin',p)
 assert plan['programs_sha256'][Path(__file__).relative_to(ROOT).as_posix()]==sha(__file__)
 sr=path(plan['scoring_directory']);ss=read(sr/'SEALED.json')
 assert plan['inputs_sha256'][plan['scoring_directory']+'/SEALED.json']==sha(sr/'SEALED.json')
 receipts=[]
 def member(name):
  p=sr/name;assert sha(p)==ss['files'][name];receipts.append({'role':'SEALED_TASK_SCALAR','file_PRIVATE':str(p),'sha256':sha(p)});return read(p)
 score_plan=read(path(plan['scoring_plan']));receipt=member('SCORING_RECEIPT.json')
 assert receipt['state']=='PASS_R51_SINGLE_POSTSEAL_CONFIGURED_ORIGINAL_BOX_TASK_SCORE'
 assert receipt['plan_sha256']==sha(path(plan['scoring_plan'])) and receipt['n']==plan['n']==score_plan['n']
 assert receipt['single_GT_scoring_pass'] is True and receipt['unknown_fallback_and_no_update_rows_remain_full_denominator'] is True
 arms=score_plan['arm_names'];refs=score_plan['reference_arms'];reused=score_plan['reused_arms'];n=plan['n']
 assert receipt['arm_names']==arms and len(arms)==len(set(arms)) and set(refs)<=set(arms)
 permitted={plan['scoring_directory']+'/SEALED.json',plan['scoring_plan'],
   score_plan['cohort_registration'],score_plan['dev_registration'],score_plan['dev_split_by_id']}
 for spec in reused.values():permitted.update((spec['registration'],spec['saved_rows']))
 assert set(plan['inputs_sha256'])==permitted,'GT/P5/metadata/prediction arrays are not allowed input pins'
 def get(rel):
  assert rel in permitted and sha(path(rel))==plan['inputs_sha256'][rel]
  receipts.append({'role':'REGISTRATION_OR_SEALED_BASELINE_SCALARS','file_PRIVATE':rel,'sha256':sha(path(rel))})
  return read(path(rel))
 rows=get(score_plan['cohort_registration']);dev=get(score_plan['dev_registration']);splits=get(score_plan['dev_split_by_id'])
 assert len(rows)==n and len({r['id'] for r in rows})==n and len({ident(r) for r in rows})==n
 assert collections.Counter(r['platform'] for r in rows)==({'drone':168,'quad':168} if n==336 else {'drone':2984,'quad':2928})
 assert len(dev)==336 and len({ident(r) for r in dev})==336 and set(splits)=={r['id'] for r in dev}
 devkeys={ident(r):r['id'] for r in dev};assert set(devkeys)<={ident(r) for r in rows}
 tasks=member('ALL_TASK_ROWS_PRIVATE.json');assert len(tasks)==n
 byid={r['id_PRIVATE']:r for r in tasks};assert len(byid)==n and set(byid)=={r['id'] for r in rows}
 for i,(r,t) in enumerate(zip(rows,tasks)):
  assert t['id_PRIVATE']==r['id'] and t['row']==f'R{i+1:05d}'
  assert t['platform']==r['platform'] and t['scene_PRIVATE']==r['scene']
  part='DEV336' if ident(r) in devkeys else 'NONDEV5576'
  assert t['part']==part and t['split']==(splits[devkeys[ident(r)]] if part=='DEV336' else 'NONDEV5576')
  assert set(t['IoU'])==set(t['queries'])==set(t['statuses'])==set(arms)
  for a in arms:
   value=t['IoU'][a];q=t['queries'][a]
   assert type(value) in (float,int) and math.isfinite(value) and 0<=value<=1
   assert type(q) is int and 0<=q<256
   assert set(t['statuses'][a])=={'unknown','fallback','no_parameter_update'} and all(type(v) is bool for v in t['statuses'][a].values())
 baseline_cells=0
 for arm,spec in reused.items():
  reg=get(spec['registration']);old=get(spec['saved_rows']);oldids={r['id_PRIVATE']:r for r in old}
  assert len(oldids)==len(old)
  route={ident(r):r['id'] for r in reg};assert len(route)==len(reg)
  for r in rows:
   base=oldids[route[ident(r)]];t=byid[r['id']]
   assert base['platform']==r['platform'] and base['scene_PRIVATE']==r['scene']
   assert t['IoU'][arm]==base['IoU'][spec['saved_arm']]
   assert t['queries'][arm]==base['queries'][spec['saved_arm']]
   baseline_cells+=1
 expected_summary,expected_scenes=aggregate(tasks,arms,refs,n)
 summary_error=check_tree(expected_summary,member('TASK_SUMMARY.json'))
 scene_error=check_tree(expected_scenes,member('ALL_SCENE_SUMMARIES_PRIVATE.json'))
 expected_neg=[]
 for r in tasks:
  for a,b in itertools.product(arms,repeat=2):
   delta=r['IoU'][a]-r['IoU'][b]
   if a!=b and delta< -1e-10:
    expected_neg.append({'row':r['row'],'part':r['part'],'split':r['split'],'platform':r['platform'],
                        'arm':a,'reference':b,'delta_pp':float(delta*100)})
 # Same saved float subtraction/order, so every negative row/value is exact.
 assert expected_neg==member('ALL_ORDERED_NEGATIVE_COMPARISONS.json')
 selected=None;calibration=score_plan.get('calibration')
 if calibration is not None:
  assert n==336 and calibration['shared_across_platforms'] is True
  targets=calibration['target_arm_by_factor'];key={.5:'0.5',1.:'1',2.:'2'}
  # Use the independently validated saved arithmetic at the tiny tie boundary;
  # also require that independent fsum picks the same registered winner.
  saved_summary=member('TASK_SUMMARY.json')['ALL336']['balanced_mIoU_percent']
  def choose(metric):
   best=max(metric[targets[key[c]]] for c in calibration['factor_values'])
   return next(c for c in calibration['tie_priority'] if best-metric[targets[key[c]]]<=1e-10)
  chosen=choose(saved_summary);assert chosen==choose(expected_summary['ALL336']['balanced_mIoU_percent'])
  selected={'selected_shared_factor':chosen,'selected_arm':targets[key[chosen]],
   'metric':'ALL336_two_platform_equal_weight_mIoU_percent','numeric_tie_tolerance_pp':1e-10,
   'tie_priority':calibration['tie_priority'],'platform_specific_calibration_forbidden':True,
   'all_trials_retained':True,'development_labels_used_for_calibration':True}
 assert selected==member('SHARED_CALIBRATION_SELECTION.json')
 assert receipt['task_values']==n*len(arms) and receipt['reused_saved_task_values']==baseline_cells
 assert receipt['new_GT_scored_task_values']==n*(len(arms)-len(reused))
 assert receipt['ordered_negative_comparisons']==len(expected_neg)
 out=path(plan['output_directory']);assert not out.exists();out.mkdir(parents=True)
 result={'state':'PASS_R51_INDEPENDENT_SAVED_SCALAR_REAGGREGATION_NO_GT','n':n,'arm_names':arms,
   'task_values':n*len(arms),'reused_baseline_values_and_queries_exact_six_field_identity_cells':baseline_cells,
   'ordered_negative_comparisons_exact':len(expected_neg),'scene_rows_checked':len(expected_scenes),
   'independent_summary_max_abs_error_pp':summary_error,'independent_scene_max_abs_error_pp':scene_error,
   'aggregate_numeric_tolerance_pp':SUMMARY_ABS_TOLERANCE_PP,
   'calibration_recomputed':calibration is not None,'no_new_GT_or_geometry_score_or_prediction_array_read':True,
   'scoring_seal_sha256':sha(sr/'SEALED.json'),'scoring_receipt_sha256':sha(sr/'SCORING_RECEIPT.json'),
   'prediction_seal_sha256':receipt['prediction_seal_sha256'],'plan_sha256':sha(args.plan),'program_sha256':sha(__file__)}
 dump(out/'AUDIT_RESULT.json',result);dump(out/'READ_RECEIPTS_PRIVATE.json',receipts)
 dump(out/'SEALED.json',{'state':result['state'],'files':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()}})

if __name__=='__main__':main()
