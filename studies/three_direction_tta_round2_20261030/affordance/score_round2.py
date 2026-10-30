"""Independent CPU scoring process. Reads native annotation column3 only after prediction seal.
This file imports no adaptation/cachebuilder/model code.
"""
import argparse,pathlib,json,hashlib,time
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=pathlib.Path(__file__).resolve().parent
METRICS=['auc','aiou','sim','mae']
def digest(p):return hashlib.file_digest(open(p,'rb'),'sha256').hexdigest()
def metrics(pred,target):
 p=pred.reshape(-1);t=target.reshape(-1);truth=t>=.5
 sim=float(np.minimum(p/(p.sum()+1e-12),t/(t.sum()+1e-12)).sum());mae=float(np.abs(p-t).mean())
 auc=float(roc_auc_score(truth,p)) if np.unique(truth).size==2 else None
 aiou=None
 if truth.any():
  values=[]
  for threshold in np.linspace(0,1,20):
   mask=p>=threshold;values.append(float(np.logical_and(mask,truth).sum()/np.logical_or(mask,truth).sum()))
  aiou=float(np.mean(values))
 return {'auc':auc,'aiou':aiou,'sim':sim,'mae':mae}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=pathlib.Path,required=True);ap.add_argument('--cohort',choices=['dev','confirm'],required=True);a=ap.parse_args();protocol=json.loads((a.run/'protocol.json').read_text());assert protocol['state']=='sealed' and protocol['cohort']==a.cohort
 execution_lock=ROOT/'EXECUTION_LOCK.json'
 if execution_lock.exists():assert protocol['execution_lock_sha256']==digest(execution_lock)
 seal=json.loads((a.run/'SEALED.json').read_text())
 for n,h in seal.items():assert digest(a.run/n)==h
 scoring=ROOT/f'cohorts/{a.cohort}_scoring.json';lock=json.loads((ROOT/'cohorts/COHORT_LOCK.json').read_text());assert digest(scoring)==lock['manifests'][scoring.name];refs=json.loads(scoring.read_text());assert not protocol.get('engineering_subset_first_locked_parent') and len(refs)==protocol['count'],'Do not score an engineering subset as the complete development cohort';result=[];reference={};undefined=[]
 for i,row in enumerate(refs):
  container=np.load(row['native_annotation_container'],allow_pickle=False);assert container.shape==(2048,4);target=np.array(container[:,3:],copy=True);assert np.isfinite(target).all();truth=target>=.5
  why={}
  if not np.any(truth):why.update(auc='all_zero_binary_target',aiou='all_zero_binary_target')
  elif np.all(truth):why['auc']='all_one_binary_target'
  if why:undefined.append({'id':row['id'],'reason':why})
  for objective in protocol['objectives']:
   for step in protocol['steps']:
    d=np.load(a.run/f'{i:04d}_{objective}_k{step}.npz',allow_pickle=False)
    for arm in ['source','ensemble','median','q75','adapted']+(['legacy32_source'] if 'legacy32_source' in d else []):
     assert d[arm].shape==(2048,1) and np.isfinite(d[arm]).all()
     if arm!='adapted':
      key=(i,arm)
      if key in reference:assert np.array_equal(reference[key],d[arm]),'Frozen reference changed across arms'
      else:reference[key]=d[arm].copy()
     if step==0:assert np.array_equal(d['adapted'],d['median']),'Zero-head median readout changed'
     result.append({'id':row['id'],'parent_id':row['parent_id'],'object_class':row['object_class'],'affordance':row['affordance'],'objective':objective,'step':step,'arm':arm,**metrics(d[arm],target)})
 averages=[]
 for objective in protocol['objectives']:
  for step in protocol['steps']:
   for arm in sorted({r['arm'] for r in result}):
    sub=[r for r in result if (r['objective'],r['step'],r['arm'])==(objective,step,arm)]
    if not sub:continue
    entry={'objective':objective,'step':step,'arm':arm,'n':len(sub),'valid_counts':{},'undefined_counts':{}}
    for metric in METRICS:
     v=[r[metric] for r in sub if r[metric] is not None];entry[metric]=float(np.mean(v)) if v else None;entry['valid_counts'][metric]=len(v);entry['undefined_counts'][metric]=len(sub)-len(v)
    averages.append(entry)
 scored={'cohort':a.cohort,'rows':result,'averages':averages,'seal_verified':True,'frozen_source_and_all_ensemble_baselines_bitexact':True,'zero_head_median_bitexact':True,'undefined_samples':undefined,'gt_annotation_column':'native container[:,3:]','gt_data_read_only_in_this_scoring_process':True,'measurement':'valid_for_locked_cohort','scoring_script_sha256':digest(pathlib.Path(__file__))}
 output=a.run/'SCORED.json';output.write_text(json.dumps(scored,indent=2));print(json.dumps(averages,indent=2))
 if a.cohort=='dev':
  candidates=[x for x in averages if x['objective']=='hm_support' and x['step']==0 and x['arm'] in ['ensemble','median','q75']];priority=['median','ensemble','q75'];candidates.sort(key=lambda x:(-(x['aiou'] if x['aiou'] is not None else -1),priority.index(x['arm'])));chosen=candidates[0]
  decision={'state':'frozen_before_confirm','primary_candidate':'hm_support_k3_median_readout','primary_comparator':chosen['arm'],'selection_rule':'maximum dev aIOU among all3 no-update K5 ensembles; ties prefer median, then mean ensemble, then q75','all_three_frozen_comparators_reported_on_confirm':True,'candidate_hyperparameters_changed_from_prelock':False,'cohort_lock_sha256':digest(ROOT/'cohorts/COHORT_LOCK.json'),'mechanism_lock_sha256':digest(ROOT/'MECHANISM_LOCK.json'),'algorithm_sha256':digest(ROOT/'algorithm.py'),'dev_scored_sha256':digest(output),'confirm_replacement_allowed':False}
  destination=ROOT/'DEV_DECISION_FROZEN.json'
  if destination.exists():assert json.loads(destination.read_text())==decision,'Refuse to change an already frozen confirmation decision'
  else:destination.write_text(json.dumps(decision,indent=2))
if __name__=='__main__':main()
