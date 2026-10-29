"""Independent CPU scorer, callable only after prediction seal; official SIM/thresholds."""
import argparse,pathlib,json,hashlib
import numpy as np
from sklearn.metrics import roc_auc_score

def metrics(pred,target):
 pred=pred.reshape(-1);target=target.reshape(-1)
 # Official SIM() sum-normalizes both maps without min-max rescaling.
 p=pred/(pred.sum()+1e-12);t=target/(target.sum()+1e-12);sim=float(np.minimum(p,t).sum())
 truth=target>=.5;values=[]
 for threshold in np.linspace(0,1,20):
  mask=pred>=threshold;union=np.logical_or(mask,truth).sum();values.append(float(np.logical_and(mask,truth).sum()/union) if union else float('nan'))
 return {'auc':float(roc_auc_score(truth,pred)) if np.unique(truth).size==2 else float('nan'),'aiou':float(np.nanmean(values)) if truth.any() else float('nan'),'sim':sim,'mae':float(np.abs(pred-target).mean())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=pathlib.Path);p.add_argument('--targets',required=True,type=pathlib.Path);a=p.parse_args()
 seal=json.loads((a.run/'SEALED.json').read_text());meta=json.loads((a.run/'protocol.json').read_text());assert meta['state']=='sealed'
 for file,digest in seal.items():assert hashlib.file_digest(open(a.run/file,'rb'),'sha256').hexdigest()==digest
 rows=json.loads(a.targets.read_text());result=[];undefined=[];reference={}
 reasons={};valid_masks={k:[] for k in ['auc','aiou','sim','mae']}
 for i,row in enumerate(rows):
  target=np.load(row['target'],allow_pickle=False);assert np.isfinite(target).all()
  truth=target.reshape(-1)>=.5
  valid={'auc':np.unique(truth).size==2,'aiou':bool(truth.any()),'sim':True,'mae':True}
  for metric,ok in valid.items():valid_masks[metric].append(bool(ok))
  why={}
  if not valid['auc']:why['auc']='all_zero_binary_target' if not truth.any() else 'all_one_binary_target'
  if not valid['aiou']:why['aiou']='all_zero_binary_target'
  if why:undefined.append({'id':row['id'],'reason':why})
  for objective in meta['objectives']:
   for step in meta['steps']:
    if objective=='entropy' and step==0:continue
    data=np.load(a.run/f'{i:04d}_{objective}_k{step}.npz',allow_pickle=False)
    for arm in ['source','ensemble','adapted']:
     assert np.isfinite(data[arm]).all(),f'Nonfinite prediction: {i}/{objective}/{step}/{arm}'
     if arm in ['source','ensemble']:
      key=(i,arm)
      if key in reference:assert np.array_equal(reference[key],data[arm]),f'Cross-arm repeat mismatch: {key}'
      else:reference[key]=data[arm].copy()
     result.append({'id':row['id'],'objective':objective,'step':step,'arm':arm,**metrics(data[arm],target)})
 averages=[]
 for objective in meta['objectives']:
  for step in meta['steps']:
   if objective=='entropy' and step==0:continue
   for arm in ['source','ensemble','adapted']:
    subset=[r for r in result if r['objective']==objective and r['step']==step and r['arm']==arm];averages.append({'objective':objective,'step':step,'arm':arm,'n':len(subset),**{k:float(np.mean([r[k] for r,ok in zip(subset,valid_masks[k]) if ok])) if any(valid_masks[k]) else None for k in ['auc','aiou','sim','mae']},'valid_counts':{k:sum(v) for k,v in valid_masks.items()},'undefined_counts':{k:len(v)-sum(v) for k,v in valid_masks.items()}})
 (a.run/'SCORED.json').write_text(json.dumps({'rows':result,'averages':averages,'seal_verified':True,'source_and_ensemble_cross_arm_identical':True,'fixed_target_valid_masks':valid_masks,'undefined_samples':undefined,'measurement':'valid_for_declared_cohort'},indent=2));print(json.dumps(averages,indent=2))
if __name__=='__main__':main()
