"""Prespecified source-own q.9 calibration, before any new target predictions."""
from common140 import *
import numpy as np,collections
def main():
 check();assert rd(H/'SOURCE_AUDIT.json')['status']=='PASS';reg=rd(R/'research/round132/SOURCE_CALIBRATION_INPUTS_PRIVATE.json');oldrows=rd(R/'research/round132/SOURCE_CALIBRATION_ROWS_PRIVATE.json');rows=[];gaps={};eligible=collections.Counter()
 for e in reg:
  src,sid=e['entry']['source'],e['entry']['row']['id'];a=np.load(B/'source'/src/(sid+'.npz'));S=a['S'];ass=a['assignment'];groups=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];rec={'source':src,'id_PRIVATE':sid,'eligible':False,'view_gaps':[]};hasB=rd(B/'source'/src/(sid+'.json'))['has_B']
  if hasB and S.sum()>=2:
   label=e['entry']['files']['label'];assert sha(R/label['path'])==label['sha256'];y=np.load(R/label['path'])['ious'];good=np.array([y[g].max()>.25 for g in groups])[S]
   if good.any():
    rec['eligible']=True;eligible[src]+=1
    for v in ['A','B']:
     lp=np.log(np.maximum([a['p'+v][g].mean() for g in groups],1e-300))[S];gap=float(lp.max()-lp[good].max());rec['view_gaps'].append(gap);gaps.setdefault(src,[]).append(gap)
  prev=next(r for r in oldrows if r['source']==src and r['id_PRIVATE']==sid);assert rec['eligible']==prev['eligible'];rows.append(rec)
 old={s:[g for r in oldrows if r['source']==s for g in r['view_gaps']] for s in ['waymo','drone','quad']};q=lambda a:float(np.quantile(a,.9));oldpool=q(sum(old.values(),[]));assert abs(oldpool-rd(R/'research/round132/SOURCE_LOG_CALIBRATION.json')['kappa'])<1e-14
 out={'status':'LOCKED_SOURCE_ONLY_NOT_TARGET_SELECTED','quantile':.9,'source_eligible_counts':dict(eligible),'oldreader_pooled':{s:oldpool for s in old},'oldreader_ownsource':{s:q(x) for s,x in old.items()},'fullreader_pooled':{s:q(sum(gaps.values(),[])) for s in gaps},'fullreader_ownsource':{s:q(x) for s,x in gaps.items()},'primary':'fullreader_ownsource','source_labels_used':True,'target_predictions_or_GT_read':False,'source_training':False,'no_target_best_level_selection':True,'source_predictions_seal_sha256':sha(B/'source/SEALED.json'),'FINAL_access':False,'utc':utc()};dump(H/'SOURCE_CALIBRATION_ROWS_PRIVATE.json',rows);dump(H/'SOURCE_KAPPAS_LOCK.json',out);print(out,flush=True)
if __name__=='__main__':main()
