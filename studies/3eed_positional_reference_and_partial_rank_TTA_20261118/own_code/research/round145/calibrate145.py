"""Own-source q90 using frozen F-point/A-image and F-point/B-image predictions."""
from common145 import *
import numpy as np,collections
def main():
 check();assert rd(H/'SOURCE_AUDIT.json')['status']=='PASS'
 for p,h in rd(H/'experiment/SOURCE_CODE_LOCK.json')['files'].items():assert sha(R/p)==h,p
 rows=[];gaps={};counts=collections.Counter()
 for e in rd(R/'research/round132/SOURCE_CALIBRATION_INPUTS_PRIVATE.json'):
  src,sid=e['entry']['source'],e['entry']['row']['id'];a=np.load(B/'source'/src/(sid+'.npz'));has=rd(B/'source'/src/(sid+'.json'))['has_B'];ass,S=a['assignment'],a['S'];gs=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];rec={'source':src,'id_PRIVATE':sid,'eligible':False,'view_gaps':[]}
  if has and S.sum()>=2:
   label=e['entry']['files']['label'];assert sha(R/label['path'])==label['sha256'];y=np.load(R/label['path'])['ious'];good=np.array([y[g].max()>.25 for g in gs])[S]
   if good.any():
    rec['eligible']=True;counts[src]+=1
    for tag in ['A','B']:
     lp=np.log(np.maximum([a['p'+tag][g].mean() for g in gs],1e-300))[S];gap=float(lp.max()-lp[good].max());rec['view_gaps'].append(gap);gaps.setdefault(src,[]).append(gap)
  prev=next(r for r in rd(R/'research/round140/SOURCE_CALIBRATION_ROWS_PRIVATE.json') if r['source']==src and r['id_PRIVATE']==sid);assert rec['eligible']==prev['eligible'];rows.append(rec)
 out={'status':'LOCKED_SOURCE_ONLY_IMAGE_READER','quantile':.9,'source_eligible_counts':dict(counts),'kappa':{s:float(np.quantile(g,.9)) for s,g in gaps.items()},'source_labels_used':True,'target_GT_parameter_choice':False,'source_weight_training':False,'source_reader':'R144positional_reference_Fpoints_Aimage_and_Fpoints_Bimage_full_native_competition','source_prediction_seal_sha256':sha(B/'source/SEALED.json'),'FINAL_access':False,'utc':utc()};dump(H/'SOURCE_CALIBRATION_ROWS_PRIVATE.json',rows);dump(H/'SOURCE_KAPPAS_LOCK.json',out);print(out,flush=True)
if __name__=='__main__':main()
