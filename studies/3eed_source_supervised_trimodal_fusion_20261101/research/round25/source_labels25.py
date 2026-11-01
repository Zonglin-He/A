"""Source TRAIN/VAL candidate IoU labels. Explicitly refuses target platforms."""
import sys,time,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round25'),str(ROOT/'research/round5/audit')]
from common25 import *
from independent_core_v2 import candidate_ious

def main():
 verify(BASE/'source_features_attempt002');out=EXP/'source_labels_attempt002';out.mkdir(exist_ok=False);(out/'samples_PRIVATE').mkdir();gt=read(EXP.parent/'input/SOURCE_GT_PRIVATE.json');start=time.perf_counter();stats={}
 for split in ['S3682','SV512']:
  rr=rows(split);vv=[];orig=[]
  for r in rr:
   assert r['platform']=='waymo';z=feature(r);b=z['boxes'].copy();b[:,3:]=np.maximum(b[:,3:],1e-6);v=candidate_ious(b,np.array(gt[r['id']],np.float32))[0];np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'.npz'),ious=v);vv.append(v.max());orig.append(v[z['scores'].argmax()])
  stats[split]={'n':len(rr),'source_mIoU':float(np.mean(orig)*100),'candidate_oracle_mIoU':float(np.mean(vv)*100),'oracle25_percent':float(np.mean(np.array(vv)>.25)*100),'oracle50_percent':float(np.mean(np.array(vv)>.5)*100),'oracle_below05':int((np.array(vv)<.05).sum())};print('SOURCE_LABELS25',split,stats[split],flush=True)
 dump(out/'SUMMARY.json',stats);seal(out,'R25_source_only_supervised_matching_labels',n=4194,target_labels_read=0,seconds=time.perf_counter()-start);print('SOURCE_LABELS25_SEALED',flush=True)
if __name__=='__main__':main()
