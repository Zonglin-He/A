"""Exposed DEV64 selects one common beta; R29 R64 and new S64 never read."""
import sys,numpy as np,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'crossplatform/round30'),str(ROOT/'research/round27')]
from common30 import *
from evaluate27 import ious

def main():
 out=BASE/'screen_attempt001';verify(out);fit=read(out/'FIT_RECEIPTS.json');summary=[];devrows=[]
 for beta in read(EXP/'PLAN_LOCK.json')['finite_beta_grid']:
  name='b'+str(beta).replace('.','p');pm={}
  for pl in ['drone','quad']:
   vals=[]
   for r in [r for r in rows('DEV64') if r['platform']==pl]:
    zz=load_npz(out/(pl+'_'+name+'_DEV')/'predictions_PRIVATE'/(r['id']+'.npz'));q=int(zz['query']);assert np.array_equal(zz['box'],feature(r)['boxes'][q]);v=float(ious(r,zz['box'][None])[0]);vals.append(v);devrows.append({'alias':r['alias'],'platform':pl,'scene_PRIVATE':r['scene'],'beta':beta,'iou':v,'q_PRIVATE':q})
   pm[pl]=float(np.mean(vals)*100)
  receipt=[x for x in fit if x['beta']==beta];Bpair=sum(x['B_pairs_satisfied'] for x in receipt);Bfinal=sum(x['B_final_positive'] for x in receipt);summary.append({'beta':beta,'name':name,'platform_mIoU_percent':pm,'balanced_mIoU_percent':float(np.mean(list(pm.values()))),'B_pairs_satisfied':Bpair,'B_final_positive':Bfinal,'eligible':bool(Bpair>=5 and Bfinal>=4)})
 eligible=[x for x in summary if x['eligible']];assert eligible,'R29beta0 should pass feasibility';best=sorted(eligible,key=lambda x:(-x['balanced_mIoU_percent'],x['beta']))[0];dump(EXP/'DEV64_ALL_OPTIONS.json',summary);dump(EXP/'DEV64_ROWS_PRIVATE.json',devrows);dump(EXP/'SELECTED_BETA_LOCK.json',{'selected':best,'criterion':read(EXP/'PLAN_LOCK.json')['selection'],'fit_prediction_seal_sha256':sha(out/'SEALED.json'),'selection_uses_exposed_DEV64_GT':True,'R29_R64_used_for_selection':False,'S64_labels_not_read':True,'selection_program_sha256':sha(__file__),'positive_betas_eligible':[x['beta'] for x in eligible if x['beta']>0]});print(json.dumps({'selected':best,'all':summary},indent=2))
if __name__=='__main__':main()
