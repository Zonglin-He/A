"""Freeze original-B log-gap labels from fixed source kappa; no GT or new model."""
from common132 import *
import numpy as np,sys,collections
sys.path.insert(0,str(R132/'research/round130'));from prepare130 import shuffle
def main():
 basecheck();cal=rd(H132/'SOURCE_LOG_CALIBRATION.json');assert cal['status']=='SOURCE_ONLY_AB_LOGRATIO_KAPPA_LOCKED';kappa=cal['kappa'];reg=rd(R132/'research/round131/INPUT_REGISTRY_PRIVATE.json');D=B132/'log_scopes_PRIVATE';D.mkdir(exist_ok=False);counts=collections.Counter()
 for e in reg:
  name=e['source']+'_to_'+e['target'];sid=e['row']['id'];f=np.load(R132/'crossplatform/round131/formal'/name/'features_PRIVATE'/(sid+'.npz'));S=f['S'];ass=f['assignment'];groups=[np.flatnonzero(ass==j) for j in range(int(ass.max()+1))];C=np.zeros_like(S)
  if S.sum()>=2:
   for v in ['A','B']:
    p=np.array([f['p'+v][g].mean() for g in groups])[S];x=np.log(np.maximum(p,1e-300));C[S]|=x.max()-x<=kappa
  Cw,swaps=shuffle(S,C,ass,e['source']+'|'+sid);active=bool(S.sum()>=2 and C.any() and (S&~C).any());identical=np.array_equal(C,f['C']);counts['active']+=active;counts['same_labels']+=identical;counts['new_task']+=active and not identical;counts['newly_active']+=active and np.array_equal(f['C'],S);counts['effective_wrong']+=active and bool(swaps);counts['C_equals_S']+=bool(S.sum()>=2 and np.array_equal(C,S));counts['n']+=1
  sub=D/name;sub.mkdir(exist_ok=True);np.savez_compressed(sub/(sid+'.npz'),S=S,C=C,wrong_C=Cw,old_C=f['C'],assignment=ass,teacher_conditional=f['teacher_conditional'],same_labels=np.array(identical),active=np.array(active))
 dump(H132/'LOG_SCOPE_SUMMARY.json',dict(counts)|{'kappa':kappa,'calibration_sha256':sha(H132/'SOURCE_LOG_CALIBRATION.json'),'target_GT_used':False,'forced_nonempty_negative':False});dump(B132/'LOG_SCOPES_SEALED.json',{'n':len(reg),'calibration_sha256':sha(H132/'SOURCE_LOG_CALIBRATION.json'),'files':{str(p.relative_to(D)):sha(p) for p in D.rglob('*.npz')}});print(dict(counts))
if __name__=='__main__':main()
