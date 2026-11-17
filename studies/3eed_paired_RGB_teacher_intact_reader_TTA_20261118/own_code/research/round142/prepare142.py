"""Conditional phase2 evidence, no target GT, calibrated native image reader."""
from common142 import *
import numpy as np,collections
sys.path.insert(0,str(R/'research/round136'));from prepare136 import direct,soft
from common142 import R,H,B,rd,sha,dump,check,utc
def main():
 check();assert rd(H/'IMAGE_DECISION.json')['passed'] and rd(H/'SOURCE_AUDIT.json')['status']=='PASS';ks=rd(H/'SOURCE_KAPPAS_LOCK.json');reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');out=B/'evidence_PRIVATE';out.mkdir(exist_ok=False);counts=collections.Counter();eng=[]
 for e in reg:
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;old=np.load(R/'crossplatform/round135/formal'/name/'evidence_PRIVATE'/(sid+'.npz'));a=np.load(B/'image'/name/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(a['arm_names'],a['scores']));ass,S=old['assignment'],old['S'];gs=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];pA,pB,p0=ps['Frozen_F'],ps['F_Bimage_only_teacher'],ps['F_RGBB_only'];C=np.zeros_like(S);k=ks['kappa'][src]
  if e['has_B'] and S.sum()>=2:
   for prob in [pA,pB]:
    lp=np.log(np.maximum([prob[g].mean() for g in gs],1e-300))[S];C[S]|=lp.max()-lp<=k
  eligible=bool(e['has_B'] and S.sum()>=2 and C.any());target=soft(np.array([np.log(max(float(pB[g].max()),1e-300)) for g in gs])[C]) if eligible else np.zeros(int(C.sum()));perm=np.arange(len(target));cids=np.flatnonzero(C);rng=np.random.default_rng(int(hashlib.sha256((src+'|'+sid+'|R142_Cconditional').encode()).hexdigest()[:16],16))
  for n in sorted({len(gs[g]) for g in cids}):
   jj=np.array([i for i,g in enumerate(cids) if len(gs[g])==n],int);order=rng.permutation(jj)
   if len(order)>1:perm[order]=np.roll(order,1)
  wrong=target[perm];effective=bool(eligible and np.max(abs(target-wrong),initial=0)>1e-12);dd=out/name;dd.mkdir(exist_ok=True);np.savez_compressed(dd/(sid+'.npz'),assignment=ass,S=S,C=C,pA=pA,pB=pB,p0=p0,target=target,wrong_target=wrong,permutation=perm,eligible=np.array(eligible),effective_wrong=np.array(effective),direct=direct(p0,gs,S,C,target) if eligible else p0.copy(),wrong_direct=direct(p0,gs,S,C,wrong) if eligible else p0.copy());counts['n']+=1;counts['eligible']+=eligible;counts['effective_wrong']+=effective;counts['C_singleton']+=eligible and C.sum()==1;counts['partial_active']+=eligible and (S&~C).any()
 for src,tgt in sorted({(e['source'],e['target']) for e in reg}):
  rr=[e for e in reg if (e['source'],e['target'])==(src,tgt)]
  for active in [True,False]:
   for e in rr:
    ev=np.load(out/(src+'_to_'+tgt)/(e['row']['id']+'.npz'))
    if bool(ev['eligible'])==active:eng.append(e);break
 dump(H/'TTA_ENGINEERING_INPUTS_PRIVATE.json',eng);dump(H/'PREPARATION.json',{k:int(v) for k,v in counts.items()}|{'GT_read':False,'point_input':'same frozenF in FAimage/FBimage/ABimage','teacher':'frozenBimageonly groupMAX|C','source_kappas_sha256':sha(H/'SOURCE_KAPPAS_LOCK.json')});dump(B/'EVIDENCE_SEALED.json',{'n':len(reg),'target_GT_read':False,'files':{str(p.relative_to(B)):sha(p) for p in out.rglob('*.npz')}});print(dict(counts),flush=True)
if __name__=='__main__':main()
