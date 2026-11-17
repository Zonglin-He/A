"""Existing components and confidence-stratified C mismatch; no target labels."""
from common143 import *
import numpy as np,collections

def partial_direct(p0,groups,S,C):
 p=np.asarray(p0,float).copy()
 if S.sum()<2 or not C.any() or not (S&~C).any():return p
 si=np.concatenate([groups[g] for g in np.flatnonzero(S)]);ci=np.concatenate([groups[g] for g in np.flatnonzero(C)]);mass=p[si].sum();p[si]=0;p[ci]=p0[ci]*mass/p0[ci].sum();assert np.isfinite(p).all() and np.allclose(p.sum(),1,atol=1e-12);return p

def rank_direct(p0,groups,C,target):
 p=np.asarray(p0,float).copy()
 if C.sum()<2:return p
 ci=np.concatenate([groups[g] for g in np.flatnonzero(C)]);mass=p[ci].sum();old=np.array([p[g].max() for g in groups])[C];shift=np.log(np.maximum(target,1e-300))-np.log(np.maximum(old,1e-300))
 for g,d in zip(np.flatnonzero(C),shift):p[groups[g]]*=np.exp(d)
 p[ci]*=mass/p[ci].sum();assert np.isfinite(p).all() and np.allclose(p.sum(),1,atol=1e-12);return p

def wrong_membership(S,C,groups,p0,salt):
 perm=np.arange(len(groups));bit=np.full(len(groups),-1,int);ids=np.flatnonzero(S);means=np.maximum(np.array([p0[g].mean() for g in groups]),1e-300)
 if len(ids):bit[ids]=np.floor((np.log(means[ids].max())-np.log(means[ids]))/np.log(2)).astype(int)
 rng=np.random.default_rng(int(hashlib.sha256((salt+'|R143_Cconfidence').encode()).hexdigest()[:16],16))
 for size,b in sorted({(len(groups[g]),int(bit[g])) for g in ids}):
  jj=np.array([g for g in ids if len(groups[g])==size and bit[g]==b],int);order=rng.permutation(jj)
  if len(order)>1:perm[order]=np.roll(order,1)
 wrong=C[perm];assert wrong.sum()==C.sum() and not (wrong&~S).any();mass=float(means[C].sum()/means[S].sum()) if C.any() else 0.;wmass=float(means[wrong].sum()/means[S].sum()) if C.any() else 0.
 if C.any():assert .5-1e-12<=wmass/mass<=2+1e-12
 return wrong,perm,bit,mass,wmass

def main():
 check();reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');out=B/'evidence_PRIVATE';out.mkdir(exist_ok=False);counts=collections.Counter();records=[]
 for e in reg:
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;ev=dict(np.load(R/'crossplatform/round142/evidence_PRIVATE'/name/(sid+'.npz')));ass=ev['assignment'];groups=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];S,C=ev['S'],ev['C'];cw,perm,bit,mass,wmass=wrong_membership(S,C,groups,ev['p0'],src+'|'+sid);eligible=bool(ev['eligible']);pa=bool(eligible and (S&~C).any());ra=bool(eligible and C.sum()>=2);effective=bool(not np.array_equal(cw,C));ev.update(C_wrong_membership=cw,C_wrong_permutation=perm,confidence_bit=bit,partial_active=np.array(pa),rank_active=np.array(ra),C_wrong_effective=np.array(effective),true_C_conditional_group_mass=np.array(mass),wrong_C_conditional_group_mass=np.array(wmass),direct_partial=partial_direct(ev['p0'],groups,S,C),direct_wrong_partial=partial_direct(ev['p0'],groups,S,cw),direct_rank=rank_direct(ev['p0'],groups,C,ev['target']),direct_wrong_rank=rank_direct(ev['p0'],groups,C,ev['wrong_target']));D=out/name;D.mkdir(exist_ok=True);np.savez_compressed(D/(sid+'.npz'),**ev);counts['n']+=1;counts['eligible']+=eligible;counts['partial_active']+=pa;counts['rank_active']+=ra;counts['effective_wrong_C']+=effective;counts['new_partial_fits']+=pa;counts['new_wrong_partial_fits']+=pa and effective;counts['new_rank_fits']+=ra;counts['new_wrong_rank_fits']+=ra and bool(ev['effective_wrong']);records.append((e,eligible,pa,ra,effective))
 eng=[];keys=set()
 def add(e):
  k=(e['source'],e['target'],e['row']['id'])
  if k not in keys:eng.append(e);keys.add(k)
 for src,tgt in sorted({(e['source'],e['target']) for e in reg}):
  for flag in [True,False]:
   for e,eligible,pa,ra,eff in records:
    if (e['source'],e['target'])==(src,tgt) and eligible==flag:add(e);break
 for predicate in [lambda x:x[1] and not x[2],lambda x:x[1] and not x[3],lambda x:x[2] and x[4],lambda x:x[2] and not x[4]]:
  for row in records:
   if predicate(row):add(row[0]);break
 dump(H/'ENGINEERING_INPUTS_PRIVATE.json',eng);dump(H/'PREPARATION.json',{k:int(v) for k,v in counts.items()}|{'engineering_n':len(eng),'GT_read':False,'original_R142_S_C_teacher_and_kappas_unchanged':True});dump(B/'EVIDENCE_SEALED.json',{'n':len(reg),'GT_read':False,'files':{str(p.relative_to(B)):sha(p) for p in out.rglob('*.npz')}});print(dict(counts),'engineering',len(eng),flush=True)
if __name__=='__main__':main()
