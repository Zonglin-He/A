"""GT-free frozen C-conditional B supervision and exact direct-use controls."""
from common136 import *
import numpy as np,collections
def soft(x):e=np.exp(x-x.max());return e/e.sum()
def direct(prob,groups,S,C,target):
 p=np.array(prob,float).copy();ids=np.concatenate([groups[i] for i in np.flatnonzero(S)]) if S.any() else np.array([],int)
 if S.sum()<2 or not C.any():return p
 if (S&~C).any():
  cc=np.concatenate([groups[i] for i in np.flatnonzero(C)]);mass=p[ids].sum();p[ids]=0;p[cc]=np.asarray(prob)[cc]*mass/np.asarray(prob)[cc].sum()
 cc=np.concatenate([groups[i] for i in np.flatnonzero(C)]);mass=p[cc].sum();old=np.array([np.log(max(float(p[g].max()),1e-300)) for g in groups])[C];new=np.log(np.maximum(target,1e-300));shift=new-old
 for i,d in zip(np.flatnonzero(C),shift):p[groups[i]]*=np.exp(d)
 p[cc]*=mass/p[cc].sum();assert np.isfinite(p).all() and np.allclose(p.sum(),1,atol=1e-12);return p
def main():
 check();reg=rd(R136/'research/round135/INPUT_REGISTRY_PRIVATE.json');out=B136/'evidence_PRIVATE';out.mkdir(exist_ok=False);counts=collections.Counter();eng=[]
 for e in reg:
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;D=R136/'crossplatform/round135/formal'/name;ev=np.load(D/'evidence_PRIVATE'/(sid+'.npz'));rec=rd(D/'records_PRIVATE'/(sid+'.json'));ass=ev['assignment'];groups=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];S,C=ev['S'],ev['C'];eligible=bool(rec['has_B'] and S.sum()>=2 and C.any());target=soft(np.array([np.log(max(float(ev['pB'][g].max()),1e-300)) for g in groups])[C]) if eligible else np.zeros(int(C.sum()));perm=np.arange(len(target));cids=np.flatnonzero(C);rng=np.random.default_rng(int(hashlib.sha256((src+'|'+sid+'|R136_Cconditional').encode()).hexdigest()[:16],16))
  for n in sorted({len(groups[g]) for g in cids}):
   jj=np.array([i for i,g in enumerate(cids) if len(groups[g])==n],int);order=rng.permutation(jj)
   if len(order)>1:perm[order]=np.roll(order,1)
  wrong=target[perm];effective=bool(eligible and np.max(abs(target-wrong),initial=0)>1e-12);a=np.load(D/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(a['arm_names'],a['scores']));p0=ps['intact_live_A'];pd=direct(p0,groups,S,C,target) if eligible else p0.copy();wd=direct(p0,groups,S,C,wrong) if eligible else p0.copy();sub=out/name;sub.mkdir(exist_ok=True);np.savez_compressed(sub/(sid+'.npz'),assignment=ass,S=S,C=C,target=target,wrong_target=wrong,permutation=perm,eligible=np.array(eligible),effective_wrong=np.array(effective),direct=pd,wrong_direct=wd);counts['n']+=1;counts['eligible']+=eligible;counts['effective_wrong']+=effective;counts['C_singleton']+=eligible and C.sum()==1;counts['old_inactive_now_qualified']+=eligible and not rec['active']
 for name in sorted({e['source']+'_to_'+e['target'] for e in reg}):
  rr=[e for e in reg if e['source']+'_to_'+e['target']==name]
  for active in [True,False]:
   for e in rr:
    ev=np.load(out/name/(e['row']['id']+'.npz'))
    if bool(ev['eligible'])==active:eng.append(name+'|'+e['row']['id']);break
 dump(H136/'INPUT_REGISTRY_PRIVATE.json',reg);dump(H136/'ENGINEERING_KEYS_PRIVATE.json',eng);dump(H136/'PREPARATION.json',{k:int(v) for k,v in counts.items()}|{'GT_read':False,'teacher_scope':'B group-max conditional over fixed qualified C','unknown_C_complement_not_in_rank_loss':True});dump(B136/'EVIDENCE_SEALED.json',{'n':len(reg),'files':{str(p.relative_to(R136)):sha(p) for p in out.rglob('*.npz')},'GT_read':False});print(dict(counts),flush=True)
if __name__=='__main__':main()
