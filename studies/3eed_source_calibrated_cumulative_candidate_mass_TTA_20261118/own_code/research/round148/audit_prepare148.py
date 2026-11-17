"""Independent source quantile and target tie-complete cumulative-set audit."""
from common148 import *
import numpy as np,collections
sys.path.insert(0,str(R/'research/round136'));from prepare136 import direct
from common148 import R,H,B,rd,sha,dump,check,utc

def cm(prob,groups,S):
 v=np.array([max(float(sum(float(prob[i]) for i in g)/len(g)),1e-300) for g in groups])[S];return v/v.sum()
def prefix(v,q):
 selected=np.zeros(len(v),bool);total=0.
 for p in sorted(set(v.tolist()),reverse=True):
  at=v==p;selected|=at;total=min(total+float(v[at].sum()),1.)
  if total>=q:break
 return selected

def main():
 check();source=rd(H/'SOURCE_MASS_LOCK.json');sv=collections.defaultdict(list);nsource=0
 for e in rd(R/'research/round132/SOURCE_CALIBRATION_INPUTS_PRIVATE.json'):
  s,sid=e['entry']['source'],e['entry']['row']['id'];a=np.load(R/'crossplatform/round145/source'/s/(sid+'.npz'));S=a['S'];ass=a['assignment'];gs=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];has=rd(R/'crossplatform/round145/source'/s/(sid+'.json'))['has_B'];nsource+=1
  if has and S.sum()>=2:
   p=e['entry']['files']['label'];assert sha(R/p['path'])==p['sha256'];y=np.load(R/p['path'])['ious'];good=np.array([max(y[g])>.25 for g in gs])[S]
   if good.any():
    for t in ['A','B']:
     v=cm(a['p'+t],gs,S);order=sorted(range(len(v)),key=lambda j:(-float(v[j]),j));total=0.
     for i in order:
      total=min(total+float(v[i]),1.)
      if good[i]:sv[s].append(total);break
 for s,ls in sv.items():assert abs(float(np.quantile(ls,.9))-source['q90'][s])<1e-12
 seal=rd(B/'EVIDENCE_SEALED.json');n=nc=0
 for p,h in seal['files'].items():assert sha(B/p)==h
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;old=np.load(R/'crossplatform/round145/evidence_PRIVATE'/name/(sid+'.npz'));ev=np.load(B/'evidence_PRIVATE'/name/(sid+'.npz'));ass,S=ev['assignment'],ev['S'];assert np.array_equal(ass,old['assignment']) and np.array_equal(S,old['S']);gs=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];C=np.zeros_like(S)
  for k in ['pA','pB','p0']:assert np.array_equal(ev[k],old[k])
  if e['has_B'] and S.sum()>=2:
   for k in ['pA','pB']:C[S]|=prefix(cm(ev[k],gs,S),source['q90'][src])
  assert np.array_equal(C,ev['C']);eligible=bool(e['has_B'] and S.sum()>=2 and C.any());assert eligible==bool(ev['eligible']);tar=np.array([ev['pB'][g].max() for g in gs])[C];tar=tar/tar.sum() if eligible else tar;assert np.allclose(tar,ev['target'],rtol=1e-12,atol=1e-15);ids=np.flatnonzero(C);perm=np.arange(len(tar));rng=np.random.default_rng(int(hashlib.sha256((src+'|'+sid+'|R142_Cconditional').encode()).hexdigest()[:16],16))
  for sz in sorted({len(gs[g]) for g in ids}):
   ii=np.array([j for j,g in enumerate(ids) if len(gs[g])==sz],int);order=rng.permutation(ii)
   if len(order)>1:perm[order]=np.roll(order,1)
  assert np.array_equal(perm,ev['permutation']) and np.array_equal(ev['wrong_target'],ev['target'][perm]);assert bool(ev['effective_wrong'])==bool(eligible and np.max(abs(ev['target']-ev['wrong_target']),initial=0)>1e-12);assert np.array_equal(ev['direct'],direct(ev['p0'],gs,S,C,ev['target']) if eligible else ev['p0']);assert np.array_equal(ev['wrong_direct'],direct(ev['p0'],gs,S,C,ev['wrong_target']) if eligible else ev['p0']);same=all(np.array_equal(ev[k],old[k]) for k in old.files);assert same==bool(ev['compatible_parent']);nc+=same;n+=1
  if not e['has_B']:assert not eligible and not C.any()
 assert nsource==313 and n==1798;dump(H/'PREPARATION_AUDIT.json',{'status':'PASS','source_n':nsource,'target_n':n,'compatible_parent':nc,'source_q90_independently_reconstructed':True,'target_C_teacher_wrong_direct_reconstructed':True,'tie_complete_prefix':True,'no_target_GT_read':True,'new_models':0,'FINAL_access':False});print(n,nc,flush=True)
if __name__=='__main__':main()
