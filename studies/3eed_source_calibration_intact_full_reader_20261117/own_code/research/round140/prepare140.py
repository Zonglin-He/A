"""Fixed threshold factorial on cached complete frozen A/B, no target GT."""
from common140 import *
import numpy as np,collections
sys.path.insert(0,str(R/'research/round136'));from prepare136 import direct,soft
from common140 import R,H,B,rd,sha,dump,check,utc
LEVELS=['oldreader_ownsource','fullreader_pooled','fullreader_ownsource']
def main():
 check();kappas=rd(H/'SOURCE_KAPPAS_LOCK.json');reg=rd(R/'research/round135/INPUT_REGISTRY_PRIVATE.json');out=B/'evidence_PRIVATE';out.mkdir(exist_ok=False);counts=collections.Counter();engineering=[]
 for e in reg:
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;old=np.load(R/'crossplatform/round135/formal'/name/'evidence_PRIVATE'/(sid+'.npz'));base=np.load(R/'crossplatform/round136/evidence_PRIVATE'/name/(sid+'.npz'));oldpred=np.load(R/'crossplatform/round136/method'/name/'samples_PRIVATE'/(sid+'.npz'));p0=dict(zip(oldpred['arm_names'],oldpred['scores']))['Frozen_A'];ass=old['assignment'];groups=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];S=old['S'];hasB=rd(R/'crossplatform/round135/formal'/name/'records_PRIVATE'/(sid+'.json'))['has_B'];a={'assignment':ass,'S':S,'pA':old['pA'],'pB':old['pB']};info={};anychange=False
  for level in LEVELS:
   k=kappas[level][src];C=np.zeros_like(S)
   if hasB and S.sum()>=2:
    for prob in [old['pA'],old['pB']]:
     lp=np.log(np.maximum([prob[g].mean() for g in groups],1e-300))[S];C[S]|=lp.max()-lp<=k
   eligible=bool(hasB and S.sum()>=2 and C.any());target=soft(np.array([np.log(max(float(old['pB'][g].max()),1e-300)) for g in groups])[C]) if eligible else np.zeros(int(C.sum()));perm=np.arange(len(target));cids=np.flatnonzero(C);rng=np.random.default_rng(int(hashlib.sha256((src+'|'+sid+'|R136_Cconditional').encode()).hexdigest()[:16],16))
   for n in sorted({len(groups[g]) for g in cids}):
    jj=np.array([i for i,g in enumerate(cids) if len(groups[g])==n],int);order=rng.permutation(jj)
    if len(order)>1:perm[order]=np.roll(order,1)
   wrong=target[perm];effective=bool(eligible and np.max(abs(target-wrong),initial=0)>1e-12);same=bool(np.array_equal(C,base['C']) and np.array_equal(target,base['target']) and np.array_equal(wrong,base['wrong_target']) and eligible==bool(base['eligible']));anychange|=eligible and not same
   for key,val in {'C':C,'target':target,'wrong_target':wrong,'permutation':perm,'eligible':np.array(eligible),'effective_wrong':np.array(effective),'direct':direct(p0,groups,S,C,target) if eligible else p0.copy(),'wrong_direct':direct(p0,groups,S,C,wrong) if eligible else p0.copy()}.items():a[level+'_'+key]=val
   info[level]={'kappa':k,'eligible':eligible,'C_groups':int(C.sum()),'partial_active':bool((S&~C).any() and C.any()),'same_R136_evidence':same,'effective_wrong':effective};counts[level+'_eligible']+=eligible;counts[level+'_same_R136']+=same;counts[level+'_changed_evidence']+=eligible and not same
  D=out/name;D.mkdir(exist_ok=True);np.savez_compressed(D/(sid+'.npz'),**a);dump(D/(sid+'.json'),info)
  if hasB and not any(x['source']==src and x['target']==tgt for x in engineering) and anychange:engineering.append(e)
 assert len(engineering)==6
 dump(H/'INPUT_REGISTRY_PRIVATE.json',reg);dump(H/'ENGINEERING_INPUTS_PRIVATE.json',engineering);dump(H/'PREPARATION.json',dict(counts));dump(B/'EVIDENCE_SEALED.json',{'n':len(reg),'source_kappas_sha256':sha(H/'SOURCE_KAPPAS_LOCK.json'),'target_GT_read':False,'files':{str(p.relative_to(B)):sha(p) for p in out.rglob('*') if p.is_file()}});print(dict(counts),flush=True)
if __name__=='__main__':main()
