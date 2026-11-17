"""Fixed R145 evidence reused except for registered late-B additions; no GT."""
from common146 import *
import numpy as np,collections,shutil
sys.path.insert(0,str(R/'research/round136'));from prepare136 import direct,soft
from common146 import R,H,B,rd,sha,dump,check,utc

def main():
 check();assert rd(H/'IMAGE_AUDIT.json')['status']=='PASS'
 for f,h in rd(H/'experiment/EVIDENCE_PREPARATION_CODE_LOCK.json')['files'].items():assert sha(R/f)==h,f
 ks=rd(H/'SOURCE_KAPPAS_LOCK.json');reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');out=B/'evidence_PRIVATE';out.mkdir(exist_ok=False);counts=collections.Counter();eng=[]
 for e in reg:
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;dd=out/name;dd.mkdir(exist_ok=True);dest=dd/(sid+'.npz')
  if not e['backfilled']:
   shutil.copyfile(R/'crossplatform/round145/evidence_PRIVATE'/name/(sid+'.npz'),dest);counts['exact_parent_evidence_reused']+=1
  else:
   old=np.load(B/'support_PRIVATE'/name/(sid+'.npz'));a=np.load(B/'image'/name/'samples_PRIVATE'/(sid+'.npz'));ps=dict(zip(a['arm_names'],a['scores']));ass,S=old['assignment'],old['S'];gs=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];pA,pB,p0=ps['bf_Aimage'],ps['bf_Bimage'],ps['bf_zero'];C=np.zeros_like(S)
   if e['has_B'] and S.sum()>=2:
    for prob in [pA,pB]:
     lp=np.log(np.maximum([prob[g].mean() for g in gs],1e-300))[S];C[S]|=lp.max()-lp<=ks['kappa'][src]
   eligible=bool(e['has_B'] and S.sum()>=2 and C.any());target=soft(np.array([np.log(max(float(pB[g].max()),1e-300)) for g in gs])[C]) if eligible else np.zeros(int(C.sum()));perm=np.arange(len(target));cids=np.flatnonzero(C);rng=np.random.default_rng(int(hashlib.sha256((src+'|'+sid+'|R142_Cconditional').encode()).hexdigest()[:16],16))
   for n in sorted({len(gs[g]) for g in cids}):
    jj=np.array([i for i,g in enumerate(cids) if len(gs[g])==n],int);order=rng.permutation(jj)
    if len(order)>1:perm[order]=np.roll(order,1)
   wrong=target[perm];effective=bool(eligible and np.max(abs(target-wrong),initial=0)>1e-12)
   np.savez_compressed(dest,assignment=ass,S=S,C=C,pA=pA,pB=pB,p0=p0,target=target,wrong_target=wrong,permutation=perm,eligible=np.array(eligible),effective_wrong=np.array(effective),direct=direct(p0,gs,S,C,target) if eligible else p0.copy(),wrong_direct=direct(p0,gs,S,C,wrong) if eligible else p0.copy());counts['fresh_evidence']+=1;counts['new_eligible']+=eligible;counts['new_effective_wrong']+=effective;counts['new_partial_active']+=eligible and (S&~C).any();counts['new_C_singleton']+=eligible and C.sum()==1
  ev=np.load(dest);counts['n']+=1;counts['total_eligible']+=bool(ev['eligible'])
 for src,tgt in sorted({(e['source'],e['target']) for e in reg}):
  rr=[e for e in reg if (e['source'],e['target'])==(src,tgt)];picked=[]
  for kind in ['backfilled','effective_wrong','original_has_B','remaining_noB']:
   for e in rr:
    ev=np.load(out/(src+'_to_'+tgt)/(e['row']['id']+'.npz'));ok=(e['backfilled'] and bool(ev['effective_wrong'])) if kind=='effective_wrong' else not e['has_B'] if kind=='remaining_noB' else e[kind]
    if ok:
     if e['row']['id'] not in picked:eng.append(e);picked.append(e['row']['id'])
     break
 dump(H/'TTA_ENGINEERING_INPUTS_PRIVATE.json',eng);dump(H/'TTA_PREPARATION.json',dict(counts)|{'GT_read':False,'new_source_calibration':False,'kappas_exact_R145':True,'source_kappas_sha256':sha(H/'SOURCE_KAPPAS_LOCK.json')});dump(B/'EVIDENCE_SEALED.json',{'n':len(reg),'target_GT_read':False,'files':{str(p.relative_to(B)):sha(p) for p in out.rglob('*.npz')}});print(dict(counts),flush=True)
if __name__=='__main__':main()
