"""New cumulativeC, all other R145 evidence identical; no target GT."""
from common148 import *
from mass148 import *
import collections
sys.path.insert(0,str(R/'research/round136'));from prepare136 import direct,soft
from common148 import R,H,B,rd,sha,dump,check,utc

def main():
 check();qs=rd(H/'SOURCE_MASS_LOCK.json')['q90'];reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');out=B/'evidence_PRIVATE';out.mkdir(exist_ok=False);seal=rd(R/'crossplatform/round145/EVIDENCE_SEALED.json')['files'];counts=collections.Counter();rows=[];eng=[]
 for e in reg:
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;oldpath=R/'crossplatform/round145/evidence_PRIVATE'/name/(sid+'.npz');assert sha(oldpath)==seal[str(oldpath.relative_to(R/'crossplatform/round145'))];old=np.load(oldpath);ass,S=old['assignment'],old['S'];pA,pB,p0=old['pA'],old['pB'],old['p0'];gs=[np.flatnonzero(ass==g) for g in range(int(ass.max()+1))];C=np.zeros_like(S)
  if e['has_B'] and S.sum()>=2:
   for p in [pA,pB]:C[S]|=mass_set(conditional_mass(p,gs,S),qs[src])
  eligible=bool(e['has_B'] and S.sum()>=2 and C.any());target=soft(np.array([np.log(max(float(pB[g].max()),1e-300)) for g in gs])[C]) if eligible else np.zeros(int(C.sum()));perm=np.arange(len(target));cids=np.flatnonzero(C);rng=np.random.default_rng(int(hashlib.sha256((src+'|'+sid+'|R142_Cconditional').encode()).hexdigest()[:16],16))
  for n in sorted({len(gs[g]) for g in cids}):
   jj=np.array([i for i,g in enumerate(cids) if len(gs[g])==n],int);order=rng.permutation(jj)
   if len(order)>1:perm[order]=np.roll(order,1)
  wrong=target[perm];effective=bool(eligible and np.max(abs(target-wrong),initial=0)>1e-12);vals={'assignment':ass,'S':S,'C':C,'pA':pA,'pB':pB,'p0':p0,'target':target,'wrong_target':wrong,'permutation':perm,'eligible':np.array(eligible),'effective_wrong':np.array(effective),'direct':direct(p0,gs,S,C,target) if eligible else p0.copy(),'wrong_direct':direct(p0,gs,S,C,wrong) if eligible else p0.copy()};compatible=all(np.array_equal(v,old[k]) for k,v in vals.items());dd=out/name;dd.mkdir(exist_ok=True);np.savez_compressed(dd/(sid+'.npz'),**vals,compatible_parent=np.array(compatible));counts['n']+=1;counts['eligible']+=eligible;counts['changed_C']+=not np.array_equal(C,old['C']);counts['compatible_parent']+=compatible;counts['new_correct_fits']+=eligible and not compatible;counts['new_wrong_fits']+=eligible and effective and not compatible;counts['new_C_equal_S']+=eligible and np.array_equal(C,S);counts['partial_active']+=eligible and bool((S&~C).any())
  bm=np.array([pB[g].max() for g in gs]);leader=int(np.flatnonzero(S)[bm[S].argmax()]) if S.any() else -1;rows.append({'source':src,'target':tgt,'id_PRIVATE':sid,'eligible':eligible,'compatible':compatible,'changed_C':not np.array_equal(C,old['C']),'old_C_fraction':float(old['C'].sum()/max(S.sum(),1)),'new_C_fraction':float(C.sum()/max(S.sum(),1)),'S_groups':int(S.sum()),'old_B_MAX_leader_retained':bool(old['C'][leader]) if leader>=0 else False,'new_B_MAX_leader_retained':bool(C[leader]) if leader>=0 else False,'teacher_top_group_changed':bool(eligible and np.flatnonzero(C)[target.argmax()]!=np.flatnonzero(old['C'])[old['target'].argmax()])})
 for src,tgt in sorted({(e['source'],e['target']) for e in reg}):
  picked=[];rr=[e for e in reg if (e['source'],e['target'])==(src,tgt)]
  for kind in ['changed','compatibleB','noB']:
   for e in rr:
    ev=np.load(out/(src+'_to_'+tgt)/(e['row']['id']+'.npz'));ok=(bool(ev['eligible']) and not bool(ev['compatible_parent'])) if kind=='changed' else (e['has_B'] and bool(ev['compatible_parent'])) if kind=='compatibleB' else not e['has_B']
    if ok:
     if e['row']['id'] not in picked:eng.append(e);picked.append(e['row']['id'])
     break
 stats=[]
 for src,tgt in sorted({(e['source'],e['target']) for e in reg}):
  rr=[x for x in rows if (x['source'],x['target'])==(src,tgt) and x['eligible']];stats.append({'source':src,'target':tgt,'n_B':len(rr),'changed_C':sum(x['changed_C'] for x in rr),'mean_old_C_fraction':float(np.mean([x['old_C_fraction'] for x in rr])),'mean_new_C_fraction':float(np.mean([x['new_C_fraction'] for x in rr])),'old_B_MAX_leader_missing':sum(not x['old_B_MAX_leader_retained'] for x in rr),'new_B_MAX_leader_missing':sum(not x['new_B_MAX_leader_retained'] for x in rr),'teacher_top_group_changed':sum(x['teacher_top_group_changed'] for x in rr)})
 dump(H/'PREPARATION.json',{k:int(v) for k,v in counts.items()}|{'GT_read':False,'source_weight_training':False,'new_models':0,'summary':stats,'prerequisite_nontrivial':bool(counts['changed_C']>0 and counts['new_correct_fits']>0)});dump(H/'EVIDENCE_ROWS_PRIVATE.json',rows);dump(H/'ENGINEERING_INPUTS_PRIVATE.json',eng);dump(B/'EVIDENCE_SEALED.json',{'n':len(reg),'GT_read':False,'files':{str(p.relative_to(B)):sha(p) for p in out.rglob('*.npz')}});print(dict(counts),stats,flush=True)
if __name__=='__main__':main()
