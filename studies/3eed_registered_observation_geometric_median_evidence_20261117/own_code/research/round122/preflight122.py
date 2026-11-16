from pathlib import Path
import sys,json,hashlib,datetime,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round122';sys.path.insert(0,str(R/'research/round118'));import core118 as C
from aggregate122 import solve122
rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 rows=[];summary=[]
 for co in ['DEV','EXP']:
  for src in ['waymo','drone','quad']:
   for tgt in ['waymo','drone','quad']:
    if src==tgt:continue
    name=src+'_to_'+tgt
    for r in C.rows(co,tgt):
     sid=r['id'];prior=R/f'crossplatform/round118/formal/{co}/{name}/records_PRIVATE/{sid}.json';alias=rd(prior)['protected_alias_fallback'];v={'cohort':co,'source':src,'target':tgt,'row':r,'protected_alias_fallback':alias,'source_alias_proof_sha256':sha(prior)}
     if not alias:
      D=R/('crossplatform/round111' if co=='DEV' else 'crossplatform/round117/formal')/name;pf=D/'prefix_PRIVATE'/(sid+'.npz');of=D/'observations_PRIVATE'/(sid+'.npz')
      with np.load(pf) as z,np.load(of) as zz:
       A=z['vis0'][0];obs=[]
       for view in ['B','C']:
        pre=('real_' if co=='DEV' else '')+view+'_'
        if pre+'vis' in zz:obs.append({k:zz[pre+k] for k in ['vis','valid']})
       assert A.shape==(1024,288) and all(o['vis'].shape==A.shape and o['valid'].shape==(1024,) for o in obs);rob,w,triple,history=solve122(A,obs);v.update(prefix=str(pf.relative_to(R)),observations=str(of.relative_to(R)),prefix_sha256=sha(pf),observations_sha256=sha(of),observations_available=len(obs),three_valid_seeds=int(triple.sum()),robust_minus_mean_L2=float(np.linalg.norm(rob.astype(float)-(np.stack([A]+[o['vis'] for o in obs]).astype(float)*(np.stack([np.ones(1024,bool)]+[o['valid'] for o in obs])/np.stack([np.ones(1024,bool)]+[o['valid'] for o in obs]).sum(0)[None])[:,:,None]).sum(0))),weight_nonuniform_seed_n=int((np.max(abs(w-1/3),axis=0)>1e-6)[triple].sum()) if len(obs)==2 else 0,objective_means=history)
     rows.append(v)
    rr=[x for x in rows if x['cohort']==co and x['source']==src and x['target']==tgt];ss=[x for x in rr if not x['protected_alias_fallback']];summary.append({'cohort':co,'source':src,'target':tgt,'n':len(rr),'protected_aliases':len(rr)-len(ss),'two_extra_observations_inputs':sum(x['observations_available']==2 for x in ss),'some_triple_supported_inputs':sum(x['three_valid_seeds']>0 for x in ss),'total_triple_supported_seeds':sum(x['three_valid_seeds'] for x in ss),'nonuniform_weight_seed_n':sum(x['weight_nonuniform_seed_n'] for x in ss)});print('PREFLIGHT122',co,name,summary[-1],flush=True)
 assert len(rows)==2304 and sum(x['protected_alias_fallback'] for x in rows)==154;out={'state':'PASS_EXISTING_SENSOR_CACHE_TRIPLE_SUPPORT_AND_ROBUST_AGGREGATOR_PREREQUISITE','n':2304,'groups':summary,'model_GT_or_media_calls':0,'FINAL_access':False};(H/'CACHE_REGISTRY_PRIVATE.json').write_text(json.dumps(rows,indent=2)+'\n');(H/'PREFLIGHT.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()
