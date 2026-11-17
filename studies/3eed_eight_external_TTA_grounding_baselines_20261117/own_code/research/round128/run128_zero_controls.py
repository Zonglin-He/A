"""Matched observation-budget zero-gradient controls, fixed before scoring."""
from core128 import *
def main():
 strict97();reg=[e for e in jread(R/'research/round122/CACHE_REGISTRY_PRIVATE.json') if e['cohort']=='DEV'];seals={}
 for src in ['waymo','drone','quad']:
  base=base97(src);S=snapshot(base)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;D=B/'formal_controls'/name
   for sub in ['samples_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
   rr=[e for e in reg if e['source']==src and e['target']==tgt]
   for j,e in enumerate(rr):
    sid=e['row']['id'];a=load(B/f'formal_fast/{name}/samples_PRIVATE/{sid}.npz');old=dict(zip(a['arm_names'],a['scores']));out={'frozen_four_view_mean':(old['source']+3*old['frozen_three_augment_mean'])/4,'memo_BNprior_four_view_mean':(old['memo_BNprior_zero']+3*old['memo_BNprior_three_mean'])/4};tick=time.perf_counter()
    if e['protected_alias_fallback']:out['BN_four_view_mean']=old['source'].copy();out['BN_three_view_mean']=old['source'].copy()
    else:
     reset(base,S);norm(base);z=load(R/e['prefix']);pc=points(e['row'],z)
     with torch.no_grad():pp=[cpu(native(base,v,e['row'],z)[0]) for v in views(pc,sid)]
     out['BN_three_view_mean']=np.mean(pp,axis=0);out['BN_four_view_mean']=(old['BN_statistics_only']+np.sum(pp,axis=0))/4
    keys=sorted(out);arr=np.stack([out[k] for k in keys]);assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-8);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(keys),scores=arr,queries=arr.argmax(1),source_boxes=a['source_boxes']);jdump(D/'records_PRIVATE'/(sid+'.json'),{'protected_alias_fallback':e['protected_alias_fallback'],'optimizer_steps':0,'parameter_updates':0,'seconds':time.perf_counter()-tick})
    if j%32==0:print('R128_ZERO',name,j+1,flush=True)
   jdump(D/'SEALED.json',{'n':len(rr),'GT_used':False,'files':{str(f.relative_to(D)):sha(f) for f in D.rglob('*') if f.is_file() and f.name!='SEALED.json'}});seals[name]=sha(D/'SEALED.json')
  reset(base,S);del base,S;torch.cuda.empty_cache()
 jdump(H/'CONTROLS_MATRIX_SEALED.json',{'n':768,'direction_seals':seals,'FINAL_access':False})
if __name__=='__main__':main()
