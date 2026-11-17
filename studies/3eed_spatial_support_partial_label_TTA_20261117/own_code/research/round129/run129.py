from core129 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
def main():
 assert jread(H129/'SCOPE_SEALED.json')['n']==2304;reg=jread(R129/'research/round122/CACHE_REGISTRY_PRIVATE.json');scopeRows=jread(H129/'SCOPE_ROWS_PRIVATE.json');bykey={(e['cohort'],e['source'],e['target'],e['id_PRIVATE']):e for e in scopeRows};seals={};n=0;checks=[]
 for src in ['waymo','drone','quad']:
  base=base97(src);finger=fingerprint(base);m,S=model97(base,src)
  for co in ['DEV','EXP']:
   for tgt in ['waymo','drone','quad']:
    if src==tgt:continue
    name=src+'_to_'+tgt;rr=[e for e in reg if e['source']==src and e['target']==tgt and e['cohort']==co]
    if args.engineering:
     active=[e for e in rr if bykey[(co,src,tgt,e['row']['id'])]['partial_eligible']];cond=[e for e in rr if bykey[(co,src,tgt,e['row']['id'])]['scope_eligible']]
     off=[e for e in rr if not e['protected_alias_fallback'] and not bykey[(co,src,tgt,e['row']['id'])]['scope_eligible']];rr=[(active or cond or off)[0],(off or cond or active)[0]];rr=list({e['row']['id']:e for e in rr}.values())
    D=B129/('engineering' if args.engineering else 'formal')/co/name
    for sub in ['samples_PRIVATE','features_PRIVATE','states_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
    for i,e in enumerate(rr):
     sid=e['row']['id'];meta=bykey[(co,src,tgt,sid)];tick=time.perf_counter();features={};states={}
     if meta['scope_eligible']:new,features,states,rec=process129(base,m,S,e,args.engineering)
     else:new={};rec={'scope':meta,'traces':{},'writebacks':[],'scope_parameters':4896,'C_used_in_gradient_or_gate':False,'no_qualified_scope_no_update':True}
     # No held-out readout is opened before all updates for this input are fixed.
     old=np.load(R129/f'crossplatform/round127/formal/{co}/{name}/samples_PRIVATE/{sid}.npz');po=dict(zip(old['arm_names'],old['scores']));refs={k:po[k].copy() for k in ['original_source','prepared_A','legacy_ABC_context','zero_F','zero_C']}
     for v in ['F','C']:refs['R127_global_'+v]=po['fab_full_'+v].copy();refs['R127_ab_native_'+v]=po['ab_native_'+v].copy();refs['direct_global_'+v]=po['direct_full_'+v].copy()
     if not meta['scope_eligible']:
      for k in NEW129:new[k]=po['zero_'+k.rsplit('_',1)[1]].copy()
     assert set(new)==set(NEW129)|({'zero_F','zero_C'} if meta['scope_eligible'] else set())
     allout=refs|new;arr=np.stack([allout[k] for k in ARMS129]);assert arr.shape==(25,256) and np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-9)
     np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS129),scores=arr,queries=arr.argmax(1),source_boxes=old['source_boxes']);old.close();rec.update(seconds=time.perf_counter()-tick,protected_alias_fallback=e['protected_alias_fallback'])
     if features:np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**features)
     if states:np.savez_compressed(D/'states_PRIVATE'/(sid+'.npz'),**states)
     jdump(D/'records_PRIVATE'/(sid+'.json'),rec);n+=1
     if args.engineering:checks.append({'cohort':co,'source':src,'target':tgt,'scope_eligible':meta['scope_eligible'],'partial_eligible':meta['partial_eligible'],'finite_difference':rec.get('finite_difference',[]),'seconds':rec['seconds']})
     if args.engineering or i%32==0:print('RUN129',args.engineering,co,name,i+1,round(rec['seconds'],3),flush=True)
    jdump(D/'SEALED.json',{'n':len(rr),'task_GT_used':False,'files':{str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file() and p.name!='SEALED.json'}});seals[co+'/'+name]=sha(D/'SEALED.json')
  assert fingerprint(base)==finger and all(np.array_equal(v,state(m)[k]) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 if args.engineering:jdump(H129/'ENGINEERING.json',{'status':'PASS_REAL_SCOPE_PARTIAL_GUARD_OPTIMIZATION','n':n,'checks':checks})
 else:assert n==2304;jdump(B129/'MATRIX_SEALED.json',{'n':n,'arms':ARMS129,'primary':PRIMARY129,'seals':seals,'FINAL_access':False,'target_GT_used':False})
 print('COMPLETE129',args.engineering,n,flush=True)
if __name__=='__main__':B129.mkdir(exist_ok=True);main()
