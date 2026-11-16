from core120 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');a=ap.parse_args()
def main():
 plan=jread(H120/'experiment/DESIGN_LOCK.json')
 for p,h in plan['files'].items():assert sha(R120/p)==h,p
 if not a.engineering:assert jread(H120/'ENGINEERING.json')['n']==24
 total=aliases=0;seals={};engineering=[]
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);registry=jread(R120/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json')['modules'][src];assert sha(R120/registry['state_path'])==registry['state_sha256'];np.savez_compressed(B120/(src+'_SOURCE_READOUT_PRIVATE.npz'),**{k:v for k,v in S.items() if k.startswith(('match.','interface.out.','rgb.','pred_depth.','image_adapter.'))})
  for co in ['DEV','EXP']:
   for tgt in ['waymo','drone','quad']:
    if tgt==src:continue
    name=src+'_to_'+tgt;rows=C118.rows(co,tgt);D=B120/('engineering' if a.engineering else 'formal')/co/name;assert not D.exists()
    if a.engineering:rows=[r for r in rows if not jread(R120/'crossplatform/round118/formal'/co/name/'records_PRIVATE'/(r['id']+'.json'))['protected_alias_fallback']][:2]
    for ix,r in enumerate(rows):
     tick=time.perf_counter();sid=r['id'];oldrec=jread(R120/'crossplatform/round118/formal'/co/name/'records_PRIVATE'/(sid+'.json'));alias=oldrec['protected_alias_fallback'];meta={'protected_alias_fallback':alias,'GN_parameters':224,'TTA':not alias,'new_source_preparation':False,'new_external_experts':False};ff={};states={}
     if alias:new=None;aliases+=1
     else:
      z0,g,z,d,proof=current120(co,src,r);restore(m,S);rt,role=role97(base,m,r,z0,g,z,d);route=tensor_route89(rt);pm,im=maskpair79(r,d);proposals,grads,train,hold,fit=proposals120(m,S,co,src,r,z,d,pm,im,a.engineering);new,ff,ev=evaluate120(base,m,S,co,src,r,z0,z,d,route,pm,im,proposals,train,hold,a.engineering);meta.update(inputs=proof,role_info=role,fit=fit,evaluation=ev);states={s:v.cpu().numpy() for s,v in proposals.items()}|{'gradient':grads['correct'].cpu().numpy(),'wrong_gradient':grads['wrong'].cpu().numpy()}
      if a.engineering:engineering.append({'cohort':co,'source':src,'target':tgt,'states':len(ev['engineering']),'old_provider_bitexact':ev['old_source_A_F_h_u_late_r_bitexact'],'scope_GN224_only':True,'DEV_gradients_bitexact_to_R116':co=='DEV','original_query_anchors_fixed':True})
     with np.load(R120/'crossplatform/round118/formal'/co/name/'samples_PRIVATE'/(sid+'.npz')) as aa:po=dict(zip(aa['arm_names'],aa['scores']));boxes=aa['source_boxes'].copy()
     if alias:new={k:po['prepared_A'].copy() for k in NEW120}
     out={k:po[v] for k,v in REFMAP120.items()}|new;arr=np.stack([out[k] for k in ARMS120]);assert arr.shape==(36,256) and np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-10)
     for sub in ['samples_PRIVATE','records_PRIVATE','features_PRIVATE','states_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
     np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),source_boxes=boxes,arm_names=np.array(ARMS120),scores=arr,queries=arr.argmax(1))
     if ff:np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff);np.savez_compressed(D/'states_PRIVATE'/(sid+'.npz'),**states)
     meta['seconds']=time.perf_counter()-tick;jdump(D/'records_PRIVATE'/(sid+'.json'),meta);restore(m,S);freeze91(m);total+=1
     if a.engineering or (ix+1)%16==0:print('RUN120',a.engineering,co,name,ix+1,round(meta['seconds'],3),flush=True)
    jdump(D/'SEALED.json',{'n':len(rows),'files':{str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file()},'GT_used':False});seals[co+'/'+name]=sha(D/'SEALED.json')
  assert fingerprint(base)==fp and all(np.array_equal(state(m)[k],v) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 if a.engineering:jdump(H120/'ENGINEERING.json',{'state':'PASS24_REAL_LIVE_CONTEXT_GN_SCOPE_NATIVE_FOUNDATION_OLD_RAW_AND_PROVIDER_CHECKS','n':total,'checks':engineering,'GT_used':False})
 else:assert total==2304 and aliases==154;jdump(B120/'MATRIX_SEALED.json',{'n':total,'arms':ARMS120,'new_arms':NEW120,'primary':PRIMARY120,'directions':seals,'aliases':aliases,'GT_used':False,'FINAL_access':False})
 print('RUN120_COMPLETE',a.engineering,total,aliases,flush=True)
if __name__=='__main__':B120.mkdir(exist_ok=True);main()
