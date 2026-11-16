from core116 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');a=ap.parse_args()
def main():
 plan=jread(H116/'experiment/DESIGN_LOCK.json')
 for p,h in plan['files'].items():assert sha(R116/p)==h,p
 B116.mkdir(exist_ok=True);seals={};n=0;eng=[]
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;rows=jread(R116/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');D=B116/('engineering' if a.engineering else 'formal')/name
   if a.engineering:rows=[r for r in rows if (B111/name/'prefix_PRIVATE'/(r['id']+'.npz')).exists()][:2]
   else:assert jread(H116/'ENGINEERING.json')['n']==12
   assert not D.exists()
   for k in ['samples_PRIVATE','features_PRIVATE','records_PRIVATE','states_PRIVATE']:(D/k).mkdir(parents=True,exist_ok=True)
   for ix,r in enumerate(rows):
    tick=time.perf_counter();sid=r['id'];oldrec=jread(B114/'formal'/name/'records_PRIVATE'/(sid+'.json'));meta={'protected_alias_fallback':oldrec['protected_alias_fallback'],'fits':{}};ff={}
    if not meta['protected_alias_fallback']:
     c=fit116(base,m,S,src,r,a.engineering);new,ff,ev=evaluate116(base,m,S,src,r,c,a.engineering);meta.update(fits=c['records'],evaluation=ev,inputs=c['proof']);np.savez_compressed(D/'states_PRIVATE'/(sid+'.npz'),**{s+'_'+t:v.cpu().numpy() for (s,t),v in c['states'].items()},gradient=c['gradient'].cpu().numpy(),wrong_gradient=c['wrong_gradient'].cpu().numpy())
     if a.engineering:eng.append({'source':src,'target':tgt,'scope_parameters':224,'gradient_L2':float(c['gradient'].norm()),'actual_native_checks':len(ev['engineering']),'all_scope_foundation_source_checks':True,'no_C_before_allfitstates':True})
    with np.load(B114/'formal'/name/'samples_PRIVATE'/(sid+'.npz')) as old:po=dict(zip(old['arm_names'],old['scores']));boxes=old['source_boxes'].copy()
    if meta['protected_alias_fallback']:new={k:po['prepared_frozen'].copy() for k in NEW116}
    out={k:po[k] for k in REF116};out.update(new);arr=np.stack([out[k] for k in ARMS116]);assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-10);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS116),scores=arr,queries=arr.argmax(1),source_boxes=boxes)
    if ff:np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff)
    meta['seconds']=time.perf_counter()-tick;jdump(D/'records_PRIVATE'/(sid+'.json'),meta);restore(m,S);freeze91(m);assert all(np.array_equal(state(m)[k],v) for k,v in S.items());n+=1
    if a.engineering or (ix+1)%16==0:print('RUN116',a.engineering,name,ix+1,round(meta['seconds'],2),flush=True)
   jdump(D/'SEALED.json',{'n':len(rows),'files':{str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file()},'GT_used':False});seals[name]=sha(D/'SEALED.json')
  assert fingerprint(base)==fp;del base,m;torch.cuda.empty_cache()
 if a.engineering:jdump(H116/'ENGINEERING.json',{'state':'PASS_EXISTING_DEPTH_NONZERO_GN224_SCOPE_NATIVE_FULL_FAST_SOURCE_RESTORE','n':n,'checks':eng,'GT_used':False})
 else:jdump(B116/'MATRIX_SEALED.json',{'n':n,'arms':ARMS116,'new_arms':NEW116,'directions':seals,'primary':PRIMARY116,'GT_used':False,'FINAL_access':False})
 print('RUN116_COMPLETE',a.engineering,n,flush=True)
if __name__=='__main__':main()
