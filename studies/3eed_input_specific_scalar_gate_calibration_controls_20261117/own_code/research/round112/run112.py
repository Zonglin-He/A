"""Fixed 2x2 causal ablation; no optimization or new evidence generation."""
from core112 import *
def main():
 assert jread(H112/'ENGINEERING.json')['n']==12
 for p,h in jread(H112/'experiment/CODE_LOCK.json')['files'].items():assert sha(R112/p)==h
 n=0;seals={}
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);fp=fingerprint(base);np.savez_compressed(B112/(src+'_HEAD_PRIVATE.npz'),**state(m.match));jdump(B112/(src+'_SOURCE_MODULE_SHA256_PRIVATE.json'),{'source_checkpoint_sha256':source97(src)['checkpoint_sha256'],'learned_source_module_state_sha256':hashlib.sha256(b''.join(k.encode()+v.tobytes() for k,v in sorted(S.items()))).hexdigest()})
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;rows=jread(R112/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');D=B112/name;assert not D.exists()
   for sub in ['samples_PRIVATE','features_PRIVATE','records_PRIVATE']:(D/sub).mkdir(exist_ok=True,parents=True)
   for ix,r in enumerate(rows):
    tick=time.perf_counter();sid=r['id'];old=load(B111/name/'samples_PRIVATE'/(sid+'.npz'));scores=dict(zip(old['arm_names'],old['scores']));oldrec=jread(B111/name/'records_PRIVATE'/(sid+'.json'));rec={'protected_alias_fallback':oldrec['protected_alias_fallback'],'fits':{},'TTA_run':False,'component_edits':{},'families':oldrec['families'],'fixed_source_A256_positions':True};features={}
    if rec['protected_alias_fallback']:scores.update({k:scores['prepared_frozen'] for k in NEW112})
    else:
     z0,g,zz,proof=pair112(src,r);d=tensor_data(g,r);restore(m,S);rt,roleinfo=role97(base,m,r,z0,g,zz['A'],d);route=tensor_route89(rt);perm=permutation94(r,d,0);dw=bound94(d,perm);rec.update(source_inputs=proof,frozen_route_after_fullR91_before_edits=True,wrong_binding_changed_visible_seeds=int(np.count_nonzero(perm!=np.arange(1024))),same_wrong_visible_mask=bool(torch.equal(d['p2i'].ge(0).any(1),dw['p2i'].ge(0).any(1))))
     with np.load(B106/name/'features_PRIVATE'/(sid+'.npz')) as prior,np.load(B111/name/'features_PRIVATE'/(sid+'.npz')) as native:
      orig=load(R112/'crossplatform/round110/original_PRIVATE'/name/(sid+'.npz'))
      for view,z in zz.items():
       nh=orig['h'] if view=='A' else native['real_fusedABC_h'];nu=orig['tokens'] if view=='A' else native['real_fusedABC_tokens'];nh=torch.as_tensor(nh,device='cuda');nu=torch.as_tensor(nu,device='cuda')
       for binding,dd in [('correct',d),('wrong',dw)]:
        for variant in VAR112:
         meta=edit112(m,S,variant);f=forward112(m,base,z,dd,route)
         if variant=='full' and binding=='correct':
          pold=prior['real_'+view+'_p'];assert float(abs(f['p'].cpu().numpy()-pold).max())<1e-7;assert np.array_equal(f['h'].cpu().numpy(),prior['real_'+view+'_h']);continue
         label=variant+'_'+binding+'_'+view;scores[label]=f['p'].cpu().numpy();meta.update(hidden_RMS_from_native=float((f['h']-nh).double().square().mean().sqrt()),text_RMS_from_native=float((f['u']-nu).double().square().mean().sqrt()));rec['component_edits'][label]=meta
         if variant=='neither':assert torch.equal(f['h'],nh) and torch.equal(f['u'],nu)
         for k,v in f.items():features[label+'_'+k]=v.cpu().numpy()
    restore(m,S);arr=np.stack([scores[a] for a in ARMS112]);assert arr.shape==(89,256) and np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-12);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS112),scores=arr,source_boxes=old['source_boxes'],queries=arr.argmax(1),anno_index=old['anno_index'])
    if features:np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**features)
    rec['seconds']=time.perf_counter()-tick;jdump(D/'records_PRIVATE'/(sid+'.json'),rec);n+=1
    if (ix+1)%16==0:print('RUN112',name,ix+1,len(rows),round(rec['seconds'],2),flush=True)
   jdump(D/'SEALED.json',{'n':len(rows),'no_GT':True,'TTA_run':False,'primary':PRIMARY112,'files':{str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*')) if p.is_file()}});seals[name]=sha(D/'SEALED.json')
  assert fingerprint(base)==fp and all(np.array_equal(state(m)[k],v) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 jdump(B112/'MATRIX_SEALED.json',{'n':n,'arms':ARMS112,'primary':PRIMARY112,'directions':seals,'no_GT':True,'TTA_run':False});print('RUN112_COMPLETE',n,len(ARMS112),flush=True)
if __name__=='__main__':main()
