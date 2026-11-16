"""Fixed 1536-input expansion of native LiDAR fusion and frozen dual-head readout."""
from core109 import *
def main():
 assert jread(H109/'BRIDGE.json')['n']==6;assert jread(B109/'registration_PRIVATE/SEALED.json')['n']==len(jread(H109/'PAIRS_PRIVATE.json'))
 lock=jread(H109/'experiment/CODE_LOCK.json')
 for p,h in lock['files'].items():assert sha(R109/p)==h
 for p,h in lock['input_sha256'].items():assert sha(R109/p)==h
 for p,h in lock['dependency_sha256'].items():assert sha(R109/p)==h
 anchors={x['A']['metapath']:x for x in jread(H109/'ANCHORS_PRIVATE.json')};pairs={x['pair_PRIVATE']:x for x in jread(H109/'PAIRS_PRIVATE.json')};seals={};n=0
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);head=base.prediction_heads[-1].sem_cls_scores_head;np.savez_compressed(B109/(src+'_CE_WEIGHTS_PRIVATE.npz'),**statece108(head))
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;O=B109/name;assert not O.exists()
   for sub in ['samples_PRIVATE','records_PRIVATE','features_PRIVATE']:(O/sub).mkdir(parents=True,exist_ok=True)
   rows=jread(H109/f'input/TARGET_{tgt.upper()}256_PRIVATE.json')
   for ix,r in enumerate(rows):
    tick=time.perf_counter();sid=r['id'];restore(m,S);old=load(R109/'crossplatform/round100'/name/'samples_PRIVATE'/(sid+'.npz'));op=dict(zip(old['arm_names'],old['scores']));scores={k:op[k] for k in REF109};ar=anchors[str(Path(r['metapath']).resolve())];rec={'protected_alias_fallback':not ar['A']['eligible_for_extra_media'],'families':{},'fits':{},'no_parameter_or_statistics_update':True};features={}
    if rec['protected_alias_fallback']:scores.update({k:op['prepared_frozen'] for k in ARMS109 if k not in scores})
    else:
     with torch.no_grad():
      z0,g,_=capture98(base,r,src);z=tensor_prefix(z0);d=tensor_data(g,r);rt,_=role97(base,m,r,z0,g,z,d);route=tensor_route89(rt);A=feature106(m,base,z,d,route,S,[]);err=float(abs(A['p']-torch.as_tensor(op['prepared_frozen'],device='cuda')).max());assert err<1e-7 and np.array_equal(z0['boxes'],old['source_boxes']);rec['source_prepared_probability_error']=err
      qq,_=ce108(head,A['h'],A['mask'])
      for label,p in pool108(A['p'],qq,r).items():scores[label+'_A']=p.cpu().numpy()
      observations={k:{} for k in FAMILIES109};proof={k:{} for k in FAMILIES109}
      for v in ['B','C']:
       for k in ['real','randomA']:
        pid=ar[v+'_pair'];obs,pf=extra109(base,ar['A'],pairs[pid],z0,k,v) if pid else (None,{'status':'no_'+v+'_candidate'});observations[k][v]=obs;proof[k][v]=pf
       observations['wrong'][v]=wrong106(observations['real'][v],r,v);proof['wrong'][v]={'status':proof['real'][v]['status'],'same_support_mask_and_values_as_real':True}
      vv={}
      for k in FAMILIES109:
       obs=observations[k];views={'A':A}
       for v in ['B','C']:views[v]=feature106(m,base,z,d,route,S,[obs[v]]) if obs[v] is not None else A
       views['fusedABC']=feature106(m,base,z,d,route,S,[obs[v] for v in ['B','C'] if obs[v] is not None],True);vv.update({k+'_'+v:f for v,f in views.items()});scores[k+'_BBF_ensemble']=torch.stack([views[v]['p'] for v in ['A','B','C']]).mean(0).cpu().numpy()
       for v in ['B','C','fusedABC']:scores[k+'_BBF_'+v]=views[v]['p'].cpu().numpy()
       f=views['fusedABC'];q,_=ce108(head,f['h'],f['mask'])
       for label,p in pool108(f['p'],q,r).items():scores[k+'_'+label+'_fusedABC']=p.cpu().numpy()
       rec['families'][k]={'B_available':obs['B'] is not None,'C_available':obs['C'] is not None,'support':proof[k]}
      features=feature_save105(vv)
    arr=np.stack([scores[a] for a in ARMS109]);assert arr.shape==(33,256) and np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-12);rec['seconds']=time.perf_counter()-tick;np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS109),scores=arr,source_boxes=old['source_boxes'],queries=arr.argmax(1),anno_index=old['anno_index']);jdump(O/'records_PRIVATE'/(sid+'.json'),rec)
    if features:np.savez_compressed(O/'features_PRIVATE'/(sid+'.npz'),**features)
    restore(m,S);n+=1
    if (ix+1)%16==0:print('RUN109',name,ix+1,len(rows),round(rec['seconds'],2),flush=True)
   jdump(O/'SEALED.json',{'n':len(rows),'no_GT':True,'primary':PRIMARY109,'files':{str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file()}});seals[name]=sha(O/'SEALED.json')
  end=state(m);assert fingerprint(base)==fp and all(np.array_equal(v,end[k]) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 jdump(B109/'MATRIX_SEALED.json',{'n':n,'arms':ARMS109,'directions':seals,'primary':PRIMARY109,'no_GT':True,'not_TTA':True});print('RUN109_COMPLETE',n,len(ARMS109),flush=True)
if __name__=='__main__':main()
