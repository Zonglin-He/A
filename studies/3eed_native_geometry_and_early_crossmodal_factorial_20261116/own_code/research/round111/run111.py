"""Frozen original/native branch ablation, exact R106 physical evidence and controls."""
from core111 import *
def main():
 assert jread(H111/'ENGINEERING.json')['n']==12
 for p,h in jread(H111/'experiment/CODE_LOCK.json')['files'].items():assert sha(R111/p)==h
 anchors={x['A']['metapath']:x for x in jread(R111/'research/round105/FULL_ANCHORS_PRIVATE.json')};pairs={p['pair_PRIVATE']:p for p in jread(R111/'research/round105/FULL_PAIRS_PRIVATE.json')};n=0;seals={}
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);np.savez_compressed(B111/(src+'_HEAD_NATIVE_PRIVATE.npz'),**state(base.contrastive_align_projection_image));np.savez_compressed(B111/(src+'_HEAD_PREPARED_PRIVATE.npz'),**state(m.match))
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;rows=jread(R111/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');D=B111/name;assert not D.exists()
   for sub in ['samples_PRIVATE','features_PRIVATE','records_PRIVATE','prefix_PRIVATE','observations_PRIVATE']:(D/sub).mkdir(exist_ok=True,parents=True)
   for ix,r in enumerate(rows):
    tick=time.perf_counter();sid=r['id'];old=load(R111/'crossplatform/round110'/name/'samples_PRIVATE'/(sid+'.npz'));scores=dict(zip(old['arm_names'],old['scores']));assert set(scores)==set(REFS111);oldrec=jread(B106/name/'records_PRIVATE'/(sid+'.json'));rec={'protected_alias_fallback':oldrec['protected_alias_fallback'],'no_TTA':True,'fits':{},'families':{},'all_source_positions_fixed':True};features={}
    if rec['protected_alias_fallback']:scores.update({k:scores['prepared_frozen'] for k in ARMS111 if k not in scores})
    else:
     prior=B111/'attempt001_PRIVATE'/name;reuse=(prior/'samples_PRIVATE'/(sid+'.npz')).exists()
     if reuse:
      z0=load(prior/'prefix_PRIVATE'/(sid+'.npz'));oldO=load(R111/'crossplatform/round110/original_PRIVATE'/name/(sid+'.npz'));A={k:torch.as_tensor(oldO[k],device='cuda') for k in ['h','query','tokens','mask','p']};receipt={'reused_failed_attempt_prefix':True,'new_sensor_or_backbone_forward':False};obs=None;proof={};obsout=load(prior/'observations_PRIVATE'/(sid+'.npz'))
     else:
      z0,receipt=sourceprefix111(base,r,src);ar=anchors[str(Path(r['metapath']).resolve())];obs,proof=observations111(base,r,z0,ar,pairs,oldrec);A=None;obsout={k+'_'+v+'_'+x:t.detach().cpu().numpy() for k in ['real','randomA'] for v in ['B','C'] if obs[k][v] is not None for x,t in obs[k][v].items()}
     z=tensor_prefix(z0)
     if A is None:A=original111(base,z)
     assert float(abs(A['p'].cpu().numpy()-scores['original_source_matchedcoverage']).max())<1e-7;np.savez_compressed(D/'prefix_PRIVATE'/(sid+'.npz'),**z0);np.savez_compressed(D/'observations_PRIVATE'/(sid+'.npz'),**obsout);rec['source_prefix_receipt']=receipt;rec['readout_amended_cached_state_reuse']=reuse
     with torch.no_grad():
      for k in FAMILIES111:
       vv={'A':A}
       if reuse:
        cached=load(prior/'features_PRIVATE'/(sid+'.npz'))
        for v in ['B','C','fusedABC']:
         vv[v]={x:torch.as_tensor(cached[k+'_'+v+'_'+x],device='cuda') for x in ['h','tokens','mask']};p,q=direct111(base.contrastive_align_projection_image,vv[v]['h'],vv[v]['tokens'],vv[v]['mask']);vv[v]['p']=p
       else:
        for v in ['B','C']:vv[v]=original111(base,prefix106(z,[obs[k][v]])) if obs[k][v] is not None else A
        vv['fusedABC']=original111(base,prefix106(z,[obs[k][v] for v in ['B','C'] if obs[k][v] is not None],True))
       vv['ensemble']={'p':torch.stack([vv[v]['p'] for v in ['A','B','C']]).mean(0)}
       for v in VIEWS111:scores['native_'+k+'_'+v]=vv[v]['p'].cpu().numpy()
       for v in ['B','C','fusedABC']:
        for x in ['h','tokens','mask','p']:features[k+'_'+v+'_'+x]=vv[v][x].cpu().numpy()
       rec['families'][k]=oldrec['families'][k] if reuse else {'B_available':obs[k]['B'] is not None,'C_available':obs[k]['C'] is not None,'support':proof.get(k,{}),'same_evidence_as_R106':True}
      with np.load(B106/name/'features_PRIVATE'/(sid+'.npz')) as cache:
       for label,p in swaps111(base,m,r,A,cache).items():scores[label]=p.cpu().numpy()
    arr=np.stack([scores[a] for a in ARMS111]);assert arr.shape==(75,256) and np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-12);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS111),scores=arr,source_boxes=old['source_boxes'],queries=arr.argmax(1),anno_index=old['anno_index'])
    if features:np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**features)
    rec['seconds']=time.perf_counter()-tick;jdump(D/'records_PRIVATE'/(sid+'.json'),rec);n+=1
    if (ix+1)%16==0:print('RUN111',name,ix+1,len(rows),round(rec['seconds'],2),flush=True)
   jdump(D/'SEALED.json',{'n':len(rows),'primary':PRIMARY111,'no_GT':True,'files':{str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*')) if p.is_file()}});seals[name]=sha(D/'SEALED.json')
  assert fingerprint(base)==fp and all(np.array_equal(state(m)[k],v) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 jdump(B111/'MATRIX_SEALED.json',{'n':n,'arms':ARMS111,'directions':seals,'primary':PRIMARY111,'no_TTA':True,'no_GT':True});print('RUN111_COMPLETE',n,len(ARMS111),flush=True)
if __name__=='__main__':main()
