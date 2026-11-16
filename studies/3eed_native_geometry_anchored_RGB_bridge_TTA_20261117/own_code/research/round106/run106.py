"""Actual and same-frame point observations, fixed semantic teacher and native TTA."""
from core106 import *
REFS106=['original_source','prepared_frozen','frozen_four','direct_FC','joint3_FC']
FAMILIES106=['real','randomA','wrong']
ARMS106=REFS106+[k+'_'+label+'_'+v for k in FAMILIES106 for label,vv in [('frozen',['B','C','ensemble','fusedABC']),('TTA',['A','B','C','ensemble','fusedABC'])] for v in vv]+['teacher_shuffle_C','teacher_shuffle_fusedABC']
def draw106(m,views,st):
 put94(m,st)
 with torch.no_grad():p={k:read105(m,f) for k,f in views.items()};p['ensemble']=torch.stack([p[k] for k in ['A','B','C']]).mean(0);return p
def main():
 assert jread(H106/'ENGINEERING.json')['n']==12
 for p,h in jread(H106/'experiment/CODE_LOCK.json')['files'].items():assert sha(R106/p)==h
 anchors={x['A']['metapath']:x for x in jread(R106/'research/round105/FULL_ANCHORS_PRIVATE.json')};pairs={p['pair_PRIVATE']:p for p in jread(R106/'research/round105/FULL_PAIRS_PRIVATE.json')};n=0;seals={}
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);head=patch94(m)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;rows=jread(R106/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');O=B106/name;assert not O.exists()
   for sub in ['samples_PRIVATE','features_PRIVATE','records_PRIVATE','states_PRIVATE']:(O/sub).mkdir(exist_ok=True,parents=True)
   for ix,r in enumerate(rows):
    tick=time.perf_counter();sid=r['id'];restore(m,S);old=load(R106/'crossplatform/round100'/name/'samples_PRIVATE'/(sid+'.npz'));oldp=dict(zip(old['arm_names'],old['scores']));scores={k:oldp[k] for k in REFS106};ar=anchors[str(Path(r['metapath']).resolve())];rec={'protected_alias_fallback':not ar['A']['eligible_for_extra_media'],'C_neural_after_all_fits':True,'families':{},'fits':{}};states={};features={}
    if rec['protected_alias_fallback']:scores.update({k:oldp['prepared_frozen'] for k in ARMS106 if k not in REFS106})
    else:
     with torch.no_grad():z0,g,_=capture98(base,r,src);z=tensor_prefix(z0);d=tensor_data(g,r);rt,_=role97(base,m,r,z0,g,z,d);route=tensor_route89(rt);A=feature106(m,base,z,d,route,S,[])
     assert np.array_equal(z0['boxes'],old['source_boxes']) and float(abs(A['p']-torch.as_tensor(oldp['prepared_frozen'],device='cuda')).max())<1e-7
     observations={k:{} for k in FAMILIES106};proof={k:{} for k in FAMILIES106}
     for kind in ['real','randomA']:
      if ar['B_pair']:obs,pf=extra106(base,ar['A'],pairs[ar['B_pair']],z0,kind,'B')
      else:obs,pf=None,{'status':'no_B_candidate'}
      observations[kind]['B']=obs;proof[kind]['B']=pf
     observations['wrong']['B']=wrong106(observations['real']['B'],r,'B');proof['wrong']['B']={'status':proof['real']['B']['status'],'same_support_mask_as_real':True}
     views={k:{'A':A} for k in FAMILIES106}
     with torch.no_grad():
      for k in FAMILIES106:views[k]['B']=feature106(m,base,z,d,route,S,[observations[k]['B']]) if observations[k]['B'] is not None else A
     targets={k:(A['p']+views[k]['B']['p'])/2 for k in FAMILIES106};targets['teacher_shuffle']=(A['p']+views['real']['B']['p'][torch.as_tensor(qperm94(r),device='cuda')])/2
     for label in FAMILIES106+['teacher_shuffle']:
      k='real' if label=='teacher_shuffle' else label
      if observations[k]['B'] is not None:
       st,trials,info=lm105(m,[A,views[k]['B']],targets[label].log()[None].repeat(2,1),head);states[label]=st;rec['fits'][label]={**info,'trials':trials,'target':targets[label].cpu().numpy().tolist()}
     # Independent C is encoded/read only after the fit loop has fully terminated.
     put94(m,head)
     for kind in ['real','randomA']:
      if ar['C_pair']:obs,pf=extra106(base,ar['A'],pairs[ar['C_pair']],z0,kind,'C')
      else:obs,pf=None,{'status':'no_C_candidate'}
      observations[kind]['C']=obs;proof[kind]['C']=pf
     observations['wrong']['C']=wrong106(observations['real']['C'],r,'C');proof['wrong']['C']={'status':proof['real']['C']['status'],'same_support_mask_as_real':True}
     frozen={};fitted={}
     with torch.no_grad():
      for k in FAMILIES106:
       ob=observations[k];views[k]['C']=feature106(m,base,z,d,route,S,[ob['C']]) if ob['C'] is not None else A;views[k]['fusedABC']=feature106(m,base,z,d,route,S,[ob[v] for v in ['B','C'] if ob[v] is not None],True);frozen[k]=draw106(m,views[k],head)
     for label,st in states.items():fitted[label]=draw106(m,views['real' if label=='teacher_shuffle' else label],st)
     for k in FAMILIES106:
      enabled=bool(observations[k]['B'] is not None and observations[k]['C'] is not None);rec['families'][k]={'B_available':observations[k]['B'] is not None,'C_available':observations[k]['C'] is not None,'updated_output_enabled':enabled,'support':proof[k]};result=fitted.get(k,frozen[k]) if enabled else frozen[k]
      for label,vv,ps in [('frozen',['B','C','ensemble','fusedABC'],frozen[k]),('TTA',['A','B','C','ensemble','fusedABC'],result)]:
       for v in vv:scores[k+'_'+label+'_'+v]=ps[v].cpu().numpy()
     shuf=fitted.get('teacher_shuffle',frozen['real']) if rec['families']['real']['updated_output_enabled'] else frozen['real']
     for v in ['C','fusedABC']:scores['teacher_shuffle_'+v]=shuf[v].cpu().numpy()
     features=feature_save105({k+'_'+v:f for k,vv in views.items() for v,f in vv.items()})
    rec['seconds']=time.perf_counter()-tick;arr=np.stack([scores[k] for k in ARMS106]);assert arr.shape==(34,256) and np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-12);np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS106),scores=arr,source_boxes=old['source_boxes'],queries=arr.argmax(1),anno_index=old['anno_index']);jdump(O/'records_PRIVATE'/(sid+'.json'),rec)
    if features:np.savez_compressed(O/'features_PRIVATE'/(sid+'.npz'),**features)
    for k,st in states.items():np.savez_compressed(O/'states_PRIVATE'/(sid+'_'+k+'.npz'),**st)
    restore(m,S);n+=1
    if (ix+1)%16==0:print('RUN106',name,ix+1,len(rows),round(rec['seconds'],2),flush=True)
   jdump(O/'SEALED.json',{'n':len(rows),'primary':'real_TTA_fusedABC','no_GT':True,'files':{str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file()}});seals[name]=sha(O/'SEALED.json')
  ss=state(m);assert fingerprint(base)==fp and all(np.array_equal(v,ss[k]) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 jdump(B106/'MATRIX_SEALED.json',{'n':n,'arms':ARMS106,'directions':seals,'primary':'real_TTA_fusedABC','no_GT':True});print('RUN106_COMPLETE',n,flush=True)
if __name__=='__main__':main()
