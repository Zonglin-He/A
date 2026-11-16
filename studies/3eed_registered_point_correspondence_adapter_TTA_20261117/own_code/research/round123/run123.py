from core123 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');a=ap.parse_args()
def main():
 plan=jread(H123/'experiment/DESIGN_LOCK.json')
 for p,h in plan['files'].items():assert sha(R123/p)==h,p
 if not a.engineering:assert jread(H123/'ENGINEERING.json')['n']==24
 allrows=jread(R123/'research/round122/CACHE_REGISTRY_PRIVATE.json');total=aliases=0;seals={};checks=[]
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);S0,radius=start123(m,S);np.savez_compressed(B123/(src+'_AUDIT_MODEL_PRIVATE.npz'),**{k:v for k,v in S0.items() if k.startswith(('point_adapter.','match.'))},original_adapter_radius=np.array(radius))
  for co in ['DEV','EXP']:
   for tgt in ['waymo','drone','quad']:
    if tgt==src:continue
    name=src+'_to_'+tgt;rows=[e for e in allrows if e['cohort']==co and e['source']==src and e['target']==tgt]
    if a.engineering:
     yes=[];no=[]
     for e in rows:
      if e['protected_alias_fallback']:continue
      with np.load(R123/e['observations']) as z:has=('real_B_vis' if co=='DEV' else 'B_vis') in z
      (yes if has else no).append(e)
     rows=[yes[0],no[0]]
    D=B123/('engineering' if a.engineering else 'formal')/co/name;assert not D.exists()
    for sub in ['samples_PRIVATE','features_PRIVATE','states_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
    for ix,e in enumerate(rows):
     t=time.perf_counter();sid=e['row']['id'];old=np.load(R123/f'crossplatform/round122/formal/{co}/{name}/samples_PRIVATE/{sid}.npz');po=dict(zip(old['arm_names'],old['scores']));hm=np.load(R123/f'crossplatform/round122/headmean_controls/{co}/{name}/{sid}.npz');po.update(dict(zip(hm['arm_names'],hm['scores'])));prior=np.load(R123/f'crossplatform/round120/formal/{co}/{name}/samples_PRIVATE/{sid}.npz');oldC=dict(zip(prior['arm_names'],prior['scores']))['context_zero_C'];refs={k:(oldC if k=='zero_C' else po[k]) for k in REF123};meta={'protected_alias_fallback':e['protected_alias_fallback'],'no_external_expert':True,'no_source_training':True};ff={};st={}
     if e['protected_alias_fallback']:new={s+'_'+v:po['prepared_A'].copy() for s in SETTINGS for v in ['F','C']};aliases+=1
     else:new,ff,st,rec=process123(base,m,S,e,a.engineering);meta.update(rec)
     out=refs|new;arr=np.stack([out[k] for k in ARMS123]);assert arr.shape==(25,256) and np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-10);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),source_boxes=old['source_boxes'],arm_names=np.array(ARMS123),scores=arr,queries=arr.argmax(1));meta['seconds']=time.perf_counter()-t;jdump(D/'records_PRIVATE'/(sid+'.json'),meta)
     if ff:np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff);np.savez_compressed(D/'states_PRIVATE'/(sid+'.npz'),**st)
     if a.engineering:checks.append({'cohort':co,'source':src,'target':tgt,'zero_h_u_bitexact':True,'actual_writeback_scope4896only':True,'functional_equivalence':meta['engineering'],'B_for_fit':meta['B_for_fit']})
     total+=1;restore(m,S);freeze91(m)
     if a.engineering or (ix+1)%16==0:print('RUN123',a.engineering,co,name,ix+1,round(meta['seconds'],3),flush=True)
    jdump(D/'SEALED.json',{'n':len(rows),'GT_used':False,'files':{str(f.relative_to(D)):sha(f) for f in D.rglob('*') if f.is_file()}});seals[co+'/'+name]=sha(D/'SEALED.json')
  assert fingerprint(base)==fp and all(np.array_equal(state(m)[k],v) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 if a.engineering:jdump(H123/'ENGINEERING.json',{'state':'PASS24_REAL_POINT_ADAPTER_WRITES_NATIVE_ZERO_FUNCTIONAL_LINK_AND_NO_C_GRADIENT','n':total,'checks':checks})
 else:assert total==2304 and aliases==154;jdump(B123/'MATRIX_SEALED.json',{'n':total,'arms':ARMS123,'new_arms':NEW123,'primary':PRIMARY123,'directions':seals,'aliases':aliases,'GT_used':False,'FINAL_access':False})
 print('COMPLETE123',a.engineering,total,flush=True)
if __name__=='__main__':B123.mkdir(exist_ok=True);main()
