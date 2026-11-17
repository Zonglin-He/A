from core127 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');a=ap.parse_args()
def main():
 plan=jread(H127/'experiment/DESIGN_LOCK.json')
 for p,h in plan['files'].items():assert sha(R127/p)==h,p
 if not a.engineering:assert jread(H127/'ENGINEERING.json')['n']==24 and jread(H127/'CPU_PROBE.json')['n']==6
 reg=jread(R127/'research/round122/CACHE_REGISTRY_PRIVATE.json');total=aliases=0;seals={};checks=[]
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);S0,radius=start123(m,S)
  np.savez_compressed(B127/(src+'_AUDIT_MODEL_PRIVATE.npz'),**{k:v for k,v in S0.items() if k.startswith(('point_adapter.','match.'))},original_adapter_radius=np.array(radius))
  for co in ['DEV','EXP']:
   for tgt in ['waymo','drone','quad']:
    if src==tgt:continue
    name=src+'_to_'+tgt;rr=[e for e in reg if e['source']==src and e['target']==tgt and e['cohort']==co]
    if a.engineering:
     yes=[];no=[]
     for e in rr:
      if e['protected_alias_fallback']:continue
      with np.load(R127/e['observations']) as ob:has=('real_B_vis' if co=='DEV' else 'B_vis') in ob
      (yes if has else no).append(e)
     rr=[yes[0],no[0]]
    D=B127/('engineering' if a.engineering else 'formal')/co/name;assert not D.exists()
    for sub in ['samples_PRIVATE','features_PRIVATE','states_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
    for ix,e in enumerate(rr):
     tick=time.perf_counter();sid=e['row']['id'];old=np.load(R127/f'crossplatform/round125/formal/{co}/{name}/samples_PRIVATE/{sid}.npz');po=dict(zip(old['arm_names'],old['scores']));refs={'original_source':po['original_source'],'prepared_A':po['prepared_A'],'legacy_ABC_context':po['mean_context']};ff={};st={};meta={'protected_alias_fallback':e['protected_alias_fallback'],'no_external_expert':True,'no_source_training':True}
     if e['protected_alias_fallback']:new={k:po['prepared_A'].copy() for k in NEW127};aliases+=1
     else:new,ff,st,rec=process127(base,m,S,e,a.engineering);meta.update(rec)
     assert set(new)==set(NEW127);out=refs|new;arr=np.stack([out[k] for k in ARMS127]);assert arr.shape==(21,256) and np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-10)
     np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),source_boxes=old['source_boxes'],arm_names=np.array(ARMS127),scores=arr,queries=arr.argmax(1))
     meta['seconds']=time.perf_counter()-tick;jdump(D/'records_PRIVATE'/(sid+'.json'),meta)
     if ff:np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff);np.savez_compressed(D/'states_PRIVATE'/(sid+'.npz'),**st)
     if a.engineering:checks.append({'cohort':co,'source':src,'target':tgt,'active':meta['fit']['active'],'scope4896':True,'finite_difference':meta['finite_difference'],'seconds':meta['seconds']})
     total+=1;restore(m,S);freeze91(m)
     if a.engineering or (ix+1)%16==0:print('RUN127',a.engineering,co,name,ix+1,round(meta['seconds'],3),flush=True)
    jdump(D/'SEALED.json',{'n':len(rr),'GT_used':False,'files':{str(f.relative_to(D)):sha(f) for f in D.rglob('*') if f.is_file()}});seals[co+'/'+name]=sha(D/'SEALED.json')
  assert fingerprint(base)==fp and all(np.array_equal(state(m)[k],v) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 if a.engineering:jdump(H127/'ENGINEERING.json',{'state':'PASS24_REAL_FACTORIAL_FULL_PATH_ACTUAL_POINT_ADAPTER_WRITES','n':total,'checks':checks})
 else:assert total==2304 and aliases==154;jdump(B127/'MATRIX_SEALED.json',{'n':total,'arms':ARMS127,'new_arms':NEW127,'primary':PRIMARY127,'directions':seals,'aliases':aliases,'GT_used':False,'FINAL_access':False})
 print('COMPLETE127',a.engineering,total,flush=True)
if __name__=='__main__':B127.mkdir(exist_ok=True);main()
