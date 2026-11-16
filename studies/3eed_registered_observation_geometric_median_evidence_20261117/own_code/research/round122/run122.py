from core122 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');a=ap.parse_args()
def main():
 plan=jread(H122/'experiment/DESIGN_LOCK.json')
 for p,h in plan['files'].items():assert sha(R122/p)==h,p
 if not a.engineering:assert jread(H122/'ENGINEERING.json')['n']==24
 rr=jread(H122/'CACHE_REGISTRY_PRIVATE.json');total=aliases=0;seals={};checks=[]
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);np.savez_compressed(B122/(src+'_MATCH_PRIVATE.npz'),**{k[6:]:v for k,v in S.items() if k.startswith('match.')})
  for co in ['DEV','EXP']:
   for tgt in ['waymo','drone','quad']:
    if tgt==src:continue
    rows=[x for x in rr if x['cohort']==co and x['source']==src and x['target']==tgt]
    if a.engineering:
     live=[r for r in rows if not r['protected_alias_fallback']];yes=next(r for r in live if r['three_valid_seeds']>0);no=next(r for r in live if r['three_valid_seeds']==0);rows=[yes,no]
    name=src+'_to_'+tgt;D=B122/('engineering' if a.engineering else 'formal')/co/name;assert not D.exists()
    for sub in ['samples_PRIVATE','records_PRIVATE','features_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
    for ix,e in enumerate(rows):
     t=time.perf_counter();sid=e['row']['id'];old=np.load(R122/f'crossplatform/round118/formal/{co}/{name}/samples_PRIVATE/{sid}.npz');po=dict(zip(old['arm_names'],old['scores']));alias=e['protected_alias_fallback'];info={'protected_alias_fallback':alias,'TTA':False,'gradient_updates':0,'no_new_source_preparation':True};ff={}
     if alias:new={k:po['prepared_A'].copy() for k in NEW122};aliases+=1
     else:new,ff,info0=process122(base,m,S,e,a.engineering);info.update(info0)
     out={k:po[v] for k,v in REF122.items()}|new;arr=np.stack([out[k] for k in ARMS122]);assert arr.shape==(20,256) and np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-10);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),source_boxes=old['source_boxes'],arm_names=np.array(ARMS122),scores=arr,queries=arr.argmax(1));info['seconds']=time.perf_counter()-t;jdump(D/'records_PRIVATE'/(sid+'.json'),info)
     if ff:np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff)
     if a.engineering:checks.append({'source':src,'target':tgt,'cohort':co,'triple_seeds':info['three_valid_seeds'],'old_mean_h_U_bitexact':info['engineering_old_mean_h_U_bitexact'],'frozen_parameters':True})
     total+=1;restore(m,S);freeze91(m)
     if a.engineering or (ix+1)%32==0:print('RUN122',a.engineering,co,name,ix+1,round(info['seconds'],3),flush=True)
    jdump(D/'SEALED.json',{'n':len(rows),'GT_used':False,'files':{str(f.relative_to(D)):sha(f) for f in D.rglob('*') if f.is_file()}});seals[co+'/'+name]=sha(D/'SEALED.json')
  assert fingerprint(base)==fp and all(np.array_equal(state(m)[k],v) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 if a.engineering:jdump(H122/'ENGINEERING.json',{'state':'PASS24_REAL_NONTRIPLE_AND_TRIPLE_FIXEDANCHOR_AND_OLD_MEAN_PARITY','n':total,'checks':checks})
 else:assert total==2304 and aliases==154;jdump(B122/'MATRIX_SEALED.json',{'n':total,'arms':ARMS122,'new_arms':NEW122,'primary':PRIMARY122,'directions':seals,'aliases':aliases,'GT_used':False,'TTA':False,'FINAL_access':False})
 print('COMPLETE122',a.engineering,total,flush=True)
if __name__=='__main__':B122.mkdir(exist_ok=True);main()
