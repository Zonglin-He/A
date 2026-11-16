from core121 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');a=ap.parse_args()
def main():
 lock=jread(H121/'experiment/DESIGN_LOCK.json');assert all(sha(R121/p)==h for p,h in lock['files'].items())
 if not a.engineering:assert jread(H121/'ENGINEERING.json')['n']==6
 registry=jread(H121/'SOURCE_CACHE_REGISTRY_PRIVATE.json');total=0;proofs={};start=time.perf_counter()
 for src in ['waymo','drone','quad']:
  rows=[e for e in registry if e['source']==src];rows=rows[:2] if a.engineering else rows;setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);D=B121/('engineering' if a.engineering else 'formal')/src;assert not D.exists()
  for sub in ['samples_PRIVATE','features_PRIVATE','states_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
  np.savez_compressed(B121/(src+'_SOURCE_READOUT_PRIVATE.npz'),**{k:v for k,v in S.items() if k.startswith(('match.','interface.out.'))})
  for ix,e in enumerate(rows):
   t=time.perf_counter();sid=e['row']['id'];sc,ff,st,rec=analyze121(base,m,S,e,a.engineering);arr=np.stack([sc[k] for k in ARMS121]);assert arr.shape==(13,256) and np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-10);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS121),scores=arr,queries=arr.argmax(1));np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff);np.savez_compressed(D/'states_PRIVATE'/(sid+'.npz'),**st);rec['seconds']=time.perf_counter()-t;jdump(D/'records_PRIVATE'/(sid+'.json'),rec);total+=1
   if a.engineering or (ix+1)%8==0:print('RUN121',a.engineering,src,ix+1,round(rec['seconds'],2),'total',round(time.perf_counter()-start,1),flush=True)
  assert fingerprint(base)==fp and all(np.array_equal(state(m)[k],v) for k,v in S.items());jdump(D/'SEALED.json',{'n':len(rows),'source_label_controlled_diagnostic_NOT_TTA':True,'files':{str(f.relative_to(D)):sha(f) for f in D.rglob('*') if f.is_file()}});proofs[src]=sha(D/'SEALED.json');del base,m;torch.cuda.empty_cache()
 if a.engineering:jdump(H121/'ENGINEERING.json',{'state':'PASS6_SOURCE_NATIVE_READOUT_SCOPE_GN224_AND_RESTORE','n':total,'source_labels_used':True,'target_data_read':False})
 else:assert total==384;jdump(B121/'MATRIX_SEALED.json',{'n':total,'arms':ARMS121,'source_seals':proofs,'source_labels_used':True,'new_source_training':False,'target_data_read':False,'FINAL_access':False,'seconds':time.perf_counter()-start})
 print('COMPLETE121',a.engineering,total,flush=True)
if __name__=='__main__':B121.mkdir(exist_ok=True);main()
