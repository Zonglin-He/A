from core124 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');a=ap.parse_args()
def main():
 plan=jread(H/'experiment/DESIGN_LOCK.json')
 for p,h in plan['files'].items():assert sha(R/p)==h,p
 if not a.engineering:assert jread(H/'ENGINEERING.json')['n']==6
 reg=jread(R/'research/round121/SOURCE_CACHE_REGISTRY_PRIVATE.json');total=0;seals={}
 for src in ['waymo','drone','quad']:
  entries=[e for e in reg if e['source']==src];entries=entries[:2] if a.engineering else entries;setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);D=B/('engineering' if a.engineering else 'formal')/src;assert not D.exists()
  for sub in ['samples_PRIVATE','features_PRIVATE','states_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
  np.savez_compressed(B/(src+'_SOURCE_HEAD_PRIVATE.npz'),**{k:v for k,v in S.items() if k.startswith('match.')})
  for ix,e in enumerate(entries):
   tick=time.perf_counter();sid=e['row']['id'];scores,features,states,rec=process(base,m,S,e,a.engineering);arr=np.stack([scores[k] for k in ARMS]);assert arr.shape==(7,256) and np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-10);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),scores=arr,queries=arr.argmax(1),arm_names=np.array(ARMS));np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**features);np.savez_compressed(D/'states_PRIVATE'/(sid+'.npz'),**states);rec['seconds']=time.perf_counter()-tick;jdump(D/'records_PRIVATE'/(sid+'.json'),rec);total+=1
   if a.engineering or (ix+1)%16==0:print('RUN124',a.engineering,src,ix+1,round(rec['seconds'],3),flush=True)
  assert fingerprint(base)==fp and all(np.array_equal(state(m)[k],v) for k,v in S.items());jdump(D/'SEALED.json',{'n':len(entries),'ephemeral_source_GT_NOT_TTA':True,'files':{str(f.relative_to(D)):sha(f) for f in D.rglob('*') if f.is_file()}});seals[src]=sha(D/'SEALED.json');del base,m;torch.cuda.empty_cache()
 if a.engineering:jdump(H/'ENGINEERING.json',{'n':total,'state':'PASS6_NATIVE_SOURCE_ZERO_EXACT_SCOPE4896_RESTORE_FINITE_DIFFERENCE','source_GT_used':True})
 else:assert total==384;jdump(B/'MATRIX_SEALED.json',{'n':total,'arms':ARMS,'seals':seals,'source_GT_used':True,'target_data_read':False,'new_source_preparation':False,'FINAL_access':False})
 print('COMPLETE124',a.engineering,total,flush=True)
if __name__=='__main__':B.mkdir(exist_ok=True);main()
