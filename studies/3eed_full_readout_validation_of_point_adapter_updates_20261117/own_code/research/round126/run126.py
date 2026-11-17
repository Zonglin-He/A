from core126 import *
import time
def main():
 torch.set_num_threads(4);plan=rd(H/'experiment/DESIGN_LOCK.json')
 for p,h in plan['files'].items():assert sha(R/p)==h,p
 assert rd(R/'research/round125/NATIVE_AUDIT.json')['n']==2304;matrix=rd(R/'crossplatform/round125/MATRIX_SEALED.json');reg=rd(R/'research/round122/CACHE_REGISTRY_PRIVATE.json');col=rd(R/'research/round97/SOURCE_PREPARED_COLLECTION_PRIVATE.json')['modules'];n=alias=0;seals={}
 for key in matrix['directions']:
  co,name=key.split('/');src,_,tgt=name.split('_');S0=np.load(R/col[src]['state_path']);S={k[6:]:S0[k] for k in S0.files if k.startswith('match.')};D=B/co/name;assert not D.exists()
  for sub in ['samples_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
  for e in [e for e in reg if e['cohort']==co and e['source']==src and e['target']==tgt]:
   tick=time.perf_counter();out,record=process(e,S);a=np.stack([out[k] for k in NEW]);assert a.shape==(15,256) and np.isfinite(a).all() and np.allclose(a.sum(1),1,atol=1e-10);sid=e['row']['id'];np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(NEW),scores=a,queries=a.argmax(1));record['seconds']=time.perf_counter()-tick;(D/'records_PRIVATE'/(sid+'.json')).write_text(json.dumps(record,indent=2)+'\n');n+=1;alias+=record['protected_alias_fallback']
  (D/'SEALED.json').write_text(json.dumps({'n':sum(e['cohort']==co and e['source']==src and e['target']==tgt for e in reg),'files':{str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file()}},indent=2)+'\n');seals[key]=sha(D/'SEALED.json');print('RUN126',key,n,flush=True)
 assert n==2304 and alias==154;(B/'MATRIX_SEALED.json').write_text(json.dumps({'n':n,'new_arms':NEW,'directions':seals,'aliases':alias,'primary':'fullval_corr_F','base_R125_seal_sha256':sha(R/'crossplatform/round125/MATRIX_SEALED.json'),'GT_used':False,'FINAL_access':False},indent=2)+'\n');print('COMPLETE126',n,flush=True)
if __name__=='__main__':main()
