"""New random repeats from sealed SA2 states; no full backbone rerun or GT."""
import sys,time,torch
from common68 import *
import random68 as s
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
H=ROOT/'research/round68';p=read(H/'experiment/DESIGN_LOCK.json');pin(p);rows=read(ROOT/p['manifest']);src=ROOT/p['r67_capture_root'];ss=read(src/'SEALED.json');cert=read(ROOT/'research/round67/independent_capture_audit_attempt001/AUDIT_RESULT.json');assert cert['capture_seal_sha256']==sha(src/'SEALED.json');out=ROOT/p['random_capture_root'];assert not out.exists();out.mkdir(parents=True);(out/'samples_PRIVATE').mkdir();m=load_source(read(ROOT/p['source_plan']));torch.set_grad_enabled(False);fp=fingerprint(m);start=time.perf_counter();calls=0
for i,r in enumerate(rows):
 rel='samples_PRIVATE/'+r['id']+'.npz';assert sha(src/rel)==ss['files'][rel];allz=load(src/rel);payload={}
 for v in 'ABC':
  z={k[len(v)+1:]:val for k,val in allz.items() if k.startswith(v+'_')};z['native_SA3_indices']=allz['native_'+v+'_SA3_indices']
  for rep in [1,2]:
   rr=s.predict(m,z,rep,r['id']+'|'+v);calls+=1
   for k in ['h','query','tokens','SA3_indices']:payload['s'+str(rep)+'_'+v+'_'+k]=rr[k]
 np.savez_compressed(out/rel,**payload)
 if(i+1)%32==0:print('RANDOM68',i+1,336,calls,flush=True)
assert fingerprint(m)==fp;dump(out/'COSTS.json',{'n':336,'new_full_source_forwards':0,'new_random_suffix_replays':calls,'source_unchanged':True,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())});seal(out,'R68_TWO_PRESPECIFIED_RANDOM_REPEATS_FIXED_SA3_AND_ANCHORS_NO_GT',n=336);print({'n':336,'replays':calls})
