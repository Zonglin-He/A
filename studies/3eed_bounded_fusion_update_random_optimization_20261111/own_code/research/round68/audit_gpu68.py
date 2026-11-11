"""Independent seed repeats and every suffix output, reuse R67 source proof."""
import sys,time,torch
from common68 import *
from independent_gpu68 import suffix,readout
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
H=ROOT/'research/round68';p=read(H/'experiment/GPU_AUDIT_PLAN_LOCK.json');pin(p);rows=read(ROOT/p['manifest']);cap=ROOT/p['random_capture_root'];cs=read(cap/'SEALED.json');model=load_source(read(ROOT/p['source_plan']));torch.set_grad_enabled(False);fp=fingerprint(model);start=time.perf_counter();count=0
for i,r in enumerate(rows):
 rel='samples_PRIVATE/'+r['id']+'.npz';assert sha(cap/rel)==cs['files'][rel];saved=load(cap/rel);allz=load(ROOT/p['r67_capture_root']/rel)
 for v in 'ABC':
  z={k[len(v)+1:]:val for k,val in allz.items() if k.startswith(v+'_')};T=lambda k:torch.tensor(z[k],device='cuda');base=allz['native_'+v+'_SA3_indices'][:256].tolist()
  for rep in [1,2]:
   ss=int(hashlib.sha256(('R68random_repeat'+str(rep)+'|'+r['id']+'|'+v).encode()).hexdigest()[:16],16);order=np.random.default_rng(ss).permutation(1024).tolist();seq=base.copy()
   for j in order:
    if j not in seq:seq.append(j)
    if len(seq)==512:break
   ids=np.array(seq,np.int64);assert np.array_equal(ids,saved['s'+str(rep)+'_'+v+'_SA3_indices']);vis=suffix(model,T('SA2_xyz'),T('SA2_feat'),ids);out=readout(model,z,vis);count+=1
   for k in ['h','query','tokens']:assert np.array_equal(out[k],saved['s'+str(rep)+'_'+v+'_'+k]),(i,v,rep,k)
 if(i+1)%64==0:print('AUDIT_GPU68',i+1,count,flush=True)
assert fingerprint(model)==fp;out=H/'independent_capture_audit_attempt001';assert not out.exists();out.mkdir();dump(out/'AUDIT_RESULT.json',{'state':'PASS_R68_TWO_FIXED_RANDOM_SEEDS_AND_ALL_SUFFIXES_BITEXACT','n':336,'independent_suffix_replays':count,'new_full_source_forwards':0,'source_unchanged':True,'max_feature_error':0.,'random_capture_seal_sha256':sha(cap/'SEALED.json'),'R67_source_provenance_audit_sha256':sha(ROOT/'research/round67/independent_capture_audit_attempt001/AUDIT_RESULT.json'),'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())});print({'n':336,'replays':count})
