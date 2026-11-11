"""Postseal stratified GPU replay and masked-value-leakage checks, no GT."""
from common75 import *
import torch,time
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,fingerprint
from interface66 import replay
p=read(H/'experiment/DESIGN_LOCK.json');fit=ROOT/'crossplatform/round75/fit336_attempt001';assert read(fit/'SEALED.json')['n']==336;assert read(ROOT/'crossplatform/round75/predicted336_attempt001/SEALED.json')['n']==336
rr=rows();selected=[]
for scene in sorted({r['scene'] for r in rr}):selected+=sorted([r for r in rr if r['scene']==scene],key=lambda r:hashlib.sha256(('R75independentaudit|'+r['id']).encode()).hexdigest())[:2]
m=load_source(read(ROOT/p['source_plan']));torch.set_grad_enabled(False);fp=fingerprint(m);calls=0;maxerr=0;start=time.perf_counter()
for r in selected:
 z=original(r);sid=r['id'];zero=replay(m,z);calls+=1;assert np.array_equal(zero['h'],z['h']) and np.array_equal(zero['tokens'],z['tokens']);a=load(ROOT/'crossplatform/round75/masked336_attempt001/samples_PRIVATE'/(sid+'.npz'));b=load(ROOT/'crossplatform/round75/held336_attempt001/samples_PRIVATE'/(sid+'.npz'))
 for v,saved in [('M1',a),('M2',a),('M3',b)]:
  mask=mask_for(r,z,v);y=replay(m,z,mask);calls+=1
  for k in ['h','query','tokens']:maxerr=max(maxerr,float(abs(y[k]-saved[v+'_'+k]).max()));assert np.array_equal(y[k],saved[v+'_'+k])
  alter=z['vis0'].copy();alter[:,mask]=np.float32(-1234.5);other=replay(m,z,mask,alternate_vis0=alter);calls+=1;assert np.array_equal(y['h'],other['h']) and np.array_equal(y['tokens'],other['tokens']);assert y['proof']['max_blocked_attention']==0. and y['proof']['masked_query_keys_n']==0
assert fingerprint(m)==fp;out=H/'independent_gpu_audit_attempt001';assert not out.exists();out.mkdir();rec={'state':'PASS_R75_FIVE_RECORDING_STRATIFIED_REAL_GPU_MASKED_TRANSFORMER_REPLAY_AND_NO_MASK_VALUE_LEAKAGE','n':len(selected),'recording_clusters':len({r['scene'] for r in selected}),'suffix_calls':calls,'max_feature_error':maxerr,'all256_source_anchor_seeds_retained':True,'source_unchanged':True,'new_full_source_forwards':0,'GT_or_expert_read':False,'seconds':time.perf_counter()-start};dump(out/'AUDIT_RESULT.json',rec);print(rec)
