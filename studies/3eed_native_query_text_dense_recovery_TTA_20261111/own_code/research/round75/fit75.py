from common75 import *
from numerics75 import *
import time

def main():
 p=read(H/'experiment/DESIGN_LOCK.json');pin(p);pin(read(H/'experiment/FIT_CODE_LOCK.json'));inputroot=ROOT/'crossplatform/round75/masked336_attempt001';ss=read(inputroot/'SEALED.json');assert ss['n']==336;initial=load(ROOT/'crossplatform/round50/episodic336_capture_attempt001/SOURCE_MATCHING_MLP_PRIVATE.npz');out=ROOT/'crossplatform/round75/fit336_attempt001';assert not out.exists();out.mkdir(parents=True)
 for f in ['weights_PRIVATE','cases_PRIVATE','reference_PRIVATE']:(out/f).mkdir()
 start=time.perf_counter();counts={k:0 for k in ADAPT};torch.set_num_threads(4)
 for i,r in enumerate(rows()):
  sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(inputroot/rel)==ss['files'][rel];z=original(r);masked=load(inputroot/rel);seed=int(hashlib.sha256(('R75rowderangement|'+sid).encode()).hexdigest()[:16],16);cycle=np.random.default_rng(seed).permutation(256);perm=np.empty(256,np.int64);perm[cycle]=np.roll(cycle,1);assert np.all(perm!=np.arange(256));d=data(initial,z,masked,perm);np.savez_compressed(out/'reference_PRIVATE'/(sid+'.npz'),query=d['q0'].numpy(),S0=d['S0'].numpy(),R0=d['R0'].numpy(),C0=d['C0'].numpy(),permutation=perm)
  for kind in ADAPT:
   states,rec=one(initial,d,kind);np.savez_compressed(out/'weights_PRIVATE'/(sid+'_'+kind+'.npz'),**states);dump(out/'cases_PRIVATE'/(sid+'_'+kind+'.json'),rec);counts[kind]+=int(rec['accepted'])
  if(i+1)%32==0:print('FIT75',i+1,counts,flush=True)
 cost={'n':336,'head_states':336*3,'accepted':counts,'updated_parameters':sum(v.size for v in initial.values()),'M3_read':False,'reference_text_fixed':True,'seconds':time.perf_counter()-start};dump(out/'COUNTS.json',cost);seal(out,'R75_ALL_M1_M2_ONLY_HEAD_STATES_SEALED_BEFORE_M3',n=336);print(cost)
if __name__=='__main__':main()
