"""Same all-raw correspondence rule and unchanged R59 fusion for every policy."""
import sys,time,torch
from common65 import *
sys.path.insert(0,str(ROOT/'research/round57'));from associate57 import affinity,z_native
sys.path.insert(0,str(ROOT/'research/round62'));from compute62 import readouts,cpu_head
from scipy.special import softmax
p=read(ROOT/'research/round65/experiment/EXPERIMENT_SPEC_LOCK.json');pin(p);rows=read(ROOT/p['manifest']);cap=ROOT/p['new_capture_root'];cs=read(cap/'SEALED.json');out=ROOT/p['prediction_root'];assert not out.exists();out.mkdir(parents=True)
for k in ['samples_PRIVATE','cases_PRIVATE','association_PRIVATE']:(out/k).mkdir()
head=cpu_head(load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz'));torch.set_grad_enabled(False);torch.set_num_threads(4);start=time.perf_counter();statuses=[]
for i,r in enumerate(rows):
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(cap/rel)==cs['files'][rel];new=load(cap/rel);old=load(ROOT/p['capture_root']/rel);prior=load(ROOT/p['r59_root']/rel);rec=read(ROOT/p['prepared_root']/'cases_PRIVATE'/(sid+'.json'));assert sha(r['pointpath'])==rec['raw_sha256'];path=Path(r['pointpath']);raw=np.load(path,allow_pickle=False) if path.suffix=='.npy' else np.fromfile(path,np.float32).reshape(-1,4);x=np.array(raw[:,:3],np.float32,copy=True)
 if r['platform']=='drone':x[:,2]+=1.8
 result={name:prior['scores'][list(prior['arm_names']).index(key)] for name,key in [('source','source'),('old_ABC','frozen_ABC'),('old_R59','direct_all')]};assocs={};details={}
 for mode in ['common']+p['sampling_modes']:
  z={k:old[k] for k in ['A_h','A_query','A_tokens','A_token_mask','A_boxes']}
  for v in 'BC':
   for k in ['h','query','tokens','token_mask','boxes']:z[v+'_'+k]=old[v+'_'+k] if mode=='common' else new[mode+'_'+v+'_'+k]
  assert all(np.array_equal(z[v+'_token_mask'],z['A_token_mask']) for v in 'BC');base=softmax(z_native(z,'A_'));g={'base':base};dd={}
  for v in 'BC':
   af=affinity(z['A_boxes'].astype(float),z[v+'_boxes'].astype(float),x.astype(float));w=af.astype(float)/(1+af.astype(float).sum(0,keepdims=True));mass=float((w@softmax(z_native(z,v+'_'))).sum());g[v+'_affinity']=af;g[v+'_W']=w;g[v+'_fixed_mass']=np.array(mass);dd[v+'_known_mass']=mass
  mean,fused,_=readouts(head,{v:z[v+'_h'] for v in 'ABC'},z,g);result[mode+('_ABC' if mode=='common' else '_mean')]=mean;result[mode+'_R59']=fused
  for k,val in g.items():assocs[mode+'_'+k]=val
  details[mode]=dd
 scores=np.stack([result[a] for a in ARMS]);q=scores.argmax(1);assert np.isfinite(scores).all();np.savez_compressed(out/rel,arm_names=np.array(ARMS),scores=scores,queries=q,source_boxes=old['A_boxes'],source_scores=old['A_scores'],selected_original_boxes=old['A_boxes'][q]);np.savez_compressed(out/'association_PRIVATE'/(sid+'.npz'),**assocs);dump(out/'cases_PRIVATE'/(sid+'.json'),details);statuses.append({'id':sid,'arm_statuses':{a:{'unknown':False,'fallback':False,'no_parameter_update':True} for a in ARMS}})
 if (i+1)%32==0:print('PREDICT65',i+1,len(rows),flush=True)
dump(out/'READOUT_STATUS_PRIVATE.json',statuses);dump(out/'COSTS.json',{'n':len(rows),'arms':len(ARMS),'seconds':time.perf_counter()-start,'new_gradient_steps':0,'raw_support_common_across_policies':True});seal(out,'R65_ALL_FIXED_A256_POLICY_PREDICTIONS_SEALED_BEFORE_SCORE',n=336,arm_names=ARMS)
