"""Locked 72-input GPU replay sample; all-input checks live in CPU audit."""
import sys,time,torch
from common69 import *
import independent69 as I
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,source_capture,fingerprint
sys.path.insert(0,str(ROOT/'crossplatform/round7/experiment'));from runner_helpers import sample_points
H=ROOT/'research/round69';p=read(H/'experiment/AUDIT_GPU_PLAN_LOCK.json');pin(p);coverage=read(H/'experiment/AUDIT_COVERAGE_LOCK.json');ids=set(coverage['chosen_ids_PRIVATE']);rows=[r for r in read(ROOT/p['manifest']) if r['id'] in ids];model=load_source(read(ROOT/p['source_plan']));fp=fingerprint(model);torch.set_grad_enabled(False);start=time.perf_counter();full=regional=random=0
for i,r in enumerate(rows):
 rel='samples_PRIVATE/'+r['id']+'.npz';ab=load(ROOT/p['AB_capture_root']/rel);c=load(ROOT/p['C_random_root']/rel);e=load(ROOT/p['evidence_root']/rel)
 for v,seed in [('A',r['seed']),('B',bseed(r)),('C',cseed(r))]:
  z={k[2:]:x for k,x in ab.items() if k.startswith(v+'_')} if v!='C' else c;pts,ix,_=sample_points(dict(r,seed=seed));capt={}
  def hook(_m,args):capt['xyz']=args[0].detach().clone();capt['feat']=args[1].detach().clone()
  hk=model.backbone_net.SA_modules[2].register_forward_pre_hook(hook)
  try:o,_=source_capture(model,pts,ix,r['caption']);full+=1
  finally:hk.remove()
  for k in ['h','query','tokens','boxes','point_indices','query_points_sample_inds']:assert np.array_equal(o[k],z[k]),('source',i,v,k)
  assert np.array_equal(capt['xyz'].cpu().numpy(),z['SA2_xyz']) and np.array_equal(capt['feat'].cpu().numpy(),z['SA2_feat'])
  if v!='C':
   controls=I.controls(z['xyz'][0].astype(float),e[v+'_region_masks'],z['query_points_sample_inds'])
   for ti,category in enumerate([False,True]):
    y=I.replay(model,z,category=category);regional+=1;pp=y['p'] if v=='A' else I.mapped(y['p'],e['B_W'],float(e['B_fixed_mass']),e['base']);assert np.array_equal(pp,e['baseline']['AB'.index(v),ti])
   for j,(control,ok) in enumerate(controls):
    assert np.array_equal(control,e[v+'_control_masks'][j])
    if not e['valid']['AB'.index(v),j]:continue
    for ti,category in enumerate([False,True]):
     for ki,mask in enumerate([e[v+'_region_masks'][j],control]):
      y=I.replay(model,z,mask,category);regional+=1;pp=y['p'] if v=='A' else I.mapped(y['p'],e['B_W'],float(e['B_fixed_mass']),e['base']);assert np.array_equal(pp,e['perturbed']['AB'.index(v),ti,ki,j])
  from ops.teed_pointnet.pointnet2_batch import pointnet2_utils as U
  D=U.furthest_point_sample(capt['xyz'].contiguous(),1024)[0].cpu().tolist();base=[]
  for j in D+list(range(1024)):
   if j not in base:base.append(j)
   if len(base)==256:break
  for repeat in range(3):
   prefix='R67random' if repeat==0 else 'R68random_repeat'+str(repeat);rng=np.random.default_rng(int(hashlib.sha256((prefix+'|'+r['id']+'|'+v).encode()).hexdigest()[:16],16));ix=base.copy()
   for j in rng.permutation(1024):
    if int(j) not in ix:ix.append(int(j))
    if len(ix)==512:break
   ix=np.array(ix,np.int64);assert np.array_equal(ix,c['s'+str(repeat)+'_'+v+'_SA3_indices']);vis=I.suffix(model,capt['xyz'],capt['feat'],ix);out=I.readout(model,z,vis);random+=1
   for k in ['h','query','tokens']:assert np.array_equal(out[k],c['s'+str(repeat)+'_'+v+'_'+k]),('random',i,v,repeat,k)
 if(i+1)%8==0:print('AUDIT_GPU69',i+1,len(rows),full,regional,random,flush=True)
assert fingerprint(model)==fp;out=H/'independent_gpu_audit_attempt001';assert not out.exists();out.mkdir();dump(out/'AUDIT_RESULT.json',{'state':'PASS_R69_PRELOCKED_STRATIFIED_SOURCE_REGION_AND_RANDOM_GPU_REPLAY','audited_inputs':len(rows),'total_inputs':p['n'],'full_source_replays':full,'region_encoder_decoder_replays':regional,'random_suffix_replays':random,'all_arrays_in_audited_sample_bitexact':True,'source_unchanged':True,'all_input_GPU_replay_not_claimed':True,'prediction_seal_sha256':sha(ROOT/p['prediction_root']/'SEALED.json'),'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated())});print({'audited_inputs':len(rows),'total':p['n']})
