"""Independent full-source/sample audit, parallel only with CPU readout work."""
from common65 import *
import audit65 as a,sys,time,torch
sys.path.insert(0,str(ROOT/'crossplatform/round51'));from model51 import load_source,source_capture,fingerprint
H=ROOT/'research/round65';p=read(H/'experiment/CAPTURE_AUDIT_PLAN_LOCK.json');pin(p);rows=read(ROOT/p['manifest']);cap=ROOT/p['new_capture_root'];cs=read(cap/'SEALED.json');prep=ROOT/p['prepared_root'];ps=read(prep/'SEALED.json');roles={r['id_PRIVATE']:np.array(r['mask'],bool) for r in read(ROOT/'research/round62/role_diagnostics_attempt001/MASKS_PRIVATE.json') if r['role']=='reference'};model=load_source(read(ROOT/p['source_plan']));torch.set_grad_enabled(False);fp=fingerprint(model);start=time.perf_counter();calls=0
for i,r in enumerate(rows):
 sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(cap/rel)==cs['files'][rel] and sha(prep/rel)==ps['files'][rel];zold=load(ROOT/p['capture_root']/rel);znew=load(cap/rel);prepared=load(prep/rel);rawrec=read(prep/'cases_PRIVATE'/(sid+'.json'));assert sha(r['pointpath'])==rawrec['raw_sha256'];path=Path(r['pointpath']);raw=np.load(path,allow_pickle=False) if path.suffix=='.npy' else np.fromfile(path,np.float32).reshape(-1,4);x=np.array(raw[:,:3],np.float32,copy=True)
 if r['platform']=='drone':x[:,2]+=1.8
 generated,targets,refs,focus,context=a.samples(x,zold,roles[sid],sid);assert np.array_equal(prepared['target_hypothesis_indices'],targets) and np.array_equal(prepared['reference_hypothesis_indices'],refs) and np.array_equal(prepared['focus_indices'],focus) and np.array_equal(prepared['context_indices'],context);actualstats=read(cap/'cases_PRIVATE'/(sid+'.json'))
 for key,ix in generated.items():
  assert np.array_equal(ix,prepared[key+'_point_indices']) and np.array_equal(ix,znew[key+'_point_indices']);extra={}
  def save(_m,_i,o):extra['seed_xyz']=o['fp2_xyz'][0].detach().cpu().numpy().copy()
  hk=model.register_forward_hook(save)
  try:zz,_=source_capture(model,x[ix],ix,r['caption']);calls+=1
  finally:hk.remove()
  for k in ['boxes','scores','h','query','tokens','token_mask']:assert np.array_equal(zz[k],znew[key+'_'+k]),('source_replay',i,key,k)
  assert np.array_equal(extra['seed_xyz'],znew[key+'_seed_xyz']);assert a.process_stats(ix,zold['A_point_indices'],focus,context,extra['seed_xyz'],zold['A_boxes'],targets,refs,len(x))==actualstats[key]
 if (i+1)%32==0:print('AUDIT_CAPTURE65',i+1,336,calls,flush=True)
assert fingerprint(model)==fp;out=H/'independent_capture_audit_attempt001';assert not out.exists();out.mkdir();rr={'state':'PASS_R65_ALL_PAIRED_SAMPLES_AND_2016_FULL_SOURCE_FORWARD_REPLAY','n':336,'new_full_source_replay_forwards':calls,'source_restored_exact':True,'all_prepared_point_indices_exact':True,'all_full_coverage_views_cover_every_raw_point':True,'all_backbone_seed_and_process_counts_exact':True,'capture_seal_sha256':sha(cap/'SEALED.json'),'preparation_seal_sha256':sha(prep/'SEALED.json'),'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'program_sha256':sha(__file__),'helper_sha256':sha(H/'audit65.py')};dump(out/'AUDIT_RESULT.json',rr);print(rr)
