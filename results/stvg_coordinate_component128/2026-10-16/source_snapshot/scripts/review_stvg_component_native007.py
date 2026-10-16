"""Pre-GT independent saved-arithmetic and full-readout audit for deletion trials."""
from pathlib import Path
import sys,math,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_component_ablation_common_v1 import *
NATIVE = BASE/'native_clock/revision002'

def fit_audit(f,ref,flags):
 import torch,numpy as np
 from scripts.audit_stvg_soft_ep_v1 import arr,err,iou,_optimizer,treeeq
 pos=[a['position']for a in ref['anchors']if a['weight']>0];ev=np.asarray([a['box_cxcywh']for a in ref['anchors']if a['weight']>0]);weights=np.asarray([a['weight']for a in ref['anchors']if a['weight']>0]);weights=np.ones_like(weights)if flags.get('uniform_reliability')else weights
 names=f.get('active_names',list(f['initial_state']));expected=[n for n in f['initial_state']if (not flags.get('query_only')or n=='spatial.query_residual')and(not flags.get('ln_only')or n!='spatial.query_residual')]
 assert names==expected and f['active_parameters']==sum(arr(f['initial_state'][n]).size for n in names)
 assert f.get('ablation_flags',{})==flags and f['observation_positions']==pos and not f['GT_read']
 cfg=f['config'];settings=f['settings'];key=f['seed_key'];prev=f['initial_state'];prevopt=None;accepted=proposed=0;stopped=False;u=f['protection_positions'];assert u==[i for i in range(len(f['final_cached_decoder_boxes']))if i not in pos]
 assert len(f['path'])==cfg['steps']
 for step,r in enumerate(f['path']):
  assert r['step']==step and treeeq(r['state_before'],prev)
  if prevopt is not None:assert treeeq(prevopt,r['optimizer_before'])
  if stopped:assert r['skipped']and r['reason']=='stopped_after_rejected_step';continue
  if 'rollout'not in r:raise AssertionError('missing live rollout')
  ro=r['rollout'];mu=arr(ro['mean']);z=arr(ro['samples']);q=arr(ro['weights']);before=arr(r['boxes_before']);assert z.shape==(len(pos),32,4)
  seed=int(hashlib.sha256(f'spatial-opd-v1|{key}|{step}'.encode()).hexdigest()[:15],16);assert ro['seed']==seed and ro['permutations']is None
  gen=torch.Generator().manual_seed(seed);noise=torch.randn(len(pos),32 if flags.get('iid_sampling')else 16,4,generator=gen)
  if not flags.get('iid_sampling'):noise=torch.cat([noise,-noise],1)
  expected_z=mu[:,None,:]+np.float32(cfg['sigma'])*noise.numpy();assert np.array_equal(expected_z,z)
  assert err(mu,np.log(before[pos].astype(float)/(1-before[pos].astype(float))))<2e-6
  rewards=iou(1/(1+np.exp(-z.astype(float))),ev[:,None,:]);assert err(rewards,ro['rewards'])<3e-6 and np.array_equal(arr(ro['rewards']),arr(ro['used_rewards']))
  logits=arr(ro['used_rewards']).astype(float)/cfg['tau'];logits-=logits.max(-1,keepdims=True);qq=np.exp(logits);qq/=qq.sum(-1,keepdims=True);assert err(qq,q)<2e-6
  logp=-np.square(z.astype(float)-mu[:,None,:]).sum(-1)/(2*cfg['sigma']**2)-2*math.log(2*math.pi*cfg['sigma']**2);pa=-((q-(0. if flags.get('no_control_variate')else 1/32))*logp).sum(-1)
  tol=2e-6+8*np.finfo(np.float32).eps*max(1.,float(np.abs(pa).max()))
  assert err(pa,r['per_anchor_loss'])<tol and abs(float((weights*pa).mean())-r['loss'])<tol and err(weights,r['reliability'])<1e-7
  assert err(iou(before[pos],ev),r['reference_iou_before'])<2e-6
  info=bool(np.any(arr(ro['used_rewards'])!=arr(ro['used_rewards'])[:,:1]));assert info==ro['informative']
  if r['skipped']:
   assert not info and not flags.get('no_flat_skip')and r['reason']=='no_information'and 'gradient'not in r
   prevopt=r['optimizer_before'];continue
  assert info or flags.get('no_flat_skip');proposed+=1
  assert max(_optimizer(r,names))<3e-6
  d=arr(r['original_adam_delta']);v=arr(r['filtered_delta']);assert len(v)==f['active_parameters']and np.isfinite(v).all()
  solver=r['solver']
  if flags.get('no_soft_direction'):assert not solver and np.array_equal(d,v)
  if solver:
   solution=arr(solver['solution']);assert err(v,solution)<1e-6
   jd=arr(solver['Jd']).astype(float);d64=d.astype(float)
   if solver['status']!='near_zero_response_identity':assert abs(solver['lambda']-solver['mu']*float(d64@d64)/float(jd@jd))<1e-4*max(1.,solver['lambda'])
   assert abs(solver['rho']-np.linalg.norm(v.astype(float))/max(1e-30,np.linalg.norm(d64)))<2e-5
   if 'normal_applied'in solver:assert err(arr(solver['cg_rhs'])-arr(solver['normal_applied']),solver['cg_residual'])<2e-6
  original_rms=float(np.sqrt(np.square(arr(r['original_proposal_boxes']).astype(float)[u]-before.astype(float)[u]).mean()))if u else 0.
  assert abs(original_rms-r['response_original_RMS'])<1e-6 and abs(r['response_budget']-original_rms*settings['response_ratio'])<1e-6
  candidates=r['candidates'];scales=[1.]if flags.get('no_backtracking')else settings['backtracking'];assert [c['scale']for c in candidates]==scales[:len(candidates)]
  for j,c in enumerate(candidates):
   train=bool(flags.get('no_reference_gate'))or float((weights*(arr(c['anchor_IoU'])-arr(r['reference_iou_before']))).sum()/weights.sum())>=-1e-7
   # Use saved FP32 reduction for exact gate bit; independently reconstruct accepted final below.
   response=bool(flags.get('no_response_gate'))or c['U_RMS']<=r['response_budget']+settings['response_abs_tolerance']
   assert response==c['response_pass']
   assert train==c['train_pass']or abs(float((weights*(arr(c['anchor_IoU'])-arr(r['reference_iou_before']))).sum()/weights.sum())+1e-7)<2e-7
   assert bool(c['accepted'])==bool(c['train_pass']and c['response_pass']) and(j==len(candidates)-1 or not c['accepted'])
  if r['accepted']:
   accepted+=1;assert candidates[-1]['accepted']and candidates[-1]['scale']==r['accepted_scale']and not r['rollback'];off=0
   for n,b in prev.items():
    if n in names:
     size=arr(b).size;want=arr(b)+v[off:off+size].reshape(arr(b).shape)*np.float32(r['accepted_scale']);off+=size;assert err(want,r['state_after'][n])<3e-7
    else:assert treeeq(b,r['state_after'][n])
   assert off==len(v)and treeeq(r['optimizer_after'],r['optimizer_proposed'])
   after=arr(r['boxes_after']).astype(float);rms=float(np.sqrt(np.square(after[u]-before.astype(float)[u]).mean()))if u else 0.
   assert abs(rms-candidates[-1]['U_RMS'])<1e-6 and err(iou(after[pos],ev),candidates[-1]['anchor_IoU'])<2e-6
   if not flags.get('no_reference_gate'):assert float((weights*(iou(after[pos],ev)-iou(before[pos],ev))).sum()/weights.sum())>=-1e-6
   if not flags.get('no_response_gate'):assert rms<=r['response_budget']+settings['response_abs_tolerance']+1e-7
  else:
   assert not any(c['accepted']for c in candidates)and r['rollback']and treeeq(r['state_after'],prev)and treeeq(r['optimizer_after'],r['optimizer_before']);stopped=not flags.get('no_rejection_stop')
  if r['norm_control']:
   nc=r['norm_control'];a=arr(nc['matched_direction']);ss=arr(nc['soft_direction']).astype(float);dd=arr(nc['original_direction']).astype(float);want=dd*(np.linalg.norm(ss)/np.linalg.norm(dd));eps=np.finfo(a.dtype).eps
   assert err(a,want)<=8*eps*max(1e-12,float(np.abs(want).max()))+1e-12
   assert abs(np.linalg.norm(a.astype(float))-np.linalg.norm(ss))<=max(2e-6,8*eps*np.linalg.norm(ss))
  if flags.get('no_equal_norm_diagnostic'):assert r['norm_control']is None
  prev=r['state_after'];prevopt=r['optimizer_after']
 assert treeeq(prev,f['final_state'])and accepted==f['optimizer_steps']and proposed==f['proposed_optimizer_steps']
 return {'accepted':accepted,'proposals':proposed}

def main():
 import torch,numpy as np
 from scripts.audit_stvg_soft_ep_v1 import treeeq,arr
 from scripts.run_stvg_table1_student004 import stable_fit_key
 final=verify_final();verify_preparation();b=NATIVE/'trials'/sys.argv[1];rt=read(b/'RUNTIME.json');design=read(b/'DESIGN_LOCK.json')
 for rel,h in {**rt['pins'],**rt['protected']}.items():assert sha(ROOT/rel)==h,rel
 bar=read(b/'STUDENT_GLOBAL_BARRIER.json');assert bar['status']=='complete'and bar['count']==128 and not bar['GT_read']and bar['runtime_sha256']==sha(b/'RUNTIME.json')
 refs={x['case_id']:x for x in read(NATIVE/'REFERENCE_LOCK.json')['records']};ts={x['case_id']:x for x in read(NATIVE/'TEMPORAL_LOCK.json')['records']};rr={x['probe_alias']:x for x in roster()};seen=set();counts={'accepted':0,'proposals':0,'no_reference':0}
 for x in bar['files']:
  p=ROOT/x['path'];receipt=p.with_suffix('.receipt.json');assert sha(p)==x['sha256']and sha(receipt)==x['receipt_sha256']and read(receipt)['sha256']==sha(p)and read(receipt)['status']=='complete'
  d=torch.load(p,map_location='cpu',weights_only=False);cid=d['case_id'];assert cid not in seen;seen.add(cid);r=rr[cid];c,sp=load_case(r);ds=r['dataset'];ref=refs[cid]
  assert d['source_restored']and not d['GT_read']and d['runtime_sha256']==sha(b/'RUNTIME.json')and d['source_file_sha256']==sha(sp)and d['source_case_sha256']==c['source_case_sha256']and d['reference_records']==ref and set(d['arms'])==set(ARMS)
  assert np.array_equal(d['arms']['F']['boxes'],c['native_boxes_xyxy_normalized'])and np.array_equal(d['arms']['T']['boxes'],d['arms']['F']['boxes'])and d['arms']['F']['interval']==c['Native_interval']and d['arms']['T']['interval']==ts[cid]['selected_interval']
  f=d['fits']['M2'];o=d['readouts']['M2']
  if not ref['anchors']:
   assert f is o is None;counts['no_reference']+=1;assert np.array_equal(d['arms']['S_M2']['boxes'],d['arms']['F']['boxes'])and np.array_equal(d['arms']['TS_M2']['boxes'],d['arms']['T']['boxes']);continue
  assert f['config']==final['configs'][ds]and f['settings']==final['response_settings'][ds]and f['seed_key']==stable_fit_key(ds,r['query_key_sha256'])and treeeq(f['initial_state'],c['source_light_initial_state'])
  flags=design['ablation_flags']
  if design['variant']=='baseline':
   ob=SEARCH/'trials'/final['actual_baseline_trial'][ds];assert read(ob/'ROOT_READBACK.json')['status']=='pass';z=torch.load(ob/'student'/(cid+'.pt'),map_location='cpu',weights_only=False);ff=dict(f);assert ff.pop('reused_fit_from_sha256')==sha(ob/'student'/(cid+'.pt'));ff.pop('reused_fit_from_trial',None);assert treeeq(ff,z['fits']['M2'])and treeeq(o,z['readouts']['M2'])
  a=fit_audit(f,ref,flags)
  for k in a:counts[k]+=a[k]
  assert torch.equal(o['boxes'].cpu(),f['final_cached_decoder_boxes']);bb=arr(o['boxes']).astype(float);xy=np.concatenate([bb[:,:2]-bb[:,2:]/2,bb[:,:2]+bb[:,2:]/2],1)
  assert np.array_equal(xy,d['arms']['S_M2']['boxes'])and np.array_equal(xy,d['arms']['TS_M2']['boxes'])and d['arms']['S_M2']['interval']==c['Native_interval']and d['arms']['TS_M2']['interval']==ts[cid]['selected_interval']
 assert seen==set(rr)and len(bar['files'])==128
 write(b/'ROOT_READBACK.json',dict(status='pass',count=128,counters=counts,GT_read=False,CPU_only=True,runtime_sha256=sha(b/'RUNTIME.json'),barrier_sha256=sha(b/'STUDENT_GLOBAL_BARRIER.json'),scope='saved source/state/reward/loss/Adam/solver witnesses and actual full readout; no independent full model backward'))
 files=list(bar['files']);files.extend({'path':x['receipt_path'],'sha256':x['receipt_sha256']}for x in bar['files'])
 for p in [b/n for n in ['DESIGN_LOCK.json','RUNTIME.json','ROOT_READBACK.json','STUDENT_GLOBAL_BARRIER.json']]+[Path(__file__),BASE/'FINAL_PARAMETER_LOCK.json']+[SEARCH/'ROSTER_LOCK.json',NATIVE/'REFERENCE_LOCK.json',NATIVE/'TEMPORAL_LOCK.json']:files.append({'path':str(p.relative_to(ROOT)),'sha256':sha(p)})
 write(b/'GLOBAL_PREDICTION_BARRIER.json',dict(status='complete',count=128,files=files,runtime_sha256=sha(b/'RUNTIME.json'),GT_read=False,student_barrier_sha256=sha(b/'STUDENT_GLOBAL_BARRIER.json'),root_readback_sha256=sha(b/'ROOT_READBACK.json'),scorer_sha256=sha(ROOT/'scripts/measure_stvg_component_native007.py')))
if __name__=='__main__':main()
