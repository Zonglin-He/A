"""Independent CPU realized-output, sparse reward, RN/control-variate and metric audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_realizable_query_opd32_v1 import D,P,PRE,CLOSED,FEAT,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha

def main():
 import torch,numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.audit_stvg_discriminative_dense32_v1 import independent_metrics
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['all32_outputs_before_NEW_GT']and bar['count']==32;z=read(P/'RESULTS_PUBLIC.json');rows={r['alias']:r for r in z['rows']};checks=collections.Counter();maxerr=0.
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
 def sparse_iou(b,pos,target):
  if not pos:return 0.
  total=[]
  for j,i in enumerate(pos):
   x=np.asarray(b[i]);y=target[j];l=np.maximum(x[:2]-x[2:]/2,y[:2]-y[2:]/2);h=np.minimum(x[:2]+x[2:]/2,y[:2]+y[2:]/2);a=np.maximum(h-l,0).prod();total.append(a/max(x[2]*x[3]+y[2]*y[3]-a,1e-30))
  return float(sum(total)/len(pos))
 def importance(a,m,c,reward,valid):
  lr=np.array([float(-np.dot(v-m,v-m)/2+np.dot(v-c,v-c)/2)for v in a]);v=np.zeros(len(a));w=v.copy()
  if not valid.any():return v,w,np.zeros(256),False
  def sm(x):y=np.exp(x-x.max());return y/y.sum()
  v[valid]=sm(lr[valid]);info=bool(np.ptp(reward[valid])>1e-12)
  if info:w[valid]=sm(lr[valid]+reward[valid]/.1)
  else:w=v.copy()
  d=np.zeros(256)
  if info:
   for j in range(len(a)):d+=(w[j]-v[j])*a[j]
  return v,w,d,info
 for row in p['records']:
  a=row['alias'];o=load(D/'predictions'/(a+'.pt'));x=load(FEAT/'features'/(a+'.pt'));src=load(PRE/'source'/(a+'.pt'));rc=load(CLOSED/'replay_inputs'/(a+'.pt'));S=src['Source'].double().numpy();E=x['E'].double().numpy();q=np.concatenate([v['query_tgt'].numpy().astype(np.float64).ravel()for v in rc['spatial_inputs']]);sigma=float(np.sqrt(np.sum(q*q)/len(q))/16);assert abs(sigma-o['sigma'])<1e-15;checks['original_query_scale_exact']+=1
  positions=[f['position']for f,k in zip(x['frames'],x['selected'])if k>=0];target=np.array([f['boxes'][k].tolist()for f,k in zip(x['frames'],x['selected'])if k>=0]).reshape(-1,4);assert positions==o['positions']and np.array_equal(target,o['targets']);checks['fixed_expert_support']+=1
  cache={r['m'].tobytes():r for r in o['cache']};zero=cache[np.zeros(256).tobytes()];assert np.array_equal(zero['native'],src['Source'])and np.array_equal(zero['boxes'],E);checks['zero_Source_E_exact']+=1
  for rec in cache.values():
   want_u=(np.float32(sigma)*rec['m'].astype(np.float32));assert np.max(abs(want_u-rec['u'].numpy()))<2e-8;native=rec['native'].double().numpy();valid=bool(np.isfinite(native).all()and(native[:,2:]>0).all());assert valid==rec['valid']
   if valid:
    out=np.concatenate((E[:,:2]+native[:,:2]-S[:,:2],E[:,2:]*native[:,2:]/S[:,2:]),1);err=float(np.max(abs(out-rec['boxes'])));maxerr=max(maxerr,err);assert err<1e-12;checks['actual_native_E_readout_scalars']+=out.size
  assert len(cache)==o['compute']['actual_decoder_trajectory_calls'];assert o['compute']['actual_PosDecoder_calls']==2*len(cache)and o['compute']['backward_calls']==0;checks['physical_call_and_no_backward_receipts']+=1
  for mode,f in o['fits'].items():
   assert np.array_equal(f['path'][0]['state'],np.zeros(256));tt=np.roll(target,1,axis=0)if mode=='Latent_shuffle'and len(target)>1 else target
   for t in f['rounds']:
    m=t['before'];c=np.zeros(256)if mode=='Latent_fixed'else m;actions=c[None]+o['noise'][t['step']];assert np.array_equal(c,t['sampling_center'])and np.array_equal(actions,t['actions']);valid=np.ones(16,bool);reward=np.zeros(16)
    if positions:
     for j,u in enumerate(actions):
      rec=cache[u.tobytes()];valid[j]=rec['valid'];reward[j]=sparse_iou(rec['boxes'],positions,tt)if valid[j]else 0.;assert np.array_equal(rec['native'],t['actual_candidates'][j]['native'])
     checks['actual_latent_candidates_verified']+=16
    else:assert not t['actual_candidates']
    assert np.array_equal(valid,t['valid'])and np.max(abs(reward-t['rewards']))<1e-12;v,w,d,info=importance(actions,m,c,reward,valid);assert np.max(abs(v-t['weights']['base']))<1e-12 and np.max(abs(w-t['weights']['teacher']))<1e-12;assert np.max(abs(d-t['weights']['delta']))<1e-12;proposal=m+.5*d;assert np.max(abs(proposal-t['proposed']))<1e-12
    new=m if t['invalid_mean_rejected']else proposal;assert np.max(abs(new-t['after']))<1e-12;rec=cache[np.asarray(t['after'],np.float64).tobytes()];assert rec['valid']and np.array_equal(rec['boxes'],f['path'][t['step']+1]['boxes']);checks['independent_RN_reward_state_rounds']+=1
   assert np.array_equal(f['final'],o['outputs'][mode]);checks['actual_mean_final_readout']+=1
  fx,lv=o['fits']['Latent_fixed'],o['fits']['Latent_live'];assert np.array_equal(fx['path'][1]['state'],lv['path'][1]['state']);checks['first_fixed_live_exact']+=1
  acts=np.concatenate([t['actions']for t in fx['rounds']]);reward=np.concatenate([t['rewards']for t in fx['rounds']]);valid=np.concatenate([t['valid']for t in fx['rounds']]);base=sparse_iou(E,positions,target);scores=np.r_[base,np.where(valid,reward,-np.inf)];best=int(scores.argmax());assert best==o['search']['best_index'];selected=zero if best==0 else cache[acts[best-1].tobytes()];assert np.array_equal(selected['boxes'],o['outputs']['Latent_search']);checks['same_budget_whole_search']+=1
  v,w,d,info=importance(acts,np.zeros(256),np.zeros(256),reward,valid);d=np.zeros(256)if o['mean_control']['invalid_mean_rejected']else d;assert np.max(abs(d-o['mean_control']['state']))<1e-12;assert np.array_equal(cache[o['mean_control']['state'].tobytes()]['boxes'],o['outputs']['Latent_mean_control']);checks['same_budget_single_mean_control']+=1
  if not positions:
   for arm,b in o['outputs'].items():assert np.array_equal(b,E);checks['no_reference_all_outputs_noop']+=1
  independent_metrics(o,plans[row['dataset']][row['query_ordinal']],gt[row['dataset']][0][row['query_ordinal']],gt[row['dataset']][1][row['query_ordinal']],rows[a],checks)
 for ds,g in z['groups'].items():
  rr=[r for r in z['rows']if r['dataset']==ds]
  for key,eps in g['paired'].items():
   a,b=key.split('_minus_')
   for ep,want in eps.items():assert paired([None if r['values'][a][ep]is None or r['values'][b][ep]is None else 100*(r['values'][a][ep]-r['values'][b][ep])for r in rr])==want;checks['independent_parent_bootstrap']+=1
 assert not torch.cuda.is_initialized();put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),max_E_readout_error=maxerr,no_GPU_model_rerun=True));put(D/'ROOT_AUDIT_COMPLETION.json',dict(status='pass',count=32));print(dict(checks),maxerr,flush=True)
if __name__=='__main__':main()
