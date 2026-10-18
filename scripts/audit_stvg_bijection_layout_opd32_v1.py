"""Independent NumPy ROI and subset-DP assignment optimum; gradients and metrics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_bijection_layout_opd32_v1 import D,P,PRE,LAY,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.audit_stvg_roi_layout_opd32_v1 import sample,sm

def optimum(cost):
 import numpy as np
 c=cost.reshape(-1,9,9);allout=[]
 for start in range(0,len(c),1024):
  q=c[start:start+1024];dp=np.full((len(q),512),-np.inf);dp[:,0]=0
  for mask in range(1,512):
   cols=[j for j in range(9)if mask&(1<<j)];prev=[mask^(1<<j)for j in cols];row=mask.bit_count()-1;dp[:,mask]=(dp[:,prev]+q[:,row,cols]).max(1)
  allout.append(dp[:,-1]/9)
 return np.concatenate(allout).reshape(cost.shape[:-2])

def main():
 import numpy as np,torch
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.audit_stvg_discriminative_dense32_v1 import independent_metrics
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['all32_before_NEW_GT']and bar['count']==32;z=read(P/'RESULTS_PUBLIC.json');rows={r['alias']:r for r in z['rows']};counts=collections.Counter();maxerr=0.
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
 for rr in p['records']:
  a=rr['alias'];x=load(PRE/'features'/(a+'.pt'));o=load(D/'predictions'/(a+'.pt'));prev=load(LAY/'predictions'/(a+'.pt'));E=x['E'].numpy();B=o['B'].numpy();amp=B.sum(1);assert np.array_equal(B,prev['B']);assert np.array_equal(o['scope'],prev['scope']);counts['same_scope_amplitude_basis']+=1
  for arm,oldarm in [('Previous_independent','Independent_scoped'),('Previous_joint','Joint_live')]:assert np.array_equal(o['outputs'][arm],prev['outputs'][oldarm]);counts['old_outputs_exact']+=1
  pos=[f['position']for f,k in zip(x['frames'],x['selected'])if k>=0];bb=np.array([f['boxes'][k].tolist()for f,k in zip(x['frames'],x['selected'])if k>=0]).reshape(-1,1,4)
  if pos:tt,tw=sample(x['maps'][pos],x['valid'][pos],bb);tt,tw=tt[:,0],tw[:,0]
  eps=load(PRE/'feedback'/(a+'.pt'))['feedback']['noise'];fixed,live=o['fits']['Bijection_fixed'],o['fits']['Bijection_live'];assert torch.equal(fixed['path'][1]['boxes'],live['path'][1]['boxes']);assert torch.equal(live['rounds'][0]['candidates'],prev['fits']['Independent_scoped']['rounds'][0]['candidates']);counts['first_fixed_live_and_previous_candidates_exact']+=1
  entries=[(o['direct']['candidates'],o['direct']['feedback'],False,None,None),(o['diagonal']['candidates'],o['diagonal']['feedback'],True,None,None)]+[(t['candidates'],t['feedback'],arm=='Diagonal_live',o['permutation']if arm=='Bijection_shuffle'else None,(arm,t))for arm,f in o['fits'].items()for t in f['rounds']]
  for b,fb,diag,perm,entry in entries:
   maps=x['maps'][perm]if perm is not None else x['maps'];valid=x['valid'][perm]if perm is not None else x['valid'];ph,w=sample(maps,valid,b.numpy());ok=w.sum(-1)>1e-6;assert np.array_equal(ok,fb['valid']);safe=ok.copy();safe[~ok.any(1)]=True
   if pos:
    C=np.einsum('nkjc,aic->ankji',ph,tt)*w[None,:,:,:,None]*tw[:,None,None,None,:];di=C.diagonal(axis1=-2,axis2=-1).sum(-1)/9
    if diag:per=di
    else:
     per=optimum(C);assignment=fb['assignment'].numpy();assert np.array_equal(np.sort(assignment,axis=-1),np.broadcast_to(np.arange(9),assignment.shape));selected=np.take_along_axis(C,assignment[...,None],axis=-1)[...,0].sum(-1)/9;assert abs(per-selected).max()<3e-5;assert np.min(per-di)>-1e-12;counts['DP_optimal_bijections']+=per.size
    err=float(abs(per-fb['per_anchor_rewards'].numpy()).max());maxerr=max(maxerr,err);assert err<3e-5;counts['independent_reward_scalars']+=per.size
   ee=torch.cat(tuple(eps),1).numpy()if entry is None else eps[entry[1]['step']].numpy();lp=-.5*(ee**2).sum(-1);lp[~safe]=-np.inf;prior=sm(lp);r=fb['rewards'].numpy();info=ok.any(1)&((np.where(ok,r,-np.inf).max(1)-np.where(ok,r,np.inf).min(1))>1e-6)if pos else np.zeros(len(E),bool);pt=sm(lp+r/.1);pt[~info]=prior[~info];assert abs(pt-fb['prob'].numpy()).max()<1e-12;counts['independent_probability_scalars']+=pt.size
   if entry is None:
    out=E.copy();best=pt.argmax(-1);use=(amp>0)&info;out[use]=b.numpy()[np.arange(len(E))[use],best[use]];name='Diagonal_readout'if diag else'Bijection_readout';assert np.array_equal(out,o['outputs'][name]);counts['direct_readouts']+=1
   else:
    mode,t=entry;eta=t['before'].numpy();center=np.zeros_like(eta)if mode=='Bijection_fixed'else eta;assert np.array_equal(center,t['sampling_center']);d=center[:,None]+ee;candidate=np.broadcast_to(E[:,None],b.shape).copy();candidate[:,:,:2]+=.25*E[:,None,2:]*amp[:,None,None]*d;assert abs(candidate-b.numpy()).max()<1e-14;sp=sm(np.where(safe,-.5*((d-eta[:,None])**2).sum(-1),-np.inf));g=((sp-pt)[...,None]*d).sum(1);g[~((amp>0)&info)]=0;assert abs(g-t['gradient'].numpy()).max()<1e-12;new=eta-.5*g;assert abs(new-t['after'].numpy()).max()<1e-12;out=E.copy();out[:,:2]+=.25*E[:,2:]*amp[:,None]*new;assert abs(out-o['fits'][mode]['path'][t['step']+1]['boxes'].numpy()).max()<1e-12;counts['actual_gradient_state_output_rounds']+=1
  for arm,out in o['outputs'].items():
   assert np.array_equal(out[:,2:],E[:,2:])and np.array_equal(out[x['planned_positions']],E[x['planned_positions']])and np.array_equal(out[~o['scope'].numpy()],E[~o['scope'].numpy()]);counts['width_anchors_scope_fixed']+=1
   if not pos:assert np.array_equal(out,E);counts['no_reference_noop']+=1
  independent_metrics(o,plans[rr['dataset']][rr['query_ordinal']],gt[rr['dataset']][0][rr['query_ordinal']],gt[rr['dataset']][1][rr['query_ordinal']],rows[a],counts)
 for ds,g in z['groups'].items():
  rowsds=[r for r in z['rows']if r['dataset']==ds]
  for key,eps in g['paired'].items():
   a,b=key.split('_minus_')
   for ep,want in eps.items():assert paired([None if r['values'][a][ep]is None or r['values'][b][ep]is None else 100*(r['values'][a][ep]-r['values'][b][ep])for r in rowsds])==want;counts['paired_parent_bootstrap']+=1
 assert not torch.cuda.is_initialized();put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(counts),max_independent_reward_error=maxerr,independent_bitmaskDP_not_scipy_solver=True,no_model_GPU_rerun=True));put(D/'ROOT_AUDIT_COMPLETION.json',dict(status='pass',count=32));print(dict(counts),maxerr,flush=True)
if __name__=='__main__':main()
