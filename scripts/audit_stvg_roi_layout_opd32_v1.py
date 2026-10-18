"""Independent NumPy bilinear ROI/score, saved KL/state and official metric audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_roi_layout_opd32_v1 import D,P,PRE,OLD,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha

def sample(maps,valid,b):
 import numpy as np
 x=maps.numpy().transpose(0,2,3,1)*valid.numpy()[...,None];v=valid.numpy().astype(np.float32);b=np.asarray(b);n,k=b.shape[:2];h,w=v.shape[-2:];vh=v.max(2).sum(1);vw=v.max(1).sum(1);a=(np.arange(3)+.5)/3-.5;yy,xx=np.meshgrid(a,a,indexing='ij');o=np.stack((xx,yy),-1).reshape(1,1,9,2);q=b[:,:,None,:2]+b[:,:,None,2:]*o;q=q*np.stack((vw/w,vh/h),-1)[:,None,None,:];grid=(2*q-1).astype(np.float32)
 xp=((grid[...,0]+np.float32(1))*np.float32(w)-np.float32(1))/np.float32(2);yp=((grid[...,1]+np.float32(1))*np.float32(h)-np.float32(1))/np.float32(2);x0=np.floor(xp).astype(int);y0=np.floor(yp).astype(int);dx=xp-x0;dy=yp-y0;out=np.zeros((n,k,9,x.shape[-1]),np.float32);weight=np.zeros((n,k,9),np.float32);ni=np.arange(n)[:,None,None]
 for ox,oy,fac in [(0,0,(1-dx)*(1-dy)),(1,0,dx*(1-dy)),(0,1,(1-dx)*dy),(1,1,dx*dy)]:
  xi=x0+ox;yi=y0+oy;ok=(xi>=0)&(xi<w)&(yi>=0)&(yi<h);xc=xi.clip(0,w-1);yc=yi.clip(0,h-1);f=(fac*ok).astype(np.float32);out+=x[ni,yc,xc]*f[...,None];weight+=v[ni,yc,xc]*f
 norm=np.sqrt((out*out).sum(-1,keepdims=True));out=out/np.maximum(norm,1e-12);return out.astype(np.float64),weight.astype(np.float64)

def sm(x):
 import numpy as np
 a=np.exp(x-np.max(x,axis=-1,keepdims=True));return a/a.sum(-1,keepdims=True)

def main():
 import numpy as np,torch
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.audit_stvg_discriminative_dense32_v1 import independent_metrics
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all32_before_NEW_GT'];z=read(P/'RESULTS_PUBLIC.json');rows={r['alias']:r for r in z['rows']};checks=collections.Counter();maxreward=0.;maxprob=0.
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
 for r in p['records']:
  a=r['alias'];x=load(PRE/'features'/(a+'.pt'));o=load(D/'predictions'/(a+'.pt'));old=load(OLD/'predictions'/(a+'.pt'));ctx=load(D/'context'/(a+'.pt'));E=x['E'].numpy();n=len(E);allowed=np.ones(n,bool);allowed[x['planned_positions']]=False
  positions=[f['position']for f,k in zip(x['frames'],x['selected'])if k>=0];bb=np.array([f['boxes'][k].tolist()for f,k in zip(x['frames'],x['selected'])if k>=0]).reshape(-1,1,4)
  if positions:
   tt,tw=sample(x['maps'][positions],x['valid'][positions],bb);tt=tt[:,0];tw=tw[:,0];assert np.max(abs(tt-ctx['templates'].numpy()))<3e-5;assert np.max(abs(tw-ctx['weights'].numpy()))<3e-6;checks['independent_anchor_ROI_coordinates']+=tt.size
  else:tt=np.empty((0,9,x['maps'].shape[1]));tw=np.empty((0,9))
  assert torch.equal(o['direct']['candidates'],old['direct']['candidates']);checks['same_initial34']+=1
  for new,oldarm in [('Forward_readout','Cosine_readout'),('Forward_Fixed','Fixed_dense'),('Forward_On','On_policy_dense'),('Forward_Shuffled','Shuffled_OPD')]:assert np.array_equal(o['outputs'][new],old['outputs'][oldarm]);checks['old_forward_exact']+=1
  fixed,on=o['fits']['Fixed_dense'],o['fits']['On_policy_dense'];assert torch.equal(fixed['path'][1]['boxes'],on['path'][1]['boxes'])and torch.equal(fixed['rounds'][0]['feedback']['prob'],on['rounds'][0]['feedback']['prob']);checks['same_first_fixed_live']+=1
  entries=[(o['direct']['candidates'],o['direct']['feedback'],False,None,None),(o['pooled']['candidates'],o['pooled']['feedback'],True,None,None)]
  for mode,fit in o['fits'].items():
   assert torch.count_nonzero(fit['path'][0]['eta'])==0;checks['query_zero_state']+=1
   for step,t in enumerate(fit['rounds']):entries.append((t['candidates'],t['feedback'],mode=='Unordered_OPD',o['permutation']if mode=='Shuffled_OPD'else None,(mode,step,t,fit)))
  for b,fb,unordered,perm,entry in entries:
   maps=x['maps'][perm]if perm is not None else x['maps'];valid=x['valid'][perm]if perm is not None else x['valid'];ph,w=sample(maps,valid,b.numpy());ok=w.sum(-1)>1e-6;assert np.array_equal(ok,fb['valid']);safe=ok.copy();safe[~ok.any(1)]=True
   if positions:
    common=w[None]*tw[:,None,None,:]
    if unordered:
     pp=(ph*w[...,None]).sum(-2)/np.maximum(w.sum(-1),1e-12)[...,None];tp=(tt*tw[...,None]).sum(-2)/np.maximum(tw.sum(-1),1e-12)[:,None];per=np.einsum('nkc,ac->ank',pp,tp)
    else:per=(np.einsum('nkjc,ajc->ankj',ph,tt)*common).sum(-1)/np.maximum(common.sum(-1),1e-12)
    reward=per.mean(0);er=float(abs(per-fb['per_anchor_rewards'].numpy()).max());maxreward=max(maxreward,er);assert er<3e-5;checks['independent_ordered_or_unordered_reward_scalar']+=per.size
   else:reward=np.zeros(b.shape[:2])
   # Probability reconstructed from saved reward tests the actual finite update,
   # while independent bilinear reward above checks its visual computation.
   if entry is None:
    noise=load(PRE/'feedback'/(a+'.pt'))['feedback']['noise'];ee=torch.cat((noise[0],noise[1]),1).numpy()
   else:mode,step,t,fit=entry;ee=load(PRE/'feedback'/(a+'.pt'))['feedback']['noise'][step].numpy()
   lp=-.5*(ee*ee).sum(-1);lp[~safe]=-np.inf;pr=sm(lp);rr=fb['rewards'].numpy();pt=sm(lp+rr/.1);info=ok.any(1)&((np.where(ok,rr,-np.inf).max(-1)-np.where(ok,rr,np.inf).min(-1))>1e-6)if positions else np.zeros(n,bool);pt[~info]=pr[~info];er=float(abs(pt-fb['prob'].numpy()).max());maxprob=max(maxprob,er);assert er<1e-12;checks['independent_probability_scalar']+=pt.size
   if entry is None:
    out=E.copy();best=pt.argmax(-1);use=allowed&info;out[use]=b.numpy()[np.arange(n)[use],best[use]];name='Unordered_readout'if unordered else'Layout_readout';assert np.array_equal(out,o['outputs'][name]);checks['actual_direct']+=1
   else:
    eta=t['eta_before'].numpy();center=np.zeros_like(eta)if mode=='Fixed_dense'else eta;assert np.array_equal(center,t['sampling_center']);d=center[:,None]+ee;sp=sm(np.where(safe,-.5*((d-eta[:,None])**2).sum(-1),-np.inf));gradient=((sp-pt)[...,None]*d).sum(1);gradient[~(allowed&info)]=0;assert abs(gradient-t['gradient'].numpy()).max()<1e-12;new=eta-.5*gradient;assert abs(new-t['eta_after'].numpy()).max()<1e-12;out=E.copy();out[:,:2]+=E[:,2:]*.25*new;out[x['planned_positions']]=E[x['planned_positions']];assert abs(out-fit['path'][step+1]['boxes'].numpy()).max()<1e-12;checks['actual_KL_gradient_state_output_rounds']+=1
  for arm,b in o['outputs'].items():
   assert np.array_equal(b[:,2:],E[:,2:])and np.array_equal(b[x['planned_positions']],E[x['planned_positions']]);checks['fixed_size_planned_time']+=1
   if not positions:assert np.array_equal(b,E);checks['no_reference_noop']+=1
  independent_metrics(o,plans[r['dataset']][r['query_ordinal']],gt[r['dataset']][0][r['query_ordinal']],gt[r['dataset']][1][r['query_ordinal']],rows[a],checks)
 for ds,g in z['groups'].items():
  rr=[r for r in z['rows']if r['dataset']==ds]
  for key,eps in g['paired'].items():
   aa,bb=key.split('_minus_')
   for ep,want in eps.items():assert paired([None if r['values'][aa][ep]is None or r['values'][bb][ep]is None else 100*(r['values'][aa][ep]-r['values'][bb][ep])for r in rr])==want;checks['independent_paired_bootstrap']+=1
 assert not torch.cuda.is_initialized();put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),max_independent_reward_error=maxreward,max_probability_error=maxprob,no_GPU_or_model_rerun=True));put(D/'ROOT_AUDIT_COMPLETION.json',dict(status='pass',count=32));print(dict(checks),maxreward,maxprob,flush=True)
if __name__=='__main__':main()
