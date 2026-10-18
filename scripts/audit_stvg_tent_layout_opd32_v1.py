"""Independent physical basis/NumPy ROI rewards, finite KL, projection and metrics."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_tent_layout_opd32_v1 import D,P,PRE,LAY,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.audit_stvg_roi_layout_opd32_v1 import sample,sm

def main():
 import numpy as np,torch
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.audit_stvg_discriminative_dense32_v1 import independent_metrics
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all32_before_NEW_GT'];z=read(P/'RESULTS_PUBLIC.json');rows={r['alias']:r for r in z['rows']};checks=collections.Counter();maxreward=0.
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
 for row in p['records']:
  a=row['alias'];x=load(PRE/'features'/(a+'.pt'));o=load(D/'predictions'/(a+'.pt'));prev=load(LAY/'predictions'/(a+'.pt'));E=x['E'].numpy();ids=np.array(x['frame_ids']);pp=sorted(set(x['planned_positions']));iv=x['interval'];B=np.zeros((len(E),3));scope=(np.r_[ids[1:],ids[-1]]>=iv[0])&(np.r_[ids[0],ids[:-1]]<iv[1]);scope[pp]=False
  for j,(l,r)in enumerate(zip(pp[:-1],pp[1:])):
   for t in range(len(E)):
    if ids[l]<ids[t]<ids[r]:B[t,j]=min(2*(ids[t]-ids[l])/(ids[r]-ids[l]),2*(ids[r]-ids[t])/(ids[r]-ids[l]))
  B*=scope[:,None];assert np.max(abs(B-o['B'].numpy()))<1e-15;assert np.array_equal(scope,o['scope']);eligible=B.sum(1)>0;active=B.sum(0)>0;checks['physical_time_basis_scope']+=1
  assert int(np.linalg.matrix_rank(B))==o['basis_rank'];assert not o['noise'][:,:,~torch.from_numpy(active)].any();checks['latent_identifiability']+=1
  pos=[f['position']for f,k in zip(x['frames'],x['selected'])if k>=0];bb=np.array([f['boxes'][k].tolist()for f,k in zip(x['frames'],x['selected'])if k>=0]).reshape(-1,1,4)
  if pos:tt,tw=sample(x['maps'][pos],x['valid'][pos],bb);tt,tw=tt[:,0],tw[:,0]
  def visual(b,perm):
   nonlocal maxreward
   maps=x['maps'][perm]if perm is not None else x['maps'];valid=x['valid'][perm]if perm is not None else x['valid'];ph,w=sample(maps,valid,b);ok=w.sum(-1)>1e-6
   if pos:
    com=w[None]*tw[:,None,None,:];r=(np.einsum('nkjc,ajc->ankj',ph,tt)*com).sum(-1)/np.maximum(com.sum(-1),1e-12);r=r.mean(0)
   else:r=np.zeros(b.shape[:2])
   return r,ok
  fixed,live=o['fits']['Joint_fixed'],o['fits']['Joint_live'];assert torch.equal(fixed['path'][1]['boxes'],live['path'][1]['boxes']);checks['first_fixed_live_exact']+=1
  entries=[('direct',None,o['direct'])]+[(m,t,t['feedback'])for m,f in o['fits'].items()for t in f['rounds']]
  for mode,t,fb in entries:
   joint=mode!='Independent_scoped';perm=o['permutation']if mode=='Joint_shuffle'else None
   if joint:
    u=torch.cat(tuple(o['noise']),0).numpy()if t is None else t['candidates'].numpy();b=np.broadcast_to(E[:,None],(len(E),len(u),4)).copy();b[:,:,:2]+=.25*E[:,None,2:]*np.einsum('nj,kjd->nkd',B,u);wantb=fb['boxes'].numpy();assert np.max(abs(b-wantb))<1e-14
    r,ok=visual(b,perm);er=float(abs(r-fb['node_rewards'].numpy()).max());maxreward=max(maxreward,er);assert er<3e-5;checks['independent_ROI_reward_scalars']+=r.size
    rr=fb['node_rewards'].numpy()[eligible].mean(0)if eligible.any()else np.zeros(len(u));assert np.max(abs(rr-fb['reward'].numpy()))<1e-12
    vk=ok[eligible].any(0)if eligible.any()else np.ones(len(u),bool)
    if not vk.any():vk[:]=True
    assert np.array_equal(vk,fb['valid']);eps=torch.cat(tuple(o['noise']),0).numpy()if t is None else t['noise'].numpy();lp=-.5*((eps*active[None,:,None])**2).sum((1,2));lp[~vk]=-np.inf;prior=sm(lp);info=bool(eligible.any()and pos and rr[vk].max()-rr[vk].min()>1e-6);pt=sm(lp+rr/.1)if info else prior;assert abs(pt-fb['prob'].numpy()).max()<1e-12;checks['joint_probability']+=pt.size
    if t is None:
     out=b[:,pt.argmax()]if info else E;assert np.max(abs(out-o['outputs']['Joint_direct']))<1e-14;checks['whole_trajectory_direct']+=1
    else:
     mu=t['before'].numpy();center=np.zeros_like(mu)if mode=='Joint_fixed'else mu;assert np.array_equal(center,t['sampling_center']);sp=sm(np.where(vk,-.5*(((u-mu)*active[None,:,None])**2).sum((1,2)),-np.inf));g=((sp-pt)[:,None,None]*u).sum(0);g[~active]=0
     if not info:g[:]=0
     assert abs(g-t['gradient'].numpy()).max()<1e-12;new=mu-.5*g;out=E.copy();out[:,:2]+=.25*E[:,2:]*np.einsum('nj,jd->nd',B,new);assert abs(new-t['after'].numpy()).max()<1e-12;assert abs(out-o['fits'][mode]['path'][t['step']+1]['boxes'].numpy()).max()<1e-12;checks['actual_joint_state_updates']+=1
   else:
    d=t['standardized_candidates'].numpy();amp=B.sum(1);b=np.broadcast_to(E[:,None],(len(E),d.shape[1],4)).copy();b[:,:,:2]+=.25*E[:,None,2:]*amp[:,None,None]*d;assert abs(b-t['candidates'].numpy()).max()<1e-14;r,ok=visual(b,None);er=float(abs(r-fb['rewards'].numpy()).max());maxreward=max(maxreward,er);assert er<3e-5;checks['independent_ROI_reward_scalars']+=r.size
    safe=ok.copy();safe[~ok.any(1)]=True;eta=t['before'].numpy();eps=d-eta[:,None];lp=np.where(safe,-.5*(eps**2).sum(-1),-np.inf);prior=sm(lp);rr=fb['rewards'].numpy();info=ok.any(1)&(np.where(ok,rr,-np.inf).max(1)-np.where(ok,rr,np.inf).min(1)>1e-6)if pos else np.zeros(len(E),bool);pt=sm(lp+rr/.1);pt[~info]=prior[~info];assert abs(pt-fb['prob'].numpy()).max()<1e-12;g=((prior-pt)[...,None]*d).sum(1);g[~(eligible&info)]=0;assert abs(g-t['gradient'].numpy()).max()<1e-12;new=eta-.5*g;out=E.copy();out[:,:2]+=.25*E[:,2:]*amp[:,None]*new;assert abs(out-o['fits'][mode]['path'][t['step']+1]['boxes'].numpy()).max()<1e-12;checks['actual_independent_updates']+=1
  for key,source in [('Independent_projection',o['outputs']['Independent_scoped']),('Previous_projection',prev['outputs']['On_policy_dense'])]:
   y=(source[:,:2]-E[:,:2])/(.25*E[:,2:]);u=np.linalg.pinv(B[eligible])@y[eligible]if eligible.any()else np.zeros((3,2));out=E.copy();out[:,:2]+=.25*E[:,2:]*(B@u);assert abs(out-o['outputs'][key]).max()<1e-12;checks['independent_LS_projection']+=1
  for arm,out in o['outputs'].items():
   assert np.array_equal(out[:,2:],E[:,2:])and np.array_equal(out[pp],E[pp])and np.array_equal(out[~scope],E[~scope]);checks['scope_width_anchors_fixed']+=1
   if not pos:assert np.array_equal(out,E);checks['no_reference_noop']+=1
  independent_metrics(o,plans[row['dataset']][row['query_ordinal']],gt[row['dataset']][0][row['query_ordinal']],gt[row['dataset']][1][row['query_ordinal']],rows[a],checks)
 for ds,g in z['groups'].items():
  rr=[r for r in z['rows']if r['dataset']==ds]
  for contrast,eps in g['paired'].items():
   a,b=contrast.split('_minus_')
   for ep,t in eps.items():assert paired([None if r['values'][a][ep]is None or r['values'][b][ep]is None else 100*(r['values'][a][ep]-r['values'][b][ep])for r in rr])==t;checks['parent_paired_bootstrap']+=1
 assert not torch.cuda.is_initialized();put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),max_independent_reward_error=maxreward,no_GPU_model_rerun=True));put(D/'ROOT_AUDIT_COMPLETION.json',dict(status='pass',count=32));print(dict(checks),maxreward,flush=True)
if __name__=='__main__':main()
