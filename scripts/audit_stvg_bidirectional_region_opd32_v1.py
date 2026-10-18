"""Independent CPU reverse-affinity, both rewards, real updates and official scores."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_bidirectional_region_opd32_v1 import D,P,PRE,OLD,OLDP,verify
from scripts.run_stvg_joint_event_critic_v1 import read,load,put

def main():
 import torch,numpy as np
 import torch.nn.functional as F
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.audit_stvg_discriminative_dense32_v1 import independent_metrics
 from vg_tta.stvg_soft_correspondence_v1 import cell_overlap
 p=verify();torch.set_num_threads(4);bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['all32_before_NEW_GT'];z=read(P/'RESULTS_PUBLIC.json');rows={r['alias']:r for r in z['rows']};checks=collections.Counter();maxr=0.;maxa=0.
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt}
 for rec in p['records']:
  a=rec['alias'];ds=rec['dataset'];q=rec['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));x=load(PRE/'features'/(a+'.pt'));ctx=load(D/'context'/(a+'.pt'))['context'];old=load(OLD/'predictions'/(a+'.pt'));E=x['E'];pos=x['planned_positions'];n=len(E);valid=x['valid'];H,W=valid.shape[-2:];allowed=torch.ones(n,dtype=torch.bool);allowed[pos]=False
  f=x['maps'].float().flatten(2).transpose(1,2);f=f/f.norm(dim=-1,keepdim=True).clamp_min(1e-12)
  for j,i in enumerate(ctx['positions']):
   logits=(f@f[i].T).transpose(-1,-2)/.07;logits=logits.masked_fill(~valid.flatten(1)[:,None,:],-torch.inf);rv=logits.softmax(-1)*valid[i].flatten()[None,:,None];err=float(abs(rv-ctx['reverse'][j]).max());maxa=max(maxa,err);assert err<2e-6;checks['independent_reverse_attention_scalar']+=rv.numel()
  assert torch.equal(o['direct']['candidates'],old['direct']['candidates']);checks['same_initial34_candidates']+=1
  for na,oa in [('Forward_readout','Cosine_readout'),('Forward_Fixed','Fixed_dense'),('Forward_On','On_policy_dense'),('Forward_Shuffled','Shuffled_OPD')]:assert np.array_equal(o['outputs'][na],old['outputs'][oa]);checks['old_forward_actual_readouts_exact']+=1
  fixed,on=o['fits']['Fixed_dense'],o['fits']['On_policy_dense'];assert torch.equal(fixed['path'][1]['boxes'],on['path'][1]['boxes'])and torch.equal(fixed['rounds'][0]['feedback']['prob'],on['rounds'][0]['feedback']['prob']);checks['matched_first_fixed_live_step']+=1
  entries=[(o['direct']['candidates'],o['direct']['feedback'],None,None)]
  for mode,fit in o['fits'].items():
   assert torch.count_nonzero(fit['path'][0]['eta'])==0;checks['zero_query_state']+=1
   for step,t in enumerate(fit['rounds']):entries.append((t['candidates'],t['feedback'],o['permutation']if mode=='Shuffled_OPD'else None,(mode,step,t,fit)))
  for b,fb,perm,fitentry in entries:
   ss=ctx['forward_soft'][:,perm]if perm is not None else ctx['forward_soft'];rv=ctx['reverse'][:,perm]if perm is not None else ctx['reverse'];vv=valid[perm]if perm is not None else valid;area=cell_overlap(b,H,W)*vv[:,None];ok=area.sum((-1,-2))>1e-12;assert torch.equal(ok,fb['valid'])
   if len(ss):
    area0=cell_overlap(b,H,W);it=(area0[None]*ss[:,:,None].double()).sum((-1,-2));den=b[None,...,2]*b[None,...,3]*H*W+ss.double().sum((-1,-2))[:,:,None]-it;rf=torch.where(den>0,it/den,torch.zeros_like(it))
    back=torch.matmul(rv.double(),area.flatten(-2).transpose(-1,-2)[None]).transpose(-1,-2)
    aa=cell_overlap(ctx['anchor_boxes'],H,W).flatten(-2);ir=(back*aa[:,None,None,:]).sum(-1);ab=ctx['anchor_boxes'];den=ab[:,None,None,2]*ab[:,None,None,3]*H*W+back.sum(-1)-ir;rr=torch.where(den>0,ir/den,torch.zeros_like(ir));reward=torch.sqrt(rf*rr).mean(0)
    err=max(float(abs(rf-fb['forward_rewards']).max()),float(abs(rr-fb['reverse_rewards']).max()));maxr=max(maxr,err);assert err<1e-12;checks['independent_both_direction_reward_scalar']+=rf.numel()+rr.numel()
   else:reward=torch.zeros_like(fb['rewards'])
   assert abs(reward-fb['rewards']).max()<1e-12
   safe=ok.clone();safe[~ok.any(1)]=True;prior=fb['prior'];student_noise=None
   if fitentry is None:
    noise=load(PRE/'feedback'/(a+'.pt'))['feedback']['noise'];ee=torch.cat((noise[0],noise[1]),1)
   else:mode,step,t,fit=fitentry;ee=load(PRE/'feedback'/(a+'.pt'))['feedback']['noise'][step]
   lp=-.5*ee.square().sum(-1);lp[~safe]=-torch.inf;pt=(lp+reward/.1).softmax(-1);informative=ok.any(1)&((reward.masked_fill(~ok,-torch.inf).amax(-1)-reward.masked_fill(~ok,torch.inf).amin(-1))>1e-12)if len(ss)else torch.zeros(n,dtype=torch.bool);pt[~informative]=lp.softmax(-1)[~informative];assert abs(pt-fb['prob']).max()<1e-12;checks['same_prior_teacher_distribution_scalar']+=pt.numel()
   if fitentry is None:
    out=E.clone();use=allowed&informative;best=pt.argmax(-1);out[use]=b[torch.arange(n)[use],best[use]];assert np.array_equal(out,o['outputs']['Bidir_readout']);checks['actual_direct_readout']+=1
   else:
    mode,step,t,fit=fitentry;center=torch.zeros_like(t['eta_before'])if mode=='Fixed_dense'else t['eta_before'];assert torch.equal(center,t['sampling_center']);d=center[:,None]+ee;sp=(-.5*(d-t['eta_before'][:,None]).square().sum(-1)).masked_fill(~safe,-torch.inf).softmax(-1);gradient=((sp-pt)[...,None]*d).sum(1);gradient[~(allowed&informative)]=0;assert abs(gradient-t['gradient']).max()<1e-12;new=t['eta_before']-.5*gradient;assert abs(new-t['eta_after']).max()<1e-12 and torch.equal(t['eta_before']-.5*t['gradient'],t['eta_after']);out=E.clone();out[:,:2]+=E[:,2:]*.25*new;out[pos]=E[pos];assert abs(out-fit['path'][step+1]['boxes']).max()<1e-12;checks['actual_KL_gradient_state_output']+=1
    if not len(ss):assert torch.equal(out,E);checks['no_reference_strict_noop']+=1
  for arm,b in o['outputs'].items():assert np.array_equal(b[:,2:],E[:,2:].numpy())and np.array_equal(b[pos],E[pos].numpy());checks['all_nine_sizes_anchors_fixed']+=1
  independent_metrics(o,plans[ds][q],gt[ds][0][q],gt[ds][1][q],rows[a],checks)
 for ds,g in z['groups'].items():
  rr=[r for r in z['rows']if r['dataset']==ds]
  for c,stats in g['paired'].items():
   a,b=c.split('_minus_')
   for k,t in stats.items():
    v=np.array([100*(r['values'][a][k]-r['values'][b][k])for r in rr if r['values'][a][k]is not None and r['values'][b][k]is not None]);ix=np.random.default_rng(20261018).integers(len(v),size=(10000,len(v)));assert abs(v.mean()-t['mean'])<1e-12 and np.max(abs(np.quantile(v[ix].mean(1),[.025,.975])-t['CI95']))<1e-12;checks['independent_paired_aggregate_bootstrap']+=1
 verify(full=True);assert not torch.cuda.is_initialized();put(P/'ROOT_INDEPENDENT_AUDIT.json',dict(status='pass',count=32,checks=dict(checks),max_reverse_attention_difference=maxa,max_both_reward_difference=maxr,all32_before_NEW_GT=True,new_GPU_Source_DINO_training=0));put(D/'ROOT_AUDIT_COMPLETION.json',dict(status='pass',count=32));print(dict(checks),maxa,maxr,flush=True)

if __name__=='__main__':main()
