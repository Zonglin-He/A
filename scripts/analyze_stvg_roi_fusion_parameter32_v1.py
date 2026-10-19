"""Postseal-only per-anchor disagreement, fixed-target fit strength and GT quality."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,importlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha

def main(kind):
 import torch,numpy as np
 from scripts.score_stvg_opd_p1_v1 import truths
 from scripts.run_stvg_dino_local_residual32_v1 import mean
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 from vg_tta.stvg_cosine_candidate_headroom32_v1 import xy,iou
 from vg_tta.stvg_discriminative_dense32_v1 import student_logits
 mod=importlib.import_module('scripts.run_stvg_fine_roi_'+kind+'32_v1');D,P=mod.D,mod.P;torch.set_num_threads(4);p=mod.verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32
 for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
 gt={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gt};rows=[]
 def KL(q,logp):return torch.where(q>0,q*(q.clamp_min(1e-300).log()-logp),0).sum(-1)
 def ent(q):return -torch.where(q>0,q*q.clamp_min(1e-300).log(),0).sum(-1)
 for r in p['records']:
  a=r['alias'];ds=r['dataset'];idx=r['query_ordinal'];o=load(D/'predictions'/(a+'.pt'));E=o['outputs']['E_MBR'];n=len(E);allowed=o['allowed'];ids=o['frame_ids'];known=np.array([int(i)in gt[ds][0][idx]for i in ids]);support=known&allowed.numpy();gtxy=np.array([gt[ds][0][idx].get(int(i),[np.nan]*4)for i in ids]);Qe=np.full(n,np.nan);Qe[known]=iou(xy(E,plans[ds][idx],ds)[known],gtxy[known]);modes={}
  for mode,fit in o['fits'].items():
   steps=[]
   for st,t in enumerate(fit['rounds']):
    f=t['feedback'];q=f['prob'];pr=f['prior'];d=t['standardized_candidates'];logbefore=student_logits(d,t['eta_before'],f['valid']).log_softmax(-1);logafter=student_logits(d,t['eta_after'],f['valid']).log_softmax(-1);kp=KL(q,pr.clamp_min(1e-300).log());kb=KL(q,logbefore);ka=KL(q,logafter);delta=fit['path'][st+1]['boxes'][:,:2]-fit['path'][st]['boxes'][:,:2];disp=(delta/torch.as_tensor(E[:,2:])).norm(dim=-1);per=f['per_anchor_rewards'];A=len(per);safe=f['valid'].clone();safe[~safe.any(1)]=True;lp=(-.5*o['noise'][st].double().square().sum(-1)).masked_fill(~safe,-torch.inf);tau=fit.get('actual_configuration',p['configuration'])['temperature'];qa=(lp[None]+per/tau).softmax(-1)if A else torch.empty_like(per);mix=qa.mean(0)if A else pr;geo=(lp+f['rewards']/tau).softmax(-1);geo[~f['informative']]=pr[~f['informative']];baseline=(lp+f['rewards']/.1).softmax(-1);baseline[~f['informative']]=pr[~f['informative']];js=ent(mix)-ent(qa).mean(0)if A else torch.zeros(n);TV=.5*(mix-geo).abs().sum(-1);suppTV=.5*(mix-q).abs().sum(-1)
    vals=dict(teacher_prior_KL=mean(kp[allowed]),student_same_teacher_KL_before=mean(kb[allowed]),student_same_teacher_KL_after=mean(ka[allowed]),same_teacher_KL_reduction=mean((kb-ka)[allowed]),normalized_Esize_step_displacement=mean(disp[allowed]),anchor_JS_nats=mean(js[allowed]),raw_mix_geometric_TV=mean(TV[allowed]),raw_mix_applied_TV=mean(suppTV[allowed]),gate_suppressed_TV=mean(suppTV[allowed&~f['informative']]),gate_suppressed_nontrivial_nodes=int(((suppTV>1e-12)&allowed&~f['informative']).sum()),anchors=A,allowed_nodes=int(allowed.sum()),updated_nodes=int((allowed&f['informative']).sum()))
    b=t['candidates'].numpy();quality=np.full(b.shape[:2],np.nan)
    for j in np.flatnonzero(known):quality[j]=iou(xy(b[j],plans[ds][idx],ds),gtxy[j])
    kk=[]
    for j in np.flatnonzero(support):
     if not f['valid'][j].any():continue
     qq=quality[j];pq=q[j].numpy();prior=pr[j].numpy();gq=geo[j].numpy();bq=baseline[j].numpy();good=(qq>Qe[j]+1e-10)&f['valid'][j].numpy();kk.append(dict(teacher_minus_prior_expected_IoU_pp=float(100*((pq-prior)*qq).sum()),teacher_minus_same_state_geometric_expected_IoU_pp=float(100*((pq-gq)*qq).sum()),teacher_minus_same_state_baseline_tau_expected_IoU_pp=float(100*((pq-bq)*qq).sum()),teacher_expected_minus_E_IoU_pp=float(100*((pq*qq).sum()-Qe[j])),teacher_minus_prior_good_mass=float((pq-prior)[good].sum())))
    vals.update({k:mean([v[k]for v in kk])for k in kk[0]}if kk else {});vals['known_scope_visual_nodes']=len(kk);steps.append(vals)
   modes[mode]=steps
  rows.append(dict(alias=a,dataset=ds,active=o['active'],modes=modes))
 groups={}
 for ds in gt:
  rr=[r for r in rows if r['dataset']==ds];groups[ds]={}
  for mode in rows[0]['modes']:
   groups[ds][mode]={str(st+1):{k:paired([r['modes'][mode][st].get(k)for r in rr])for k in sorted(set(k for r in rr for k in r['modes'][mode][st]))}for st in range(2)}
 put(P/'POST_SEAL_DIAGNOSTICS.json',dict(status='complete',count=32,groups=groups,rows=rows,GT_offline_after_all_outputs=True,second_round_actual_candidates_GT_evaluated=True,KL_same_support_and_same_detached_teacher=True,all_optimization_diagnostics_use_predicted_allowed_not_GT=True,GT_quality_diagnostics_conditional_scope_not_main_denominator=True,raw_per_anchor_posteriors_in_private_sealed_paths=True,teacher_distributions_not_calibrated_identity_probability=True,extra_ROI_or_model_evaluations=0));print(kind,'diagnostics complete',flush=True)
if __name__=='__main__':main(sys.argv[1])
