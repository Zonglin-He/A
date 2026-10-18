"""Only sealed-cache E-relative comparisons; no new candidates, labels or models."""
from pathlib import Path
import os,sys
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_bidirectional_region_opd32_v1 import D,P,OLD,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load

def main():
 import numpy as np,torch
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 from scripts.run_stvg_dino_local_residual32_v1 import mean
 from vg_tta.stvg_cosine_candidate_headroom32_v1 import summarize_frame
 torch.set_num_threads(4);plan=verify();rows=[]
 for r in plan['records']:
  a=r['alias'];o=load(D/'predictions'/(a+'.pt'));old=load(OLD/'predictions'/(a+'.pt'));q=load(ROOT/'artifacts/stvg_cosine_candidate_headroom32_v1/diagnostics'/(a+'.pt'));fb=o['direct']['feedback'];b=o['direct']['candidates'];assert torch.equal(b,old['direct']['candidates']);valid=fb['valid'].clone();valid[:,0]=True;lp=fb['prior'].log();allowed=torch.ones(len(b),dtype=torch.bool);allowed[o['planned_positions']]=False
  rankings={};scales={}
  for kind,raw in [('Forward',old['direct']['feedback']['rewards']),('Reverse',fb['reverse_rewards'].mean(0)if o['active']else fb['rewards']),('Symmetric',fb['rewards'])]:
   pt=(lp+raw/.1).softmax(-1)if o['active']else fb['prior'];frames=[]
   for t in np.flatnonzero(q['mask']):
    f=summarize_frame(q['quality'][t],raw[t].numpy(),pt[t].numpy(),valid[t].numpy(),q['quality'][t,0],o['active'],q['quality'][t,0],q['quality'][t,0])
    for k in ['on_gain_pp','step1_gain_pp','posterior_good_but_On_nonpositive']:f.pop(k)
    f.update(E_raw_rank=float(1+(raw[t][valid[t]]>raw[t,0]+1e-12).sum()),E_probability=float(pt[t,0]));frames.append(f)
   rankings[kind]={k:mean([f[k]for f in frames])for k in frames[0]}if frames else {}
   rr=raw[allowed];pp=pt[allowed];pr=fb['prior'][allowed];term=torch.where(pp>0,pp*(pp.clamp_min(1e-300).log()-pr.clamp_min(1e-300).log()),torch.zeros_like(pp))
   scales[kind]=dict(reward_std=float(rr.std(-1,unbiased=False).mean()),reward_range=float((rr.max(-1).values-rr.min(-1).values).mean()),contrast_over_temperature=float(rr.std(-1,unbiased=False).mean()/.1),prior_ESS=float((1/pr.square().sum(-1)).mean()),teacher_ESS=float((1/pp.square().sum(-1)).mean()),teacher_prior_KL=float(term.sum(-1).mean()))
  changes={}
  for mode in ['Fixed_dense','On_policy_dense']:
   for tag,fit in [('Bidir',o['fits'][mode]),('Forward',old['fits'][mode])]:
    E=fit['path'][0]['boxes'];d=(fit['final'][:,:2]-E[:,:2])[allowed]/(.25*E[:,2:][allowed]);norm=d.norm(dim=-1)
    changes[tag+'_'+mode]=dict(sigma_normalized_mean=float(norm.mean()),sigma_normalized_RMS=float(norm.square().mean().sqrt()),sigma_normalized_max=float(norm.max()),changed_nodes=int((norm>1e-12).sum()))
  shift=(fb['rewards']-fb['rewards'][:,0,None]-(old['direct']['feedback']['rewards']-old['direct']['feedback']['rewards'][:,0,None]))/.1
  controls={}
  for name,mask in [('good',q['quality']>q['quality'][:,0,None]+1e-10),('bad',q['quality']<q['quality'][:,0,None]-1e-10)]:
   mask=mask&q['mask'][:,None]&valid.numpy();controls[name+'_symmetric_minus_forward_E_relative_contrast']=float(shift.numpy()[mask].mean())if mask.any()and o['active']else None
  rows.append(dict(alias=a,dataset=r['dataset'],rankings=rankings,scales=scales,output_response=changes,E_relative_contrast=controls))
 groups={}
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];rk={}
  for kind in ['Forward','Reverse','Symmetric']:
   keys=next(r['rankings'][kind]for r in rr if r['rankings'][kind]).keys();rk[kind]={k:paired([r['rankings'][kind].get(k)for r in rr])for k in keys}
  groups[ds]=dict(count=16,ranking=rk,scales={kind:{k:mean([r['scales'][kind][k]for r in rr])for k in rr[0]['scales'][kind]}for kind in rr[0]['scales']},response={kind:{k:mean([r['output_response'][kind][k]for r in rr])for k in rr[0]['output_response'][kind]}for kind in rr[0]['output_response']},E_relative_contrast={k:paired([r['E_relative_contrast'][k]for r in rr])for k in rr[0]['E_relative_contrast']})
 put(P/'SEALED_CACHE_REVIEW.json',dict(status='complete',groups=groups,rows=rows,no_new_predictions_or_GT_reads=True,initial34_candidates_exact=True,new_Live_second_round_candidate_GT_quality='unknown; never reused initial candidate qualities',interpretation='Forward/Reverse/Symmetric rankings are diagnostic only; no new deployed arm or threshold selection'))
 print('sealed cache review complete',flush=True)
if __name__=='__main__':main()
