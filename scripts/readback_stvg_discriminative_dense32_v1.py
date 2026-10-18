"""Actual target-frame feature/score/probability alignment, without GT access."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load
from scripts.run_stvg_discriminative_dense32_v1 import D,P,verify
from vg_tta.stvg_soft_correspondence_v1 import cell_overlap
import torch,numpy as np

def main():
    torch.set_num_threads(4);p=verify();assert read(D/'FEEDBACK_BARRIER.json')['all32_before_GT'];checks=collections.Counter();coverage={};maxerr=0
    for rec in p['records']:
        a=rec['alias'];z=load(D/'feedback'/(a+'.pt'));x=load(D/'features'/(a+'.pt'));f=z['feedback'];b=f['candidates'];teachers=z['teachers'];n=len(b);stat=coverage.setdefault(rec['dataset'],dict(selected_E=0,usable=0,queries_without_teacher=0,unusable_no_background=0,positive_background_contrast=[]));stat['selected_E']+=sum(k>=0 for k in x['selected']);stat['usable']+=len(teachers);stat['queries_without_teacher']+=not bool(teachers);stat['unusable_no_background']+=sum(not t['usable']and t['reason']=='no_disjoint_background'for t in z['training_diagnostics'])
        expected=[g['position']for g,k in zip(x['frames'],x['selected'])if k>=0];assert expected==f['cosine_anchor_positions'];checks['all_E_cosine_support']+=1
        for j in x['planned_positions']:assert f['permutation'][j]==j
        for key,order in [('real',list(range(n))),('shuffled',f['permutation'])]:
            phi=[];okay=[]
            for i,t in enumerate(order):
                area=cell_overlap(b[i],*x['maps'].shape[-2:])*x['valid'][t];mass=area.sum((-1,-2));raw=torch.stack([(aa[None]*x['maps'][t].double()).sum((-1,-2))/mm.clamp_min(1e-12)for aa,mm in zip(area,mass)]);raw=raw/raw.norm(dim=-1,keepdim=True).clamp_min(1e-12);phi.append(torch.cat((raw,torch.ones(len(raw),1)),1));okay.append(mass>1e-12)
            phi=torch.stack(phi);okay=torch.stack(okay);assert torch.equal(okay,f[key]['valid'])
            if teachers:
                score=torch.stack([phi@t['w']for t in teachers]);err=float((score-f[key]['scores']).abs().max());maxerr=max(err,maxerr);assert err<1e-12;checks['own_target_frame_classifier_score_scalars']+=score.numel()
                noise=torch.cat((f['noise'][0],f['noise'][1]),1);lp=-.5*noise.square().sum(-1);safe=okay.clone();safe[~safe.any(1)]=True;lp[~safe]=-torch.inf;pa=(lp[None]+score/p['configuration']['temperature']).softmax(-1);assert (pa-f[key]['per_anchor']).abs().max()<1e-12;checks['per_anchor_distribution_scalars']+=pa.numel()
                prob=pa.mean(0);special=~f[key]['informative'];prob[special]=f[key]['prior'][special];assert (prob-f[key]['prob']).abs().max()<1e-12;checks['probability_mixture_scalars']+=prob.numel()
            else:assert torch.equal(f[key]['prob'],f[key]['prior']);checks['no_teacher_prior_exact']+=1
        for t in z['training_diagnostics']:
            if t['usable']:stat['positive_background_contrast'].append(t['positive_score']-t['background_mean_score'])
    for g in coverage.values():
        v=g.pop('positive_background_contrast');g['positive_background_contrast']=dict(count=len(v),mean=float(np.mean(v))if v else None,positive=sum(x>0 for x in v))
    verify(full=True);put(P/'ROOT_FEATURE_ALIGNMENT.json',dict(status='pass',count=32,checks=dict(checks),max_score_repool_error=maxerr,coverage=coverage,GT_read=False,all_anchors_background_failures_retained=True));print(coverage,flush=True)
if __name__=='__main__':main()
