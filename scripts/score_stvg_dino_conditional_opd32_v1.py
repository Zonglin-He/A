"""Full32 seal then GT; no GT-dependent candidate creation or adaptation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_dino_conditional_opd32_v1 import D,P,ARMS,verify
from scripts.score_stvg_joint_robust_correction32_v1 import paired


def mean(v):
    a=[float(x)for x in v if x is not None];return float(np.mean(a))if a else None


def main():
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube,box_iou
    p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==32 and bar['all_outputs_before_GT']
    for rec in bar['files']:assert sha(ROOT/rec['path'])==rec['sha256']
    gtall={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']}
    plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in gtall}
    prior={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-18/query_e64/RESULTS.json')['rows']}
    rows=[];checks=0;tick=time.perf_counter()
    for r in p['records']:
        z=load(D/'predictions'/(r['alias']+'.pt'));ds=r['dataset'];q=r['query_ordinal'];row=plans[ds][q];gt,span=gtall[ds][0][q],gtall[ds][1][q];ids=np.array(z['frame_ids'])
        un=np.array([int(t)in gt for t in ids])&~np.isin(np.arange(len(ids)),z['planned_positions']);values={}
        for a,b in z['outputs'].items():
            v=official(b,row,gt,span,z['interval'],ds);other=DenseTube(b,row,gt,span,clip=ds=='hc2').score(z['interval']);assert max(abs(v[k]-other[k])for k in ['v','t','s'])<2e-10;checks+=3
            xy=np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2]*np.array([row['input']['width'],row['input']['height']]*2)
            if ds=='hc2':xy=np.maximum(xy,0)
            si=box_iou(xy[un],np.array([gt[int(t)]for t in ids[un]]))if un.any()else []
            values[a]=dict(**v,unobserved_actual_s=mean(si),known_unobserved_frames=int(un.sum()))
        assert len(set(v['t']for v in values.values()))==1
        for a in ARMS[:2]:
            for ep in ['v','t','s','unobserved_actual_s']:
                x,y=values[a][ep],prior[r['alias']]['values'][a][ep]
                assert x is None and y is None or x is not None and y is not None and abs(x-y)<1e-12;checks+=1
        diagnostics={}
        for a,f in z['fits'].items():
            details=[d for step in f['rounds']for d in step['details']]
            diagnostics[a]=dict(rounds=len(f['rounds']),logical_teacher_queries=f['logical_teacher_queries'],
                legal_actions=sum(x['legal']for x in details),invalid_actions=sum(not x['legal']for x in details),
                no_accepted_anchors=sum(x['accepted_anchors']==0 for x in details),accepted_anchor_feedback=sum(x['accepted_anchors']for x in details),
                mean_student_steps=[v['mean_step_l2']for v in f['rounds']],distill_losses=[[v['loss_before'],v['loss_after']]for v in f['rounds']],
                output_displacement_l2=float(np.linalg.norm(z['outputs'][a]-z['outputs']['E_MBR'])),
                mean_correction_from_sample_l2=mean([np.linalg.norm(x['teacher_action']-x['action'])for x in details]))
        rows.append(dict(alias=r['alias'],dataset=ds,active=z['active'],values=values,diagnostics=diagnostics,compute=z['compute'],qualifications=z['qualifications']))
    pairs=[(a,b)for a in ARMS[2:]for b in ['E_MBR','Source_MBR']]+[('On_policy_distill','Fixed_sample_distill'),('On_policy_distill','One_correction')]
    groups={}
    for ds in gtall:
        rr=[r for r in rows if r['dataset']==ds]
        groups[ds]=dict(queries=len(rr),parents=len(rr),active=sum(r['active']for r in rr),
            metrics={a:{ep:mean([r['values'][a][ep]for r in rr])for ep in ['v','t','s','unobserved_actual_s']}for a in ARMS},
            paired={a+'_minus_'+b:{ep:paired([None if r['values'][a][ep]is None or r['values'][b][ep]is None else 100*(r['values'][a][ep]-r['values'][b][ep])for r in rr])for ep in ['v','s','unobserved_actual_s']}for a,b in pairs})
    macro={};rng=np.random.default_rng(20261018)
    for a,b in pairs:
        vals=[np.array(groups[d]['paired'][a+'_minus_'+b]['v']['per_parent_pp'])for d in gtall]
        bs=[v[rng.integers(len(v),size=(10000,len(v)))].mean(1)for v in vals]
        macro[a+'_minus_'+b]=dict(mean_pp=float(np.mean([v.mean()for v in vals])),CI95_pp=np.quantile(np.mean(bs,0),[.025,.975]).tolist())
    bad=any(groups[d]['paired']['One_correction_minus_E_MBR']['v']['mean']<0 for d in gtall)
    dynamic=all(groups[d]['paired']['On_policy_distill_minus_'+c][ep]['mean']>0 for d in gtall for c in ['E_MBR','Fixed_sample_distill']for ep in ['v','unobserved_actual_s'])
    decision=dict(status='stop_this_teacher_interface_no_LR_rescue'if bad else 'conditional_dynamic_support_requires_new_confirmation'if dynamic else 'teacher_interface_positive_but_on_policy_increment_not_established',
        one_correction_worse_than_E_in_any_domain=bad,dynamic_positive_both_domains_both_endpoints_over_E_and_fixed=dynamic,automatic_expansion=False,
        not_all_OPD_or_DINO_refinement_impossible=True,full_video_dense_independent_evidence=False)
    put(P/'RESULTS_PUBLIC.json',dict(status='complete',count=32,rows=rows,groups=groups,equal_domain_macro=macro,decision=decision,
        all32_outputs_before_new_GT=True,old_exposed_development=True,independent_confirmation=False,independent_metrics_and_baseline_checks=checks))
    put(P/'DECISION.json',decision);put(D/'CPU_COMPLETION.json',dict(status='complete',count=32,score_seconds=time.perf_counter()-tick,result_sha256=sha(P/'RESULTS_PUBLIC.json')))
    print(dict(groups={d:g['metrics']for d,g in groups.items()},decision=decision))


if __name__=='__main__':main()
