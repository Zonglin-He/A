"""Primary: no-current-expert arrivals, conditioned on the saved online history."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_external_opd20_online64_v1 import D,P,verify
from vg_tta.stvg_external_opd20_v1 import ARMS
from scripts.score_stvg_joint_robust_correction32_v1 import paired


def mean(v):
    q=[float(x)for x in v if x is not None];return float(np.mean(q))if q else None


def summarize(rows):
    metrics={a:{k:mean([r['values'][a][k]for r in rows])for k in ['v','t','s','actual_decoder_s','R03','R05']}for a in ARMS}
    contrasts={}
    for a,b in [('D_dense_OPD',x)for x in ['A_no_learning','B_Direct','C_sparse_OPD','F_fixed_rollout']]+[('B_Direct','A_no_learning'),('C_sparse_OPD','A_no_learning'),('F_fixed_rollout','A_no_learning')]:
        contrasts[a+'_minus_'+b]={}
        for ep in ['v','s','actual_decoder_s']:
            v=paired([None if r['values'][a][ep]is None or r['values'][b][ep]is None else 100*(r['values'][a][ep]-r['values'][b][ep])for r in rows]);x=np.array(v.get('per_parent_pp',[]));v.update(severe_loss_below_minus20pp=int((x< -20).sum()),severe_loss_fraction=None if not len(x)else float((x< -5).mean()),CI_conditioned_on_fixed_saved_history=True);contrasts[a+'_minus_'+b][ep]=v
    return dict(queries=len(rows),parents=len(rows),metrics=metrics,paired=contrasts)


def main():
    import torch
    from scripts.score_stvg_opd_p1_v1 import truths
    from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube,box_iou
    torch.set_num_threads(4);p=verify();bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==64 and bar['all64_five_arm_outputs_before_GT']
    for f in bar['files']:assert sha(ROOT/f['path'])==f['sha256']
    targets={ds:truths(ds,dict(parents=[r['query_ordinal']for r in p['records']if r['dataset']==ds]))for ds in ['hc2','vidstg']};plans={ds:read(ROOT/f'artifacts/decota_paper_experiments_v1/{ds}/PLAN.json')['rows']for ds in targets}
    old={r['alias']:r for r in read(ROOT/'results/stvg_error_state/2026-10-18/source_expanded256/RESULTS.json')['rows']};rows=[];checks=0;started=time.perf_counter()
    for rec in p['records']:
        z=load(D/'predictions'/(rec['alias']+'.pt'));ds=rec['dataset'];q=rec['query_ordinal'];row=plans[ds][q];gt,span=targets[ds][0][q],targets[ds][1][q];ids=np.array(z['frame_ids']);known=np.array([int(t)in gt for t in ids]);vals={}
        for a,b in z['outputs'].items():
            m=official(b,row,gt,span,z['interval'],ds);v=DenseTube(b,row,gt,span,clip=ds=='hc2').score(z['interval']);assert max(abs(m[k]-v[k])for k in ['v','t','s'])<2e-10;checks+=3
            xy=np.c_[b[:,:2]-b[:,2:]/2,b[:,:2]+b[:,2:]/2]*np.array([row['input']['width'],row['input']['height']]*2)
            if ds=='hc2':xy=np.maximum(xy,0)
            actual=mean(box_iou(xy[known],np.array([gt[int(t)]for t in ids[known]])))if known.any()else None
            vals[a]=dict(**m,actual_decoder_s=actual,known_decoder_frames=int(known.sum()),R03=float(m['v']>.3),R05=float(m['v']>.5))
        assert len({v['t']for v in vals.values()})==1
        original='Source_no_correction'if rec['role']=='no_expert'else'E_DINO_support'
        for k in ['v','t','s']:assert abs(vals['A_no_learning'][k]-old[rec['alias']]['values'][original][k])<1e-12;checks+=1
        if rec['role']=='expert':assert all(vals[a]==vals['A_no_learning']for a in ARMS)
        fitstats={}
        for a,f in z['fits'].items():
            rr=[x for x in f['trace']if x.get('loss')is not None];fitstats[a]=dict(actual_steps=f['actual_steps'],distinct_rollouts=f['distinct_rollouts'],W_change_norm=float((f['W_after']-f['W_before']).norm()),
                mean_parameter_step=mean([x['parameter_step_norm']for x in rr]),mean_output_step=mean([x['output_displacement_l2']for x in rr]),
                mean_reward_variance=mean([x['reward_variance']for x in rr]),mean_ESS=mean([x['ESS']for x in rr]),
                mean_IoU_before=mean([x['reference_IoU_before']for x in rr]),mean_IoU_after=mean([x['reference_IoU_after']for x in rr]))
        rows.append(dict(alias=rec['alias'],dataset=ds,role=rec['role'],domain_arrival=rec['domain_arrival'],values=vals,valid_expert=z['valid_expert'],fit_diagnostics=fitstats,
            predicted_W_norm={a:float(z['W_before'][a].norm())for a in ARMS},CPU_arrival_seconds=z['compute']['CPU_arrival_seconds']))
    groups={ds:{role:summarize([r for r in rows if r['dataset']==ds and(role=='all'or r['role']==role)])for role in ['no_expert','expert','all']}for ds in targets}
    macro={};rng=np.random.default_rng(20261018)
    for pair in groups['hc2']['no_expert']['paired']:
        macro[pair]={}
        for ep in ['v','s','actual_decoder_s']:
            vv=[np.array(groups[ds]['no_expert']['paired'][pair][ep]['per_parent_pp'])for ds in targets];boot=[a[rng.integers(len(a),size=(10000,len(a)))].mean(1)for a in vv]
            macro[pair][ep]=dict(mean_pp=float(np.mean([a.mean()for a in vv])),CI95_pp=np.quantile(np.mean(boot,0),[.025,.975]).tolist(),equal_domain_weight=True,conditional_on_one_saved_history=True)
    for ds in targets:
        for ep in ['v','s']:
            for a in ARMS:
                assert abs(groups[ds]['all']['metrics'][a][ep]-.5*(groups[ds]['expert']['metrics'][a][ep]+groups[ds]['no_expert']['metrics'][a][ep]))<1e-12
    put(P/'RESULTS.json',dict(status='complete',count=64,primary='no_expert16/domain',groups=groups,equal_domain_no_expert_macro=macro,rows=rows,
        fixed_online_history_conditional_intervals_not_resampled_streams=True,expert_output_identical_all_arms=True,
        old_A_Source_E_scalars_exact=True,all_temporal_outputs_identical=True,independent_metric_checks=checks,
        teacher_not_read_at_no_expert_arrivals=True,historically_exposed_development=True,independent_confirmation=False))
    screens={c:all(groups[ds]['no_expert']['paired']['D_dense_OPD_minus_'+c][ep]['mean']>0 for ds in targets for ep in ['v','s'])for c in ['A_no_learning','B_Direct','C_sparse_OPD','F_fixed_rollout']}
    other={a:all(groups[ds]['no_expert']['paired'][a+'_minus_A_no_learning'][ep]['mean']>0 for ds in targets for ep in ['v','s'])for a in ARMS[1:]}
    put(P/'DECISION.json',dict(status='complete',D_positive_both_domains_v_s_against=screens,learning_arm_positive_both_domains_v_s_against_A=other,
        action='report_development_transfer_signal_no_automatic_expansion'if all(screens.values())else'report_fixed_external20_limits_no_sweep_or_route_level_impossibility_claim',
        no_automatic_expansion_or_parameter_change=True,no_backbone_LN_or_query_updates=True,second_backbone_remains_deferred=True))
    put(D/'CPU_COMPLETION.json',dict(status='complete',count=64,GT_read=True,result_sha256=sha(P/'RESULTS.json'),CPU_score_seconds=time.perf_counter()-started));assert not torch.cuda.is_initialized()


if __name__=='__main__':main()
