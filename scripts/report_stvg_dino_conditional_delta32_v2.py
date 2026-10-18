"""Fixed-estimator32 measured report, no model or target-derived tuning."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_dino_conditional_delta32_v2 import D,P,ARMS,verify


def closing_verify():
    # Only a post-measurement report syntax repair; scientific runtime immutable.
    repair=read(D/'REPORT_REPAIR002.json');runtime=read(D/'RUNTIME_LOCK.json')
    assert sha(D/'RUNTIME_LOCK.json')==repair['original_runtime_sha256']
    report='scripts/report_stvg_dino_conditional_delta32_v2.py'
    assert runtime['pins'][report]==repair['original_report_sha256']
    assert sha(ROOT/repair['preserved_original_report'])==repair['original_report_sha256']
    for f,h in runtime['pins'].items():
        assert sha(ROOT/f)==(repair['repaired_report_sha256']if f==report else h),f
    from main_method.runtime import resource_guard
    resource_guard();return read(D/'PLAN_LOCK.json')


def main():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    p=closing_verify();r=read(P/'RESULTS_PUBLIC.json');audit=read(P/'ROOT_INDEPENDENT_AUDIT.json');assert audit['status']=='pass'
    prior=read(ROOT/'results/stvg_error_state/2026-10-18/dino_conditional32/RESULTS_PUBLIC.json');old={x['alias']:x for x in prior['rows']}
    cfg={k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']}
    cfg.update(source_checkpoints=read(ROOT/'results/stvg_error_state/2026-10-18/dino_conditional32/CONFIGURATION.json')['source_checkpoints'],
        DINO_weight_SHA256='1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3',
        old_commit='60c617cc431f829c1b1a34dffb065dae201ccf4e',old_teacher_implementation_verbatim=True)
    put(P/'CONFIGURATION.json',cfg);put(P/'CPU_PROPERTY_COMPLETION.json',read(D/'CPU_PROPERTY_COMPLETION.json'));put(P/'CPU_REPORT_RECOVERY.json',read(D/'REPORT_REPAIR002.json'))
    rr=[];groups={};cost=collections.Counter();allidentity=0
    for rec in p['records']:
        z=load(D/'predictions'/(rec['alias']+'.pt'));case=dict(alias=rec['alias'],dataset=rec['dataset'],active=z['active'],arms={})
        for name in ['Fixed_sample_distill','On_policy_distill']:
            f=z['fits'][name];rounds=[]
            for st in f['rounds']:
                d=st['details'];identity=sum(x['accepted_anchors']==0 for x in d)
                rounds.append(dict(round=st['round'],candidates=len(d),legal_candidates=sum(x['legal']for x in d),invalid_candidates=sum(not x['legal']for x in d),
                    candidates_with_any_feedback=len(d)-identity,identity_candidates=identity,accepted_anchor_feedback=sum(x['accepted_anchors']for x in d),
                    attempted_anchor_references=sum(len(x['anchors'])for x in d),
                    exact_noop_if_all_identity=not all(x['accepted_anchors']==0 for x in d)or np.array_equal(st['student_mean_before'],st['student_mean_after']),
                    mean_displacement_norm=st['mean_step_l2']))
                allidentity+=int(st['all_identity_exact_noop'])
            case['arms'][name]=dict(rounds=rounds,any_feedback=any(t['candidates_with_any_feedback']>0 for t in rounds),
                wholly_identity=bool(rounds)and all(t['candidates_with_any_feedback']==0 for t in rounds),
                output_E_exact=np.array_equal(z['outputs'][name],z['outputs']['E_MBR']),
                mean_E_displacement_l2=float(np.linalg.norm(f['mean']-np.asarray(load(ROOT/rec['prior_prediction'])['estimate']['delta'])/p['scale'][rec['dataset']])))
        rr.append(case)
        for k,v in z['compute'].items():
            if k!='GPU_seconds_so_far':cost[k]+=v
    for ds in ['hc2','vidstg']:
        groups[ds]={}
        for a in ['Fixed_sample_distill','On_policy_distill']:
            ss=[x for x in rr if x['dataset']==ds];st=[t for x in ss for t in x['arms'][a]['rounds']]
            groups[ds][a]=dict(queries=16,active=sum(x['active']for x in ss),queries_with_any_feedback=sum(x['arms'][a]['any_feedback']for x in ss),
                all_identity_queries=sum(x['arms'][a]['wholly_identity']for x in ss),E_exact_outputs=sum(x['arms'][a]['output_E_exact']for x in ss),
                candidates=sum(t['candidates']for t in st),legal_candidates=sum(t['legal_candidates']for t in st),
                candidates_with_any_feedback=sum(t['candidates_with_any_feedback']for t in st),accepted_anchor_feedback=sum(t['accepted_anchor_feedback']for t in st),
                actual_attempted_anchor_refs=sum(t['attempted_anchor_references']for t in st),all_identity_rounds=sum(t['candidates_with_any_feedback']==0 for t in st),
                all_identity_rounds_exact_noop=all(t['exact_noop_if_all_identity']for t in st))
    put(P/'FEEDBACK_COVERAGE.json',dict(status='complete',groups=groups,rows=rr,all_identity_rounds_exact_noop=allidentity,GT_not_used_for_coverage=True))
    costs=dict(**dict(cost),GPU_lease_seconds=read(D/'GLOBAL_PREDICTION_BARRIER.json')['GPU_seconds'],score_seconds=read(D/'CPU_COMPLETION.json')['score_seconds'],
        CPU_contract_checks=8,first2=read(D/'PILOT2_TIMING.json'),one_correction_group_reused=True,
        conditional_decoder_calls_include_clone_qualification=8,conditional_decoder_seconds_include_clone_qualification=True,
        full_forward_seconds_include_native_decoder=True,old_E_generation_and_Source_cost_excluded=True,
        not_complete_first_online_end_to_end_cost=True)
    put(P/'ACTUAL_COSTS.json',costs)
    pairs=['On_policy_distill_minus_E_MBR','On_policy_distill_minus_Fixed_sample_distill']
    comparisons={d:{a:dict(old_vIoU=prior['groups'][d]['metrics'][a]['v'],fixed_vIoU=r['groups'][d]['metrics'][a]['v'],
        change_pp=100*(r['groups'][d]['metrics'][a]['v']-prior['groups'][d]['metrics'][a]['v']))for a in ['Fixed_sample_distill','On_policy_distill']}for d in groups}
    put(P/'OLD_NEW_MATCHED_COMPARISON.json',dict(status='complete',groups=comparisons,rows=[dict(alias=x['alias'],dataset=x['dataset'],
        old={a:old[x['alias']]['values'][a]for a in ['Fixed_sample_distill','On_policy_distill']},new={a:x['values'][a]for a in ['Fixed_sample_distill','On_policy_distill']})for x in r['rows']],same_cohort_scale_seed_E_MBR_teacher=True,only_estimator_changed=True))
    lines=['# Teacher-relative correction estimator: original32 re-evaluation','','Single preauthorized change: mu_new=mu+mean(y-a). The former uncentered empirical KL optimum is not claimed. Same population mean at fixed current mean under zero-mean noise; finite-sample paths change. No parameter search. Fixed24 and dynamic8x3 teacher feedback were actually recomputed, not spliced from v1. All32 outputs sealed before new GT; old target exposure acknowledged.','','| Domain | E+MBR vIoU% | One correction (exact old)% | Fixed24 repaired% | Dynamic3 repaired% | Old dynamic% |','|---|---:|---:|---:|---:|---:|']
    for ds,g in r['groups'].items():
        m=g['metrics'];lines.append('| '+ds+' | '+' | '.join(f'{100*m[a]["v"]:.6f}'for a in ARMS[1:])+f' | {100*prior["groups"][ds]["metrics"]["On_policy_distill"]["v"]:.6f} |')
    lines+=['','| Domain | Contrast | Complete vIoU pp [95% CI] | Unobserved actual sIoU pp [95% CI] | v loss >5pp /16 |','|---|---|---:|---:|---:|']
    def fmt(v):return 'unknown'if v['mean']is None else f'{v["mean"]:+.6f} [{v["CI95"][0]:+.6f},{v["CI95"][1]:+.6f}]'
    for ds,g in r['groups'].items():
        for pair in pairs+['Fixed_sample_distill_minus_E_MBR']:
            v=g['paired'][pair]['v'];u=g['paired'][pair]['unobserved_actual_s'];lines.append(f'| {ds} | {pair} | {fmt(v)} | {fmt(u)} | {v["severe_loss_below_minus5pp"]}/16 |')
    lines+=['','## Feedback coverage','','| Domain/arm | Query with feedback | Legal candidates | Candidates with feedback | Accepted anchor feedback | All-identity queries | E-exact outputs |','|---|---:|---:|---:|---:|---:|---:|']
    for ds,gg in groups.items():
        for a,g in gg.items():lines.append(f'| {ds}/{a} | {g["queries_with_any_feedback"]}/16 | {g["legal_candidates"]}/{g["candidates"]} | {g["candidates_with_any_feedback"]} | {g["accepted_anchor_feedback"]} | {g["all_identity_queries"]} | {g["E_exact_outputs"]}/16 |')
    lines+=['','All-identity feedback now leaves every round mean exactly unchanged. CPU contract covers K1/8/24/32 identity,100 identity rounds, partial feedback, different sample means and zero empirical noise mean. Actual saved paths independently verify the same invariant. No-feedback candidates contribute zero; denominator remains all sampled candidates. No clipping, rejection resampling or extra threshold. Unknown GT is not failure.','','## Decision','','`'+r['decision']['status']+'`. No automatic data expansion or teacher-interface tuning. Return near E alone establishes estimator repair, not added visual-correction or on-policy utility. Original E+MBR and one-correction are exact prior baselines. Three-round time intervals/tIoU remain identical. All new output movement must come from actual admitted correction residuals.','','This is exposed HC16/Vid16 development, not independent confirmation. Frozen nearest-native-proposal content initialization may produce very little target-qualified feedback; native replay/clone qualification does not establish arbitrary-box grounding reliability. The four image positions are sparse visual evidence even with many sampled states. No new Qwen/SAM, critic, association training, QR/LN/backbone fit, second backbone or historical Before replay.','','## Cost and integrity','',f'Actual GPU lease {costs["GPU_lease_seconds"]:.3f}s; full anchor encode/native-forward {costs["encoding_full_forward_seconds"]:.3f}s; additional decoder {costs["conditional_decoder_seconds"]:.3f}s (includes8 clone qualifications). Existing E generation cost excluded. No new Source call.112 new anchor image/text encodings for28 active queries;4 unsupported Vid queries remain no-op. Not complete first online latency.','','Old predictions preserved. Fixed24 and dynamic round1 actions and teacher outputs checked exact against v1 before update; later dynamic states used fresh teacher calls. Root audit independently reconstructs every correction, output and official metric, signed bootstrap intervals and all identity no-op states. Raw images, Query/parent text, GT coordinates, weights and tensors stay private.']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n');(P/'README.md').write_text('See REPORT.md, RESULTS_PUBLIC.json, CONFIGURATION.json, FEEDBACK_COVERAGE.json, ROOT_INDEPENDENT_AUDIT.json, OLD_NEW_MATCHED_COMPARISON.json and ACTUAL_COSTS.json.\n')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5));fig.subplots_adjust(bottom=.25,wspace=.3)
    for ax,ds in zip(axs,groups):
        names=['E_MBR','Fixed_sample_distill','On_policy_distill'];vals=[100*r['groups'][ds]['metrics'][a]['v']for a in names]
        ax.bar(range(3),vals,color=['#528b99','#7564a5','#bc5353']);ax.set_xticks(range(3),['E','Fixed24 repaired','Dynamic3 repaired'],rotation=15);ax.set_ylabel('Complete tube vIoU (%)');ax.set_title(ds+'16, exposed development')
    fig.text(.07,.04,'Only mean update changed: mu + mean(teacher correction). No-feedback states stay fixed.\nSame source scales, sampled epsilon, original4 DINO frames, .35 threshold, frozen teacher and final-MBR.',fontsize=8)
    review=D/'root_review';review.mkdir(exist_ok=True);fig.savefig(P/'MEASURED_REPAIR.svg');fig.savefig(review/'MEASURED_REPAIR.png',dpi=140);plt.close(fig)
    from scripts.run_decota_paper_main_v1 import read_row,frames_for
    cache=collections.OrderedDict();panels=[]
    for ds in groups:
        rec=next(x for x in p['records']if x['dataset']==ds);z=load(D/'predictions'/(rec['alias']+'.pt'));row=read_row(ds,rec['query_ordinal']);frames,ids,_=frames_for(ds,row,cache)
        fig,axs=plt.subplots(1,4,figsize=(16,4));fig.subplots_adjust(top=.86,bottom=.05)
        for ax,pos in zip(axs,z['planned_positions']):
            rgb=frames[pos];h,w=rgb.shape[:2];ax.imshow(rgb)
            for arm,col in [('E_MBR','lime'),('On_policy_distill','magenta')]:
                cx,cy,bw,bh=z['outputs'][arm][pos];ax.add_patch(Rectangle(((cx-bw/2)*w,(cy-bh/2)*h),bw*w,bh*h,fill=False,color=col,lw=1.4))
            ax.set_title('anchor '+str(pos));ax.axis('off')
        fig.suptitle(rec['alias']+' actual original4: green E, magenta repaired dynamic; no semantic label');f=review/(ds+'_actual_pixels.png');fig.savefig(f,dpi=100);plt.close(fig)
        panels.append(dict(alias=rec['alias'],dataset=ds,path=str(f.relative_to(ROOT)),sha256=sha(f),anchors=4))
    put(D/'ACTUAL_PIXEL_PANELS.json',dict(status='pending_root_actual_view',panels=panels,new_semantic_labels=0))
    own=['vg_tta/stvg_dino_conditional_delta32_v2.py','scripts/run_stvg_dino_conditional_delta32_v2.py','scripts/score_stvg_dino_conditional_delta32_v2.py','scripts/audit_stvg_dino_conditional_delta32_v2.py','scripts/report_stvg_dino_conditional_delta32_v2.py','scripts/publish_stvg_dino_conditional_delta32_v2.py','protocols/stvg_dino_conditional_delta32_v2.md']
    files=['RESULTS_PUBLIC.json','CONFIGURATION.json','DECISION.json','ROOT_INDEPENDENT_AUDIT.json','CPU_PROPERTY_COMPLETION.json','ACTUAL_COSTS.json','FEEDBACK_COVERAGE.json','OLD_NEW_MATCHED_COMPARISON.json','REPORT.md','README.md','MEASURED_REPAIR.svg','CPU_REPORT_RECOVERY.json']
    put(D/'PUBLIC_FILES.json',dict(files=own+[str((P/f).relative_to(ROOT))for f in files]));print(dict(status='report_ready',files=len(own)+len(files)))


if __name__=='__main__':main()
