"""Measured E-neighborhood feedback qualification; anonymous export."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha
from scripts.run_stvg_e_tube_critic64_v1 import D,P,verify


def main():
    import matplotlib.pyplot as plt
    p=verify(True);r=read(P/'RESULTS.json');assert read(P/'ROOT_INDEPENDENT_AUDIT.json')['status']=='pass'
    put(P/'RESULTS_PUBLIC.json',r)
    cfg={k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']};cfg['backbone']='TA-STVG';cfg['source_checkpoints']=read(ROOT/'results/stvg_error_state/2026-10-18/query_e_response64/CONFIGURATION.json')['source_checkpoints'];cfg['critic_checkpoints']={d:dict(source_domain='vidstg'if d=='hc2'else'hc2',sha256=read(D/'RUNTIME_LOCK.json')['pins'][f'artifacts/stvg_joint_event_critic_v1/critics/{"vidstg"if d=="hc2"else"hc2"}.pt'])for d in ['hc2','vidstg']}
    put(P/'CONFIGURATION.json',cfg)
    fig,axs=plt.subplots(1,3,figsize=(15,4.7));fig.subplots_adjust(bottom=.24,wspace=.35)
    for ax,name,label in zip(axs[:2],['spatial_ranking_v','temporal_ranking_v'],['Spatial order','Temporal order (tube vIoU)']):
        for j,ds in enumerate(['hc2','vidstg']):
            v=r['groups'][ds]['ranking'][name]
            if v['mean']is not None:ax.errorbar(j,v['mean'],yerr=np.array([[v['mean']-v['CI95'][0]],[v['CI95'][1]-v['mean']]]),fmt='o',capsize=4,color=['#217f88','#bc7744'][j])
        ax.axhline(.5,color='.5',ls='--');ax.set_xticks([0,1],['HC32','Vid32']);ax.set_ylim(0,1);ax.set_title(label);ax.set_ylabel('Mean query pair accuracy')
    for ds,col in [('hc2','#217f88'),('vidstg','#bc7744')]:
        rr=[x for x in r['rows']if x['dataset']==ds];axs[2].scatter([100*(x['values']['Oracle']['v']-x['values']['E_MBR']['v'])for x in rr],[100*(x['values']['Joint']['v']-x['values']['E_MBR']['v'])for x in rr],s=20,c=col,label=ds,alpha=.7)
    axs[2].axhline(0,color='.6',lw=.8);axs[2].set_xlabel('Candidate oracle gain vs E (pp)');axs[2].set_ylabel('Critic-selected gain vs E (pp)');axs[2].set_title('Headroom versus realized selection');axs[2].legend(fontsize=8)
    fig.text(.06,.045,'Every query retained for selection means. Pair accuracy uses query-level bootstraps; all-GT-tie rankings remain unknown.\nOracle is retrospective diagnosis only. This is feedback qualification, not OPD performance.',fontsize=9)
    review=D/'root_review';review.mkdir(exist_ok=True);fig.savefig(P/'FEEDBACK_QUALIFICATION.svg');fig.savefig(review/'FEEDBACK_QUALIFICATION.png',dpi=140);plt.close(fig)
    lines=['# Frozen critic feedback near E:64-query qualification','','Same original HC32/Vid32 exposed development cohort. Three spatial states(Source, half E delta, full E) and final MBR plus three other highest-probability distinct native physical intervals. Old critic weights are frozen. No OPD, QR/LN update, expert call or training occurs in this stage. All candidate scores sealed before new GT.','','| Domain | Tubes | E+MBR vIoU% | Joint-selected vIoU% | E-only time choice% | GT candidate oracle% | Joint-E pp [95% CI] |','|---|---:|---:|---:|---:|---:|---:|']
    for ds,g in r['groups'].items():
        m=g['metrics'];d=g['deltas_vs_E']['Joint']['v'];lines.append(f'| {ds} | {sum(x["candidate_count"]for x in r["rows"]if x["dataset"]==ds)} | {100*m["E_MBR"]["v"]:.6f} | {100*m["Joint"]["v"]:.6f} | {100*m["E_time_selection"]["v"]:.6f} | {100*m["Oracle"]["v"]:.6f} | {d["mean"]:+.6f} [{d["CI95"][0]:+.6f},{d["CI95"][1]:+.6f}] |')
    lines+=['','| Domain | Spatial v-order mean [95% CI] | Temporal v-order mean [95% CI] | Temporal t-order mean [95% CI] |','|---|---:|---:|---:|']
    for ds,g in r['groups'].items():
        cells=[]
        for name in ['spatial_ranking_v','temporal_ranking_v','temporal_ranking_t']:
            v=g['ranking'][name];cells.append('unknown'if v['mean']is None else f'{v["mean"]:.6f} [{v["CI95"][0]:.6f},{v["CI95"][1]:.6f}], n={v["n"]}')
        lines.append('| '+ds+' | '+' | '.join(cells)+' |')
    lines+=['','The unit is a query, not a candidate pair. All strict pairs are retained in each query record, exact critic ties receive half credit, GT-quality ties use the prelocked1e-8 tolerance; no strict pairs means unknown. Unsupported E queries are retained with duplicate spatial states. Spatial/time v-order here evaluates complete-tube quality; t-order separately tests event-boundary direction.','','## Headroom and time-support checks','','| Domain | Oracle-E pp | Time oracle at E-E pp | Spatial oracle at MBR-E pp | Joint tIoU change pp | Shortened queries | v up/t down |','|---|---:|---:|---:|---:|---:|---:|']
    for ds,g in r['groups'].items():
        d=g['deltas_vs_E'];lines.append(f'| {ds} | {d["Oracle"]["v"]["mean"]:+.6f} | {d["Oracle_time_at_E"]["v"]["mean"]:+.6f} | {d["Oracle_space_at_MBR"]["v"]["mean"]:+.6f} | {d["Joint"]["t"]["mean"]:+.6f} | {g["time_change"]["shortened"]} | {g["time_change"]["v_improved_t_decreased"]} |')
    lines+=['','Complete per-query deleted/added actual decoder-frame counts, exact-known GT support, E sIoU on removed/retained/added locations, interval duration/start fractions, scores and all12 v/t/s metrics are public. Missing GT support stays unknown. A vIoU gain through discarding hard frames is not automatically event-boundary improvement. No GT oracle candidate is used for deployment.','','## Prelocked decision','','```json',json.dumps(r['decision'],indent=2),'```','',f'Equal-domain macro Joint-E: {r["macro"]["mean_pp"]:+.6f}pp,95% CI {r["macro"]["CI95_pp"]}.',
        '','The advancement rule is a resource decision: query-mean spatial and temporal(v-order)95% lower>.5 in both domains, both domain Joint-E means>=0, macro paired lower>0. Uncertain ranking is not a pass and does not automatically expand data/change thresholds. Old source32/33 to target9/30 spatial failure stays recorded; changing the candidate neighborhood does not presume the old critic has become reliable.','','If the gate fails, stop this critic before any optimizer experiment; do not search a temperature, reward, step size or new model to rescue this same feedback. This does not prove all critics, joint inference or OPD impossible. If it passes, only the prelocked source-scaled five-round external time2/space4 experiment is eligible, with fresh finite resource lock and first-two-case actual timings. This stage is not an OPD score or a backbone-agnostic system: the critic is source trained and consumes TA-STVG features.','','## Actual cost and audit','','Source caches held exact boxes/logits/E but no full maps for new regions, so one frozen Source capture per query was required; no historical Before replay or new expert. Cached Source ROI features were checked exact against the original critic extraction within each domain. Checkpoints, Source output/logits and frozen model states were verified. Native final-envelope candidates are independently rebuilt by direct two-offset enumeration. All64 rankings/metrics/bootstrap comparisons are independently audited.','','[Complete anonymous results](RESULTS_PUBLIC.json) · [Exact configuration and source-only conditional stage2 lock](CONFIGURATION.json) · [Independent audit](ROOT_INDEPENDENT_AUDIT.json) · [Actual cost](ACTUAL_COSTS.json) · [Qualification plot](FEEDBACK_QUALIFICATION.svg).']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n');(P/'README.md').write_text('# E-neighborhood frozen feedback64\n\n[Report](REPORT.md). Complete candidate/selection diagnostics; not an OPD experiment.\n')
    bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');put(P/'ACTUAL_COSTS.json',dict(GPU_seconds=bar['GPU_seconds'],first_stage_wall_seconds=time_now()-p['created_epoch'],CPU_score_seconds=read(D/'CPU_COMPLETION.json')['CPU_seconds'],new_Source_captures=64,new_DINO_calls=0,new_Qwen_calls=0,new_training=0,new_OPD_fits=0,total_tubes=r['total_tubes'],Source_capture_seconds=sum(x['compute']['Source_capture_seconds']for x in r['rows']),source_only_optional_stage2_scale_calibration_critic_evals=144,not_complete_first_online_expert_cost=True))
    put(P/'VERIFICATION.json',dict(status='pass',count=64,independent=read(P/'ROOT_INDEPENDENT_AUDIT.json'),actual_chart_view_pending=True,old_predictions_and_registries_unchanged=True,private_maps_boxes_queries_GT_weights_not_exported=True,new_semantic_labels=0))
    files=['vg_tta/stvg_e_tube_critic64_v1.py','scripts/run_stvg_e_tube_critic64_v1.py','scripts/continue_stvg_e_tube_critic64_v1.py','scripts/score_stvg_e_tube_critic64_v1.py','scripts/audit_stvg_e_tube_critic64_v1.py','scripts/report_stvg_e_tube_critic64_v1.py','scripts/publish_stvg_e_tube_critic64_v1.py','protocols/stvg_e_tube_critic64_v1.md']
    files +=[str((P/f).relative_to(ROOT))for f in ['RESULTS_PUBLIC.json','CONFIGURATION.json','DECISION.json','ROOT_INDEPENDENT_AUDIT.json','ACTUAL_COSTS.json','VERIFICATION.json','REPORT.md','README.md','FEEDBACK_QUALIFICATION.svg']]
    put(D/'PUBLIC_FILES.json',dict(files=files));print(dict(status='generated',files=len(files)))


def time_now():
    import time
    return time.time()


if __name__=='__main__':main()
