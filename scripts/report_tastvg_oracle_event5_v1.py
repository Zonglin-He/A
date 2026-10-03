"""Source-backed tables and figures; final research recommendation is a root record."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from scripts.tastvg_oracle_event5_common_v1 import *
import numpy as np
def estimate(m,absolute=False):
    return f'{m["mean"]*100:+.4f} [{m["ci95"][0]*100:+.4f}, {m["ci95"][1]*100:+.4f}]'
def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    verified();assert read(BASE/'EXPERIMENT1_ROOT_AUDIT.json')['status']=='pass' and read(BASE/'EXPERIMENT2_ROOT_AUDIT.json')['status']=='pass'
    lines=['# Fixed-A oracle ceilings and GT-event5 observation intervention',
      '', 'Both experiments completed. All values below are vIoU percentage points or absolute percent as labelled; intervals are 95% paired source bootstrap (10000 draws, seed 20261003). This is a GT-assisted diagnostic, not an unlabelled method score.',
      '', '## Frozen setting and cohort', '',
      'The predecessor is verified GitHub 3b3ebd297f621667ce03f1291d130b6fd71236db. Each dataset reuses its exact 32 development and 16 within-batch disjoint confirmation sources, one query/source, two fixed orders, clean plus five 5% corruptions and 25% expert arrivals. All sources have historical exposure. Confirmation is a descriptive existing cohort, with no new parameter selection. A is the Uniform persistent spatial learner plus original temporal Fast, rather than Frozen. Its actual pre-arrival states, nine probes and persistent K1/K8 trajectories are read-only.',
      '', 'Vid: lr .033761698432507946, teacher .34902548789596055, K1; HC: lr .006097133675874025, teacher 1, K8. Both rho .05, student 1, D4 and 1792 parameters. Same-domain official checkpoints, original Paper48 observed grid and exact old corruption pixels. GT interval substitution affects evaluation only. No GT is supplied to the decoder.',
      '', '## Experiment 1: complete-stream ideal readouts', '',
      'GT space is the true dense annotated tube, rather than sparse GT interpolation. Joint GT scores exactly 100% at every arrival under the literal official dense metric. A-box trajectories are interpolated only inside original sampled support. HC endpoint conventions remain literal. These readout interventions are not additive causal contributions.', '',
      '| Split / dataset / subset | Cells / sources | A | GT time | GT space | Joint GT |',
      '|---|---:|---:|---:|---:|---:|']
    summary1={};summary2={}
    for split in SPLITS:
        for ds in DATASETS:
            z=read(PUBLIC/'experiment1'/split/ds/'SUMMARY.json');summary1[split,ds]=z
            for sub in ['all','expert','nonexpert']:
                a=z['corruption'][sub];m=a['metrics']
                lines.append(f'| {split} / {ds} / {sub} | {a["cells"]} / {a["sources"]} | '+' | '.join(estimate(m[f]) for f in ['A_v','GT_time_v','GT_space_v','Joint_GT_v'])+' |')
    lines+=['', '## Candidate coverage and selection on expert arrivals', '',
      'Only the already cached 288 expert arrivals have 8 temporal candidates and 9 spatial tubes; all 20736 combinations are evaluated on CPU. No support is generated for nonexpert arrivals. Candidate-native/center outputs and duplicates remain included. Expert source-macro means have different source/order membership from full-stream means; do not multiply them by 25% to estimate full-stream effects.', '',
      '| Split / dataset | Temporal selection H | Temporal coverage H | Spatial selection H | Spatial coverage H | Joint over best single | T recovered |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for split in SPLITS:
        for ds in DATASETS:
            m=summary1[split,ds]['corruption']['expert']['metrics']
            lines.append(f'| {split} / {ds} | '+' | '.join(estimate(m[f]) for f in ['H_selection','H_coverage','H_spatial_selection','H_spatial_coverage','joint_minus_best_single','T_recovered'])+' |')
    lines+=['', 'H_temporal=GT-time−A=H_selection+H_coverage. H_spatial=GT-space−A=H_spatial_selection+H_spatial_coverage. The upper-bound identity holds cellwise under the official full-GT-span denominator. Joint increments and synergy, T remaining selection error, clean, nonexpert and order values and candidate uniqueness are preserved in anonymous rows/SUMMARY. This bounds current cached supports, not all possible temporal decoders or spatial interfaces.', '']
    for split in SPLITS:
        for ds in DATASETS:
            t=summary1[split,ds]['corruption']['expert']['T_recovery']
            lines.append(f'{split}/{ds}: T recovers {100*t["ratio_of_macro_means"]:.2f}% of existing temporal selection headroom (ratio of macro means), zero-gap cells {t["zero_selection_gap_cells"]}. Negative recovery is retained.')
    lines += ['', '## Experiment 2: event-only observations at fixed A state', '',
      'GT-event is five fixed quantiles of existing observed frames strictly within the saved half-open GT interval. Fewer than five unique event frames means unsupported; no outside padding. All unsupported cells remain in Experiment 1. Primary comparisons below use the same eligible donor subset for all four observations; full-scheduled no-op sensitivity is also published. U/U2/R masks, actual reward selection and ordinary one-step outputs are cached. Only GT-event observes new images. Temporary offsets expire after this query, while persistent A remains unchanged. HC temporary one-step results do not establish its K8 sustained learning value.', '',
      '| Split / dataset | Eligible / scheduled donors | U selected | U2 selected | R selected | GT-event selected | U update | U2 update | R update | GT-event update |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for split in SPLITS:
        for ds in DATASETS:
            z=read(PUBLIC/'experiment2'/split/ds/'SUMMARY.json');summary2[split,ds]=z
            a=z['corruption']['eligible_matched'];tot=z['corruption']['all_scheduled_with_noop_unsupported'];m=a['metrics']
            fields=[f'delta_{b}_{k}_GT' for k in ['select','temp'] for b in BRANCHES]
            lines.append(f'| {split} / {ds} | {a["cells"]} / {tot["cells"]} | '+' | '.join(estimate(m[f]) for f in fields)+' |')
    lines += ['', 'The previous table reports selected-center and one-step-center increments evaluated at GT time. Spatial output alone changes; GT time is never fed through the decoder.', '',
      '| Split / dataset / interval | GT-event−R selection | GT-event−U2 selection | GT-event−R update | GT-event−U2 update |',
      '|---|---:|---:|---:|---:|']
    for split in SPLITS:
        for ds in DATASETS:
            m=summary2[split,ds]['corruption']['eligible_matched']['metrics']
            for interval in ['GT','A']:
                fields=[f'GT_minus_{b}_{k}_{interval}' for k in ['select','temp'] for b in ['R','U2']]
                lines.append(f'| {split} / {ds} / {interval} | '+' | '.join(estimate(m[f]) for f in fields)+' |')
    lines += ['', 'Eligible corrupted donors are Vid development 75/80 (15 distinct expert sources), Vid confirmation 35/40 (7), HC development 80/80 (14), HC confirmation 40/40 (7). The 12 unsupported donors include 10 corrupt and 2 clean Vid inputs; those remain in complete-stream Experiment 1 and the published scheduled no-op sensitivity.', '',
      'Clean is a matched control, without a corruption-specific gain claim. At GT interval:', '',
      '| Split / dataset | GT-event−R selection | GT-event−U2 selection | GT-event−R update | GT-event−U2 update |',
      '|---|---:|---:|---:|---:|']
    for split in SPLITS:
        for ds in DATASETS:
            m=summary2[split,ds]['clean']['eligible_matched']['metrics']
            fields=[f'GT_minus_{b}_{k}_GT' for k in ['select','temp'] for b in ['R','U2']]
            lines.append(f'| {split} / {ds} | '+' | '.join(estimate(m[f]) for f in fields)+' |')
    lines += ['', '## Evidence and execution diagnostics', '',
      '| Split / dataset / observation | Empty requests | Valid/scorable event frames | Event GT IoU, valid-only | Event GT IoU, empty=0 observed | Better selected tube but harmful update, GT / A |',
      '|---|---:|---:|---:|---:|---:|']
    for split in SPLITS:
        for ds in DATASETS:
            z=summary2[split,ds]['corruption']['eligible_matched']
            for b in BRANCHES:
                q=z['quality'][b];e=z['execution'][b]
                def v(f):return 'NA' if q[f] is None else f'{q[f]*100:.2f}%'
                lines.append(f'| {split} / {ds} / {b} | {q["empty_requests"]}/{q["requests"]} | {q["event_valid_frames"]}/{q["event_scorable_valid_frames"]} | {v("event_box_GT_IoU_valid_weighted")} | {v("event_box_GT_IoU_observed_zero_empty")} | {e["GT"]["selected_better_but_update_harm"]} / {e["A"]["selected_better_but_update_harm"]} |')
    lines += ['', 'Evidence-quality ratios are frame-count weighted within each observation; task outcomes and pairwise contrasts use source-macro bootstrap. Valid-only IoU excludes empty masks and no-event observation positions, so it cannot be interpreted without the denominators. Empty request/frame rates, event annotation gaps, gross gains/losses, strict .3/.5 correctness changes, >5pp harms and both positive/negative cases remain in machine-readable outputs. A good selected tube is a relative gain; it is not automatically threshold-correct.', '', '## Root decision and limits', '']
    decision=read(BASE/'DECISION.json')
    lines += decision['report_paragraphs']
    resources=dict(expert=read(BASE/'EXPERT_RESOURCES.json'),updates={d:read(BASE/d/'updates/RESOURCES.json') for d in DATASETS},
        cached_U_readout={d:read(BASE/d/'cached_U_readout/RESOURCES.json') for d in DATASETS})
    writes=sum(resources['updates'][d]['suffix_replays']+resources['cached_U_readout'][d]['suffix_replays'] for d in DATASETS)
    backs=sum(resources['updates'][d]['backwards'] for d in DATASETS)
    seconds=resources['expert']['worker_wall_seconds']+sum(v['worker_wall_seconds'] for g in ['updates','cached_U_readout'] for v in resources[g].values())
    lines += ['', '## Actual incremental resources and verification', '',
      f'GT-event logical donors 288; eligible {resources["expert"]["eligible"]}; unsupported {resources["expert"]["unsupported"]}; actual new Sa2VA calls {resources["expert"]["new_calls"]}; input-matched reused requests {resources["expert"]["reused"]}. Extra specialist controls 0. Suffix replays {writes}; backwards {backs} (including Uniform bitwise gradient controls). New TA-STVG backbone calls and regenerated probes are both zero. Sum of successful model worker process wall {seconds/60:.3f} minutes, including model loading and I/O; not pure GPU kernel time or deployment end-to-end latency.', '',
      'The old confirmation U caches retained exact first-step states and gradients but omitted 96 post-update tubes. These were reconstructed with 96 suffix-only forwards from the saved states plus four bitwise pre-state prediction controls. No learning or raw GT was used in those model calls. This additional readout work is included in the resource totals; it is not a new online run. Valid development results emitted before this cache-contract repair are retained and byte-compared against the completed scoring output.', '',
      'Those U readouts were reconstructed after the first, failed CPU scoring attempt had read diagnostic GT. All U states and gradients predate that exposure, were bitwise fixed, and no U parameter/output was selected with GT. Both U barriers precede the final complete CPU scoring. Serialization and revision-order failures, as well as the missing-readout cache-contract repair, are recorded in ENGINEERING_HISTORY with original saved sources and valid partial outputs; no scientific rule changed.', '',
      'Experiment1 official scalar checks and Experiment2 reward/KL/SGD/state seals are recorded in ROOT_AUDIT. The A trajectory and CURRENT_METHOD remain unchanged. All derived anonymous rows, candidate matrices, paired uncertainty and negative cases are exported; captions/media/annotations/GT coordinates, weights and state/gradient/H tensors stay private. No new loss, Specific correction, temporal view, full stream, parameter search or production promotion is authorized by this report.', '']
    out=ROOT/'docs/TA_ORACLE_EVENT5_REVIEW.md';out.write_text('\n'.join(lines))
    (PUBLIC/'figures').mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.5),layout='constrained')
    keys=['A_v','temporal_oracle_v','spatial_oracle_v','joint_oracle_v','GT_time_v'];labels=['A','Time oracle','Space oracle','Joint oracle','GT time'];colors=['#7f8792','#6586ba','#5fa996','#9865aa','#c9864b']
    for ax,ds in zip(axes,DATASETS):
        for j,split in enumerate(SPLITS):
            m=summary1[split,ds]['corruption']['expert']['metrics'];xx=np.arange(5)+(j-.5)*.32
            vals=np.array([m[k]['mean'] for k in keys])*100;ci=np.array([m[k]['ci95'] for k in keys]).T*100
            ax.errorbar(xx,vals,yerr=np.maximum(np.vstack([vals-ci[0],ci[1]-vals]),0),fmt='o' if split=='confirm' else 's',
                color='#2e3540' if split=='confirm' else '#8493a6',capsize=3,label=split,markersize=5,linestyle='none')
        ax.set(xticks=np.arange(5),xticklabels=labels,title=ds,ylabel='Dense vIoU (%)');ax.tick_params(axis='x',rotation=18);ax.grid(axis='y',alpha=.18);ax.legend(frameon=False)
    for ext in ['png','pdf']:fig.savefig(PUBLIC/'figures'/f'oracle_ceilings.{ext}',dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10.5,6.2),layout='constrained')
    names=['select vs R','select vs U2','update vs R','update vs U2'];keys=[f'GT_minus_{b}_{k}' for k in ['select','temp'] for b in ['R','U2']]
    for col,ds in enumerate(DATASETS):
        for row,interval in enumerate(['GT','A']):
            ax=axes[row,col]
            for j,split in enumerate(SPLITS):
                m=summary2[split,ds]['corruption']['eligible_matched']['metrics'];yy=np.arange(4)+(j-.5)*.15
                vals=np.array([m[k+'_'+interval]['mean'] for k in keys])*100;ci=np.array([m[k+'_'+interval]['ci95'] for k in keys]).T*100
                ax.errorbar(vals,yy,xerr=np.maximum(np.vstack([vals-ci[0],ci[1]-vals]),0),fmt='o' if split=='confirm' else 's',
                    color='#345f96' if split=='confirm' else '#b3753b',capsize=3,label=split,markersize=5,linestyle='none')
            ax.axvline(0,color='#606a74',lw=.8);ax.set(yticks=np.arange(4),yticklabels=names,title=f'{ds}, fixed {interval} interval',xlabel='GT-event incremental vIoU (pp)')
            ax.grid(axis='x',alpha=.18);ax.invert_yaxis();ax.legend(frameon=False)
    for ext in ['png','pdf']:fig.savefig(PUBLIC/'figures'/f'event5_paired.{ext}',dpi=220)
    plt.close(fig)
    write(PUBLIC/'DECISION.json',decision);write(PUBLIC/'RESOURCES.json',resources)
    write(BASE/'REPORT_RECEIPT.json',dict(report=str(out.relative_to(ROOT)),sha256=sha(out),figures={str(p.relative_to(ROOT)):sha(p) for p in (PUBLIC/'figures').glob('*')},time=time.time()))
    print('REPORT_AND_FIGURES_WRITTEN',flush=True)
if __name__=='__main__':run()
