"""Tables and exportable charts from verified, paired fixed-box outcomes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.tastvg_temporal_coverage_common_v1 import *
import numpy as np

def estimate(m):return f'{100*m["mean"]:+.4f} [{100*m["ci95"][0]:+.4f}, {100*m["ci95"][1]:+.4f}]'
def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assert read(BASE/'FINAL_ROOT_AUDIT.json')['status']=='pass';verify(labels=True)
    decision=read(BASE/'DECISION.json')
    lines=['# Fixed-budget temporal candidate coverage and grid oracle','',
        'One replacement allocation completed on both datasets and both existing panels. Eight intervals per expert arrival, the original temporal scorer, fixed A boxes and persistent Uniform updates. This separates an unlabelled readout contrast from privileged candidate/grid oracle diagnostics. Values are dense vIoU percent or percentage points as labelled, with 95% paired source-bootstrap intervals (10,000 draws, seed20261003).','',
        '## Matched setting','',
        'Each dataset retains the same32 development and16 within-batch disjoint confirmation sources, one query/source, two orders, clean plus five5% transient corruptions and25% scheduled expert arrivals. All sources have historical exposure, including preceding mechanism diagnostics. Confirmation is descriptive, with no result-based generator selection. Total1,152 arrivals;288 expert candidate comparisons. Same-domain official TA-STVG checkpoints and original Paper48 observed pixels/grid. A means the existing Uniform persistent spatial learner plus the original single-view temporal Fast, not Frozen.',
        '', 'Vid K1/lr .033761698432507946/teacher .34902548789596055; HC K8/lr .006097133675874025/teacher1; rho .05/student1/D4/1792. All actual A pre/post states and full spatial boxes are unchanged. No current correction, spatial loss, temporal view or new specialist observation is added.',
        '', '## The one predefined replacement pool','',
        'Slot0 retains current native. Six slots cross early/middle/late center with short/medium duration; the final slot is long at any center. Fixed thirds divide positions and lengths on the physical observed window. Within each slot, frozen source-head endpoint products select the best unused legal interval; exact ties use start then end index. Start/end probabilities are separately normalized within each original offset, then interleaved with equal offset mass. This is an allocation prior, not a calibrated likelihood of the native two-offset envelope. It is independent of UniversalVTG scoring. The preserved native is checked against the source cache rather than substituted.',
        '', 'Both old and new pools use the literal original max(confidence*interval-IoU) critic, its same cached first-view proposals/confidences, and native-first np.argmax ties. New intervals are student-grid readouts, not specialist proposal coordinates. The previous two-view min rule is not used. All new pools/selected readouts across both datasets and both panels were globally sealed before new GT scoring. No new expert, backbone, suffix or backward calls occurred.',
        '', '## Actual readout: complete corruption streams','',
        '| Panel / dataset | Cells / sources | A old8 (%) | New8 actual (%) | New−A vIoU (pp) | New−A tIoU (pp) |',
        '|---|---:|---:|---:|---:|---:|']
    summaries={}
    for split in SPLITS:
        for ds in DATASETS:
            z=read(PUBLIC/split/ds/'SUMMARY.json');summaries[split,ds]=z;a=z['corruption']['all'];m=a['metrics']
            lines.append(f'| {split}/{ds} | {a["cells"]}/{a["sources"]} | '+' | '.join(estimate(m[k]) for k in ['A_v','new_v','actual_gain','actual_t_gain'])+' |')
    lines+=['', 'Nonexpert output is exactly A at all864 positions; full-flow increments are computed from complete source/condition/order rows, not from an assumed25% multiplier. No persistent state is changed, so this measures current temporal readout only, not future parameter transfer.',
        '', '## Oracle opportunity versus realized selection: corruption experts','',
        '| Panel / dataset | Cells / sources | Old8 oracle (%) | New8 oracle (%) | New−old oracle (pp) | Actual new−old (pp) | New selection gap (pp) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for split in SPLITS:
        for ds in DATASETS:
            a=summaries[split,ds]['corruption']['expert'];m=a['metrics']
            lines.append(f'| {split}/{ds} | {a["cells"]}/{a["sources"]} | '+' | '.join(estimate(m[k]) for k in ['old_oracle_v','new_oracle_v','oracle_gain','actual_gain','new_selection_gap'])+' |')
    lines+=['', 'The oracle selects with GT only after sealing and is not an implementable result. Both pools retain native but the new pool does not contain the entire old pool, so its per-cell oracle can decrease; all negative changes are preserved.',
        '', '## Where the old coverage gap resides','',
        '| Panel / dataset | GT-time−old8 oracle (pp) | Grid−old8 oracle (pp) | GT-time−grid oracle (pp) | Grid−new8 oracle (pp) |',
        '|---|---:|---:|---:|---:|']
    for split in SPLITS:
        for ds in DATASETS:
            m=summaries[split,ds]['corruption']['expert']['metrics']
            lines.append(f'| {split}/{ds} | '+' | '.join(estimate(m[k]) for k in ['old_coverage_gap','grid_minus_old_oracle','GT_time_minus_grid','grid_minus_new_oracle'])+' |')
    lines+=['', 'Cellwise GT-time−old8=(grid−old8)+(GT-time−grid), with the same identity for new8. Grid enumerates every i<j endpoint pair on the original MERGED observed frame grid, using [frame_i,frame_j+1). It allows cross-offset endpoints that the original two-offset envelope MAP need not visit. It does not invent unseen intermediate endpoints, change sampling or send GT time into the decoder. Official dense interpolation, no spatial extrapolation and full annotated GT span in the denominator remain literal.',
        '', '| Panel / dataset | Full-flow grid oracle (%) | Full-flow GT-time (%) | GT-time−grid (pp) |',
        '|---|---:|---:|---:|']
    for split in SPLITS:
        for ds in DATASETS:
            m=summaries[split,ds]['corruption']['all']['metrics']
            lines.append(f'| {split}/{ds} | '+' | '.join(estimate(m[k]) for k in ['grid_oracle_v','GT_time_v','GT_time_minus_grid'])+' |')
    diagnostic=read(PUBLIC/'SUPPORT_AND_SELECTOR_DIAGNOSIS.json')
    lines+=['', '## Candidate eviction versus ranking damage','',
        '| Panel / dataset | Harmed corrupt expert cells | All new candidates worse than old A | A-quality candidate exists but scorer harms | Old A interval still in new pool | Higher critic score yet harms |',
        '|---|---:|---:|---:|---:|---:|']
    for split in SPLITS:
        for ds in DATASETS:
            d=diagnostic[f'{split}/{ds}/corruption']
            lines.append(f'| {split}/{ds} | {d["harmed"]} | {d["new_pool_forced_harm"]} | {d["preservable_harm"]} | {d["old_A_in_new"]}/{d["cells"]} | {d["higher_critic_score_harm"]} |')
    lines+=['', 'Retaining central native does NOT retain the old critic-selected A interval. A forced-harm cell has new-pool oracle below old A, so no scorer over that new pool can preserve old performance there. A preservable-harm cell still has an A-quality candidate but the critic chooses worse. These are distinct limitations; negative actual results cannot be assigned entirely to the scorer. Counts are per expert arrival, not independent sources.', '']
    lines+=['', '## Clean, ordering and negative outcomes','',
        '| Panel / dataset | Clean all actual gain (pp) | Corrupt expert order1 / order2 gain (pp) | Improved / harmed / unchanged expert cells | >5pp harm |',
        '|---|---:|---:|---:|---:|']
    for split in SPLITS:
        for ds in DATASETS:
            z=summaries[split,ds];e=z['corruption']['expert'];c=e['counts'];m=e['metrics']['actual_gain']
            lines.append(f'| {split}/{ds} | {estimate(z["clean"]["all"]["metrics"]["actual_gain"])} | '+
                '/'.join(f'{100*v:+.4f}' for v in m['order_values'])+f' | {c["improved"]}/{c["harmed"]}/{c["unchanged"]} | {c["severe_harm_gt5pp"]} |')
    lines+=['', 'Gross gain/loss, strict .3/.5 correctness transitions, all clean/expert/nonexpert means, cell means, source influence, leave-one-source-out ranges and positive/negative anonymous cases are retained in ROWS/SUMMARY/CASES. Expert subsets have only7–16 distinct sources and differing membership across orders. Many correlated diagnostics are reported without a multiplicity-adjusted significance claim.',
        '', '## Root interpretation and next variable','']+decision['paragraphs']
    gen=read(BASE/'GENERATION_RESOURCES.json');score=read(BASE/'SCORE_ROOT_CHECKS.json');root=read(BASE/'FINAL_ROOT_AUDIT.json')
    lines+=['', '## Actual incremental cost and audit','',
        f'Generation CPU process wall {gen["worker_wall_seconds"]:.3f}s; scoring wall {score["worker_wall_seconds"]:.3f}s; independent root audit wall {root["worker_wall_seconds"]:.3f}s. These include loading/I/O and are not GPU kernel time or deployment latency. New model/expert/backbone/suffix/backward calls all0. Exhaustive grid pairs scored: {score["checks"]["enumerated_grid_intervals"]:,}. Literal official scalar checks: {score["checks"]["official_scalar_checks"]:,}; maximum dense discrepancy {score["max_dense_error"]:.3g}.',
        '', f'All288 current native intervals match cached source native; every old critic score/choice reproduces exactly; every new pool has8 distinct intervals,0stratum fallback slots. {root["private_input_hashes_verified"]} immutable input files are verified. Both historical A scalar outcomes and old eight-candidate oracle reproduce the preceding report. New global seal precedes GT exposure and all metric computation. Startup import-path failure before preparation was repaired and preserved; no prediction, label read or scientific rule changed in that repair.',
        '', 'Only this fixed allocation was tested. A large grid oracle is an interface-capacity result, not a proof that an unlabelled generator or critic can find the best interval. Temporal headroom is conditional on fixed A boxes, cannot be added to the spatial conditional upper bound, and does not establish temporal as the only overall bottleneck. The production registry and every old paused queue remain unchanged. Code, protocol, all anonymous outcomes and figures are public-exported; media/captions/annotations/GT coordinates/weights/raw tensor caches remain excluded.', '']
    report=ROOT/'docs/TA_TEMPORAL_CANDIDATE_COVERAGE_REVIEW.md';report.write_text('\n'.join(lines))
    write(PUBLIC/'DECISION.json',decision)
    figures=PUBLIC/'figures';figures.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    names={'vidstg':'VidSTG','hc2':'HC-STVG-v2'}
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.4),layout='constrained')
    labels=['Old8 actual','New8 actual','Old8 oracle','New8 oracle','Grid oracle','GT time']
    fields=['A_v','new_v','old_oracle_v','new_oracle_v','grid_oracle_v','GT_time_v']
    for ax,ds in zip(axes,DATASETS):
        for j,split in enumerate(SPLITS):
            m=summaries[split,ds]['corruption']['expert']['metrics'];x=np.arange(6)+(j-.5)*.22
            v=np.array([100*m[k]['mean'] for k in fields]);ci=np.array([m[k]['ci95'] for k in fields]).T*100
            ax.errorbar(x,v,yerr=np.maximum(np.vstack([v-ci[0],ci[1]-v]),0),fmt='o' if split=='confirm' else 's',
                color='#2e6599' if split=='confirm' else '#9b7754',capsize=3,label='Confirmation' if split=='confirm' else 'Development')
        ax.set(title=names[ds],xticks=np.arange(6),xticklabels=labels,ylabel='Dense vIoU (%)')
        ax.tick_params(axis='x',rotation=24);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=8)
    for ext in ['png','pdf','svg']:fig.savefig(figures/f'coverage_and_selection.{ext}',dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.2,3.0),layout='constrained')
    fields=['oracle_gain','actual_gain','grid_minus_old_oracle','GT_time_minus_grid']
    labels=['New8 oracle − old8','New actual − old','Grid − old8 oracle','GT time − grid']
    for ax,ds in zip(axes,DATASETS):
        for j,split in enumerate(SPLITS):
            m=summaries[split,ds]['corruption']['expert']['metrics'];y=np.arange(4)+(j-.5)*.17
            v=np.array([100*m[k]['mean'] for k in fields]);ci=np.array([m[k]['ci95'] for k in fields]).T*100
            ax.errorbar(v,y,xerr=np.maximum(np.vstack([v-ci[0],ci[1]-v]),0),fmt='o' if split=='confirm' else 's',
                color='#2e6599' if split=='confirm' else '#9b7754',capsize=3,label='Confirmation' if split=='confirm' else 'Development')
        ax.axvline(0,color='#5d626a',lw=.8);ax.set(title=names[ds],yticks=np.arange(4),yticklabels=labels,xlabel='Paired vIoU difference (pp)')
        ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend(frameon=False,fontsize=8)
    for ext in ['png','pdf','svg']:fig.savefig(figures/f'paired_coverage_gaps.{ext}',dpi=220)
    plt.close(fig)
    write(BASE/'REPORT_RECEIPT.json',dict(report=str(report.relative_to(ROOT)),sha256=sha(report),
        figures={str(f.relative_to(ROOT)):sha(f) for f in figures.glob('*')},time=time.time()))
    print('REPORT_AND_FIGURES_WRITTEN',flush=True)
if __name__=='__main__':run()
