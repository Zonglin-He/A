"""Figures and review for the fixed C1 comparator; no method selection."""
import sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_decota_c1_common_v1 import *
import numpy as np


def pp(x):return f'{100*x:.4f}'
def estimate(m):return f"{pp(m['mean'])} [{pp(m['ci95'][0])}, {pp(m['ci95'][1])}]"


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=read(PUB/'SUMMARY.json');cost=read(PUB/'COST.json');rows=read(PUB/'ROWS.json')
    assert read(PUB/'ROOT_AUDIT.json')['status']=='pass'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(8.1,2.7),sharey=True)
    conds=['clean','frame_drop_5','frame_freeze_5','motion_blur_5','occlusion_5','exposure_5']
    for ax,ds,label in zip(axs,DATASETS,['VidSTG','HC-STVG-v2']):
        ms=[summary[ds]['confirm']['conditions'][c]['metrics']['delta_v'] for c in conds]
        y=np.array([m['mean']*100 for m in ms]);ci=np.array([m['ci95'] for m in ms])*100
        ax.errorbar(np.arange(6),y,yerr=np.array([y-ci[:,0],ci[:,1]-y]),fmt='o',color='#2861a1',capsize=3,lw=1.3)
        ax.axhline(0,color='#555555',lw=.8);ax.set_xticks(np.arange(6),['Clean','Drop','Freeze','Blur','Occlude','Exposure'],rotation=27)
        ax.set_title(label);ax.grid(axis='y',alpha=.16);ax.set_xlabel('Fixed corruption condition')
    axs[0].set_ylabel('DeCoTA − Frozen vIoU (pp)');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(PUB/f'c1_same_domain_conditions.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(8.1,2.6),sharey=True)
    names=['inherited_delta_v','spatial_delta_v','temporal_delta_v','delta_v']
    for ax,ds,label in zip(axs,DATASETS,['VidSTG','HC-STVG-v2']):
        ms=[summary[ds]['confirm']['corruption']['metrics'][n] for n in names]
        y=np.array([m['mean']*100 for m in ms]);ci=np.array([m['ci95'] for m in ms])*100
        ax.errorbar(range(4),y,yerr=np.array([y-ci[:,0],ci[:,1]-y]),fmt='s',color='#32837b',capsize=3,lw=1.3)
        ax.axhline(0,color='#555555',lw=.8);ax.set_xticks(range(4),['Inherited LN','Spatial after','Temporal only','Full DeCoTA'],rotation=20)
        ax.set_title(label);ax.grid(axis='y',alpha=.16)
    axs[0].set_ylabel('Readout − Frozen vIoU (pp)');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(PUB/f'c1_same_domain_readouts.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    lines=['# C1/Scale06 online DeCoTA in the within-domain corruption setting','',
        'This evaluation runs the user-confirmed later **C1/Scale06 online** version. It compares the same official within-domain TA-STVG EMA checkpoint with Frozen, without parameter search or method promotion. Current outputs follow the original after-current-query correction timing; only spatial LN persists.', '',
        '| Dataset / panel | Frozen vIoU % | DeCoTA vIoU % | Paired change pp [95% CI] | >5 pp harm / cells |','|---|---:|---:|---:|---:|']
    for ds,label in zip(DATASETS,['Vid','HC2']):
        for split in ['search','confirm']:
            s=summary[ds][split]['corruption'];m=s['metrics']
            lines.append(f"| {label} {split} | {pp(m['frozen_v']['mean'])} | {pp(m['decota_v']['mean'])} | {estimate(m['delta_v'])} | {s['harm_gt5pp']} / {s['cells']} |")
    lines+=['','All primary rows are corruption-only. The aggregation unit is source: average both orders and five conditions within a source, then average sources; paired source bootstrap has 10,000 draws, seed 20261004. Each dataset has 32 development and 16 disjoint confirmation sources. All have historical exposure; confirmation is not fresh.','',
        '## Exact method and comparison boundary','',
        '- Spatial: query256 + LN1536, unchanged locked Scale06 Adam .03, ten updates, minimum original reference loss over steps0..10, query and Adam reset every query, selected LN displacement writes back at 1/16.','- Temporal: original full66306 head, NLL + hinge, fresh AdamW five steps and eta .25 shrink, then discarded. Vid lr .1 / center .5; HC lr .001 / center1. No new persistent temporal head is introduced.','- DINO-tiny original context/target-token reference acquisition uses up to four native-interval frames on every query. This exceeds recent A’s 25% expert-arrival budget and uses a different expert. No A-versus-DeCoTA budget-matched claim is made.','- Official same-domain VidSTG / HC-STVG-v2 checkpoints, original Paper48 pixels/sampling/two offsets. Existing Frozen H is reused with exact native and original C1 full-forward/trajectory/reinsertion checks. Scale06 was historically locked on Vid; the fixed same spatial rule is newly evaluated on HC2.','- State chains reset independently at dataset/split/condition/order boundaries. Sixteen/thirty-two-query streams are a small-panel online evaluation, not a completed full-dataset benchmark.','',
        '## Current correction versus inherited state','',
        '| Confirmation corruption | Inherited LN only pp | Spatial after pp | Temporal only pp | Full tIoU change pp | Dense sIoU change pp |','|---|---:|---:|---:|---:|---:|']
    for ds,label in zip(DATASETS,['Vid','HC2']):
        m=summary[ds]['confirm']['corruption']['metrics']
        lines.append('| '+label+' | '+' | '.join(estimate(m[n]) for n in ['inherited_delta_v','spatial_delta_v','temporal_delta_v','delta_t','delta_s'])+' |')
    lines+=['','These are matched counterfactual readouts relative to Frozen. Their vIoU changes are not additive branch contributions because temporal support changes the scored spatial frames. Inherited-LN-only is generated before this query’s update; full DeCoTA includes current spatial and episodic temporal correction.','',
        '## Clean and negative tails','',
        '| Confirmation | Clean vIoU change pp | Corruption >20 pp harms | vIoU .3 correct→wrong / wrong→correct | vIoU .5 correct→wrong / wrong→correct |','|---|---:|---:|---:|---:|']
    for ds,label in zip(DATASETS,['Vid','HC2']):
        s=summary[ds]['confirm'];c=s['corruption']['correctness'];
        lines.append(f"| {label} | {estimate(s['clean']['metrics']['delta_v'])} | {s['corruption']['harm_gt20pp']} | {c['0.3']['destroyed']} / {c['0.3']['recovered']} | {c['0.5']['destroyed']} / {c['0.5']['recovered']} |")
    lines+=['','Order-specific and every-condition tables, source values, gross gain/loss and all 1,152 anonymous metric rows are saved in SUMMARY.json and ROWS.json. UPDATE_DIAGNOSTICS.json keeps each reference-loss path, chosen step, temporal losses and state magnitude. Negative cases are retained; no thresholds or configurations are selected after looking at these results.','',
        '## Cost and checks','',
        '| Dataset | New DINO forwards | Logical observation requests | Empty reference arrivals | Spatial backward calls | Evidence wall min | Online wall min |','|---|---:|---:|---:|---:|---:|---:|']
    for ds,label in zip(DATASETS,['Vid','HC2']):
        c=cost[ds];lines.append(f"| {label} | {c['new_DINO_forward_count']} | {c['logical_observation_requests']} | {c['empty_reference_arrivals']} | {c['spatial_backwards']} | {c['evidence_worker_seconds']/60:.2f} | {c['online_worker_seconds']/60:.2f} |")
    lines+=['','Temporal adaptation is evaluated once per identical query-condition and reused across its two orders; it has no persistent state. Evidence wall time includes expert loading. Online wall time starts after STVG loading and includes the complete prediction loop, host I/O and logging; it is not pure GPU kernel time. Two extra full-backbone smoke checks verify the frozen-cache interface; formal streams reuse encoder caches.', '',
        'All prediction workers deny GT and scored outcomes. Their global barrier precedes CPU scoring. The root inspected the existing label-container/key schema before launch; those historical labels did not determine the configuration, cohort, order or selection. Independent CPU checks verify the original spatial objective, FP32 Adam displacement, query reset, exact 1/16 LN recurrence, temporal NLL/hinge and last/shrunken states, and official/vectorized dense scorer agreement. Source weights and production registries are unchanged.', '',
        'A cache-device argument omission failed the first smoke invocation before predictions. Its original worker/log were preserved under recovery/cache_device_argument_001; an explicit CUDA-device argument fixed the adapter plumbing, with no scientific configuration change. The complete paired smoke checks were then regenerated and passed.', '',
        'The scope of any positive finding is this fixed historically exposed small panel, checkpoint and after-update readout. No fresh-test, general DeCoTA, long-stream retention or automatic production-promotion claim follows.', '',
        'Artifacts: [protocol](../protocols/tastvg_decota_c1_same_domain_v1.md), [all anonymous results](../results/tastvg_decota_c1_same_domain/2026-10-04/), [root audit](../results/tastvg_decota_c1_same_domain/2026-10-04/ROOT_AUDIT.json).']
    report=ROOT/'docs/TA_DECOTA_C1_SAME_DOMAIN_REVIEW.md';report.write_text('\n'.join(lines)+'\n')
    write(PUB/'REPORT_BINDING.json',dict(report_sha256=sha(report),figures={f.name:sha(f) for f in PUB.glob('*.png')},time=time.time()))
    print('REPORT_READY',str(report),flush=True)


if __name__=='__main__':run()
