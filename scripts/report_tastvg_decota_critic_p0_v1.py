"""Review and exportable figures for completed fixed P0; no new experiments."""
import sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_decota_critic_common_v1 import *
import numpy as np


def pp(x):return f'{100*x:.3f}'
def estimate(m):return f"{pp(m['mean'])} [{pp(m['ci95'][0])}, {pp(m['ci95'][1])}]"


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=read(PUB/'SUMMARY.json');d=read(PUB/'UPDATE_DIAGNOSTICS.json');cost=read(PUB/'COST.json')
    decision=read(PUB/'DECISION.json'); binding=read(PUB/'PROTOCOL_BINDING.json')
    assert read(PUB/'ROOT_AUDIT.json')['status']==read(PUB/'PUBLIC_AUDIT.json')['status']=='pass'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42})
    fig,axs=plt.subplots(1,2,figsize=(8.4,3.0),sharey=True)
    for ax,ds,label in zip(axs,DATASETS,['VidSTG','HC-STVG-v2']):
        for j,(a,c) in enumerate([('direct','#2878b5'),('critic','#d97828')]):
            ms=[s[ds][sp]['corruption']['metrics'][f'{a}_delta_v'] for sp in ['search','confirm']]
            y=np.array([m['mean'] for m in ms])*100;ci=np.array([m['ci95'] for m in ms])*100
            ax.errorbar(np.arange(2)+(j-.5)*.13,y,yerr=[np.maximum(y-ci[:,0],0),np.maximum(ci[:,1]-y,0)],
                fmt='o',color=c,capsize=3,lw=1.3,label=a.capitalize())
        ax.axhline(0,color='#555',lw=.8);ax.set_xticks([0,1],['Development','Confirmation'])
        ax.set_title(label);ax.grid(axis='y',alpha=.16)
    axs[0].set_ylabel('Current spatial correction − Frozen vIoU (pp)');axs[1].legend(frameon=False)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(PUB/f'decota_critic_p0_effects.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(8.4,3.0))
    for ax,ds,label in zip(axs,DATASETS,['VidSTG','HC-STVG-v2']):
        rr=[h for h in d if h['dataset']==ds and h['split']=='confirm' and h['condition']!='clean']
        labels=['Zero initial\ngradient','Step zero\nselected','Proxy improved,\nGT harmed']
        for j,(a,c) in enumerate([('direct','#2878b5'),('critic','#d97828')]):
            ar=[h for h in rr if h['arm']==a];n=len(ar)
            y=[100*sum(h[k] for h in ar)/n for k in ['initial_gradient_zero','selected_step_zero','GT_harmed_despite_proxy_improved']]
            ax.bar(np.arange(3)+(j-.5)*.34,y,width=.34,color=c,label=a.capitalize())
        ax.set_title(label);ax.set_xticks(range(3),labels);ax.set_ylim(0,100);ax.grid(axis='y',alpha=.16)
    axs[0].set_ylabel('Unique confirmation corruption inputs (%)');axs[1].legend(frameon=False)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(PUB/f'decota_critic_p0_diagnostics.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)
    lines=['# Spatial-DeCoTA P0: original Direct versus proposal-distribution energy critic','',
        'This is the user-authorized episodic spatial qualification of the later C1–Scale06 interface. '
        'Both arms reset spatial LN/query/Adam to the same checkpoint source for each input, '
        'adapt only the original 1,792 spatial parameters, and emit the current tube with the unchanged native temporal interval. '
        'No temporal adaptation or LN inheritance is executed.','',
        '| Corruption panel | Frozen vIoU % | Direct − Frozen pp [95% CI] | Critic − Frozen pp [95% CI] | Critic − Direct pp [95% CI] |',
        '|---|---:|---:|---:|---:|']
    for ds,label in zip(DATASETS,['Vid','HC2']):
        for split in ['search','confirm']:
            m=s[ds][split]['corruption']['metrics']
            lines.append(f"| {label} {split} | {pp(m['frozen_v']['mean'])} | {estimate(m['direct_delta_v'])} | {estimate(m['critic_delta_v'])} | {estimate(m['critic_minus_direct_v'])} |")
    lines+=['','Paired source-macro differences average both orders and five corruptions within source; '
        '10,000 paired source bootstrap draws, seed 20261004. Each dataset has 32 development and 16 source-disjoint '
        'confirmation sources; all are historically exposed. These are not fresh tests or full-dataset benchmarks. '
        'There are 576 unique inputs / 1,152 logical arrivals per arm. The episodic second-order readouts are identical and reused, '
        'not additional independent measurements.','',
        '## What changed and what stayed fixed','',
        '- Direct uses the unchanged original C1 Scale06 admission, accepted top-one DINO anchors, and sum(5 L1 + 2 GIoU)/planned4.',
        '- Critic uses all nonempty geometrically valid cached supports, M≤3 on each observed frame, '
        'without the old score/margin admission. Its objective is the valid-frame mean of '
        '`−logsumexp(logsoftmax(existing_score/1) + IoU(box, proposal)/1)`. Both temperatures are prelocked to one; no sweep.',
        '- The original context path retains phrase-score NMS .5 and truncate-three-then-valid filtering. '
        'The historical fallback cached only a single selected proposal; it remains singleton, without invented candidates.',
        '- Both use original four native-interval observations, matching input hashes and checkpoints, '
        'fresh Adam .03 / (.9,.999) / eps1e-8 / wd0, ten updates, and the earliest minimum of their own objective over steps0..10.',
        '- Empty evidence is a no-op. Best-step selection never uses GT. Ordinary IoU is retained even when disjoint boxes give zero gradient.',
        '- This changes the supervision-construction block (support/admission/reduction/objective), '
        'so an effect cannot be uniquely attributed to energy or uncertainty preservation. '
        'It remains a DINO-coordinate proxy, not a demonstrated native-policy OPD mechanism or elimination of pseudo supervision.',
        '- Old C1 online spatial-after gains +4.3812/+3.7314 pp included inherited LN. '
        'They are not the episodic Direct control measured here. Current-query benefit does not establish persistent benefit.','',
        '## Empty support, zero gradients and proxy/task mismatch','',
        '| Confirmation corruption, unique inputs | Arm | Empty inputs | Zero initial gradient | Step0 selected | Proxy improved but vIoU harmed | All observed support initially disjoint |',
        '|---|---|---:|---:|---:|---:|---:|']
    for ds,label in zip(DATASETS,['Vid','HC2']):
        for arm in ARMS:
            rr=[h for h in d if h['dataset']==ds and h['split']=='confirm' and h['condition']!='clean' and h['arm']==arm]
            values=[sum(h[k] for h in rr) for k in ['skipped','initial_gradient_zero','selected_step_zero','GT_harmed_despite_proxy_improved','initial_all_evidence_disjoint']]
            lines.append(f"| {label} (n={len(rr)}) | {arm} | "+' | '.join(str(v) for v in values)+' |')
    lines+=['','A zero gradient is the evaluated proxy derivative, not proof of task correctness. '
        'No-op can protect good tubes while also preventing correction of wrong, disjoint tubes. '
        'Per-step post-hoc GT values diagnose execution, and never choose the returned step. '
        'UPDATE_DIAGNOSTICS.json retains every path and actual displacement; reduced motion alone does not prove a better objective.','',
        '## Clean and negative tails','',
        '| Confirmation | Arm | Clean vIoU change pp [95% CI] | Corruption >5 / >20 pp harms, logical | .3 correct→wrong / wrong→correct | .5 correct→wrong / wrong→correct |',
        '|---|---|---:|---:|---:|---:|']
    for ds,label in zip(DATASETS,['Vid','HC2']):
        for arm in ARMS:
            tails=s[ds]['confirm']['corruption']['tails'][arm];c=tails['correctness']
            lines.append(f"| {label} | {arm} | {estimate(s[ds]['confirm']['clean']['metrics'][f'{arm}_delta_v'])} | "
                f"{tails['harm_gt5pp']} / {tails['harm_gt20pp']} | {c['0.3']['destroyed']} / {c['0.3']['recovered']} | "
                f"{c['0.5']['destroyed']} / {c['0.5']['recovered']} |")
    lines+=['','Every condition, order, dense sIoU, correctness transition, source value and gross gain/loss is public. '
        'The harm counts include the two identical order readouts; they are not independent samples. '
        'Temporal tIoU is exactly invariant for every arm/input.','',
        '## Largest positive and negative current-query cases','',
        '| Dataset | Arm | Panel / anonymous source / corruption | Selected step | ΔvIoU pp | Proxy loss change |',
        '|---|---|---|---:|---:|---:|']
    for ds in DATASETS:
        for arm in ARMS:
            rr=[h for h in d if h['dataset']==ds and h['split']=='confirm' and h['condition']!='clean' and h['arm']==arm]
            rr=sorted(rr,key=lambda h:h['posthoc_GT_vIoU_path'][h['selected_step']]-h['posthoc_GT_vIoU_path'][0])
            for h in [rr[0],rr[-1]]:
                j=h['selected_step'];delta=h['posthoc_GT_vIoU_path'][j]-h['posthoc_GT_vIoU_path'][0]
                lines.append(f"| {ds} | {arm} | {h['split']} / {h['source_id']} / {h['condition']} | {j} | {pp(delta)} | "
                    f"{h['objective_losses'][j]-h['objective_losses'][0]:.6f} |")
    lines+=['','These examples illustrate tails; they do not establish a general causal mechanism or override paired source results.','',
        '## Actual cost and audits','',
        '| Dataset | Unique paired inputs | New DINO calls | Formal full-backbone forwards | Spatial backwards | Worker wall min |',
        '|---|---:|---:|---:|---:|---:|']
    for ds in DATASETS:
        c=cost['datasets'][ds]
        lines.append(f"| {ds} | {c['unique_inputs']} | 0 | 0 | {c['spatial_backward_calls']} | {c['worker_seconds']/60:.2f} |")
    lines+=['','Two extra full-backbone clean smoke inputs check native bitwise parity, '
        'all-step cached/full Direct and Critic trajectories, and selected-state reinsertion with unchanged native temporal logits. '
        'They are separate from formal cache-only replay. All formal payloads are sealed globally before GT scoring. '
        'Independent NumPy objectives and Adam arithmetic, every source reset, best-state selection, official/vectorized dense scoring '
        'and public summary regeneration pass. Worker wall time includes replay, host IO and logging; it is not pure GPU kernel time.','',
        '## Scoped decision','']
    for ds in DATASETS:
        lines.append(f"- {ds}: `{decision['datasets'][ds]['scoped_conclusion']}` for this fixed critic P0.")
    lines+=['','No production registry is promoted. No LN-persistence P1, cross-domain, new teacher, gate, preservation term or loss sweep '
        'is launched by this P0. A later experiment needs a concrete mechanism supported by these measured outcomes.','',
        'Artifacts: [protocol](../protocols/tastvg_decota_critic_p0_v1.md), '
        '[complete anonymous results](../results/tastvg_decota_critic_p0/2026-10-04/), '
        '[root audit](../results/tastvg_decota_critic_p0/2026-10-04/ROOT_AUDIT.json), '
        '[public audit](../results/tastvg_decota_critic_p0/2026-10-04/PUBLIC_AUDIT.json).']
    report=ROOT/'docs/TA_DECOTA_CRITIC_P0_REVIEW.md';report.write_text('\n'.join(lines)+'\n')
    write(PUB/'REPORT_BINDING.json',dict(report_sha256=sha(report),
        figures={f.name:sha(f) for f in PUB.glob('decota_critic_p0_*')},time=time.time()))
    print('REPORT_READY',report,flush=True)


if __name__=='__main__':run()
