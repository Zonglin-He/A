"""P1 before/after/readout comparisons and inherited-state tails."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_decota_critic_ln_common_v1 import *
import numpy as np


def pp(v):return f'{100*v:.3f}'
def estimate(m):return f"{pp(m['mean'])} [{pp(m['ci95'][0])}, {pp(m['ci95'][1])}]"


def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s=read(PUB/'SUMMARY.json');rows=read(PUB/'ROWS.json');d=read(PUB/'UPDATE_DIAGNOSTICS.json');cost=read(PUB/'COST.json')
    assert read(PUB/'ROOT_AUDIT.json')['status']==read(PUB/'PUBLIC_AUDIT.json')['status']=='pass'
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    comparisons=[('inherited_v','Before − Frozen','#2878b5'),('online_vs_episodic_v','Online − Episodic','#d97828'),
        ('online_vs_frozen_v','Online − Frozen','#428b63')]
    fig,axs=plt.subplots(1,2,figsize=(9,3.3),sharey=True)
    for ax,ds in zip(axs,DATASETS):
        for j,(key,label,color) in enumerate(comparisons):
            ms=[s[ds][sp]['corruption']['metrics'][key] for sp in ['search','confirm']]
            y=np.array([m['mean'] for m in ms])*100;ci=np.array([m['ci95'] for m in ms])*100
            ax.errorbar(np.arange(2)+(j-1)*.17,y,yerr=np.maximum(np.stack([y-ci[:,0],ci[:,1]-y]),0),fmt='o',capsize=3,color=color,label=label)
        ax.axhline(0,color='#555',lw=.7);ax.set_xticks([0,1],['Development','Confirmation']);ax.set_title('VidSTG' if ds=='vidstg' else 'HC-STVG-v2')
    axs[0].set_ylabel('Corruption ΔvIoU (pp)');axs[1].legend(frameon=False,fontsize=8);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(PUB/f'decota_critic_ln_p1_effects.{ext}',dpi=220)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,3.3))
    for ax,ds in zip(axs,DATASETS):
        for order,color in [('order1','#2878b5'),('order2','#d97828')]:
            rr=[r for r in rows if r['dataset']==ds and r['split']=='confirm' and r['condition']!='clean' and r['order']==order]
            xs=sorted({r['arrival'] for r in rr});ys=[np.mean([r['inherited_v'] for r in rr if r['arrival']==at])*100 for at in xs]
            ax.plot(xs,ys,'o-',ms=3,label=order,color=color)
        ax.axhline(0,color='#555',lw=.7);ax.set_xlabel('Arrival within the reset chain');ax.set_ylabel('Before − Frozen (pp)')
        ax.set_title('VidSTG' if ds=='vidstg' else 'HC-STVG-v2');ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(PUB/f'decota_critic_ln_p1_inheritance.{ext}',dpi=220)
    plt.close(fig)
    lines=['# Critic-DeCoTA: matched LN1/16 inheritance P1','',
        'The user explicitly authorized LN inheritance after the mixed P0 readback. This experiment freezes the P0 energy critic and adds only the original C1 LN1/16 writeback. '
        'It is a small, historically exposed research panel; production registries are unchanged.','',
        '| Corruption panel | Frozen vIoU % | Before − Frozen pp [95% CI] | Online − Episodic pp [95% CI] | Online − Frozen pp [95% CI] |',
        '|---|---:|---:|---:|---:|']
    for ds in DATASETS:
        for sp in ['search','confirm']:
            m=s[ds][sp]['corruption']['metrics'];lines.append(f"| {ds} {sp} | {pp(m['frozen_v']['mean'])} | {estimate(m['inherited_v'])} | {estimate(m['online_vs_episodic_v'])} | {estimate(m['online_vs_frozen_v'])} |")
    lines+=['','The bootstrap resamples sources, averages both orders and the five corruptions within source, and uses 10,000 paired draws, seed20261004. '
        'Each dataset has32 development/16 confirmation sources, one query/source; all are historically exposed. '
        'All1,152 online arrivals actually run. P0 episodic readouts are hash-bound reused controls, not rerun or duplicated independent observations.','',
        '## Exact state lifecycle','',
        '- At each arrival inherit only1,536 last-spatial-block norm1/norm3/norm4 affine coordinates; reset the256-dimensional query residual and construct fresh Adam.',
        '- Run the unchanged P0 critic: temperatures1/1, Adam lr.03,10 updates, earliest minimum of its own energy among steps0..10.',
        '- Emit the selected current correction; commit `LN_next = LN_arrival + (LN_selected − LN_arrival)/16`. The selected query residual never enters the next state.',
        '- Reset chains at dataset/development-or-confirmation/corruption/order boundaries. Native source I0 stays fixed; no temporal adaptation, new DINO calls, new full-backbone forward, gate or sweep.',
        '- Every query still has four cached native-I0 expert observations. InheritedBefore is measured before current-query expert correction; it is not a stream with25% expert arrivals and not a formal nonexpert endpoint.',
        '- First arrivals independently reproduce every P0 critic step bitwise. All predecessor hashes, coordinate arithmetic, resets, independent NumPy energies/Adam updates and official/vectorized dense metrics pass after the global prediction barrier.','',
        '## Clean, order effects and negative tails','',
        '| Confirmation comparison | Clean ΔvIoU pp [95% CI] | Corrupt >5/>20pp harms | Correct→wrong / wrong→correct at .3 | At .5 | Positive sources |',
        '|---|---:|---:|---:|---:|---:|']
    for ds in DATASETS:
        for name in ['inherited','online_vs_episodic','online_vs_frozen']:
            t=s[ds]['confirm']['corruption']['tails'][name];c=t['correctness']
            lines.append(f"| {ds} / {name} | {estimate(s[ds]['confirm']['clean']['metrics'][name+'_v'])} | {t['harm_gt5pp']}/{t['harm_gt20pp']} | {c['0.3']['destroyed']}/{c['0.3']['recovered']} | {c['0.5']['destroyed']}/{c['0.5']['recovered']} | {t['positive_sources']}/16 |")
    lines+=['','Harm counts are actual order-specific arrivals, correlated within source; bootstrap units remain sources. '
        'Current correction and inherited utility are separate endpoints. tIoU is exactly unchanged.','',
        '| Confirmation order | Before − Frozen pp | Online − Episodic pp | Online − Frozen pp |',
        '|---|---:|---:|---:|']
    for ds in DATASETS:
        for order in ['order1','order2']:
            m=s[ds]['confirm']['orders'][order]['metrics'];lines.append(f"| {ds}/{order} | {estimate(m['inherited_v'])} | {estimate(m['online_vs_episodic_v'])} | {estimate(m['online_vs_frozen_v'])} |")
    lines+=['','## Positive and negative inherited cases','',
        '| Dataset | Anonymous source/condition/order/arrival | Before − Frozen pp | Online − Episodic pp |',
        '|---|---|---:|---:|']
    for ds in DATASETS:
        rr=sorted([r for r in rows if r['dataset']==ds and r['split']=='confirm' and r['condition']!='clean'],key=lambda r:r['inherited_v'])
        for r in [rr[0],rr[-1]]:lines.append(f"| {ds} | {r['source_id']}/{r['condition']}/{r['order']}/{r['arrival']} | {pp(r['inherited_v'])} | {pp(r['online_vs_episodic_v'])} |")
    lines+=['','Every source, condition, order, per-step energy/GT diagnostic, gross gain/loss and correctness transition is exported, including severe negative cases. '
        'No GT-selected state or new parameter is introduced.','',
        '## Actual cost and scoped decision','',
        '| Dataset | Actual online arrivals | Spatial backward calls | Worker wall minutes | New expert/full backbone calls |',
        '|---|---:|---:|---:|---:|']
    for ds in DATASETS:
        c=cost['datasets'][ds];lines.append(f"| {ds} | 576 | {c['spatial_backward_calls']} | {c['worker_seconds']/60:.2f} | 0/0 |")
    lines+=['','Four extra cached smoke fits test source-start parity and second-arrival state inheritance before GT. '
        'Worker wall time includes cache IO, replay, parameter copying and logging; it is not pure GPU kernel time.','']
    for ds in DATASETS:
        m=s[ds]['confirm']['corruption']['metrics'];v=m['inherited_v'];a=m['online_vs_episodic_v']
        lines.append(f"- {ds}: inherited before gain {'has positive CI' if v['ci95'][0]>0 else 'has negative CI' if v['ci95'][1]<0 else 'is inconclusive'}; current online versus episodic {'has positive CI' if a['ci95'][0]>0 else 'has negative CI' if a['ci95'][1]<0 else 'is inconclusive'} on this panel.")
    lines+=['','No method is promoted. Small-panel mean differences do not establish full-data, cross-domain or future25%-expert-stream benefit. '
        'P0 Vid uncertainty and severe harms remain part of the evidence.','',
        'Artifacts: [protocol](../protocols/tastvg_decota_critic_ln_p1_v1.md), '
        '[all anonymous results](../results/tastvg_decota_critic_ln_p1/2026-10-04/), '
        '[root audit](../results/tastvg_decota_critic_ln_p1/2026-10-04/ROOT_AUDIT.json).']
    report=ROOT/'docs/TA_DECOTA_CRITIC_LN_P1_REVIEW.md';report.write_text('\n'.join(lines)+'\n')
    write(PUB/'REPORT_BINDING.json',dict(report_sha256=sha(report),figures={f.name:sha(f) for f in PUB.glob('decota_critic_ln_p1_*')},time=time.time()))
    print('P1_REPORT_READY',flush=True)


if __name__=='__main__':run()
