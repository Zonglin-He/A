"""Source-bound Q0 tables and two scalar PDF figures after actual root passes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_query_q0_common_v1 import BASE,PUB,GROUPS,read,write,sha


def run():
    assert read(BASE/'ROOT_STATISTICS_READBACK.json')['status']=='pass'
    assert read(BASE/'GLOBAL_PREDICTION_BARRIER.json')['cells']==64
    s=read(PUB/'SUMMARY.json');design=read(BASE/'DESIGN_LOCK.json')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    out=ROOT/'output/pdf/stvg_opd_query_observability_q0';out.mkdir(parents=True,exist_ok=True)
    files=[];labels=['Random Query','Observable Query','Observable LN']
    fields=['oracle_delta_v','expert_top1_delta_v','parameter_mean_delta_v']
    legends=['Whole-tube oracle','Expert reward top1','True parameter mean']
    colors=['#777777','#c57529','#2865a3']
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.8),constrained_layout=True)
    for ax,ds in zip(axes,['hc2','vidstg']):
        for i,group in enumerate(GROUPS):
            block=s['datasets'][ds]['groups'][group]['statistics']
            for j,field in enumerate(fields):
                a=block[field];lo,hi=np.array(a['ci95'])*100;y=i+(j-1)*.18
                ax.plot([lo,hi],[y,y],color=colors[j],linewidth=2)
                ax.plot(a['mean']*100,y,'o',color=colors[j],label=legends[j] if i==0 else None)
        ax.axvline(0,color='.6',linewidth=1);ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('vIoU change from common Before (percentage points)');ax.set_title(ds.upper()+' / 32 development parents');ax.legend(fontsize=8)
    fig.suptitle('Q0: candidate reachability, expert selection and mean retention',fontsize=13)
    for suffix in ['pdf','png']:
        p=out/('query_readouts.'+suffix);assert not p.exists();fig.savefig(p,dpi=180);files.append(p)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.8),constrained_layout=True)
    for ax,ds in zip(axes,['hc2','vidstg']):
        for i,group in enumerate(GROUPS):
            block=s['datasets'][ds]['groups'][group]['statistics']['parameter_mean_unobserved_decoder_delta_IoU']
            if block['mean'] is None:
                ax.text(0,i,'No scoreable decoder frames',fontsize=8)
            else:
                lo,hi=np.array(block['ci95'])*100
                ax.plot([lo,hi],[i,i],color='#2865a3',linewidth=2);ax.plot(block['mean']*100,i,'o',color='#2865a3')
        ax.axvline(0,color='.6',linewidth=1);ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('Unobserved decoder event-frame IoU change (pp)');ax.set_title(ds.upper()+' / true mean readout')
    fig.suptitle('Original Native support; decoder and interpolated frames separated',fontsize=13)
    for suffix in ['pdf','png']:
        p=out/('query_propagation.'+suffix);assert not p.exists();fig.savefig(p,dpi=180);files.append(p)
    plt.close(fig)
    lines=['# Q0 frozen Query observability diagnostic','',
        'The complete 64-query/three-group candidate, reward and true parameter-mean readouts were sealed before Q0 GT access. All samples are previously exposed development parents, one saved order1 Before state each. This is deterministic frozen-state exploration, not Gaussian Query OPD or a new online stream.', '',
        f"The one pooled output RMS budget is {design['pooled_output_RMS_budget']:.10f}, from the median of all 64 old Full first-step movements including zeros. Candidate amplitudes are bounded at 1.0 and unmet budgets remain in every table. Mean outputs are real decoder outputs at weighted latent parameters; candidate box averages are only Jensen diagnostics.", '',
        '| Dataset | Group | Whole-tube oracle Δv (pp) | Expert top1 Δv (pp) | Parameter mean Δv (pp) | Mean unobserved decoder ΔIoU (pp) | Mean harm >20pp | Attained / tested directions |',
        '|---|---|---|---|---|---|---:|---:|']
    def fmt(value):
        if value['mean'] is None:return 'undefined (0 parents)'
        a,b=value['ci95'];return f"{value['mean']*100:+.3f} [{a*100:+.3f}, {b*100:+.3f}]; n={value['parents']}"
    for ds in ['hc2','vidstg']:
        for group in GROUPS:
            g=s['datasets'][ds]['groups'][group];v=g['statistics']
            cells=[fmt(v[f]) for f in fields]+[fmt(v['parameter_mean_unobserved_decoder_delta_IoU'])]
            lines.append('| '+ds+' | '+group+' | '+' | '.join(cells)+f" | {v['parameter_mean_delta_v']['harm_20pp_parents']} | {g['attained_directions']}/{g['total_signed_directions']} |")
    lines += ['', '## Direct paired group comparisons','',
        '| Dataset | Contrast | Mean-readout Δv difference (pp) | Mean unobserved decoder difference (pp) |','|---|---|---|---|']
    for ds in ['hc2','vidstg']:
        for contrast,block in s['datasets'][ds]['paired_contrasts'].items():
            lines.append(f"| {ds} | {contrast} | {fmt(block['parameter_mean_delta_v'])} | {fmt(block['parameter_mean_unobserved_decoder_delta_IoU'])} |")
    lines += ['', 'Candidate oracle uses one whole tube chosen by vIoU, with central Before included; it never stitches GT-best frames. Expert top1 is selected before GT with central-first stable ties. The complete anonymous rows retain no-expert/equal-reward cases, common-k and spectral energy, Jensen gaps, output-budget attainment, all readouts, all observation/support partitions and both 5pp/20pp tails.', '',
        'Candidate reachability, reward identifiability and mean retention are separate claims. A small oracle restricts this finite direction set and budget; it is not an upper bound on all Query space. A positive expert top1 with a weak mean indicates projection loss. An effective Query mean without an advantage over LN does not establish Query-specific value. Wide 32-parent intervals indicate uncertainty; no arbitrary effect-size gate launches further training.', '',
        'Actual autograd observed-box Jacobians and repeat outputs were checked on GPU; CPU root independently checks saved math/scalars, rewards, state restoration and dense scores. This is not an independently reconstructed complete decoder Jacobian. This experiment does not establish long-stream benefit, object identity correctness or Gaussian-policy distillation performance. Two PDFs still require actual pixel view, anonymized publication/remote content verification, archive and true FINAL.']
    p=PUB/'REPORT.md';assert not p.exists();p.write_text('\n'.join(lines)+'\n');files.append(p)
    write(BASE/'FIGURE_EXPORT.json',dict(status='pending_actual_PDF_pixel_view_publication_archive',PDF_count=2,
        files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files},actual_PDF_view=False,task_complete=False,GT_read=True,CPU_only=True,time=time.time()))


if __name__=='__main__':run()
