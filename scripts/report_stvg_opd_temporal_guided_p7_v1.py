"""Actual P7 phase reports and scientific figures after independent roots."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_temporal_guided_p7_common_v1 import *

def phase_A():
    verify();assert read(A/'ROOT_READBACK.json')['status']=='pass'
    s=read(OUT/'P7_A/SUMMARY.json');decision=read(A/'P7_B_BRANCH_DECISION.json')
    pub=OUT/'P7_A';dest=ROOT/'output/pdf/stvg_opd_temporal_guided_candidate/P7_A';dest.mkdir(parents=True,exist_ok=False)
    fields=['delta_T_OPD_v','interaction_OPD_Native_v','OPD_minus_Direct_T1_v','OPD_minus_Capped_Direct_T1_v']
    labels=['OPD: T1 - T0','Temporal interaction: OPD - Native','At T1: OPD - Direct','At T1: OPD - cap']
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    lines=['# P7-A: expanded fixed-roster temporal × spatial study','',
        'This study includes the original 128 historical non-development parents per target, two original orders, 512 clean cells and 4,096 sealed eight-way readouts. Complete P6 temporal records are reused for the old32 per target; the added96 per target have192 new frozen capture forwards, after8 independent qualification captures. No spatial/temporal optimizer, new DINO or trainable parameter was used in P7-A.', '',
        'Temporal Native T0 and original Actionness Projection T1 use identical original P6 recipe. Native is frozen source space; OPD, Direct and output-capped Direct share the saved original Full initial state/history. Captures are reused between orders only under exact source-state/pixels/IDs/input equality. Thus this is an exposed matched replication, not untouched confirmation or eight new independent online streams.', '',
        '| Target | Subset | Parents | Paired comparison | Mean (pp) | 95% CI (pp) | >5 pp harm parents | >20 pp harm parents |',
        '|---|---|---:|---|---:|---|---:|---:|']
    for ds in ['hc2','vidstg']:
        for subset in ['all128','previous_P6_32','added96']:
            block=s[ds][subset]['paired_both_orders']
            for f in fields:
                v=block['statistics'][f];lines.append(f"| {ds} | {subset} | {block['parents']} | {f} | {v['mean']*100:+.3f} | [{v['ci95'][0]*100:+.3f}, {v['ci95'][1]*100:+.3f}] | {v['harm_5pp']} | {v['harm_20pp']} |")
    lines+=['','Full Native/OPD/Direct/cap T0/T1 mvIoU, mtIoU, fixed-event msIoU and strict R@.3/.5, each order, complete parent vectors/tails, gained/lost support and event-outside expansion remain in SUMMARY and all anonymous rows. Temporal-only changes leave fixed-GT-event spatial IoU unchanged. Missing decoder support is zero; no extrapolation or GT spatial filling.', '',
        'The original dataset and source roster is fixed. Both orders are averaged within each parent before 10,000 paired bootstrap resamples with seed20261012; intervals are unadjusted for multiple comparisons. Old32 and added96 are displayed separately. Differences that favor Direct or output-capped Direct are not omitted.', '',
        '## Locked resource branch to one P7-B', '',
        'Eligibility requires added96 and all128 OPD T1−T0 means to be positive in both target datasets, after complete independent A root. This was locked before new GT. The CI is reported as uncertainty, rather than fitted into another gate. A positive mean with an interval spanning zero only licenses a bounded mechanism pilot, not a confirmed new method.', '',
        f"Actual post-root branch: **P7_B_eligible={decision['P7_B_eligible']}**. No parent, interval, learning rate, reward, temperature or observation frame is chosen using target GT. B has one fixed full512-cell four-arm experiment; A/B exact old fits and C/D new fits share original Full Before. Production registration and full official-query expansion remain unperformed."]
    (pub/'REPORT.md').write_text('\n'.join(lines)+'\n')
    colors=['#777777','#25709e','#d16a2e'];subsets=['previous_P6_32','added96','all128']
    with PdfPages(dest/'expanded_factorial_contrasts.pdf') as pdf:
        fig,axes=plt.subplots(1,2,figsize=(12.4,6.2),constrained_layout=True)
        for ax,ds in zip(axes,['hc2','vidstg']):
            for j,subset in enumerate(subsets):
                block=s[ds][subset]['paired_both_orders']['statistics']
                for i,f in enumerate(fields):
                    v=block[f];lo,hi=np.array(v['ci95'])*100;mean=v['mean']*100;y=i+(j-1)*.18
                    ax.plot([lo,hi],[y,y],color=colors[j],lw=2);ax.plot(mean,y,'o',color=colors[j],label=subset if i==0 else None)
            ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.axvline(0,color='.5',lw=1);ax.grid(axis='x',alpha=.2);ax.legend(fontsize=8);ax.set_xlabel('Paired vIoU difference (pp), 95% CI');ax.set_title('HC-STVG-v2' if ds=='hc2' else 'VidSTG')
        fig.suptitle('P7-A - original128 parents per target, old32 and added96 separated',fontsize=13);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12.4,6.2),constrained_layout=True)
        for ax,ds in zip(axes,['hc2','vidstg']):
            m=s[ds]['added96']['paired_both_orders']['statistics'];x=np.arange(4)
            for offset,t,color in [(-.1,'T0','#6f7d8b'),(.1,'T1','#207c82')]:
                y=np.array([m[f'{a}_{t}_v']['mean'] for a in ['Native','OPD','Direct','Capped_Direct']])*100
                ci=np.array([m[f'{a}_{t}_v']['ci95'] for a in ['Native','OPD','Direct','Capped_Direct']])*100
                ax.errorbar(x+offset,y,yerr=[y-ci[:,0],ci[:,1]-y],fmt='o',capsize=3,color=color,label=t)
            ax.set_xticks(x,['Native','OPD','Direct','Capped Direct']);ax.set_ylabel('Mean vIoU (%)');ax.set_title(('HC' if ds=='hc2' else 'Vid')+' - added96 parents');ax.legend();ax.grid(axis='y',alpha=.2)
        fig.suptitle('Primary expanded-support group: all eight fixed readouts',fontsize=13);pdf.savefig(fig);plt.close(fig)
    with PdfPages(dest/'coverage_and_negative_tails.pdf') as pdf:
        fig,axes=plt.subplots(1,2,figsize=(12.4,5.8),constrained_layout=True)
        for ax,ds in zip(axes,['hc2','vidstg']):
            for j,subset in enumerate(subsets):
                v=s[ds][subset]['paired_both_orders']['statistics']['delta_T_OPD_v'];y=np.sort(np.array(v['parent_values'])*100)
                ax.plot(np.arange(len(y)),y,'.-',ms=3,lw=.8,color=colors[j],label=subset)
            ax.axhline(0,color='.5',lw=1);ax.axhline(-5,color='#b44936',ls='--');ax.axhline(-20,color='#b44936',ls=':');ax.set_xlabel('Parent rank within each subset');ax.set_ylabel('OPD T1 - T0 vIoU (pp)');ax.set_title('HC' if ds=='hc2' else 'Vid');ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.suptitle('All temporal gains and severe negative parents; no case filtering',fontsize=13);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12.4,5.8),constrained_layout=True)
        for ax,ds in zip(axes,['hc2','vidstg']):
            for j,subset in enumerate(subsets):
                m=s[ds][subset]['paired_both_orders']['statistics']
                for i,f in enumerate(['delta_event_coverage','gained_event_frames','lost_event_frames']):
                    v=m[f];scale=100 if i==0 else 1;y=i+(j-1)*.18;lo,hi=np.array(v['ci95'])*scale
                    ax.plot([lo,hi],[y,y],color=colors[j],lw=2);ax.plot(v['mean']*scale,y,'o',color=colors[j],label=subset if i==0 else None)
            ax.set_yticks(range(3),['Event coverage change (pp)','Added event frames per parent','Lost event frames per parent']);ax.invert_yaxis();ax.set_title('HC' if ds=='hc2' else 'Vid');ax.legend(fontsize=8);ax.grid(axis='x',alpha=.2)
        fig.suptitle('Coverage and support are reported separately from spatial localization',fontsize=13);pdf.savefig(fig);plt.close(fig)
    write(A/'FIGURE_EXPORT.json',dict(status='pending_actual_pixel_view_publication_archive',PDFs=2,pages=4,
        files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in dest.glob('*.pdf')},CPU_only=True,GT_read=True,time=time.time()))

def phase_B():
    from scripts.score_stvg_opd_temporal_capture_p7a_v1 import rows_read
    assert read(B/'ROOT_READBACK.json')['status']=='pass';s=read(OUT/'P7_B/SUMMARY.json');d=read(B/'CANDIDATE_MECHANISM_DECISION.json')
    pub=OUT/'P7_B';dest=ROOT/'output/pdf/stvg_opd_temporal_guided_candidate/P7_B';dest.mkdir(parents=True,exist_ok=False)
    lines=['# P7-B: temporal support guides sparse spatial observations','',
        'All512 clean cells reuse the same original Full initial state in each paired comparison. A uses old OPD boxes/time I0/observations I0; B uses those exact same boxes with projected time I1. C uses unchanged OPD and observations reallocated into I1; D uses unchanged Direct L1/GIoU5/2 in I1. All1,024 complete C/D fits globally sealed before new B scoring. No learning rate, reward, temperature, roster, new gate or teacher was selected from B results.', '',
        'The 256 unique four-observation evidence packs are shared across two orders and C/D. Actual paid DINO calls and planned/admitted positions are reported from the real evidence receipts; no claim that sharing a cached benchmark pack eliminates a candidate deployment expert cost. TA-STVG and DINO were loaded in separate GPU processes; the original baseline PIDs/optimizer/prefix were preserved and resumed after the GPU phase.', '',
        '| Target | Subset | Paired contrast | Valid parents | Mean (pp) | 95% CI (pp) | >5 pp harm parents | >20 pp harm parents |',
        '|---|---|---|---:|---:|---|---:|---:|']
    fields=['coupling_C_minus_B_v','D_minus_B_v','C_minus_D_v','C_minus_B_common_unobserved_real_IoU','C_minus_B_gained_event_unobserved_real_IoU']
    for ds in ['hc2','vidstg']:
        for subset in ['all128','previous_P6_32','added96']:
            for f in fields:
                v=s[ds][subset]['both']['statistics'][f]
                mean='undefined' if v['mean'] is None else f"{v['mean']*100:+.3f}";ci='undefined' if v['ci95'] is None else f"[{v['ci95'][0]*100:+.3f}, {v['ci95'][1]*100:+.3f}]"
                lines.append(f"| {ds} | {subset} | {f} | {v['parents']} | {mean} | {ci} | {v.get('harm_5pp','-')} | {v.get('harm_20pp','-')} |")
    eb=read(B/'EXPERT_EVIDENCE_BARRIER.json')
    lines+=['',f"Actual new DINO calls: **{eb['actual_new_DINO_calls']}**, unique evidence packs256, planned observations{eb['planned_observation_positions']}. Qualification used16 complete repeated fits; formal used1,024 new complete fits, A/B reuse recorded separately.",'',
        'Common unobserved real frames exclude the union of both arms\' actually visited DINO frames; they are genuine original decoder frames within the GT event, not interpolation-only evidence. Empty partitions remain undefined with counts. New/lost event support, event-outside visits, admitted reference quality and update geometry are strictly offline diagnostics; they never choose observations or gates.', '',
        f"Added96 mechanism decision: **both-target positive coupling plus unobserved-real improvement supported={d['misallocation_hypothesis_supported_in_both_added96']}**. Even when useful, this is one-way temporal evidence -> observation allocation -> spatial update, not bidirectional joint optimization or temporal parameter TTA. Direct superiority is retained. Conditional original Full history is not an independently evolving online candidate stream. No production promotion or full official-query expansion is performed.", '',
        'All four task readouts, R@.3/.5, all128/old32/new96, each order, entire positive/negative parent vectors, zero/undefined partitions and actual resource costs are available in the complete anonymous scalar files. Confidence intervals use10,000 paired parent resamples, seed20261012, unadjusted for multiplicity; all source cohorts have historical evaluation exposure.']
    (pub/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    labels=['C - B: observation coupling','D - B: direct allocation','C - D: OPD versus Direct'];colors=['#777777','#25709e','#d16a2e'];subsets=['previous_P6_32','added96','all128']
    fig,axes=plt.subplots(1,2,figsize=(12.4,5.8),constrained_layout=True)
    for ax,ds in zip(axes,['hc2','vidstg']):
        for j,subset in enumerate(subsets):
            for i,f in enumerate(fields[:3]):
                v=s[ds][subset]['both']['statistics'][f];y=i+(j-1)*.18;lo,hi=np.array(v['ci95'])*100
                ax.plot([lo,hi],[y,y],color=colors[j],lw=2);ax.plot(v['mean']*100,y,'o',color=colors[j],label=subset if i==0 else None)
        ax.set_yticks(range(3),labels);ax.invert_yaxis();ax.axvline(0,color='.5',lw=1);ax.set_xlabel('Paired vIoU difference (pp), 95% CI');ax.set_title('HC' if ds=='hc2' else 'Vid');ax.legend(fontsize=8);ax.grid(axis='x',alpha=.2)
    fig.suptitle('P7-B - same output time, state, budget and frozen hyperparameters',fontsize=13);fig.savefig(dest/'observation_coupling.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12.4,5.8),constrained_layout=True)
    for ax,ds in zip(axes,['hc2','vidstg']):
        for j,subset in enumerate(subsets):
            v=s[ds][subset]['both']['statistics']['coupling_C_minus_B_v'];y=np.sort(np.array(v['parent_values'])*100)
            ax.plot(range(len(y)),y,'.-',ms=3,lw=.8,color=colors[j],label=subset)
        ax.axhline(0,color='.5',lw=1);ax.axhline(-5,color='#b44936',ls='--');ax.axhline(-20,color='#b44936',ls=':');ax.set_xlabel('Parent rank within subset');ax.set_ylabel('C - B vIoU (pp)');ax.set_title('HC' if ds=='hc2' else 'Vid');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Complete coupling gains and severe negative tails',fontsize=13);fig.savefig(dest/'coupling_negative_tails.pdf');plt.close(fig)
    write(B/'FIGURE_EXPORT.json',dict(status='pending_actual_pixel_view_and_case_diagnosis_publication_archive',PDFs=2,pages=2,
        files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in dest.glob('*.pdf')},CPU_only=True,GT_read=True,time=time.time()))

if __name__=='__main__':{'A':phase_A,'B':phase_B}[sys.argv[1]]();print('P7_PHASE_REPORT_AND_PDFS_WRITTEN',sys.argv[1])
