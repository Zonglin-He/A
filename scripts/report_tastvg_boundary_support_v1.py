"""Write complete matched-support results and a source-backed scientific figure."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.tastvg_boundary_support_common_v1 import *

def run():
    verify();assert read(BASE/'FINAL_ROOT_AUDIT.json')['status']=='pass'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    panels=[('vidstg','search'),('hc2','search'),('vidstg','confirm'),('hc2','confirm')]
    sums={(ds,sp):read(PUBLIC/sp/ds/'SUMMARY.json') for ds,sp in panels}
    def fmt(m,ci=False):
        s=f"{100*m['mean']:+.4f}"
        return s+f" [{100*m['ci95'][0]:+.4f}, {100*m['ci95'][1]:+.4f}]" if ci else s
    def met(ds,sp,sub,f,group='corruption'):return sums[ds,sp][group][sub]['metrics'][f]
    capacity=[met(ds,'confirm','expert','capacity_gain') for ds in DATASETS]
    gain=[met(ds,'confirm','all','D32_vs_A32_v') for ds in DATASETS]
    decision=dict(status='completed_root_verified',measurement='valid_in_locked_scope',
        capacity_evidence='positive_paired_CI_both_confirmation' if all(m['ci95'][0]>0 for m in capacity) else 'not_positive_CI_both_confirmation',
        boundary_evidence='positive_paired_CI_both_confirmation' if all(m['ci95'][0]>0 for m in gain) else
            'positive_mean_both_confirmation_only' if all(m['mean']>0 for m in gain) else 'not_positive_mean_both_confirmation',
        original_A_retained=True,Expanded32_integrated=False,D_integrated=False,C_removed=True,
        CURRENT_METHOD_changed=False,confirmation_used_to_reselect=False,next_experiment_started=False,
        limits='one_deterministic_capacity_allocation_and_one_1second_PE_transition_proxy_not_all_boundary_signals',time=time.time())
    write(BASE/'DECISION.json',decision);write(PUBLIC/'DECISION.json',decision)
    lines=['# Temporal coverage and boundary-quality review','',
        'The 2-support x (oracle + A/B/D) experiment is complete and independently audited. '
        'Every Old8 entry and its original A selection is retained in Expanded32. Spatial C is removed. '
        'Original experimental A and the deployed CURRENT_METHOD remain unchanged.','',
        '## Configuration','',
        'VidSTG and HC-STVG-v2 each retain the same 32 development +16 confirmation sources, one query/source, '
        'two orders, clean + five fixed 5% transient corruptions, 25% scheduled experts. '
        'All 1,152 readouts are covered; 288 expert positions have both supports and six selectors '
        '(240 corruption/48 clean). The 864 nonexpert positions are exactly A. All sources have historical '
        'exposure; confirmation is source-disjoint within this batch, previously diagnosed, and not fresh.','',
        'The reference is **A8**, the saved experimental A with Uniform5 persistent spatial Rank-RKL and '
        'original UVTG Fast, not Frozen. Full boxes, 1,792 parameters, Vid K1/HC K8, prior sealed learning '
        'rates/temperatures, pixels, expert schedule, sampling and checkpoints do not change. No backbone '
        'execution, backward, new expert, new temporal view or learned quality head is used.','',
        'Expanded32 preserves the eight original entries in their order and appends 24 deterministic maximin '
        'intervals in normalized physical (start,end) coordinates from all existing merged-grid i<j pairs. '
        'This is a capacity control, not a trained candidate generator. UVTG A, previous semantic contrast B '
        '(outer ratio .25), and new D share each identical support. D uses min(start rise,end fall), '
        'w=1 second prelocked, fractional integration, inner clipping to the interval, outer clipping to '
        'observed support, and neutral zero for a missing exterior side. Missing sides cannot count as '
        'positive transitions. Flat/unavailable curves retain A on the same support; other ties are '
        'native first at 1e-12. No width tuning, positive gate, confidence multiplier, C or B+D.','',
        '[BAM-DETR](https://arxiv.org/html/2312.00083v2) separates matching from localization quality with '
        'an IoU-supervised quality head. This experiment tests an **untrained PE transition proxy**, not '
        'that head and not an IoU estimator.','',
        '## Capacity on corruption expert arrivals','',
        'Values are vIoU percentage points. Oracle changes only offline temporal readout of fixed A boxes.','',
        '| Dataset/panel | Expert cells/sources | O8 vIoU % | O32 vIoU % | O32-O8 pp [paired 95% CI] |',
        '|---|---:|---:|---:|---:|']
    for ds,sp in panels:
        z=sums[ds,sp]['corruption']['expert'];m=z['metrics']
        lines.append(f"| {ds}/{sp} | {z['cells']}/{z['sources']} | {100*m['O8_v']['mean']:.4f} | {100*m['O32_v']['mean']:.4f} | {fmt(m['capacity_gain'],True)} |")
    lines+=['','## Actual complete corruption flow relative to A8','',
        'This includes scheduled and nonscheduled arrivals; do not scale expert means by .25. '
        'Values below are paired delta-vIoU pp with 95% source-bootstrap intervals.','',
        '| Dataset/panel | A8 vIoU % | B8-A8 | D8-A8 | A32-A8 | B32-A8 | D32-A8 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for ds,sp in panels:
        m=sums[ds,sp]['corruption']['all']['metrics']
        lines.append(f"| {ds}/{sp} | {100*m['A8_v']['mean']:.4f} | "+' | '.join(fmt(m[f'{a}_gain'],True) for a in ['B8','D8','A32','B32','D32'])+' |')
    lines+=['','## Same-support actual comparison','',
        '| Dataset/panel | B32-A32 vIoU pp [CI] | D32-A32 vIoU pp [CI] | D32-A32 tIoU pp [CI] |',
        '|---|---:|---:|---:|']
    for ds,sp in panels:
        m=sums[ds,sp]['corruption']['all']['metrics']
        lines.append(f"| {ds}/{sp} | {fmt(m['B32_vs_A32_v'],True)} | {fmt(m['D32_vs_A32_v'],True)} | {fmt(m['D32_vs_A32_t'],True)} |")
    lines+=['','## Oracle regret, ordering and harms','',
        'All following rows are corruption expert arrivals. Regret is oracle-vIoU minus selected-vIoU '
        'on the same support; it is nonnegative. Pairwise ordering compares every strictly unequal GT '
        'candidate pair, gives score ties half credit, excludes cells with no strict pairs, and uses '
        'source aggregation. vIoU ordering and tIoU ordering are distinct diagnostics. Severe harm is '
        'an arrival delta below -5 pp relative to A8; destroyed Fast gains count old A8>native followed '
        'by a lower readout.','',
        '| Dataset/panel | Arm | vIoU % | tIoU % | regret pp | pair-v % | pair-t % | better/harm/same vs A8 | harm>5pp | old-positive Fast destroyed |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds,sp in panels:
        z=sums[ds,sp]['corruption']['expert'];m=z['metrics']
        for arm in ARMS:
            n=8 if arm in ['A8','B8','D8'] else 32;cnt=z['arms'][arm]
            pv=z['pairwise'][f'{n}_v']['metrics'][f'{arm}_pair_v']['mean']
            pt=z['pairwise'][f'{n}_t']['metrics'][f'{arm}_pair_t']['mean']
            lines.append(f"| {ds}/{sp} | {arm} | {100*m[f'{arm}_v']['mean']:.4f} | {100*m[f'{arm}_t']['mean']:.4f} | "
                f"{100*m[f'{arm}_regret']['mean']:.4f} | {100*pv:.2f} | {100*pt:.2f} | "
                f"{cnt['improved']}/{cnt['harmed']}/{cnt['unchanged']} | {cnt['severe_harm_gt5pp']} | "
                f"{cnt['old_positive_fast_destroyed']}/{cnt['old_positive_fast']} |")
    lines+=['','A lower regret and a higher selected vIoU on identical support are the same paired difference; '
        'they are not two independent pieces of evidence. Gross gains/losses, .3/.5 correctness, paired '
        'regret CI, both order values, leave-one-source-out ranges, all clean and nonexpert rows are in '
        'the complete anonymous ROWS/SUMMARY files.','',
        '## Boundary support and clean control','',
        '| Dataset/panel | D32 missing exterior candidates / mean32 | D32 chosen both-positive fraction | Clean D32-A8 pp [CI] | Clean D32-A32 pp [CI] |',
        '|---|---:|---:|---:|---:|']
    for ds,sp in panels:
        e=sums[ds,sp]['corruption']['expert']['metrics'];cl=sums[ds,sp]['clean']['all']['metrics']
        lines.append(f"| {ds}/{sp} | {e['D32_missing_candidates']['mean']:.3f} / 32 | "
            f"{100*e['D32_selected_both_positive']['mean']:.2f}% | {fmt(cl['D32_gain'],True)} | {fmt(cl['D32_vs_A32_v'],True)} |")
    lines+=['','## Positive and negative arrival cases','',
        'These examples diagnose this fixed rule; they do not set thresholds or candidate allocation. '
        'Source IDs are anonymous within the current cohort. Full curves, all scores, normalized candidate '
        'intervals and candidate GT metric values (without annotations) are available in anonymous ROWS.','']
    for ds in DATASETS:
        rows=read(PUBLIC/'confirm'/ds/'ROWS.json');rr=[r for r in rows if r['expert_scheduled'] and r['condition']!='clean']
        for label,r in [('largest D32 harm vs A32',min(rr,key=lambda r:r['D32_vs_A32_v'])),
                        ('largest D32 gain vs A32',max(rr,key=lambda r:r['D32_vs_A32_v']))]:
            d=r['details']['D32'][r['choices']['D32']]
            lines.append(f"- {ds}, {label}: source {r['source_id']}, {r['condition']}, {r['order']}, arrival {r['arrival']}; "
                f"A32 v={100*r['A32_v']:.4f}%, D32 v={100*r['D32_v']:.4f}%, paired {100*r['D32_vs_A32_v']:+.4f} pp; "
                f"D transitions {d['start_transition']:+.6f}/{d['end_transition']:+.6f}, "
                f"chosen indices {r['choices']['A32']}→{r['choices']['D32']}.")
    lines+=['','## Decision and limits','',
        f"- Expanded-support capacity: `{decision['capacity_evidence']}`.",
        f"- D32 vs A32 on both confirmation corruption flows: `{decision['boundary_evidence']}`.",
        '- Retain original A. Neither Expanded32 nor D is integrated or promoted; no new experiment is started.',
        '- Capacity is restricted to this one deterministic 24-interval addition and these fixed A boxes. '
        'A negative proxy cannot reject all unlabelled boundary estimation. B concerns this frozen PE global '
        'semantic signal, not all event semantics. D still uses that same signal, so it is not independent '
        'boundary evidence. One-second windows and missing exterior context are disclosed limitations.',
        '- Expert membership varies by order; inference is clustered by independent source, not by cells. '
        'Small expert source counts and earlier GT exposure limit generalization. No fresh-test, new '
        'persistent-transfer or universal temporal-bottleneck claim follows.','',
        '## Verification and resources','',
        'The predecessor A8/B8 predictions and scores reproduce exactly. All old support and A-selected '
        'intervals are retained. Both support generation and all decisions seal before this experiment’s '
        'GT access. Independent root scoring uses official dense metrics; the public auditor separately '
        'recomputes allocation, every B/D score/selection, candidate arithmetic and source-bootstrap summaries. '
        'All frozen inputs and production-method hashes are verified before and after scoring.','']
    for name in ['PREPARATION','GENERATION_RESOURCES','SCORE_CHECKS','FINAL_ROOT_AUDIT']:
        z=read(BASE/f'{name}.json');lines.append(f"- {name}: {z['worker_wall_seconds']:.3f} seconds worker wall time; CUDA initialized=false.")
    lines+=['','No GPU-kernel time is claimed from worker wall time. All new model/expert calls and backwards '
        'are zero. No private captions/media/GT coordinates/features/parameters are in the public export.']
    text='\n'.join(lines)+'\n';(ROOT/'docs/TA_TEMPORAL_BOUNDARY_SUPPORT_REVIEW.md').write_text(text)
    (PUBLIC/'REVIEW.md').write_text(text)
    fig,axs=plt.subplots(1,3,figsize=(15.2,4.0),layout='constrained');labels=['Vid dev','HC dev','Vid confirm','HC confirm']
    x=np.arange(4);palette=['#547a98','#a39a84','#c07e65','#668f7e']
    for ax,field,title,subset in [(axs[0],'capacity_gain','(a) Added candidate capacity','expert'),
        (axs[1],'A_support_v','(b) UVTG actual support gain','all'),
        (axs[2],'D32_vs_A32_v','(c) Boundary vs UVTG, same32','all')]:
        ms=[met(ds,sp,subset,field) for ds,sp in panels];y=np.array([100*m['mean'] for m in ms]);ci=np.array([m['ci95'] for m in ms])*100
        ax.bar(x,y,color=palette,width=.58);ax.errorbar(x,y,yerr=np.maximum(np.array([y-ci[:,0],ci[:,1]-y]),0),fmt='none',ecolor='#333333',capsize=3,lw=1.2)
        ax.axhline(0,color='#444444',lw=.8);ax.set_xticks(x,labels);ax.set_title(title,fontsize=11);ax.set_ylabel('Paired Δ vIoU (pp)')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for ext in ['png','pdf','svg']:fig.savefig(PUBLIC/f'TEMPORAL_BOUNDARY_SUPPORT.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig);print('REPORT_READY',decision,flush=True)

if __name__=='__main__':run()
