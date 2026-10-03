"""Report all fixed-Old8 outcomes and paired-source figures, without promotion."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.tastvg_temporal_quality_common_v1 import *

def run():
    verify();root=read(BASE/'FINAL_ROOT_AUDIT.json');assert root['status']=='pass'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    sums={(ds,sp):read(PUBLIC/sp/ds/'SUMMARY.json') for ds in DATASETS for sp in SPLITS}
    def fmt(m,ci=False):
        z=f"{100*m['mean']:+.4f}"
        return z+f" [{100*m['ci95'][0]:+.4f}, {100*m['ci95'][1]:+.4f}]" if ci else z
    panels=[('vidstg','search'),('hc2','search'),('vidstg','confirm'),('hc2','confirm')]
    conf=[sums[ds,'confirm']['corruption']['all']['metrics']['actual_gain'] for ds in DATASETS]
    evidence='positive_mean_both_confirmation_panels' if all(m['mean']>0 for m in conf) else 'not_positive_in_both_confirmation_panels'
    if all(m['ci95'][0]>0 for m in conf):evidence='positive_paired_interval_both_confirmation_panels'
    decision=dict(status='completed_root_verified',measurement='valid_in_locked_scope',
        experimental_signal='PE_query_inner_outer_contrast_alpha025',evidence=evidence,
        candidate_support='exact_Old8_preserved',retained_baseline='A',new_signal_integrated=False,
        CURRENT_METHOD_changed=False,next_experiment_started=False,confirmation_used_to_reselect=False,
        decision_scope='one_predeclared_query_similarity_quality_proxy_not_all_quality_signals',time=time.time())
    write(BASE/'DECISION.json',decision);write(PUBLIC/'DECISION.json',decision)
    lines=['# Fixed Old8 temporal localization-quality review','',
        'The fixed-support experiment is complete and independently audited. The one new signal is '
        'cached PE pooled-query inner-minus-outer temporal contrast (outer ratio .25). '
        'A remains the baseline; this result does not promote a new deployment method.','',
        '## Configuration and data boundary','',
        'VidSTG and HC-STVG-v2 each use the same 32 development + 16 source-disjoint confirmation '
        'sources within the current batch, one query/source, two orders, clean + five fixed 5% transient '
        'corruptions, 25% scheduled experts. All have historical exposure. There are 1,152 readouts, '
        '288 expert positions (240 corruption + 48 clean), and 864 positions that remain exactly A. '
        'The comparison is against A, not Frozen. No formula was chosen on confirmation.','',
        'A boxes, pre/post persistent spatial states, Uniform Rank-RKL writes, Old8 intervals and their '
        'order, corruption pixels, checkpoints and expert schedule were identical. Vid K1 and HC K8 '
        'use their sealed best learning rates and teacher temperatures. No model forward, backward, '
        'new expert observation, new temporal view, target training, or source quality-head training occurred.','',
        '## Signal and its limits','',
        'The original critic takes the maximum proposal-confidence-weighted interval overlap. The '
        'new critic uses cosine of each cached projected PE video feature with text column zero '
        '(the projected pooled query), and ranks the **same Old8** by mean similarity inside minus '
        'mean similarity in two adjacent bands, each .25 times the interval length. Fractional bin '
        'overlap maps original phase-zero 2-Hz features to the unchanged physical observed window. '
        'There is no proposal confidence multiplier or consensus. A full-window candidate has no '
        'observed outer context and receives neutral zero contrast; it remains in the pool. Constant '
        'evidence falls back to A; numerical score ties follow native-first original candidate order.','',
        'The contrast geometry is inspired by [AutoLoc, ECCV 2018](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Zheng_Shou_AutoLoc_Weakly-supervised_Temporal_ECCV_2018_paper.pdf), '
        'which uses trained class activations and video-level action supervision. Here frozen query '
        'similarities are only an inference proxy. This is not AutoLoc training or calibrated interval '
        'IoU. The [official PE interface](https://github.com/facebookresearch/perception_models/blob/main/apps/pe/README.md) '
        'and the cached poolandtoken extraction contract were inspected before the rule was locked. '
        'Persistent objects/background similarity and incomplete action semantics can make this proxy '
        'prefer a wrong or too-short segment.','',
        '## Actual corruption readouts','',
        'Equal-source means and paired 95% source-cluster bootstrap intervals; vIoU/tIoU differences '
        'are percentage points. All-flow gains are scored directly, not expert means multiplied by .25.','',
        '| Panel | A vIoU | New vIoU | All-flow ΔvIoU [95% CI] | Expert ΔvIoU [95% CI] | All-flow ΔtIoU [95% CI] |',
        '|---|---:|---:|---:|---:|---:|']
    for ds,sp in panels:
        s=sums[ds,sp]['corruption'];a=s['all']['metrics'];e=s['expert']['metrics']
        lines.append(f"| {ds} {sp} | {100*a['A_v']['mean']:.4f} | {100*a['quality_v']['mean']:.4f} | {fmt(a['actual_gain'],True)} | {fmt(e['actual_gain'],True)} | {fmt(a['actual_t_gain'],True)} |")
    lines+=['','## Same-support oracle regret','',
        'The oracle is GT-best selection from the unchanged Old8 under A boxes. It is privileged '
        'offline evaluation, not an attainable unlabelled selector. Regret reduction is algebraically '
        'the same selected-vIoU improvement on a fixed pool; it is not independent confirmation. '
        'Negative recovery fractions mean the new selector increases remaining selection error.','',
        '| Corruption expert panel | Cells / independent sources | Old regret pp | New regret pp | Regret reduction pp [95% CI] | Fraction recovered |',
        '|---|---:|---:|---:|---:|---:|']
    for ds,sp in panels:
        z=sums[ds,sp]['corruption']['expert'];m=z['metrics'];fraction=z['regret_recovered_fraction']
        ff=f'{100*fraction:.2f}%' if fraction is not None else 'undefined'
        lines.append(f"| {ds} {sp} | {z['cells']} / {z['sources']} | {100*m['old_regret']['mean']:.4f} | {100*m['quality_regret']['mean']:.4f} | {fmt(m['regret_reduction'],True)} | {ff} |")
    lines+=['','## Harm, original good results, and clean controls','',
        '| Corruption expert panel | Better / worse / equal | >5pp harm | Old positive rerank harmed / available | Correct→wrong at .3 / .5 | Wrong→correct at .3 / .5 |',
        '|---|---:|---:|---:|---:|---:|']
    for ds,sp in panels:
        z=sums[ds,sp]['corruption']['expert'];c=z['counts'];x=z['correctness']
        lines.append(f"| {ds} {sp} | {c['improved']} / {c['harmed']} / {c['unchanged']} | {c['severe_harm_gt5pp']} | {c['old_rerank_gain_destroyed']} / {c['old_positive_rerank']} | {x['0.3']['correct_to_wrong']} / {x['0.5']['correct_to_wrong']} | {x['0.3']['wrong_to_correct']} / {x['0.5']['wrong_to_correct']} |")
    lines+=['','| Panel | Clean all-flow ΔvIoU pp [95% CI] | Corrupt all-flow order values pp | Leave-one-source-out range pp |',
        '|---|---:|---:|---:|']
    for ds,sp in panels:
        cl=sums[ds,sp]['clean']['all']['metrics']['actual_gain'];m=sums[ds,sp]['corruption']['all']['metrics']['actual_gain']
        orders=', '.join(f'{100*x:+.4f}' for x in m['order_values']);loo=', '.join(f'{100*x:+.4f}' for x in m['leave_one_out_range'])
        lines.append(f'| {ds} {sp} | {fmt(cl,True)} | {orders} | {loo} |')
    lines+=['','## Ranking evidence and concrete successes/failures','',
        'Every harmed position still contains the old A choice in its unchanged pool, so support '
        'deletion cannot explain this new harm. There are no missing/constant embedding fallbacks '
        'in the actual 288 expert positions; only one candidate has no outer observation. '
        'Pair accuracy is diagnostic rather than the primary endpoint: better ordering of some '
        'pairs does not ensure that the top-ranked interval has higher tube quality.','',
        '| Corruption expert panel | Strict-pair cells / sources | Old pair accuracy % | New pair accuracy % | Difference pp [95% CI] |',
        '|---|---:|---:|---:|---:|']
    for ds,sp in panels:
        z=sums[ds,sp]['corruption']['expert']['pairwise_v'];m=z['metrics']
        lines.append(f"| {ds} {sp} | {z['cells']} / {z['sources']} | {100*m['old_pair_v_accuracy']['mean']:.3f} | {100*m['quality_pair_v_accuracy']['mean']:.3f} | {fmt(m['pair_v_accuracy_gain'],True)} |")
    lines+=['','| Confirmation example | Anonymous source / condition / order | ΔvIoU pp | Old→new tIoU | Contrast of A choice→new choice |',
        '|---|---|---:|---:|---:|']
    for ds in DATASETS:
        rr=[r for r in read(PUBLIC/'confirm'/ds/'ROWS.json') if r['expert_scheduled'] and r['condition']!='clean']
        for tag,r in [('harm',min(rr,key=lambda r:r['actual_gain'])),('gain',max(rr,key=lambda r:r['actual_gain']))]:
            oi=r['old_selected'];qi=r['quality_selected']
            lines.append(f"| {ds} {tag} | {r['source_id']} / {r['condition']} / {r['order']} | {100*r['actual_gain']:+.4f} | {r['A_t']:.4f}→{r['quality_t']:.4f} | {r['quality_scores'][oi]:.6f}→{r['quality_scores'][qi]:.6f} |")
    lines+=['',
        'For HC confirmation source43/exposure/order2, the contrast increases from .001099 to '
        '.009487, but tIoU falls from .8750 to .5882 and vIoU falls by 11.7804 pp. The chosen '
        'interval extends farther to the right; its lower average outer similarity outweighs '
        'its slightly lower inner similarity. This is a direct counterexample to treating '
        'higher PE contrast as higher localization quality. It does not establish that this '
        'mechanism explains every failure. HC also has a positive source47/frame-freeze case '
        '(+8.1310 pp), and both types are preserved rather than retuning the formula.']
    resources=read(BASE/'GENERATION_RESOURCES.json');score=read(BASE/'SCORE_ROOT_CHECKS.json')
    lines+=['','## Verification and costs','',
        f"All {root['immutable_inputs']} private input files were hash-checked before and after. "
        'Every A box/state and original critic score/selection was checked; all 288 candidate pools '
        'and their old oracle values match the predecessor. The new activation was independently '
        'recomputed from cached projected features. Official dense metrics and anonymous arithmetic '
        'were independently checked, including group means, paired bootstrap, counts and decisions. '
        f"Root maximum official error: {root['max_official_error']:.3g}. "
        'Both datasets and panels were sealed before this new offline GT scoring. No GT entered '
        'the candidate scorer or decoder.','',
        f"CPU generation/readout wall time {resources['worker_wall_seconds']:.3f}s; score wall time "
        f"{score['worker_wall_seconds']:.3f}s; root audit wall time {root['worker_wall_seconds']:.3f}s. "
        f"There are {resources['unique_feature_inputs']} distinct reused feature inputs. "
        'These timings include cache IO and verification, and are not GPU kernel time or an '
        'uncached deployment estimate. New model/expert calls and backwards are all zero.','',
        'A syntax typo at pre-lock startup was preserved in recovery/startup_syntax_001 and repaired '
        'before any predictions or label read. All subsequent fixes, if any, are enumerated in '
        'ENGINEERING_HISTORY; no outcome-dependent science revision is permitted.','',
        '## Decision scope','',
        f"Evidence label: `{evidence}`. Preserve A, the previous New8 negative result, and all positive "
        'and negative examples from this test. A single unlabelled semantic contrast cannot establish '
        'that every localization-quality signal works or fails. Fixed-Old8 scoring cannot repair '
        'the large Grid−Old8 candidate coverage deficit. Any next mechanism or expansion is a new '
        'decision; no further experiment, spatial change, memory or method promotion starts here.','',
        'Full anonymous cell values, curves, eight candidate scores/utilities, cases, source summaries '
        'and order sensitivity are supplied alongside the report. Predicted interval indices and '
        'normalised observed-window coordinates are included; target annotations, GT coordinates, '
        'captions, raw features, boxes, states and weights are not.']
    text='\n'.join(lines)+'\n'
    write(PUBLIC/'REPORT_CONTEXT.json',dict(status='root_verified',decision=decision,primary_comparison='quality_vs_A_fixed_Old8',
        all_readouts=1152,expert_readouts=288,source_macro=True,paired_source_bootstrap=10000))
    (PUBLIC/'REPORT.md').write_text(text);(ROOT/'docs/TA_TEMPORAL_QUALITY_OLD8_REVIEW.md').write_text(text)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(10.8,3.8),layout='constrained')
    labels=[f'{ds.upper()} '+('development' if sp=='search' else 'confirmation') for ds,sp in panels]
    for ax,subset,title in zip(axes,['all','expert'],['All corruption arrivals','Scheduled expert arrivals']):
        means=[];lows=[];highs=[]
        for ds,sp in panels:
            m=sums[ds,sp]['corruption'][subset]['metrics']['actual_gain'];means.append(100*m['mean'])
            lows.append(100*(m['mean']-m['ci95'][0]));highs.append(100*(m['ci95'][1]-m['mean']))
        colors=['#567B9D','#D18B50','#567B9D','#D18B50']
        for i in range(4):ax.errorbar(means[i],i,xerr=[[lows[i]],[highs[i]]],fmt='o',color=colors[i],capsize=3,ms=6,lw=1.6)
        ax.axvline(0,color='#5A5A5A',lw=1,ls='--');ax.set_yticks(range(4),labels);ax.invert_yaxis()
        ax.set_title(title,fontweight='bold');ax.set_xlabel('New quality scorer − A: vIoU (pp)');ax.grid(axis='x',alpha=.15)
    figure=PUBLIC/'fixed_old8_quality_effect'
    for suffix in ['png','pdf','svg']:fig.savefig(figure.with_suffix('.'+suffix),dpi=230,bbox_inches='tight')
    plt.close(fig)
    status(BASE/'STATUS.json',dict(status='report_complete_pending_publication',time=time.time()))
    archive('1152正式读出与288同支持regret完成独立核验/报告，保留A；待公开逐远端核验')
    print('REPORT_READY',decision,flush=True)

if __name__=='__main__':run()
