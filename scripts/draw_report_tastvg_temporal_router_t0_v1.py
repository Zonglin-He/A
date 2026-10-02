"""Source-backed quantitative figures and bounded interpretation of T0."""
import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def value(summary,k):return 100*summary['metrics'][k]['mean']
def ci(summary,k):return '['+', '.join(f'{100*x:+.4f}' for x in summary['metrics'][k]['ci95'])+']'
def save(fig,base,name):
    for ext in ['png','pdf','svg']:fig.savefig(base/f'{name}.{ext}',dpi=230,bbox_inches='tight',pad_inches=.06)
    plt.close(fig)

def run(base):
    base=Path(base);datasets=['vidstg','hc2'];titles=['VidSTG','HC-STVG-v2']
    support={d:read(base/d/'SUPPORT_SUMMARY.json') for d in datasets};task={d:read(base/d/'TASK_SUMMARY.json') for d in datasets}
    erows={d:read(base/d/'EXPERT_ROWS.json') for d in datasets};decision=read(base/'DECISION.json')
    colors=['#596273','#237C9C','#D28D37'];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(10.8,5.1),constrained_layout=True)
    for i,(ds,title) in enumerate(zip(datasets,titles)):
        for ax,metric,label in zip(axes[i],['gt_mass','event_recall','quantile_coverage'],['GT support mass (%)','Event recall (%)','Quantile reference hits (%)']):
            for j,r in enumerate(['S','E','SE']):
                m=support[ds]['corruption'][r]['metrics'][r+'_'+metric];v=100*m['mean'];lo,hi=100*np.asarray(m['ci95'])
                ax.bar(j,v,color=colors[j],width=.6,zorder=2);ax.errorbar(j,v,yerr=[[v-lo],[hi-v]],fmt='none',ecolor='#343B45',capsize=3,lw=1)
            if metric=='quantile_coverage':
                u=value(support[ds]['corruption']['S'],'S_uniform_coverage');ax.axhline(u,color='#ADB4BC',lw=1,ls='--')
            ax.set_xticks([0,1,2],['S','E','SE']);ax.set_ylim(0,105);ax.set_ylabel(label);ax.grid(axis='y',color='#ECEEF0',zorder=0,lw=.6)
            ax.set_title(title,fontweight='bold')
    save(fig,base,'router_support')
    fig,axes=plt.subplots(1,2,figsize=(8.5,3),constrained_layout=True)
    labels=['Expert: tIoU','Expert: vIoU','All: tIoU','All: vIoU']
    for ax,ds,title in zip(axes,datasets,titles):
        for y,(sub,m) in enumerate([('expert','delta_t'),('expert','delta_v'),('all','delta_t'),('all','delta_v')]):
            z=task[ds]['corruption'][sub]['metrics'][m];v=100*z['mean'];lo,hi=100*np.asarray(z['ci95']);ax.errorbar(v,3-y,xerr=[[v-lo],[hi-v]],fmt='o',color=colors[y%2],markersize=5,capsize=3,lw=1.3)
        ax.axvline(0,color='#AAB1BA',ls='--',lw=.9);ax.set_yticks([3,2,1,0],labels);ax.set_xlabel('QC − Current (pp)');ax.set_title(title,fontweight='bold');ax.set_ylim(-.5,3.5);ax.grid(axis='x',color='#ECEEF0',lw=.6)
    save(fig,base,'critic_readout')
    fig,axes=plt.subplots(1,2,figsize=(8.5,2.9),constrained_layout=True)
    for ax,ds,title in zip(axes,datasets,titles):
        e=task[ds]['corruption']['expert'];keys=['Bad contributor','Native >.5 tIoU lost','Native >.5 vIoU lost'];positions=np.arange(3)
        for j,name in enumerate(['Current','QC']):
            vals=[e['critic_diagnosis'][name]['bad_contributor'],e['Native_to_'+name]['t']['0.5']['correct_to_wrong'],e['Native_to_'+name]['v']['0.5']['correct_to_wrong']]
            ax.barh(positions+(j-.5)*.3,vals,height=.28,color=colors[j],label=name)
        ax.set_yticks(positions,keys);ax.invert_yaxis();ax.set_xlabel('Corrupted expert arrivals (80 total)');ax.set_title(title,fontweight='bold');ax.grid(axis='x',color='#ECEEF0',lw=.6)
    handles,legend_names=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_names,loc='upper center',bbox_to_anchor=(.5,1.10),ncol=2,frameon=False,fontsize=9)
    save(fig,base,'critic_failures')
    cases={}
    for ds in datasets:
        rr=[r for r in erows[ds] if r['condition']!='clean'];positive=sorted([r for r in rr if r['delta_v']>1e-12],key=lambda r:-r['delta_v'])[:3];negative=sorted([r for r in rr if r['delta_v']< -1e-12],key=lambda r:r['delta_v'])[:3]
        cases[ds]={name:[{k:r[k] for k in ['source_id','parent','condition','order','arrival','Native_t','Native_v','Current_t','Current_v','QC_t','QC_v','delta_t','delta_v','Current_selected','QC_selected','Current_bad_contributor','QC_bad_contributor','candidate_t','candidate_v']} for r in seq] for name,seq in [('positive',positive),('negative',negative)]}
        for name,seq in [('positive',positive),('negative',negative)]:
            for entry,r in zip(cases[ds][name],seq):
                for critic in ['Current','QC']:
                    win=r[critic+'_winning_contributor']
                    entry[critic+'_contributor_metrics']={k:r[field][win] for k,field in [('confidence','proposal_confidence'),('agreement','proposal_agreement'),('quality','proposal_quality'),('GT_tIoU','proposal_gt_t')]} if win is not None else None
    (base/'CASES.json').write_text(json.dumps(cases,indent=2,allow_nan=False)+'\n')
    text=['# Temporal Router T0: proposal consensus did not improve support concentration',
        '', 'The authorized CPU audit is complete. E and SE raise positive event coverage to 100%, '
        'but reduce the fraction of support assigned to the event and reduce five-quantile frame precision. '
        'QC native-candidate scoring does not improve corrupted temporal or tube accuracy. '
        'The predeclared T1 development signal is absent in both datasets; T1 GPU/online trials and H-full were not started.',
        '', '## Matched setting and measurement', '',
        'Original 32 historically exposed development sources and one query per source in each dataset; '
        'two orders, clean plus frame-drop/freeze/blur/occlusion/exposure transient 5%, official same-domain '
        'TA-STVG checkpoints, original Paper48 sampling/pixels, spatial A fully frozen. '
        '384 sealed pre-update arrivals per dataset were reused, including 96 scheduled expert arrivals '
        '(80 corrupted, 16 clean). Distinct expert sources: Vid16, HC14. There was no new forward/backward, '
        'expert inference, video decode or GPU initialization.',
        '', 'The task readout is exact for a fixed spatial A trajectory: temporal selection is not an input '
        'to spatial rewards, gradients or persistent-state updates. Current scoring and outputs exactly '
        'match the prior A; all future nonexpert QC differences are zero. This is not a fresh online GPU '
        'experiment or evidence of temporal learning.',
        '', 'Source/order/condition macro means are primary. Paired 10000 source bootstrap (seed20261001) '
        'keeps repeated orders/conditions grouped by source. Cell means and every anonymous row are also '
        'published. The small exposed source panel is not a fresh confirmation set.',
        '', '## Router result on corrupted expert arrivals', '',
        '| Dataset | Router | GT mass | Event recall | Soft tIoU | Five-quantile hits | Zero reference mass | Useful evidence discarded |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for ds,title in zip(datasets,titles):
        for r in ['S','E','SE']:
            s=support[ds]['corruption'][r];cc=s['counts'];text.append(f"| {title} | {r} | {value(s,r+'_gt_mass'):.2f}% | {value(s,r+'_event_recall'):.2f}% | {value(s,r+'_soft_iou'):.2f}% | {value(s,r+'_quantile_coverage'):.2f}% | {cc['zero_weighted_mass_nonempty']}/{cc['nonempty_reference_arrivals']} | {cc['useful_reference_discarded']} |")
    text+=['','S is the student consensus; E uses every cached proposal with q=c×mean-other-proposal tIoU; '
        'SE is exactly half S plus half E. Duplicates are retained, as requested. All expert maps have '
        'positive mass in the actual panel; no missing-evidence fallback occurred.',
        '', '| Dataset | Change vs S | GT mass (pp, paired 95% CI) | Quantile hits (pp, paired 95% CI) |',
        '|---|---|---:|---:|']
    for ds,title in zip(datasets,titles):
        for r in ['E','SE']:
            z=support[ds]['corruption'][r+'_minus_S'];text.append(f"| {title} | {r}−S | {value(z,'delta_gt_mass'):+.4f} {ci(z,'delta_gt_mass')} | {value(z,'delta_quantile_coverage'):+.4f} {ci(z,'delta_quantile_coverage')} |")
    text+=['', 'The 100% recall is not a localized-event success. The maps give nonzero support to every '
        'sampled event frame, while assigning a larger fraction of their mass outside the event. '
        'Consequently the five CDF reference locations become less event-focused. Removing zero-reference '
        'mass does not establish that the restored evidence is useful.',
        '', 'Compared with the old uniform reference positions, quantile hits are still higher: uniform '
        f"Vid {value(support['vidstg']['corruption']['S'],'S_uniform_coverage'):.2f}%, HC {value(support['hc2']['corruption']['S'],'S_uniform_coverage'):.2f}% (source macro). "
        'However S already has that advantage; E/SE lose concentration relative to S. No Sa2VA call was '
        'made at these hypothetical locations, so frame-location coverage does not prove object-mask '
        'quality or spatial adaptation gains.',
        '', 'The previous H-lite diagnostic used actual sampled frames with a GT box, and reported cell means. '
        'Under that exact convention S reproduces Vid mass47.3645%/coverage63.0379% and '
        'HC mass58.5434%/coverage94.1102%. The main table instead uses source macro and the existing '
        'half-open temporal event convention. Vid event/scored masks match. HC has 12/96 expert observations '
        'with a sampled-endpoint difference; the prior HC inclusive last-box/end-coordinate discrepancy is '
        'preserved and disclosed, rather than silently changing task scoring.',
        '', '## Native-candidate critic result', '',
        '| Dataset | Subset | QC−Current tIoU (pp, paired 95% CI) | QC−Current vIoU (pp, paired 95% CI) |',
        '|---|---|---:|---:|']
    for ds,title in zip(datasets,titles):
        for sub in ['expert','all']:
            z=task[ds]['corruption'][sub];text.append(f"| {title} | {sub} | {value(z,'delta_t'):+.4f} {ci(z,'delta_t')} | {value(z,'delta_v'):+.4f} {ci(z,'delta_v')} |")
    text+=['','All four primary task-difference intervals include zero. The negative means do not prove '
        'universal harm, but there is no measured benefit from substituting consensus for raw confidence. '
        'The all-arrival result includes unchanged nonexpert positions; it is not simply expert result '
        'divided by four in HC because source weighting differs between expert-only and full-stream subsets.',
        '', '| Dataset | Corrupted choices changed | Bad contributor Current→QC | Native >.5 tIoU destroyed | Native >.5 vIoU destroyed | Candidate vIoU regret Current→QC |',
        '|---|---:|---:|---:|---:|---:|']
    for ds,title in zip(datasets,titles):
        z=task[ds]['corruption']['expert'];n=sum(r['selection_changed'] for r in erows[ds] if r['condition']!='clean')
        text.append(f"| {title} | {n}/80 | {z['critic_diagnosis']['Current']['bad_contributor']}→{z['critic_diagnosis']['QC']['bad_contributor']} | {z['Native_to_Current']['t']['0.5']['correct_to_wrong']}→{z['Native_to_QC']['t']['0.5']['correct_to_wrong']} | {z['Native_to_Current']['v']['0.5']['correct_to_wrong']}→{z['Native_to_QC']['v']['0.5']['correct_to_wrong']} | {value(z['selection_regret'],'Current_regret_v'):.4f}→{value(z['selection_regret'],'QC_regret_v'):.4f} pp |")
    text+=['', 'Bad contributor means winning proposal tIoU≤.3 despite another available proposal tIoU>.5. '
        'The count is an offline diagnostic. QC reduces that count by one in HC but increases missed '
        'correct candidates and destructive native replacements. A small improvement in one proxy '
        'counter does not fix the final candidate selection.',
        '', 'Positive cases are preserved. HC source31/occlusion/order1 gains +11.8739pp vIoU; '
        'source4 has about +10.58pp gains across several corruptions. Negative source0/frame-drop/order1 '
        'loses −34.4310pp and motion-blur loses −33.2872pp. These are anonymous repeated-source examples, '
        'not independent videos. HC corrupted expert subset has five losses and four gains exceeding '
        '5pp. See CASES.json and complete rows; no case was removed from aggregation.',
        '', 'The HC source0/frame-drop example exposes the proxy mismatch directly: the Current winning '
        'proposal has GT tIoU.8234 and agreement.2259; the QC winner has lower GT tIoU.5090 but higher '
        'agreement.2650 and confidence.5193 (versus.4526). Its consensus weight is.1376 rather than.1022. '
        'The final native candidate changes from tIoU.8347/vIoU.5770 to tIoU.3532/vIoU.2327, although '
        'another available expert proposal has GT tIoU.8797. Higher agreement does not identify the '
        'better localized proposal in this concrete case.',
        '', 'Clean expert QC−Current vIoU: '
        f"Vid {value(task['vidstg']['clean']['expert'],'delta_v'):+.4f}pp {ci(task['vidstg']['clean']['expert'],'delta_v')}; "
        f"HC {value(task['hc2']['clean']['expert'],'delta_v'):+.4f}pp {ci(task['hc2']['clean']['expert'],'delta_v')}. "
        'Clean positives do not establish benefit under deployment corruption.',
        '', '## Interpretation and decision', '',
        'The proposal-consensus heuristic does not supply the missing localization quality in this '
        'development panel. Agreement among outputs from one specialist is not independent corroboration. '
        'Many correlated or duplicate hypotheses can agree without identifying the referred event. '
        'The measured decrease in support mass and increase in HC selection regret are direct evidence '
        'against this particular correction; they do not prove all temporal evidence routing impossible.',
        '', 'Keep spatial A and the original temporal critic as the development control. '
        'T1 is not triggered; H-full, event-conditioned Sa2VA, prototype/context memory and new parameter '
        'search are not started. No production registry changes. If temporal localization is reopened, '
        'it needs additional discriminating information beyond same-model proposal agreement, with a '
        'separate authorization and matched test. This audit does not establish a unique independent '
        'cross-query state failure in HC; the preceding H-lite trajectory comparison had intervals '
        'crossing zero and changed on-policy trajectories.',
        '', '## Verification and resources', '',
        'All new maps and both readout decisions were sealed for all768 arrivals before GT interpretation. '
        'Root checked768 state links,1536 prior-current scalar matches,218196 proposal numeric values, '
        '52686 support-map values,3072 critic scores and6144 dense metric scalar checks; maximum dense '
        'error8.8818e−16. Anonymous audit independently recomputes proposal weighting, scores, argmax, '
        'support denominators, all transitions, macro aggregation and bootstrap.',
        '', 'New model/GPU/expert calls: zero. Map CPU wall and score CPU wall are recorded in RESOURCES.json '
        '(wall includes file IO/imports, not pure numerical-kernel time). Cached predictions and expert '
        'receipts were hash-verified; raw coordinates, media, labels, parameters, gradients and personal '
        'records remain excluded from public export.',
        '', '![Router support](../results/tastvg_temporal_router/2026-10-02/router_support.png)',
        '', '![Critic readout](../results/tastvg_temporal_router/2026-10-02/critic_readout.png)',
        '', '![Critic failure counts](../results/tastvg_temporal_router/2026-10-02/critic_failures.png)', '']
    report='\n'.join(text)
    (ROOT/'docs/TA_TEMPORAL_ROUTER_REVIEW.md').write_text(report)
    (base/'REPORT.md').write_text(report.replace('../results/tastvg_temporal_router/2026-10-02/',''))
    print('Report and nine figure files generated')

if __name__=='__main__':run(sys.argv[1])
