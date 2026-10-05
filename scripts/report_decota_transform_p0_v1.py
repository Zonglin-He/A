"""Source-backed report and three standalone, inspectable research figures."""
import sys,csv,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_transform_common_v1 import *
from scripts.decota_public_result_io_v1 import read as public_read

def run():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assert read(PUB/'ROOT_AUDIT.json')['status']==read(PUB/'PUBLIC_AUDIT.json')['status']=='pass'
    ts=public_read(PUB/'temporal/SUMMARY.json');ass=public_read(PUB/'spatial/ASSOCIATIONS.json');dec=read(PUB/'DECISION.json')
    tr=public_read(PUB/'temporal/ROWS.json');sr=public_read(PUB/'spatial/ROWS.json');ss=read(PUB/'spatial/SUMMARY.json');cost=read(PUB/'COST.json');cov=read(PUB/'COVERAGE.json')
    def interval(q,pp=False):
        if q['mean'] is None:return 'undefined'
        mult=100 if pp else 1
        return f"{q['mean']*mult:+.4f} [{q['ci95'][0]*mult:+.4f}, {q['ci95'][1]*mult:+.4f}]" if q['ci95'] else str(q['mean'])
    lines=['# DeCoTA real-input transformation P0 review','',
        'Two matched P0s are complete. All new predictions and unlabeled consistency scores were globally sealed before GT. The only new inputs are known temporal padding/cropping and spatial flip/brightness transforms; no new expert, optimization, parameter write, or old online-stream rerun. Native time and Uniform4/admitted Top1 remain the research working point until qualification; CURRENT is not promoted.','',
        '## Setting and actual controls','',
        'Official same-domain TA-STVG EMA; each dataset32 development +16 source-disjoint but historically exposed confirmation sources, one query/source, two orders, clean +five5% deployment corruptions with identical original pixels. There are576 unique original inputs and1152 logical arrivals. The spatial correction is the sealed Uniform4/single DINO/admitted Top1 energy, joint1792 parameters, Adam.03/10 steps/first own-loss minimum. Query state resets; online100 uses the old actual LN1/16 carryover chain. This P0 has no new persistent state.','',
        'Temporal view1 prepends ceil(N/8) repeated first frames on an exactly transported virtual frame grid. View2 retains half the available sampled context on each side of the Native interval. Every nonidentity view runs the encoder anew, then both official offsets and the original envelope readout; inverse intervals are clipped to original support and invalid ones fall back. The sole deployable P0 readout is the endpoint median of Native/shift/crop. Individual shift/crop arms are controls, not a pool from which a winner can be selected.','',
        'Spatial views are a real horizontal RGB flip and deterministic brightness.95. Both run the encoder anew. Only within that same new input do we reuse its frozen prefix to decode saved prearrival and selected states. Flip boxes are inverse-mapped; primary consistency is mean cross-view IoU inside the fixed Native interval, full-clip IoU is secondary. Episodic uses source Native prestate, while online100 uses its own inherited prearrival prestate.','',
        '## Temporal P0: consensus minus Native time','',
        'Fixed episodic corrected spatial boxes; corruption source-macro means and paired95% CIs in pp. Native-spatial and actual online100 corrected-box controls, clean, both orders and all five conditions are in temporal/SUMMARY.json.','',
        '| Dataset/panel | ΔtIoU pp | ΔvIoU pp | >5/>20 pp harm |','|---|---:|---:|---:|']
    for ds in DATASETS:
        for sp in ['search','confirm']:
            z=ts[ds][sp]['consensus_episodic']['corruption'];m=z['metrics'];tails=z['tails']
            lines.append(f"| {ds}/{sp} | {interval(m['delta_t'],True)} | {interval(m['delta_v'],True)} | {tails['harm_gt5pp']}/{tails['harm_gt20pp']} |")
    lines+=['',f"Temporal qualification: **{dec['temporal']['qualified']}**. Per-dataset locked gates: {dec['temporal']['dataset_gates']}. Conditional514-head stage: {dec['conditional_temporal514']}.",'',
        '## Spatial P0: does stability predict current help/harm?','',
        'Fixed score ΔC, no training/fitted gate. Source-disjoint confirmation is evaluated directly. Pearson correlation keeps zero utility; ROC help/harm excludes exact zero utility and reports denominator. Source bootstrap resamples each source with all its orders and conditions together.','',
        '| Dataset/panel/state | corr(ΔC,ΔvIoU) | help/harm AUC | help/harm/zero cells |','|---|---:|---:|---:|']
    for ds in DATASETS:
        for sp in ['search','confirm']:
            for stream in ['episodic','online100']:
                z=ass[ds][sp][stream]['corruption']['delta_event']
                lines.append(f"| {ds}/{sp}/{stream} | {interval(z['correlation'])} | {interval(z['AUC'])} | {z['help']}/{z['harm']}/{z['zero_utility']} |")
    lines+=['',f"Spatial qualification: **{dec['spatial']['qualified']}**. Locked dataset gates: {dec['spatial']['dataset_gates']}. Conditional acceptance stage: {dec['conditional_acceptance']}.",'',
        'No adapted state or optimizer step is accepted/rejected using these scores in P0. An association would authorize a separate matched acceptance trial; it would not establish an online gain. Wide intervals or a failed qualification do not prove that every input-stability signal is useless. The full-clip and non-directional panels cannot override the locked primary decision.','',
        '## Task preservation and coverage','',
        'A known coordinate transport law is exact; correct localization and semantic invariance are not guaranteed. Cropping is chosen by the predicted Native interval, so it can remove an unobserved true event. Repeated padding changes motion context. The same directional text can refer to a different entity after horizontal flip. All affected cells remain in the main panel; non-directional results are separately reported. Neither GT nor a good-case list chooses the transform.','',
        '| Dataset | unique inputs | identity crops | clipped shift predictions | invalid shift fallback | directional inputs |','|---|---:|---:|---:|---:|---:|']
    for ds,z in cov['dataset_counts'].items():lines.append(f"| {ds} | {z['unique_inputs']} | {z['crop_noop']} | {z['shift_clipped']} | {z['shift_invalid']} | {z['directional_unique']} |")
    lines+=['','Post-seal GT-only support audit (does not change crops or filter the primary results):','',
            '| Dataset/panel | GT event not fully retained | newly cut by crop | already incomplete original grid | total inputs |','|---|---:|---:|---:|---:|']
    for ds,panels in cov['crop_actual_GT_support_diagnostic'].items():
        for sp,z in panels.items():lines.append(f"| {ds}/{sp} | {z['event_not_fully_contained']} | {z['newly_cut_GT_event']} | {z['original_sample_support_incomplete']} | {z['inputs']} |")
    lines+=['','## Compute and verification','',
        '| Dataset | full two-offset view passes | individual offset forwards | state suffix replays | wall seconds | peak allocated GiB |','|---|---:|---:|---:|---:|---:|']
    for ds,z in cost.items():lines.append(f"| {ds} | {z['actual_full_view_passes']} | {z['actual_offset_forwards']} | {z['suffix_replays']} | {z['seconds']:.2f} | {z['peak_allocated_bytes']/2**30:.2f} |")
    lines+=['',
        'Smoke adds the separately recorded first-two clean native/full-corrected parity forwards. Cost is measured worker wall time, not isolated GPU kernel time. New encoder H is not persisted; only predictions, input hashes, view geometry and old-state receipts remain private. No DINO weights are loaded for this P0 and new backward/expert counts are zero.','',
        'ROOT_AUDIT independently verifies crop/shift inverse maps, scalar box IoU, saved parameter hashes and every official dense row. PUBLIC_AUDIT independently recomputes source bootstrap, source-pair-kernel AUC, means, tails and the fixed gates from anonymous rows. It cannot re-run private RGB or GT. All positive/negative/zero rows are retained. A status-only smoke completion error was saved and repaired without recomputing or changing completed predictions; see ENGINEERING_RECOVERY.json.','',
        '## Research decision','',
        'If neither P0 qualifies, stop these two variants and retain Native WHEN +Uniform4 +single DINO admitted Top1 current correction +LN1/16 as the research working point. This is not a production registry change and not a claim of universal failure. No full official-query benchmark, baseline, new memory/scorer/gate, third transform, or old paused queue is automatically started.','',
        '## Representative cases','',
        '| Experiment/dataset/panel | source/condition/order | change pp | stability ΔC |','|---|---|---:|---:|']
    cases=[]
    for typ,rows,arm,field in [('temporal',tr,'consensus_episodic','delta_v'),('spatial',sr,'episodic','correction_v'),('online_spatial',sr,'online100','correction_v')]:
        for ds in DATASETS:
            for sp in ['search','confirm']:
                rr=[r for r in rows if r['dataset']==ds and r['split']==sp and r['arm']==arm and r['condition']!='clean']
                for tail,fn in [('worst',min),('best',max)]:
                    r=fn(rr,key=lambda z:z[field]);case=dict(experiment=typ,tail=tail,**r);cases.append(case)
                    lines.append(f"| {typ}/{ds}/{sp}/{tail} | {r['source_id']}/{r['condition']}/{r['order']} | {100*r[field]:+.4f} | {r.get('delta_event',0):+.4f} |")
    write(PUB/'CASES.json',cases);(ROOT/'docs/TA_DECOTA_TRANSFORM_P0_REVIEW.md').write_text('\n'.join(lines)+'\n')
    # All anonymous numeric rows also available as conventional flat CSV.
    for name,rr in [('temporal',tr),('spatial',sr)]:
        fields=['dataset','split','condition','order','arrival','source_id','arm','v','t','s']+(['delta_v','delta_t','native_v','native_t'] if name=='temporal' else ['before_v','correction_v','delta_event','delta_full','selected_step','loss_gain'])
        with (PUB/name/'ROWS.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rr)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    figs=PUB/'figures';figs.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(1,2,figsize=(10,3.7),constrained_layout=True)
    labels=[];qq=[]
    for ds in DATASETS:
        for sp in ['search','confirm']:labels.append(ds+' '+sp);qq.append(ts[ds][sp]['consensus_episodic']['corruption']['metrics'])
    for a,metric in zip(ax,['delta_t','delta_v']):
        for i,q in enumerate(qq):
            x=q[metric];a.errorbar(i,100*x['mean'],yerr=np.array([[100*(x['mean']-x['ci95'][0])],[100*(x['ci95'][1]-x['mean'])]]),fmt='o',color=['#337ab7','#ca6542'][i//2],capsize=4)
        a.axhline(0,color='.5',linewidth=.8);a.set_xticks(range(4),labels,rotation=12);a.set_ylabel('Consensus − Native (pp)');a.set_title('tIoU' if metric=='delta_t' else 'vIoU')
    for ext in ['png','pdf']:fig.savefig(figs/('temporal_consensus.'+ext),dpi=220)
    plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
    labels=[];aa=[]
    for ds in DATASETS:
        for stream in ['episodic','online100']:
            labels.append(ds+' '+stream);aa.append(ass[ds]['confirm'][stream]['corruption']['delta_event'])
    for a,name,baseline in zip(ax,['correlation','AUC'],[0,.5]):
        for i,q in enumerate(aa):
            x=q[name]
            if x['mean'] is not None and x['ci95']:a.errorbar(x['mean'],i,xerr=np.array([[x['mean']-x['ci95'][0]],[x['ci95'][1]-x['mean']]]),fmt='o',color=['#337ab7','#ca6542'][i//2],capsize=4)
        a.set_yticks(range(4),labels);a.axvline(baseline,color='.5',linewidth=.8);a.set_title(name+' on source-held-out confirmation');a.invert_yaxis()
    for ext in ['png','pdf']:fig.savefig(figs/('correction_stability.'+ext),dpi=220)
    plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(10,3.8),constrained_layout=True)
    for a,ds in zip(ax,DATASETS):
        for sp,color in [('search','#337ab7'),('confirm','#ca6542')]:
            rr=[r for r in sr if r['dataset']==ds and r['split']==sp and r['arm']=='episodic' and r['condition']!='clean'];ids=sorted({r['source_id'] for r in rr})
            xx=[np.mean([r['delta_event'] for r in rr if r['source_id']==i]) for i in ids];yy=[100*np.mean([r['correction_v'] for r in rr if r['source_id']==i]) for i in ids]
            a.scatter(xx,yy,color=color,label=sp,alpha=.8,s=27)
        a.axvline(0,color='.6',lw=.8);a.axhline(0,color='.6',lw=.8);a.set_xlabel('Source-mean change in view consistency');a.set_ylabel('Source-mean current vIoU gain (pp)');a.set_title(ds);a.legend(frameon=False)
    for ext in ['png','pdf']:fig.savefig(figs/('source_diagnostic.'+ext),dpi=220)
    plt.close(fig)
    write(PUB/'ENGINEERING_RECOVERY.json',dict(recovery=['smoke_status_001','cpu_wait_002'],
        reasons=['duplicate status metadata key after completed smoke','Conda Python optional pidfd_open API absent'],
        prediction_rules_changed=False,predictions_rerun=False,GT_preseal=False,revision001_status_only=True,CPU_preseal_wait_repair=True))
    print('TRANSFORM_REPORT_AND_THREE_FIGURES_COMPLETE',flush=True)

if __name__=='__main__':run()
