"""Evidence-bound report and scientific plots; anonymous scalar inputs only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,csv,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_cross_domain_common_v1 import *
from scripts.audit_tastvg_negative_evidence_v1 import summary
ARMS=('F','T','S','A','U')

def formatted(z):
    return f"{z['mean']*100:+.3f} [{z['ci95'][0]*100:+.3f}, {z['ci95'][1]*100:+.3f}]"

def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    assert read(BASE/'ROOT_AUDIT.json')['status']=='pass'
    s={d:read(PUB/d/'SUMMARY.json') for d in DIRECTIONS};rr={d:read(PUB/d/'ROWS.json') for d in DIRECTIONS}
    st={d:read(PUB/d/'STEP_ROWS.json') for d in DIRECTIONS}
    GO=all(s[d]['panels']['all']['metrics']['A_minus_F_v']['ci95'][0]>0 for d in DIRECTIONS) and any(
        s[d]['panels']['all']['metrics']['A_minus_T_v']['ci95'][0]>0 for d in DIRECTIONS)
    decision=dict(GO=GO,production_promoted=False,full_vs_frozen_positive_ci={d:
        s[d]['panels']['all']['metrics']['A_minus_F_v']['ci95'][0]>0 for d in DIRECTIONS},
        persistent_positive_ci={d:s[d]['panels']['all']['metrics']['A_minus_T_v']['ci95'][0]>0 for d in DIRECTIONS},
        nonexpert_persistent_positive_ci={d:s[d]['panels']['nonexpert']['metrics']['A_minus_T_v']['ci95'][0]>0 for d in DIRECTIONS},
        no_target_reselection=True,no_followup_job_launched=True,time=time.time())
    write(PUB/'DECISION.json',decision)
    lines=['# Unchanged A under clean cross-dataset shift','',
        f"Predeclared qualification decision: **{'GO in this exposed qualification scope' if GO else 'GO criteria not established'}**. "
        'No method promotion, target retuning, arm selection or automatic extension.','',
        'Both Full−Frozen contrasts are approximately one percentage point in this run. '
        'This is modest qualification evidence, not reproduction of the historical 4–8pp DeCoTA gains.','',
        'Official source checkpoints and source-domain development bundles; one query per historical target source; '
        'three fixed orders, clean inputs and nominal 25% specialist availability. Vid→HC2: 135 sources/405 arrivals; '
        'HC2→Vid: 384 sources/1152 arrivals. 1557 arrivals, 390 scheduled specialist arrivals. '
        'The cohorts and development history are exposed; this is not a fresh test.','',
        '| Source→target | Method | vIoU % | Δ vs source Frozen, pp [95% CI] | tIoU % | sIoU % | @.3 % | @.5 % |',
        '|---|---|---:|---|---:|---:|---:|---:|']
    names={'F':'Frozen','T':'Fast-only','S':'Spatial-only','A':'Full A','U':'Target-trained reference*'}
    for d in DIRECTIONS:
        z=s[d]['panels']['all']['metrics'];rows=rr[d]
        for a in ('F','T','S','A','U'):
            delta='—' if a=='F' else formatted(summary([dict(r,dd=r[a+'_v']-r['F_v']) for r in rows],['dd'])['metrics']['dd'])
            lines.append(f"| {d} | {names[a]} | {z[a+'_v']['mean']*100:.3f} | {delta} | "
                f"{z[a+'_t']['mean']*100:.3f} | {z[a+'_s']['mean']*100:.3f} | "
                f"{z[a+'_at03']['mean']*100:.3f} | {z[a+'_at05']['mean']*100:.3f} |")
    lines += ['', '*Target-trained is a supervised comparison reference, not a mathematical upper bound or fair TTA baseline. '
        'All methods use this run’s actual TA outputs; old TubeDETR/DeCoTA scores are not substituted. '
        'Native lowercase text preprocessing and original parent sampling grids are preserved.','',
        '## Persistent state versus current expert correction','',
        '| Direction | Full−Fast, all, pp [95% CI] | Full−Fast, nonexpert | Spatial−Frozen, nonexpert | Current Fast on inherited state |',
        '|---|---|---|---|---|']
    for d in DIRECTIONS:
        a=s[d]['panels']['all']['metrics'];n=s[d]['panels']['nonexpert']['metrics']
        lines.append(f"| {d} | {formatted(a['A_minus_T_v'])} | {formatted(n['A_minus_T_v'])} | "
            f"{formatted(n['S_minus_F_v'])} | {formatted(a['A_minus_S_v'])} |")
    lines += ['', 'Current outputs precede the current spatial write. Full−Fast therefore measures consequences of prior spatial '
        'state, including its effects on native temporal candidates. It is not uniquely a box-coordinate causal effect. '
        'Fast-disabled live controls verify identical spatial states, outputs and gradients for two scheduled inputs per direction; '
        'all persistent and inner-step state chains are audited. No query or probe offset is carried in Frozen/Fast-only.','',
        '## Where correct outputs are lost','']
    for d in DIRECTIONS:
        diag=read(PUB/d/'PIPELINE_DIAGNOSIS.json');vals=diag['paired_source_effects'];rows=rr[d];ex=[r for r in rows if r['expert_scheduled']]
        lines += [f'### {d}','',
            f"Fixed-time inherited boxes: {formatted(vals['delta_boxes']['metrics']['delta_boxes'])} pp; "
            f"inherited native interval: {formatted(vals['delta_native_interval']['metrics']['delta_native_interval'])} pp; "
            f"current Fast: {formatted(vals['delta_fast']['metrics']['delta_fast'])} pp. "
            'This is one ordered accounting path; the three summing contrasts do not establish independent causal modules.','']
        ordered=sorted(diag['path_transitions'],key=lambda name:-diag['path_transitions'][name]['gross_loss_pp'])
        primary_path=['IB_minus_F','S_minus_IB','A_minus_S']
        greatest=max(primary_path,key=lambda name:diag['path_transitions'][name]['gross_loss_pp'])
        lines += [f"The largest gross loss along this ordered three-stage path is **{greatest}** "
            f"({diag['path_transitions'][greatest]['gross_loss_pp']:.3f} pp). "
            'This identifies the largest measured damage on this accounting path, not a unique causal explanation.','']
        lines += ['| Transition | Gross gain pp | Gross loss pp | >5pp harm | >20pp harm | Correct→wrong at .3 / at .5 |',
            '|---|---:|---:|---:|---:|---|']
        for name in ordered:
            z=diag['path_transitions'][name]
            lines.append(f"| {name} | {z['gross_gain_pp']:.3f} | {z['gross_loss_pp']:.3f} | "
                f"{z['severe_harm_gt5pp']} | {z['severe_harm_gt20pp']} | "
                f"{z['0.3']['correct_to_wrong']}/{z['0.3']['correct_before']} / "
                f"{z['0.5']['correct_to_wrong']}/{z['0.5']['correct_before']} |")
        oracle=summary(ex,['temporal_regret','temporal_coverage_gap'])['metrics']
        lines += ['',f"On {len(ex)} scheduled arrivals: temporal selection regret "
            f"{formatted(oracle['temporal_regret'])} pp; fixed-space GT-time minus candidate oracle "
            f"{formatted(oracle['temporal_coverage_gap'])} pp. "
            'These conditional upper bounds locate available opportunity, not deployable gains.','',
            '| vIoU threshold | Good interval exists | Good interval missed | No good interval in support | Native correct destroyed |',
            '|---|---:|---:|---:|---:|']
        for threshold,z in diag['temporal'].items():
            lines.append(f"| {threshold} | {z['good_candidate_present']} | {z['good_candidate_missed']} | "
                f"{z['no_good_candidate']} | {z['native_correct_destroyed']} |")
        lines += ['', 'The preceding vIoU thresholds require the fixed spatial trajectory to be correct as well. '
            'An absent correct tube does not prove an absent correct time interval. Temporal-only counts separate these cases:', '',
            '| tIoU threshold | Good interval exists | Good interval missed | No good interval in support | Native time correct destroyed |',
            '|---|---:|---:|---:|---:|']
        for threshold,z in diag['temporal_tIoU'].items():
            lines.append(f"| {threshold} | {z['good_candidate_present']} | {z['good_candidate_missed']} | "
                f"{z['no_good_candidate']} | {z['native_correct_destroyed']} |")
        lines += ['', f"For final Full A output, replacing only time with GT at fixed A boxes adds "
            f"{formatted(vals['headroom_A_GT_time']['metrics']['headroom_A_GT_time'])} pp; "
            f"ideal GT boxes at fixed A time add {formatted(vals['headroom_A_GT_space']['metrics']['headroom_A_GT_space'])} pp. "
            'These are conditional evaluation replacements, not decoder inputs or additive independent contributions.', '']
        evidence=diag['evidence'];quality=[]
        for field,q in evidence['measures'].items():
            quality.append(field+': '+(formatted(q['metrics'][field])+'%' if q['sources'] else 'N/A'))
        lines += [f"Spatial evidence at {evidence['scheduled']} scheduled arrivals: {evidence['empty']} empty; "
            f"{evidence['nonempty_without_event']} nonempty but no reference on the GT event. "
            +'; '.join(quality)+'. Nonempty/event-eligible denominators are saved separately; missing evidence is not assigned zero localization quality.', '']
        lines += ['', '| Inner step | Calls | Empty evidence | Nonempty but no GT-event reference | Loss↓ / GT-time quality↓ | Useful top / harmful update | Net GT-time update pp |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for at,z in diag['spatial'].items():
            lines.append(f"| {int(at)+1} | {z['steps']} | {z['empty_evidence']} | {z['no_event_reference']} | "
                f"{z['loss_down_GT_time_harm']} | {z['good_top_but_harm']} | {z['mean_post_minus_pre_GT_v']*100:.3f} |")
        lines += ['', 'The useful-top count allows a tied top group; unique useful-top/harm counts are ' +
            ', '.join(f"step {int(at)+1}: {z['unique_good_top_but_harm']}" for at,z in diag['spatial'].items())+'.',
            '', '| Step / threshold | Correct support exists | No correct support | Reward-top group misses correct support | Correct central output destroyed by update |',
            '|---|---:|---:|---:|---:|']
        for at,z in diag['spatial'].items():
            for threshold,q in z['correct_support'].items():
                lines.append(f"| {int(at)+1} / {threshold} | {q['present']} | {q['absent']} | {q['top_group_misses']} | {q['correct_update_destroyed']} |")
        lines += ['', '| Step | Observed GT-event frame sIoU change, pp [95% CI] | Other dense GT-event frames |',
            '|---|---|---|']
        for at,z in diag['spatial'].items():
            cells=[]
            for field in ['observed_GT_sIoU_delta','unobserved_GT_sIoU_delta']:
                q=z[field];cells.append(formatted(q['metrics'][field]) if q['sources'] else 'N/A: no eligible GT-event frames')
            lines.append(f"| {int(at)+1} | {cells[0]} | {cells[1]} |")
        lines += ['', 'Step counts are repeated online arrivals, not independent sample sizes. Updates are evaluated on fixed '
            'actual output time and separately GT time; loss decrease is not task correctness. Immediate post-update diagnosis '
            'does not count as a formal current prediction or establish future transfer. Correct-support miss counts and empty '
            'evidence are retained, including negative tails. The nine probe upper bound does not bound arbitrary gradient adaptation.','',
            '#### Dataset characteristics and limits of attribution','']
        net=diag['arrival_update']['metrics']
        lines += [f"Net complete-arrival spatial update (all executed steps): actual A-time "
            f"{formatted(net['net_update_A_v'])} pp; GT-time {formatted(net['net_update_GT_v'])} pp. "
            'The latter removes current temporal masking; both are post-update diagnostics, not the sealed current output.', '']
        ch=['event_duration_seconds','event_fraction','GT_box_area_fraction','caption_words','observed_frames','input_grid_GT_frame_fraction']
        cc=summary(rows,ch)['metrics']
        lines.append('; '.join(f"{k}: {cc[k]['mean']:.4f}" for k in ch)+'.')
        for name,z in s[d]['characteristic_slices'].items():
            if z['sources']:
                lines.append(f"- {name}: {z['sources']} sources; Full−Frozen {formatted(z['metrics']['A_minus_F_v'])} pp; "
                    f"Full−Fast {formatted(z['metrics']['A_minus_T_v'])} pp.")
        lines += ['', 'Short-event (<25% observed clip) and small-box (<2% image area) slices are posthoc diagnostic associations, '
            'not online gates or causal effects. Exposure, one-query selection and available-media filtering limit population claims.','']
        lines += ['A one-source slice has no estimable between-source uncertainty; its degenerate bootstrap interval is not population evidence.', '']
        gap=s[d]['gap_recovery']
        if gap['ratio'] is None:
            lines.append('Domain-gap recovery is undefined: the supervised-reference denominator is nonpositive or tiny.')
        else:
            lines.append(f"Domain-gap recovery point estimate: {gap['ratio']*100:.2f}%; denominator "
                f"{gap['denominator']*100:.3f} pp [{gap['denominator_ci95'][0]*100:.3f}, {gap['denominator_ci95'][1]*100:.3f}]. "
                f"Stable positive denominator: {gap['stable_positive_denominator']}. The saved ratio CI is conditional on positive "
                'denominator bootstrap draws; an unstable denominator prevents a reliable recovered-gap claim.')
    lines += ['', 'The original [VidSTG paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Where_Does_It_Exist_Spatio-Temporal_Video_Grounding_for_Multi-Form_Sentences_CVPR_2020_paper.html) '
        'defines untrimmed object/relation grounding and multiple sentence forms. This selected historical pool is declarative; '
        'it does not evaluate unknown-object interrogatives. The official [HC-STVG dataset](https://github.com/tzhhhh123/HC-STVG) '
        'focuses on a person among multiple people in 20-second movie clips. These differences motivate checking identity ambiguity, '
        'event coverage and temporal extent, but do not by themselves prove why a particular update failed. Numerical evidence above '
        'takes priority over a dataset-level narrative.','',
        '## Cases, resources and verification','',
        'CASES.json saves the eight strongest positive and eight strongest negative arrivals for each specified pipeline contrast, '
        'using a fixed metric ordering. ROWS.json contains every anonymous source/order cell; STEP_ROWS.json contains every saved '
        'inner step. Summaries average eligible order cells within source and bootstrap sources with 10,000 paired draws. '
        'Recency bins distinguish no prior write; scheduled observations cannot imply a successful write when evidence is empty. '
        'Empty bins are N/A, not zero benefit.','',
        'RESOURCES.json records actual replay/backward/provider counts and worker/CPU wall times. Wall time is not pure GPU-kernel '
        'time. Exact metadata and receipt hashes permit historical expert-cache reuse; specialists remain frozen. Full H capture is '
        'not repeated per inner SGD step. Root audit independently checks dense metrics, rank/KL/SGD, masks, source state and target '
        'checkpoint bindings, temporal critic/selection, inner/persistent state chains, live arm parity and all anonymous aggregates. '
        'It does not recompute the complete model Jacobian on CPU.','',
        'Prediction barriers precede new GT exposure; GT is only for evaluation/diagnosis and never used to select updates, samples '
        'or parameters. No new backbone, loss, memory, teacher, scorer, corruption or followup job was launched.']
    report=ROOT/'docs/TA_CROSS_DOMAIN_QUALIFICATION_REVIEW.md';report.write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.5),layout='constrained')
    palette=['#8c929a','#c8953e','#4385b8','#d65d58','#62a48c']
    for ax,d in zip(axes,DIRECTIONS):
        z=s[d]['panels']['all']['metrics'];values=np.array([z[a+'_v']['mean']*100 for a in ARMS])
        lo=np.array([z[a+'_v']['ci95'][0]*100 for a in ARMS]);hi=np.array([z[a+'_v']['ci95'][1]*100 for a in ARMS])
        ax.barh(range(5),values,color=palette,height=.65,xerr=np.stack([np.maximum(values-lo,0),np.maximum(hi-values,0)]),error_kw={'capsize':3,'lw':1})
        ax.set_yticks(range(5),[names[a] for a in ARMS]);ax.invert_yaxis();ax.set_xlabel('Source-macro vIoU (%)')
        ax.set_title('Vid → HC2' if d=='vid_to_hc2' else 'HC2 → Vid');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    for ext in ['png','pdf']:fig.savefig(PUB/f'qualification.{ext}',dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
    fs=['delta_boxes','delta_native_interval','delta_fast'];labs=['Inherited boxes','Inherited native time','Current Fast']
    for ax,d in zip(axes,DIRECTIONS):
        z=s[d]['panels']['all']['metrics'];v=np.array([z[k]['mean']*100 for k in fs]);ci=np.array([z[k]['ci95'] for k in fs])*100
        ax.bar(range(3),v,color=['#4385b8','#789cc0','#c8953e'],yerr=np.stack([np.maximum(v-ci[:,0],0),np.maximum(ci[:,1]-v,0)]),capsize=3)
        ax.axhline(0,color='#444',lw=.8);ax.set_xticks(range(3),labs,rotation=15);ax.set_ylabel('Paired vIoU change (pp)')
        ax.set_title('Vid → HC2' if d=='vid_to_hc2' else 'HC2 → Vid');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for ext in ['png','pdf']:fig.savefig(PUB/f'pipeline_accounting.{ext}',dpi=220)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    for ax,d in zip(axes,DIRECTIONS):
        steps=st[d];at=sorted({r['step'] for r in steps});vals=[];low=[];high=[]
        for j in at:
            q=[dict(r,delta=r['post_GT_v']-r['pre_GT_v']) for r in steps if r['step']==j]
            z=summary(q,['delta'])['metrics']['delta'];vals.append(z['mean']*100);low.append(z['ci95'][0]*100);high.append(z['ci95'][1]*100)
        v=np.array(vals);ax.errorbar(np.array(at)+1,v,yerr=np.stack([np.maximum(v-np.array(low),0),np.maximum(np.array(high)-v,0)]),fmt='o-',color='#4385b8',capsize=3)
        ax.axhline(0,color='#444',lw=.8);ax.set_xlabel('Spatial inner step');ax.set_xticks(np.array(at)+1)
        ax.set_ylabel('Fixed GT-time update vIoU (pp)');ax.set_title('Vid source, K1 → HC2' if d=='vid_to_hc2' else 'HC2 source, K8 → Vid')
        ax.grid(alpha=.15)
    for ext in ['png','pdf']:fig.savefig(PUB/f'spatial_execution.{ext}',dpi=220)
    plt.close(fig)
    write(PUB/'FIGURE_MANIFEST.json',dict(files={f.name:sha(f) for f in PUB.glob('*.*') if f.suffix in ['.png','.pdf']}))
    print('REPORT_AND_FIGURES',report,flush=True)

if __name__=='__main__':run()
