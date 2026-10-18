"""Anonymous postseal analysis/report. No fitting, target GT loading or inference.

Also refines competition diagnostics to distinct self-consistent association
modes. The sealed runner's nearest distinct proposal need not be a stationary
mode; it is preserved, explicitly named as such, never used as a reliability gate.
"""
import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,itertools
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_joint_robust_correction32_v1 import D,P,BASES,ARMS,verify


def modes(pools,scale,cutoff,selected_delta):
    distinct={}
    for assignment in itertools.product(*[range(-1,len(v))for v in pools]):
        active=[v[k]for v,k in zip(pools,assignment)if k>=0]
        d=np.mean(active,axis=0)if active else np.zeros(4)
        distances=[np.square((v-d)/scale).sum(1)for v in pools]
        choice=tuple(int(v.argmin())if len(v)and v.min()<cutoff else -1 for v in distances)
        use=[v[k]for v,k in zip(pools,choice)if k>=0]
        if not use:continue
        if np.linalg.norm((np.mean(use,axis=0)-d)/scale)>1e-8:continue
        obj=float(sum(min(cutoff,float(v.min()))if len(v)else cutoff for v in distances))
        distinct[choice]=(obj,d)
    ranked=sorted(distinct.items(),key=lambda x:(x[1][0],x[0]));delta=np.asarray(selected_delta)
    if not ranked:return dict(mode_count=0,distinct_mode_gap=None,normalized_mode_separation=None,delta_direction_cosine=None)
    alt=next((v for k,v in ranked if np.linalg.norm((v[1]-delta)/scale)>1e-9),None)
    if alt is None:return dict(mode_count=len(ranked),distinct_mode_gap=None,normalized_mode_separation=None,delta_direction_cosine=None)
    x=delta/scale;y=alt[1]/scale;den=float(np.linalg.norm(x)*np.linalg.norm(y))
    return dict(mode_count=len(ranked),distinct_mode_gap=float(alt[0]-ranked[0][1][0]),
        normalized_mode_separation=float(np.linalg.norm(y-x)),
        delta_direction_cosine=float(np.dot(x,y)/den)if den else None)


def main():
    import matplotlib.pyplot as plt
    verify(inputs=True);x=read(P/'RESULTS.json');audit=read(P/'ROOT_INDEPENDENT_AUDIT.json');assert audit['status']=='pass'
    cal=read(D/'SOURCE_CALIBRATION.json');derived={};case=[]
    for r in x['rows']:
        z=load(D/'predictions'/(r['alias']+'.pt'));sc=cal['source_domains'][cal['target_to_source'][r['dataset']]]
        bybase={}
        for base in BASES:
            b=z['bases'][base]['Base'];chart=np.concatenate((b[:,:2],np.log(b[:,2:])),axis=1)
            pp=[np.concatenate((f['boxes'][:,:2],np.log(f['boxes'][:,2:])),axis=1)-chart[f['position']]for f in z['frames']]
            cm=modes(pp,np.array(sc['scale']),sc['truncation'],z['bases'][base]['estimate_C']['delta'])
            assert cm['distinct_mode_gap']is None or cm['distinct_mode_gap']>=-1e-9
            bybase[base]=cm
        case.append(dict(alias=r['alias'],dataset=r['dataset'],competition=bybase))
    for ds in ['hc2','vidstg']:
        rr=[r for r in x['rows']if r['dataset']==ds];derived[ds]={}
        for base in BASES:
            gg=[g for r in rr for g in r['geometry'][base]if g['exact_GT_available']];selected=[g for g in gg if g['selected_index']>=0]
            costs=[(r['behavior'][base]['B_robust_top1']['objective'],r['behavior'][base]['C_joint']['objective'])for r in rr]
            assert all(c<=b+1e-8 for b,c in costs)
            gcounts=dict(known_reference_positions=len(gg),unknown_reference_positions=sum(not g['exact_GT_available']for r in rr for g in r['geometry'][base]),
                retained_pool_good_geometric_count=sum(g['has_geometric_IoU_ge_half']for g in gg),
                original_Top1_good_geometric_count=sum(g['original_Top1_IoU']>=.5 for g in gg),
                C_selected_good_geometric_count=sum(g['selected_geometric_IoU_ge_half']for g in gg),
                available_good_but_not_selected=sum(g['available_good_but_not_selected']for g in gg),
                no_good_in_retained_pool=sum(not g['has_geometric_IoU_ge_half']for g in gg),
                C_selected_known_references=len(selected),
                original_Top1_mean_IoU=float(np.mean([g['original_Top1_IoU']for g in gg])),
                best_retained_mean_IoU=float(np.mean([g['best_retained_IoU']for g in gg])),
                C_selected_conditional_mean_IoU=float(np.mean([g['selected_IoU']for g in selected]))if selected else None,
                C_selected_fixed_support_mean_IoU_abstention_zero=float(np.mean([g['selected_IoU']or 0. for g in gg])),
                selected_conditional_and_fixed_support_denominators_differ=True,geometric_quality_not_identity=True)
            harms=[]
            for r in rr:
                v=r['values'][base];dv=100*(v['C_joint']['metrics']['v']-v['Base']['metrics']['v']);du=100*(v['C_joint']['supports']['unobserved']['s']-v['Base']['supports']['unobserved']['s'])
                ggq=[g for g in r['geometry'][base]if g['exact_GT_available']and g['selected_index']>=0]
                if ggq and all(g['selected_geometric_IoU_ge_half']for g in ggq)and(dv<0 or du<0):
                    harms.append(dict(alias=r['alias'],known_selected=len(ggq),other_unknown_selected=sum(g['selected_index']>=0 and not g['exact_GT_available']for g in r['geometry'][base]),delta_v_pp=dv,delta_unobserved_pp=du))
            derived[ds][base]=dict(geometry=gcounts,mean_truncated_objective_B=float(np.mean([b for b,c in costs])),
                mean_truncated_objective_C=float(np.mean([c for b,c in costs])),
                objective_nonincrease_C_vs_B_all_queries=True,
                all_known_selected_geometric_good_but_output_harmed=harms,
                note='unknown/event-outside references remain unknown; known-good subset is not proof all selected references are good')
    put(P/'DIAGNOSTICS.json',dict(status='complete',groups=derived,per_query_competition=case,
        competition_clarification='RESULTS competitor_gap is nearest distinct enumerated proposal; these derived gaps restrict to distinct self-consistent association modes. Neither was used for selection or gating.',
        old_outputs_and_scores_unchanged=True,new_fitting_GT_or_model_calls=0))
    put(D/'COMPETITION_DIAGNOSTIC_CLARIFICATION.json',dict(status='clarified',raw_runner_fields_preserved=True,
        runner_gap='nearest distinct finite proposal; not necessarily an independently stationary association mode',
        derived_gap='DIAGNOSTICS per_query_competition: distinct self-consistent candidate/outlier assignment modes',
        scientific_prediction_or_metric_changes=False,no_new_reliability_gate=True))
    old=read(ROOT/'results/stvg_error_state/2026-10-18/cached32/ACTUAL_PROTOCOL.json')
    sourcecal={ds:{k:v for k,v in cc.items()if k!='model_hash'}for ds,cc in cal['source_domains'].items()}
    actual=dict(status='complete',count=32,development=True,backbone=old['backbone'],
        source_domains_and_checkpoint_hashes=old['source_domains_and_checkpoint_hashes'],cached_DINO_expert=old['cached_DINO_expert'],
        source_calibration=sourcecal,target_to_source_calibration=cal['target_to_source'],source_train=72,source_validation_used=0,
        original_admitted_frames=dict(hc2=56,vidstg=50),retained_candidates=dict(hc2=168,vidstg=144),
        pool='original <=3 post-NMS probe boxes at original admitted frames; no raw900 recovery or new admission',
        A_exact_old_global_correction=True,D_exact_C_choices_and_delta_with_sizes_identity=True,
        candidate_selection='exact finite global truncatedLS; no absolute-delta/Student-distance prior',
        source_scale_and_cutoff_locked_before_target_outputs=True,all32_outputs_sealed_before_target_GT_and_old_labels=True,
        old_labels='model-assisted original Top1 only; unknown roles conservatively UN; no labels for alternative identities',
        competition_diagnostic_clarification=read(D/'COMPETITION_DIAGNOSTIC_CLARIFICATION.json'),
        no_new_GPU_models_experts_training_Student_updates=True,new_pixels_or_identity_labels=0,
        CPU_generation_seconds=read(D/'GLOBAL_PREDICTION_BARRIER.json')['CPU_generation_seconds'],
        CPU_score_seconds=read(D/'CPU_COMPLETION.json')['CPU_score_seconds'],cached_CPU_cost_not_online_end_to_end=True,
        result_sha256=sha(P/'RESULTS.json'),calibration_sha256=sha(D/'SOURCE_CALIBRATION.json'),
        code_pins=read(D/'RUNTIME_LOCK.json')['pins'],registries_unchanged=True,no_confirmation_dense_OPD_or_new_gate=True)
    put(P/'ACTUAL_PROTOCOL.json',actual)
    fig,axs=plt.subplots(2,2,figsize=(13,9));fig.subplots_adjust(left=.085,right=.97,bottom=.12,top=.88,wspace=.3,hspace=.4)
    colors=['#6d7d8b','#b08a47','#317f9a','#668c58']
    for i,ds in enumerate(['hc2','vidstg']):
        for j,ep in enumerate(['v','unobserved']):
            ax=axs[i,j];g=x['groups'][ds]['Before']['paired']
            for k,arm in enumerate(ARMS):
                m=g[arm+'_minus_Base'][ep];lo,hi=m['CI95'];ax.errorbar(k,m['mean'],yerr=[[m['mean']-lo],[hi-m['mean']]],fmt='o',color=colors[k],capsize=5,ms=7)
            ax.axhline(0,color='.4',lw=.8);ax.set_xticks(range(4),['A mean','B Top1 robust','C joint','D center only'],rotation=10,fontsize=9)
            ax.set_ylabel('Change vs Before (pp)');ax.set_title(f"{['HC','VidSTG'][i]} — {['full vIoU','unobserved actual-frame sIoU'][j]}",loc='left',fontsize=11)
            ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Original32 development sources · fixed final MBR · global output correction',fontsize=14)
    fig.text(.085,.035,'Paired parent bootstrap95% intervals; fixed inherited predictions, not new online-TTA replay.\nNo training or new expert calls. Source-only calibration fixed before all target outputs.',fontsize=9)
    fig.savefig(P/'FOUR_ARMS.svg');(D/'root_review').mkdir(exist_ok=True);fig.savefig(D/'root_review/FOUR_ARMS.png',dpi=140);plt.close(fig)
    def fmt(d):return f"{d['mean']:+.6f} [{d['CI95'][0]:+.6f}, {d['CI95'][1]:+.6f}]"
    lines=['# Multi-candidate joint robust correction: fixed development32','','## Finding','',
        'C improves HC relative to Top1 mean and Top1 robust estimation, but loses VidSTG gains. C has positive mean full-vIoU and unobserved actual-frame sIoU changes in both domains; it nevertheless fails the locked two-domain superiority comparison against A/B. Stop this fixed geometry-only candidate estimator; no threshold/kernel sweep, new model, confirmation or dense OPD is launched. D is an explanatory size intervention, not a selected method.',
        '', '## Frozen setup and actual computation','',
        'Same16 exposed HC and16 exposed Vid parents; original saved query-start Before state primary, Source sensitivity separate; final-distribution MBR, original planned4 positions, original admitted DINO frame set, normalized center/log-size chart, four-dimensional constant correction and whole-trajectory application fixed. This is an output-layer experiment, not new Student parameter adaptation or online stream replay.',
        '', 'A exactly replays old DINO_full mean correction. B uses identical Top1 references with truncatedLS. C uses the original retained <=3 candidate boxes in each of the SAME admitted frames plus an outlier option. D shares C candidate choices, outlier assignments and full fitted delta, but applies only center components; original sizes stay bit exact. No new frame admission, target-score reranking, NMS, raw900 pool expansion or target-GT availability filter. There are56/50 admitted HC/Vid frames with168/144 retained candidate boxes; all106 original Top1 boxes reproduce exactly and separately match the old-probe recapture. The recapture is not substituted as a new expert output.',
        '', 'Source calibration reads only existing source-train36/domain caches and exact GT at original planned positions. Source-HC35 queries/113 positions and source-Vid22 queries/76 positions have>=2 known anchors; remaining queries cannot estimate within-query variation and are reported. Source-validation is unused. Each target uses the opposite source-domain calibration matching its Student checkpoint. The source/target parent sets are disjoint. Coordinate scales are parent-balanced within-query true-correction RMS; the cutoff is its parent-balanced95th percentile of normalized squared residuals. This is an explicit additional source-data requirement and engineering convention, not a target-tuned threshold or calibrated probability. See ACTUAL_PROTOCOL for all numeric scales/cutoffs and checkpoint hashes.',
        '', 'The exact global truncatedLS optimum is enumerated over at most4^4 candidate/outlier assignments. There is one vote per frame, no penalty for a correction far from the Student, no minimum-support whole-query rejection gate, and no newly fitted network. Single-frame-supported solutions are retained. All32 outputs for A/B/C/D and both bases were sealed before new target-GT scoring or old-label analysis.',
        '', '## Primary results: differences against original Before (pp)','',
        '| Arm | HC full ΔvIoU [95% CI] | HC unobserved ΔsIoU [95% CI] | Vid full ΔvIoU [95% CI] | Vid unobserved ΔsIoU [95% CI] |','|---|---:|---:|---:|---:|']
    for arm in ARMS:
        lines.append('| '+arm+' | '+' | '.join(fmt(x['groups'][ds]['Before']['paired'][arm+'_minus_Base'][ep])for ds in ['hc2','vidstg']for ep in ['v','unobserved'])+' |')
    lines+=['','All16 parents/domain remain, including Vid two no-reference no-ops. Unobserved means actual Decoder frames with exact GT, excluding all four planned positions. Missing GT stays unknown. Full s/t/R03/R05, fixed-time equality, within-MBR/interpolated supports, Source sensitivity and common-scale strata are in RESULTS.',
        '', '| Contrast | HC Δv / unobserved Δs pp | Vid Δv / unobserved Δs pp |','|---|---:|---:|']
    for a,b in [('C_joint','A_mean'),('C_joint','B_robust_top1'),('D_C_center','C_joint')]:
        lines.append('| '+a+' − '+b+' | '+' | '.join(' / '.join(f"{x['groups'][ds]['Before']['paired'][a+'_minus_'+b][ep]['mean']:+.6f}"for ep in ['v','unobserved'])for ds in ['hc2','vidstg'])+' |')
    lines+=['', 'D improves unobserved means slightly in both domains, but HC full-v is lower than C. Source sensitivity reverses some D−C signs. These results do not justify disabling sizes or choosing a different variant by domain.',
        '', '## Is the retained pool useful, and can geometry choose it?','',
        '| Domain | Known original admitted positions | Pool has IoU>=.5 | Original Top1 IoU>=.5 | C selected IoU>=.5 | Available good but not selected | No good retained candidate |','|---|---:|---:|---:|---:|---:|---:|']
    for ds in ['hc2','vidstg']:
        c=derived[ds]['Before']['geometry'];lines.append('| '+ds+' | '+' | '.join(str(c[k])for k in ['known_reference_positions','retained_pool_good_geometric_count','original_Top1_good_geometric_count','C_selected_good_geometric_count','available_good_but_not_selected','no_good_in_retained_pool'])+' |')
    lines+=['', 'These counts use only exact-GT anchors AFTER output seal and are geometry, not identity labels. Pool coverage is30/43 HC and22/24 Vid; C gets22 and8 good known references, while Top1 has20 and11. Thus candidates exist in many locations but geometry-only association still fails to select them, particularly in Vid. The absent-candidate statement is only about the original retained <=3 pool, not all raw DINO detections. Original Top1/best-retained/C-selected-conditional mean IoU is.454628/.658595/.499967 HC and.423859/.790515/.363094 Vid; Vid C abstains on3 known positions, so conditional selected means use a different denominator. Fixed-support abstention-zero means are separately saved.',
        '', 'C never increases the truncated geometric objective relative to B on any query. However, Vid actual tube gains are lower. This directly shows that a better geometric-consistency objective is not a certificate of target correctness. Old model-assisted WI/CP/MX/CI labels are used only for the ORIGINAL Top1; six old role-unknown queries remain UN. Alternative candidates have no new identity labels, and IoU does not certify the participant or role.',
        '', '## Protection, active correction and negative tails','',
        'HC C rejects0 of56 admitted frames and makes0 query no-ops; it changes28 candidate choices,26 below the original per-winner.35 score. HC improvement is therefore not produced by simply refusing to update, but low-score alternatives remain a material risk. C improves12/16 HC queries over Base, reduces<-5pp full-v harms from5(A) to2, and rescues4 A-negative queries to nonnegative full-v. It also introduces severe failure H011: A+1.054pp full-v/+1.425pp unobserved becomes C−22.536/−59.327. Without new pixel review this is a geometric failure, not a newly certified identity error.',
        '', 'HC H002 moves A−26.171/−34.112 to C+6.818/+8.074, and H010−14.197/−8.286 to+2.017/+1.692. H005 largely recovers but still has−1.244pp unobserved loss. H009 gets worse. All cases remain in the table; no query was routed by GT or old labels.',
        '', 'Vid C rejects9 of50 frames and has the same2 no-reference query no-ops as A. It changes3 candidate choices; the remaining difference from A also includes robust rejection/averaging of Top1 frames. It rescues one A-negative query but creates a new<-5pp full-v failure (V007). V010 loses much of a good A correction: full-v gain+8.932 ->+1.255pp, unobserved+57.588 ->+7.921pp. B alone is also below A, so top1 robustness did not establish a simpler two-domain improvement.',
        '', 'C HC saved-score leave-one full-v mean range is[+.665227,+4.598835]pp, but unobserved range[−.995676,+5.625624]pp; the positive HC unobserved mean is not robust to every single-parent omission. The primary HC C bootstrap intervals cross0. All severe tails, positive/negative sums, parent signs, source sensitivity and small strata are retained. Intervals describe a repeatedly exposed32-parent development panel and fixed inherited trajectories, not independent confirmation or altered-stream replay.',
        '', '## Reliability quantities and an audit clarification','',
        'Support count, candidate/outlier choices and leave-one normalized-delta distances were saved without GT. HC C has11 four-frame,2 three-frame and3 two-frame supports; Vid includes one single-frame support, which is inherently ambiguous. The sealed runner also saved the nearest distinct enumerated proposal gap. Auditing identified that such a proposal need not itself be a stationary association explanation. Original fields, predictions and metrics are preserved; DIAGNOSTICS additionally enumerates distinct self-consistent candidate/outlier modes and records their objective gap, normalized delta separation and direction cosine. Neither diagnostic was used as a selection/rejection threshold, and neither is called a correctness probability.',
        '', f"Independent auditing checks{audit['independent_assignment_games']} assignment/mean games and{audit['conditional_assignments_enumerated']} conditional assignments, {audit['leave_one_checks']} leave-one checks, {audit['output_reconstructions_and_identity_checks']} output/identity reconstructions (maximum box error{audit['maximum_box_error']}), {audit['independent_metric_and_actual_frame_IoU_checks']} metric/actual-frame checks and{audit['aggregate_CI_quality_checks']} quality/aggregate/CI checks. Source calibration is independently recomputed. A is bit-exact to old outputs; D shares C selection and changes no size bits. All old inputs, code pins and production registries remain unchanged.",
        '', '## Decision and scope','',
        'C satisfies positive two-domain mean changes against Base, but fails two-domain improvement against A and B. This fixed source-calibrated geometry-only estimator does not advance to independent confirmation or dense OPD. The experiment supports a limited HC candidate-selection opportunity and exposes unhandled wrong-consensus/constant-correction failures; it does not validate geometric consistency as a semantic association criterion. No source-scale, cutoff, pool or robust-kernel search follows, and no production method changes.',
        '', f"New GPU/model/expert/training/Student-update/visual-label counts are0. Cached CPU generation took{actual['CPU_generation_seconds']:.6f}s and scoring{actual['CPU_score_seconds']:.6f}s, excluding original captures, calibration preflight and independent audit. These are not fresh deployment end-to-end costs.",
        '', '[Full anonymous results](RESULTS.json) · [Decision](DECISION.json) · [Coverage and true competing modes](DIAGNOSTICS.json) · [Actual protocol](ACTUAL_PROTOCOL.json) · [Independent audit](ROOT_INDEPENDENT_AUDIT.json) · [Aggregate figure](FOUR_ARMS.svg)',
        '', 'Raw GT coordinates, original query/parent text, media, weights, raw candidate tensors and correction vectors are excluded from the public export.']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (P/'README.md').write_text('# Four-arm robust correction, development32\n\n[Report](REPORT.md), [all results](RESULTS.json), [actual protocol](ACTUAL_PROTOCOL.json).\n\nC improves HC but loses Vid relative to A/B; no automatic confirmation, tuning or dense OPD.\n')
    put(P/'VERIFICATION.json',dict(status='pass',count=32,independent_audit=audit,
        all32_sealed_before_target_GT=True,source_only_calibration=True,all_negative_neutral_unknown_retained=True,
        competition_diagnostic_refined_without_prediction_changes=True,production_registries_unchanged=True))
    files=['vg_tta/stvg_joint_robust_correction32_v1.py','scripts/run_stvg_joint_robust_correction32_v1.py',
        'scripts/score_stvg_joint_robust_correction32_v1.py','scripts/audit_stvg_joint_robust_correction32_v1.py',
        'scripts/report_stvg_joint_robust_correction32_v1.py','scripts/publish_stvg_joint_robust_correction32_v1.py',
        'protocols/stvg_joint_robust_correction32_v1.md']
    files += [str((P/f).relative_to(ROOT))for f in ['RESULTS.json','DECISION.json','DIAGNOSTICS.json','ACTUAL_PROTOCOL.json','ROOT_INDEPENDENT_AUDIT.json','VERIFICATION.json','REPORT.md','README.md','FOUR_ARMS.svg']]
    put(D/'PUBLIC_FILES.json',dict(files=files))
    print('REPORT',len(files),'files',flush=True)


if __name__=='__main__':main()
