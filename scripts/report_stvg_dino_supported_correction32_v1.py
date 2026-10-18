"""Report the sole added score-supported group and its prelocked diagnostics."""
import os
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.run_stvg_joint_event_critic_v1 import read,put,sha
from scripts.run_stvg_dino_supported_correction32_v1 import D,P,OLD,OLDP,BASES,verify


def main():
    import matplotlib.pyplot as plt
    verify(inputs=True);x=read(P/'RESULTS.json');audit=read(P/'ROOT_INDEPENDENT_AUDIT.json');assert audit['status']=='pass'
    bc=read(D/'BETA_CALIBRATION.json');prior=read(OLDP/'ACTUAL_PROTOCOL.json');decision=read(P/'DECISION.json')
    source_public={k:v for k,v in bc.items()if k not in ['source_train_diagnostics','CPU_calibration_seconds']}
    source_public['source_train_transform_diagnostics_summary']={ds:dict(queries=sum(r['dataset']==ds for r in bc['source_train_diagnostics']),
        retained_candidates=sum(r['candidates']for r in bc['source_train_diagnostics']if r['dataset']==ds))for ds in ['hc2','vidstg']}
    put(P/'SOURCE_BETA_CALIBRATION.json',source_public)
    diagnostics={}
    for ds in ['hc2','vidstg']:
        rows=[r for r in x['rows']if r['dataset']==ds];diagnostics[ds]={}
        for base in BASES:
            known=[g for r in rows for g in r['geometry'][base]if g['exact_GT_available']];selected=[g for g in known if g['E_index']>=0]
            unchanged_B=sum(r['values'][base]['E_DINO_support']==r['values'][base]['B_robust_top1']for r in rows)
            unchanged_C=sum(r['values'][base]['E_DINO_support']==r['values'][base]['C_joint']for r in rows)
            diagnostics[ds][base]=dict(known_original_admitted_positions=len(known),E_selected_known=len(selected),
                unknown_original_admitted_positions=sum(not g['exact_GT_available']for r in rows for g in r['geometry'][base]),
                pool_has_geometric_IoU_ge_half=sum(g['best_retained_IoU']>=.5 for g in known),
                Top1_good=sum(g['Top1_IoU']>=.5 for g in known),C_selected_good=sum(g['C_selected_IoU']is not None and g['C_selected_IoU']>=.5 for g in known),
                E_selected_good=sum(g['E_selected_IoU']is not None and g['E_selected_IoU']>=.5 for g in known),
                E_misses_available_good=sum(g['best_retained_IoU']>=.5 and(g['E_selected_IoU']is None or g['E_selected_IoU']<.5)for g in known),
                E_fixed_support_mean_reference_IoU_abstention_zero=float(np.mean([g['E_selected_IoU']or 0. for g in known])),
                E_conditional_selected_mean_reference_IoU=float(np.mean([g['E_selected_IoU']for g in selected]))if selected else None,
                E_same_all_saved_metrics_as_B=unchanged_B,E_same_all_saved_metrics_as_C=unchanged_C,
                note='Metric identity is not a claim of bit-identical boxes; the independent audit reconstructs actual E boxes separately.',
                geometric_quality_not_new_identity_annotation=True)
    put(P/'DIAGNOSTICS.json',dict(status='complete',groups=diagnostics,
        source_grid_fixed_before_target=True,candidate_acceptance_decomposition='presealed two-factor output intervention, finite Shapley averaging both orders',
        latent_rejected_candidate_convention_not_unique_causal_semantics=True,no_new_identity_labels=True))
    actual=dict(status='complete',count=32,new_formal_groups=1,primary='Before/final_MBR',Source_sensitivity=True,development_exposed=True,
        backbone=prior['backbone'],source_domains_and_checkpoint_hashes=prior['source_domains_and_checkpoint_hashes'],cached_DINO_expert=prior['cached_DINO_expert'],
        original_admitted_frames=prior['original_admitted_frames'],retained_candidates=prior['retained_candidates'],
        beta=bc['selected_beta'],beta_shared_both_domains=True,beta_grid=bc['grid'],zero_reference_not_eligible=True,
        score_transform='log(max(max(scores,1e-12)))-log(max(score,1e-12))',penalty='beta*old cutoff*relative log cost',
        scores='original DINO target-token occurrence scores in context branch; original score in non-context branch; not event/role probabilities',
        old_geometry_source_calibration_unchanged=True,old_geometry_source_calibration=prior['source_calibration'],
        source_train_only_transform_diagnostics=72,source_validation_only_shared_beta_selection=24,
        source_validation_primary='actual unobserved Decoder sIoU change, equal-domain/equal-query',source_validation_secondary='full vIoU under FIXED source Native time, not invented source MBR',
        all_new_E_and_factorial_diagnostic_outputs_sealed_before_target_GT=True,old_ABCD_results_sha256=sha(OLDP/'RESULTS.json'),
        no_new_models_experts_training_Student_updates=True,no_new_visual_labels=True,
        CPU_cost=dict(source_calibration_seconds=bc['CPU_calibration_seconds'],target_generation_seconds=read(D/'GLOBAL_PREDICTION_BARRIER.json')['CPU_generation_seconds'],
            serialized_scoring_pass_seconds=read(D/'CPU_COMPLETION.json')['CPU_score_seconds'],
            first_unserialized_CPU_scoring_pass_repeated_for_export=True,not_fresh_online_end_to_end=True),
        JSON_recovery=read(D/'recovery/json_scalar002/RECOVERY.json'),
        original_prediction_and_scorer_pins_unchanged=True,code_pins=read(D/'RUNTIME_LOCK.json')['pins'],
        production_registries_unchanged=True,no_more_combination_rule_search=True,no_dense_OPD_or_confirmation_started=True)
    put(P/'ACTUAL_PROTOCOL.json',actual);put(P/'VERIFICATION.json',dict(status='pass',count=32,independent_audit=audit,
        source_choice_before_target_outputs=True,target_outputs_before_target_GT=True,ABCD_reused_unchanged=True,
        JSON_scalar_conversion_only_no_scientific_change=True,all_signed_negative_noop_unknown_retained=True))
    fig,axs=plt.subplots(2,2,figsize=(12,8.5));fig.subplots_adjust(left=.085,right=.975,bottom=.12,top=.88,hspace=.38,wspace=.3)
    for i,ds in enumerate(['hc2','vidstg']):
        g=x['groups'][ds]['Before']
        for j,ep in enumerate(['v','unobserved']):
            ax=axs[i,j]
            for k,arm in enumerate(['A_mean','B_robust_top1','C_joint','E_DINO_support']):
                # Paired old A/B/C vs Base intervals are copied from frozen old results.
                stat=g['E_minus']['Base'][ep]if arm=='E_DINO_support'else read(OLDP/'RESULTS.json')['groups'][ds]['Before']['paired'][arm+'_minus_Base'][ep]
                m=stat['mean'];lo,hi=stat['CI95'];ax.errorbar(k,m,yerr=[[m-lo],[hi-m]],fmt='o',color=['#718392','#ab8850','#328199','#b65e52'][k],capsize=4,ms=7)
            ax.axhline(0,color='.4',lw=.8);ax.set_xticks(range(4),['A mean','B robust','C geometry','E + DINO'],rotation=10)
            ax.set_title(f"{['HC','VidSTG'][i]} — {['full vIoU','unobserved actual-frame sIoU'][j]}",loc='left',fontsize=10)
            ax.set_ylabel('Change vs Before (pp)');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('One DINO-support supplement · source-only shared beta = 1/16',fontsize=14)
    fig.text(.085,.035,'Original32 exposed development parents; all targets sealed before GT. Paired95% bootstrap.\nFixed Before/MBR/global4D correction; zero new model, expert or Student-update calls.',fontsize=9)
    fig.savefig(P/'DINO_SUPPORT.svg');(D/'root_review').mkdir(exist_ok=True);fig.savefig(D/'root_review/DINO_SUPPORT.png',dpi=140);plt.close(fig)
    def fp(p):return f"{p['mean']:+.6f} [{p['CI95'][0]:+.6f},{p['CI95'][1]:+.6f}]"
    lines=['# One DINO-supported supplement to joint four-parameter correction','','## Result and fixed decision','',
        'Adding source-calibrated relative DINO support improves HC and improves over geometry-only C in Vid, but does NOT restore Vid performance relative to Top1 mean A. The single supplement fails the prelocked two-domain mean screen. Stop this family of simple geometry+DINO-score combination rules; no further beta/transform/kernel/gate search, independent confirmation, dense OPD or production change. Four-parameter correction remains a research representation; this is not a universal rejection of all correction methods.',
        '', '## What changed, and what did not','',
        'E is the only new formal method group. It reuses original exposed HC16/Vid16, saved Before query-start trajectories with inherited state, Source sensitivity, four planned Native positions, ORIGINAL admitted56/50 frames and168/144 retained candidates, constant(cx,cy,logw,logh) correction, full-trajectory scope and final-distribution MBR. Frozen A/B/C/D outputs and scores are reused unchanged; no old method is rerun as a new experiment.',
        '', 'Only the candidate cost changes: l_ik=log(s_max)-log(s_ik), with1e-12 numerical floor; E minimizes sum_i min{lambda,min_k[d_ik²+beta*lambda*l_ik]}. The old geometric scale and lambda remain bit-identical; beta is one dimensionless value shared across domains. Top1 relative cost is0; low-score candidates remain available, without a.35 filter. These scores are original DINO target-token occurrence support (context branch) or its original non-context score, not complete-event/role judgments, calibrated identity probabilities or geometric variances. No Student-distance prior is added.',
        '', 'Exact finite candidate/outlier assignment enumeration is used. For fixed assignments the added score cost is constant in delta, so the conditional arithmetic mean remains the exact four-parameter optimum. Neither energy reduction nor semantic-score improvement is treated as a task metric.',
        '', '## Source-only choice before any new target output','',
        'Original source-train72 is used only for score-range/pool/transform diagnostics, not new network training. The original exposed source-validation12/domain chooses ONE shared positive beta. The complete predeclared grid is{1/16,1/8,1/4,1/2,1,2}; beta0 is an explicitly reported reference control, not an eligible supplement. All source-grid outputs are sealed before source evaluation. Selection maximizes equal-domain/equal-query unobserved actual-Decoder sIoU gain; fixed source Native full-v is only a tie-break. The source domains have no parent overlap with target32. No target score, old identity label, per-domain beta, temperature or target gate is used in generation.',
        '', '| beta | Pooled source-validation unobserved gain pp | Pooled source-validation Native full-v gain pp |','|---:|---:|---:|']
    for b in bc['grid']:
        v=bc['pooled'][str(b)];lines.append(f"| {b:g} | {v['unobserved_gain_pp']:+.6f} | {v['full_v_gain_pp']:+.6f} |")
    lines+=['', 'The selected beta is1/16, also the smallest predeclared positive value. No lower-beta extension follows. The source pooled gain increases7.550803 ->7.882438pp relative to beta0; this is calibration evidence, not independent source validation or proof of target transfer. All individual source rows and both domains are disclosed in SOURCE_BETA_CALIBRATION.',
        '', '## Primary target results: changes against original Before (pp)','',
        '| Arm | HC full ΔvIoU | HC unobserved ΔsIoU | Vid full ΔvIoU | Vid unobserved ΔsIoU |','|---|---:|---:|---:|---:|']
    oldresults=read(OLDP/'RESULTS.json')
    for arm in ['A_mean','B_robust_top1','C_joint','D_C_center','E_DINO_support']:
        vals=[]
        for ds in ['hc2','vidstg']:
            for ep in ['v','unobserved']:
                vals.append(x['groups'][ds]['Before']['E_minus']['Base'][ep]['mean']if arm=='E_DINO_support'else oldresults['groups'][ds]['Before']['paired'][arm+'_minus_Base'][ep]['mean'])
        lines.append('| '+arm+' | '+' | '.join(f'{v:+.6f}'for v in vals)+' |')
    lines+=['', 'E full-vIoU is29.161837% HC /9.668131% Vid; unobserved actual-frame sIoU44.539224% /46.884240%. Before baselines are23.570083% /8.494004% full-v, not raw Frozen. All16 parents/domain remain, including Vid two no-reference no-ops. Time is identical across arms. Missing GT is unknown, not zero. The observations use actual Decoder frames; benchmark interpolation is reported separately.',
        '', '| E comparison | HC full-v pp [95% CI] | HC unobserved pp [95% CI] | Vid full-v pp [95% CI] | Vid unobserved pp [95% CI] |','|---|---:|---:|---:|---:|']
    for c in ['Base','A_mean','B_robust_top1','C_joint','D_C_center']:
        lines.append('| E − '+c+' | '+' | '.join(fp(x['groups'][ds]['Before']['E_minus'][c][ep])for ds in ['hc2','vidstg']for ep in ['v','unobserved'])+' |')
    lines+=['', 'E retains HC candidate benefit and is above C in both domains, but Vid E−A remains−.757150pp full-v/−.559813pp unobserved. Vid E−A full-v95% interval[−1.710875,−.030887] is negative on this exposed panel; it is still not independent confirmation. The locked mean decision fails because Vid A gain was not restored, not because an arbitrary all-CIs-significant condition was imposed. Source-start sensitivity has the same E−A directional conclusion.',
        '', '## Candidate changes versus accepted-frame changes','',
        'A prelocked two-factor diagnostic crosses final C/E candidate indices with C/E accepted-frame masks, refits the conditional four-parameter mean and globally applies it. For rejected frames only, an unaccepted latent best candidate is defined at each final estimate; it is not an actual accepted reference. All diagnostic outputs seal with E before target GT. Endpoints exactly reconstruct C and E; hybrids are not selectable method groups. Finite two-factor Shapley values average both replacement orders. This algebraic attribution depends on that latent-index convention and is not a unique semantic/causal decomposition.',
        '', '| E−C attribution | HC full-v / unobserved pp | Vid full-v / unobserved pp |','|---|---:|---:|']
    for axis in ['candidate_choices','accepted_frame_mask']:
        lines.append('| '+axis+' | '+' | '.join(' / '.join(f"{x['groups'][ds]['Before']['E_minus_C_factorial_attribution'][ep][axis]['mean']:+.6f}"for ep in ['v','unobserved'])for ds in ['hc2','vidstg'])+' |')
    lines+=['', 'HC E and C both retain all56 admitted frames:10 candidate choices change, with0 acceptance changes. All HC E−C gains therefore come from the choice change under this fixed model. E still chooses18 non-Top1 boxes,16 below.35; this is not hard restoration of the old winner threshold. It preserves H002/H010 gains and removes C’s H011 failure, but also loses H008 benefit. H009 remains a severe failure.',
        '', 'Vid E and C both reject9/50 frames, but their masks are NOT identical: one previously accepted frame is rejected and one rejected frame is accepted. There is one candidate change on commonly accepted frames. The finite decomposition attributes+.513459pp full-v/+3.298362pp unobserved to candidate changes and−.052544/−.324075pp to acceptance changes, summing exactly to+.460915/+2.974288. Equal rejection counts do not imply equal rejected frames or zero acceptance effect.',
        '', 'All Vid E−C metric gain is concentrated in V010: full-v gain over Base+1.255 ->+8.629pp, unobserved+7.921 ->+55.510pp. The other15 have unchanged saved metrics. E is also equal to B in saved metrics on15/16 Vid cases, with only+.013583pp cohort full-v/+ .045550pp unobserved over B. Thus the remaining Vid deficit is close to the earlier Top1-robust deficit, rather than evidence that all remaining loss is caused by switching candidates. No new cutoff adjustment is attempted.',
        '', '## Reference quality, negative tails and uncertainty','',
        '| Domain | Exact-known references | Pool IoU>=.5 | Top1 good | C selected good | E selected good | E selected known |','|---|---:|---:|---:|---:|---:|---:|']
    for ds in ['hc2','vidstg']:
        g=diagnostics[ds]['Before'];lines.append('| '+ds+' | '+' | '.join(str(g[k])for k in ['known_original_admitted_positions','pool_has_geometric_IoU_ge_half','Top1_good','C_selected_good','E_selected_good','E_selected_known'])+' |')
    lines+=['', 'These are postseal geometric IoU counts, not newly certified participant/role labels. Alternative boxes receive no old Top1 identity tag; six old role-unknown queries remain unknown. Candidate scores and frame acceptance improve some references but do not certify a correct whole-tube correction.',
        '', 'HC E is positive versus Before on14/16 full-v queries, versus12/16 C; full-v<-5pp harms are1(E),2(C),5(A). E removes H011’s C−22.536pp full-v/−59.327pp unobserved collapse, restoring A’s+1.054/+1.425. It preserves H002+6.818/+8.074 and H010+2.017/+1.692. It retains H009−10.169/−22.027 and H003−2.630 full-v. Corrected positive/negative examples are all retained.',
        '', 'HC E−Before full-v95% interval is[+.469179,+13.515140]pp while unobserved interval[−.724748,+14.595241] crosses0. Its saved-score leave-one full-v range[+2.156914,+6.642484] and unobserved[+2.297279,+7.726283] are positive, but this is fixed inherited output sensitivity, not stream replay or independent confirmation. Vid full-v severe harms remain1(E) versus0(A), notably V007; E does not restore all A successes. All signs, severe tails, per-parent values, source sensitivity and strata are public. No subgroup is selected as an operational rule.',
        '', '## Verification, recovery and cost','',
        f"Independent auditing checks{audit['independent_conditional_assignment_games']} conditional-assignment games/{audit['assignments_enumerated']} assignments, {audit['source_choice_and_source_metric_checks']} source metric/selection checks, {audit['output_reconstruction_checks']} output reconstructions (maximum error{audit['maximum_output_error']}), {audit['independent_metric_checks']} metric checks, {audit['factorial_attribution_checks']} factorial identities, {audit['old_inputs_and_ABCD_metric_exact_checks']} old-input/ABCD identities and{audit['aggregate_and_CI_checks']} aggregate/CI checks. The source-selected beta is independently reproduced. All old inputs/pins and CURRENT/MAIN/PAPER are unchanged.",
        '', 'The first complete CPU scoring pass failed ONLY at strict JSON serialization because a NumPy int64 behavior count was not a native JSON scalar. No model or output generation was repeated. The original pinned scorer is unchanged; a separate json002 wrapper converts NumPy scalars at the JSON boundary and repeats postseal CPU scoring for export. The original failure and wrapper SHA are preserved, and this repetition is not counted as independent evidence. Reported serialized-scoring cost excludes the lost first pass; no online latency claim is made.',
        '', f"Source calibration CPU{actual['CPU_cost']['source_calibration_seconds']:.6f}s, cached target generation{actual['CPU_cost']['target_generation_seconds']:.6f}s, successful serialized scoring pass{actual['CPU_cost']['serialized_scoring_pass_seconds']:.6f}s. Original feature/expert work, preflight, repeated first scoring pass and audits are separate. New GPU/model/expert/training/Student-update/new-visual-label counts are0.",
        '', 'The permitted finding is narrower than method adoption: relative DINO candidate support contains useful information discarded by C, especially in HC, but this fixed simple combination still fails to recover Vid Top1-mean performance. Stop this low-cost selector-rule family without declaring all four-parameter correction impossible, without choosing E for HC/A for Vid, and without additional dense OPD or source/target tuning.',
        '', '[All results](RESULTS.json) · [Fixed decision](DECISION.json) · [Full source beta calibration](SOURCE_BETA_CALIBRATION.json) · [Reference diagnostics](DIAGNOSTICS.json) · [Actual protocol](ACTUAL_PROTOCOL.json) · [Independent audit](ROOT_INDEPENDENT_AUDIT.json) · [Figure](DINO_SUPPORT.svg)',
        '', 'Original query/parent text, GT coordinates, raw candidates/correction vectors, media, weights and personal records are excluded from public files.']
    (P/'REPORT.md').write_text('\n'.join(lines)+'\n');(P/'README.md').write_text('# One DINO-support correction supplement\n\n[Report](REPORT.md), [results](RESULTS.json), [source beta choice](SOURCE_BETA_CALIBRATION.json).\n\nHC improves; Vid does not restore A. Stop this fixed simple-combination family, without new confirmation or dense OPD.\n\nCPU reproduction requires the original private cache receipts and official datasets (not included in this public export). Run run_stvg_dino_supported_correction32_v1.py prepare, calibrate, measure; then run_stvg_dino_supported_score_json002.py; audit_stvg_dino_supported_correction32_v1.py; report_stvg_dino_supported_correction32_v1.py. Use the json002 scoring entry: the original pinned scorer is preserved, including its NumPy-scalar JSON failure. No model inference is required.\n')
    files=['vg_tta/stvg_dino_supported_correction32_v1.py','scripts/run_stvg_dino_supported_correction32_v1.py',
        'scripts/score_stvg_dino_supported_correction32_v1.py','scripts/run_stvg_dino_supported_score_json002.py',
        'scripts/audit_stvg_dino_supported_correction32_v1.py','scripts/report_stvg_dino_supported_correction32_v1.py',
        'scripts/publish_stvg_dino_supported_correction32_v1.py','protocols/stvg_dino_supported_correction32_v1.md']
    files += [str((P/f).relative_to(ROOT))for f in ['RESULTS.json','DECISION.json','SOURCE_BETA_CALIBRATION.json','DIAGNOSTICS.json','ACTUAL_PROTOCOL.json','VERIFICATION.json','ROOT_INDEPENDENT_AUDIT.json','REPORT.md','README.md','DINO_SUPPORT.svg']]
    put(D/'PUBLIC_FILES.json',dict(files=files));print('REPORT',len(files),'files',flush=True)


if __name__=='__main__':main()
