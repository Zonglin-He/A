"""Save bounded R61 decisions and complete tables after independent audits."""
from common61 import *
H=ROOT/'research/round61';p=read(H/'experiment/EXPERIMENT_SPEC_LOCK.json');a=read(H/'ANALYSIS_SUMMARY.json');rank=read(H/'RANK_SUMMARY.json');ranks={r['arm']:r for r in rank['metrics']};audit=read(H/'prediction_audit_attempt001/AUDIT_RESULT.json');scalar=read(H/'saved_scalar_audit_attempt001/AUDIT_RESULT.json');diagnostic=read(H/'independent_diagnostic_audit_attempt001/AUDIT_RESULT.json');ra=read(H/'independent_rank_audit_attempt001/AUDIT_RESULT.json');assert all(q['state'].startswith('PASS_') for q in [audit,scalar,diagnostic,ra]);cost=read(ROOT/p['prediction_root']/'COSTS.json');capture=read(ROOT/p['capture_root']/'COSTS.json');eng=read(H/'engineering_capture_attempt001/COSTS.json');m={r['arm']:r for r in a['metrics']};process=read(H/'ALL_PROCESS_SCALARS_PUBLIC_SAFE.json')
decision={'state':'R61_COMPLETED_CONTROLLED_INTERVENTIONS_TINY_LAST_LAYER_SIGNAL_NO_BROAD_FAULT_IDENTIFIED','R60_failure_explained_by_weakness_alone':False,'matching_stronger_damping_adopted':False,'new_TTA_method_adopted':False,'retain_language_cross_attention_gain_probe':True,'retain_visual_head6_probe':True,'primary_next_interface_if_local_work_continues':'decoder5_cross_l_output_reading_strength_not_matching_channel_deletion','broad_wrong_ranking_fault_localized':False,'all_controlled_rescue_union25':4,'recoverable_ABC_errors25':126,'fixed_box_unrecoverable_errors25':68,'language15_gain_vs_ABC_pp':m['dec_language_boost']['gain_vs_frozen_ABC_pp'],'language15_gain_vs_R59direct_pp':m['dec_language_boost']['gain_vs_R59direct_pp'],'head6off_gain_vs_ABC_pp':m['dec_vision_head6_off']['gain_vs_frozen_ABC_pp'],'all_tested_fixed_interventions_below_R59direct':all(m[arm]['balanced_mIoU_percent']<m['r59_direct_all']['balanced_mIoU_percent'] for arm in HEAD_ARMS+DECODER_ARMS+HEAD_OFF_ARMS),'scope':'exposed_development_diagnostic_not_independent_confirmation','no_source_training':True,'no_FINAL_access':True,'no_new_loss_or_fit':True,'no_further_experiment_scheduled':True};dump(H/'DECISION.json',decision)
config={k:v for k,v in p.items() if not k.endswith('_root') and k not in ['manifest','source_plan','programs_sha256','inputs_sha256']};config['cohort']='exposed_DEV336_168_per_platform_2Drone_3Quad_scene_clusters';dump(H/'PUBLIC_FIXED_CONFIGURATION.json',config)
dump(H/'NORMALIZED_COSTS.json',{'counts':{'n':336,'arms':34,'new_full_source_prefix_forwards':capture['new_full_source_forwards'],'engineering_full_source_forwards':eng['new_full_source_forwards'],'native_decoder_replays':cost['native_decoder_forward_calls'],'native_head_replays':cost['actual_controlled_head_forward_calls'],'new_gradient_steps':0,'pseudo_label_fits':0,'external_model_calls':0,'task_values':11424,'new_rawGT_or_geometry_scores':0,'directed_negative_comparisons':5159,'independent_native_readouts':9072,'independent_process_rows':8736,'independent_rank_rows':11424},'timing_seconds':{'full_prefix_capture':capture['seconds'],'engineering_prefix':eng['seconds'],'controlled_replay':cost['seconds'],'independent_native_audit':audit['seconds'],'independent_scalar_audit':scalar['seconds'],'independent_diagnostic_audit':diagnostic['seconds']},'peak_bytes':{'full_prefix_GPU':capture['peak_GPU_bytes'],'controlled_replay_GPU':cost['peak_GPU_bytes']},'flags':{'unique_GPU_serial':True,'source_unchanged':True,'all_cases_retained':True,'FINAL_access':False,'new_source_training':False,'new_TTA_fitting':False,'raw_encoder_latency_is_not_per_intervention_cost':True}})
dump(H/'NORMALIZED_HISTORY.json',{'events':[{'stage':'postseal_diagnostic_export','attempt':1,'status':'failed','failure_type':'numpy_integer_JSON_serialization','repair_type':'explicit_python_int_count','valid_outputs':0,'producer_predictions_changed':False,'tolerance_changed':False,'prior_attempt_preserved':True},{'stage':'postseal_diagnostic_export','attempt':2,'status':'passed','valid_outputs':8736,'producer_predictions_changed':False,'tolerance_changed':False,'prior_attempt_preserved':True}]})
table='\n'.join(f"| {r['arm']} | {r['Drone_mIoU_percent']:.6f} | {r['Quad_mIoU_percent']:.6f} | {r['balanced_mIoU_percent']:.6f} | {r['gain_vs_frozen_ABC_pp']:+.6f} | {r['balanced_Acc25_percent']:.6f} | {r['balanced_Acc50_percent']:.6f} | {r['query_changes_vs_frozen_ABC']} | {r['improved_vs_frozen_ABC']}/{r['worsened_vs_frozen_ABC']} | {r['recoverable_errors_rescued25']}/{r['original_good_predictions_broken25']} |" for r in a['metrics'])
ot='\n'.join(f"| {r['family']} | {r['tested_arms']} | {r['recoverable_error_query_change_any']} | {r['recoverable_errors_rescued_by_any_arm25']} | {r['GT_best_of_tested_arms_with_ABC_fallback_mIoU_percent']:.6f} |" for r in a['family_oracle_diagnostics'])
rankt='\n'.join(f"| {r['arm']} | {r['best_good_rank_improved']} | {r['best_good_rank_worsened']} | {r['best_good_rank_unchanged']} | {r['median_best_good_rank_after']:.1f} |" for r in rank['metrics'])
scenes=read(H/'ALL_SCENE_METRICS_PUBLIC_SAFE.json');sct='\n'.join(f"| {r['platform']} | {r['scene_cluster']} | {r['n']} | {r['arm']} | {r['mIoU_percent']:.6f} |" for r in scenes if r['arm'] in ['frozen_ABC','r59_direct_all','mlp_noise8_off','mlp_noise32_off','dec_language_boost','dec_vision_head6_off'])
pt=[]
for arm in HEAD_ARMS+DECODER_ARMS+HEAD_OFF_ARMS:
 rr=[r for r in process if r['arm']==arm];pt.append({'arm':arm,'median_relative_edited_matrix_norm':float(np.median([r['relative_edited_matrix_norm'] for r in rr])),'median_max_hidden_change':float(np.median([r['max_h_change'] for r in rr]))})
dump(H/'PROCESS_SUMMARY.json',pt)
text=f'''# R61 — controlled matching-head and last-decoder interventions

**Decision:** R60 was not merely underpowered. Making its same selected channels fully inactive changes decisions but loses overall mIoU; increasing to32does not fix it. Last-decoder language-reading strength and visual-head6show small task-relevant signals, but every fixed intervention remains below R59direct fusion. No broad harmful component or new TTA method is established.

## Fixed experiment

Waymo-only epoch100; exposed DEV336,168per platform,2Drone/3Quadsceneclusters; originalA256boxes and R57ABCsamplings/transport/UNKNOWN.34prespecifiedarms,26actualcontrolled interventions. No GT in intervention construction, no fitting/gradient/pseudo-label/source training/newloss/external model. No FINAL384or full5912access.

Original A prefix reused; old B/C prefix was missing, recovered using the exact saved seeds and pointindices. Engineering4andfull672sourceforwards; all originalboxes/scores/h/query/tokens/mask/pointindicesbitexact; source fingerprint unchanged. Every controlled intervention replays allABCviews, outputs originalAboxes only, and restores all source parameters. Source-zero replay hiddenerror0and maxABCprobabilityerror2.22e-16; every originalTop1preserved. Physical matching-head/out-projection column assignments and native residual-output interventions are separate diagnostic mechanisms, not optimized testtime updates.

Matching tests: R60noise8/common8fully off;noise32highest same excess ranks fully off;random8sameR60destinations;first/secondhidden32largest source-energy contributions removed. Noise32can include nonpositive-excess ranks; expansion is a sensitivity intervention, not a claim all32are harmful. Lastdecoder self/language/vision/FFNresidual gains0,.5,1.5; each of8visual heads separately suppressed through its36out-projection input columns. All downstream attention/norm/FFN/native matching is recalculated. No best arm is chosen per input during inference.

## Complete DEV336 task results

IoUbetter/worse refers to continuousIoU, not independent instance-identity truth. Rescue/break counts refer to crossingIoU>.25relative to frozenABC. All cases including no changes remain in denominators.

| Arm | Drone mIoU% | Quad mIoU% | Balanced mIoU% | Delta ABC pp | Acc25% | Acc50% | Query changes | IoU better/worse | Rescued25/broken25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

## Weak intervention versus wrong component

The same R60noise8selector at original soft attenuation changes0queries. Full removal changes10queries(4IoUimprovements/3worsenings)but mIoUdecreases0.009784pp. Expanding to32changes11queries but decreases0.060699pp. Common8full removal changes14queries,decreases0.100700pp. First/secondhidden32removal changes17/39queries,and decreases0.107849/0.474628pp. Median changed-matrix norm fractions are16.73%fornoise8,33.30%fornoise32,and34.37%forsecondhidden32; this is not a nonzero-but-negligible numerical edit. It does not identify a beneficial selector or demonstrate arbitrary edits of all matched magnitude would fail.

Thus larger perturbations prove matching-head sensitivity but do not repair overall grounding. R60's lack of effects had an amplitude component, yet amplitude alone does not explain lack of beneficial task edits. Do not continue that channel-damping rule by just increasing scale.

## Which last-decoder computation has evidence?

Language residual gain gives22.184478%(off),22.297528%(half),22.302460%(native),22.504647%(1.5x). The direction is monotone in overall mIoUacross these fixed probes.1.5xrescues3of126recoverableerrors,breaks0originalgood25predictions,and changes6queries(4IoUbetter/1worse). Gains occur inbothDroneclusters,Quadmeanunchanged; two scene means improve,noneworsen. This is a small exposed-development signal,still0.106468ppbelowR59directandnot independent confirmation.

Visualhead6off gives22.468224%(+0.165764pp),rescues2,breaks0at25,and changes10queries(5IoUbetter/2worse);3scene means improve,noneworsen. Its rank effects below are mixed,so it is not globally certified as a harmful head. Turningoff the entirevisualresidualinstead loses0.094106pp. Keep output contribution versus attentionQ/K changes distinct: this round does not test a learnedQ/Kupdate.

Selfattentionhalf is+0.105834ppbut its dose relation is not monotone;FFNgains are smaller/mixed. If local-interface work continues, the most justified first probe is finaldecoder cross_l output/read-strength,withvisualhead6as a secondary probe. Matchingchanneldeletionis lowerpriority. No combo withR58orR59was run,and no combo benefit is implied. Closing lastdecoder language does NOT remove earlier language information; closing its vision path does NOT remove earlier point information.

## Recoverability and finite intervention envelope

ABChas142good25predictions and194errors.126errorshave at leastone original candidateIoU>.25;68have noneand cannot be rescued atthisthresholdby any ranking-only method. These labels are offline geometric task criteria,not instance identity annotations.

| Family | Tested arms | Recoverable errors with any query change | Recoverable errors rescued by any arm25 | GT-best tested arm or ABC fallback mIoU% |
|---|---:|---:|---:|---:|
{ot}

Even selecting the best of the26controlledarms perinputwithGTrescues only4/126recoverableerrors;only22/126changeTop1under any ofthese probes. The GT-best-envelope withABCfallback is22.805084%. This finite-arm diagnostic is NOT a deployable method or a universal architecture capacity bound; interactions/different directions were not exhaustively tested. It weakens a broad 'just suppress a few bad terminal channels/branches' explanation. It is consistent with many incorrect preferences already being stable in the frozen upstream representation, but does not locate an earlier fault by itself.

## Actual ranking, not probability softening

On126recoverableerrors,the bestIoU>.25candidate's baseline rank medianis25.0. A softened negative probability gap can improve without rank improvement,so we added an explicitly postseal descriptive rank check; no method or predictions changed. Language1.5ximproves best-good rankon30,worsenson12,leaves84unchanged. Visualhead6off improves18,worsens23,leaves85unchanged despite its smallTop1task gain. Source singleAcan improve medianrankwhile reducingtotalmIoU;R59can improvetotalmIoUwith medianrank26.0. Therefore rank summaries explain behavior,not replace the registered overall task metric.

| Arm | Recoverable rank improved | Worsened | Same | Median best-good rank after |
|---|---:|---:|---:|---:|
{rankt}

## Five exposed scenes

| Platform | Anonymous cluster | n | Arm | mIoU% |
|---|---:|---:|---|---:|
{sct}

## Audit, cost and failure history

Independent functional reconstruction checks sourcecheckpointsubset,oldABCstates,selectors,and9072noninheritednative readouts. Maxindependenthiddenerror{audit['source_prefix_hidden_max_error']:.3g},scoreerror{audit['independent_native_score_max_error']:.3g};allselectedqueriesexact. Difference from producerbitexactzerois due to functionalmodule operation ordering and is within the prespecified tolerance,no tolerance change. All original boxes preserved. One postseal saved-IoUscore11424values,zero new rawGT/geometrycalls. Independent aggregates/sceneserror{scalar['independent_summary_max_abs_error_pp']:.3g}pp,5159directednegativecomparisons exact. Independentlychecked8736process/error/marginrecords and11424rankrecords. OriginalGOODandallnegativecasesretained.

SourceB/Ccapture{capture['seconds']:.2f}s,controlleddownstreamreplay{cost['seconds']:.2f}s,independentnativeaudit{audit['seconds']:.2f}s,serialsingleGPU. Fullcapturepeak{capture['peak_GPU_bytes']}bytes,replaypeak{cost['peak_GPU_bytes']}bytes;nooptimizersteps. StandardunusedRobertaPooler initialization warning occurs before strict sourcecheckpointload; fullsource fingerprint and all oldforward outputs verify the actual source,so no training was initiated. One postseal diagnostic serialization attempt failed on numpyint64; preserved draft/error and fixed explicitPythonint only,no predictions/scores/tolerances changed. No full/source model inference failed.

## Scope and stopping decision

NoTTArecipeadopted,no automatic newloss/fitting/fulltarget/holdout/source training. PreserveR59directfusionandR58references. Retaincross_lread-strengthandvisualhead6onlyasdevelopmentmechanismprobes. The answer is not simply 'make the edit stronger':strongerterminalperturbations can change errors but mostly do not fix them,and the best limited signals remain belowdirectuse. Further work needs a better-founded entity/readout mechanism or an explicitly tested earlier-stage issue,not anotherblindamplitudeincrease. DEV336has historical label exposure and onlyfiveclusters;no independent confirmation or frame-independentCI is claimed.
'''
(H/'REPORT.md').write_text(text);print(decision)
