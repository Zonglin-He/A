"""Summarize fixed diagnostic outputs; no extra predictions or tuning."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'research/round52'
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 A=H/'analysis_attempt001';metrics=read(A/'ARM_METRICS.json');transitions=read(A/'ALL_FACTORIAL_TRANSITIONS.json');scenes=read(A/'ALL_SCENE_METRICS.json');scale=read(A/'CUE_SCALE_DIAGNOSTICS.json');tasks=read(H/'score_dev336_attempt001/ALL_TASK_ROWS_PRIVATE.json');ss=read(H/'score_dev336_attempt001/TASK_SUMMARY.json');arms=[r['arm'] for r in metrics];by={r['arm']:r for r in metrics}
 native=[];margins=[];folder=ROOT/'crossplatform/round52/predicted336_attempt001';seal=read(folder/'SEALED.json')
 for row in tasks:
  f=folder/'samples_PRIVATE'/(row['id_PRIVATE']+'.npz');assert sha(f)==seal['files'][f.relative_to(folder).as_posix()]
  with np.load(f,allow_pickle=False) as z:
   s=z['source_scores'];assert np.isfinite(s).all() and s.min()>=0 and s.max()<=1
   native.append(float(np.ptp(s)));ordered=np.sort(s.astype(float));margins.append(float(ordered[-1]-ordered[-2]))
 native_stats={'native_source_range_median':float(np.median(native)),'native_top_two_gap_median':float(np.median(margins)),'weighted_new_P_range_median':scale['new_P']['range_weighted']['median'],'source_scores_bounded_0_1_verified':True}
 dump(A/'SOURCE_SCORE_SCALE.json',native_stats)
 lines=['# R52: cached color / position / reference evidence decomposition','',
 'Decision: do not adopt the current point-carrier rule-evidence G and do not start TTA. Its position term is the largest demonstrated source of damage. Target-color measurement is weaker on the point carrier, but deleting color makes the existing complete G worse. Removing position improves the complete G but does not beat source.','',
 'This is a fixed, exposed DEV336 diagnostic (168 Drone + 168 Quad; two Drone and three Quad scene clusters). It is not a new independent test or full5912 result. No FINAL384 artifacts, new models, raw media, training, parameter updates or full-target rescoring were accessed. All original 256 source boxes and the Waymo-only epoch100 initialization are unchanged.','',
 '## Frozen design','',
 'V is cached target color evidence, P is cached image-position evidence, R is cached reference-relation evidence. Scores are native source score plus 2V + 2P + 2R for included cues, in this order. Existing anchors, clipping, UNKNOWN masks, support weights and shared color factor 0.5 are unchanged. Every subset of the three cues is evaluated; no coefficient search or best-subset deployment is performed.','',
 'The new family uses the final R51 whole-proposal point-support medoid carrier. The old family is the R51 appearance-only replacement diagnostic: learned masks, measured target color and old reference proposals. It is not the full original external G. Its reference path and coverage also differ, so old-versus-new R is not an isolated carrier experiment. With R absent, V/P comparisons do not use reference scores, but still bundle carrier, measurement and association differences.','',
 'PR removes target appearance only. Reference color reweighting already inside R remains fixed; this is not an all-RGB-off experiment. Missing evidence remains UNKNOWN/zero rather than negative. All cases, fallbacks and negatives stay in the denominator.','',
 '## Overall results','',
 '| Readout | Drone mIoU % | Quad mIoU % | Balanced mIoU % | Delta vs source pp | Acc25 % | Acc50 % | Improved / worsened | Rescue / break25 |',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in metrics:lines.append(f"| {r['arm']} | {r['drone_mIoU_percent']:.6f} | {r['quad_mIoU_percent']:.6f} | {r['balanced_mIoU_percent']:.6f} | {r['gain_vs_source_pp']:+.6f} | {r['balanced_Acc25_percent']:.6f} | {r['balanced_Acc50_percent']:.6f} | {r['improved']} / {r['worsened']} | {r['rescue25']} / {r['break25']} |")
 lines+=['','Improved/worsened uses continuous IoU delta tolerance 1e-10. Rescue/break25 means crossing the official strict IoU>0.25 threshold relative to source. This is an operational localization success measure, not semantic instance-identity gold. Equal platform sizes make micro and balanced means coincide here; the primary criterion remains equal-platform mIoU.','',
 '## What changed in the diagnosis','',
 '1. Position is not merely a fusion interaction: source+P alone loses 6.243781 pp, with 9 threshold rescues and 48 breaks. It is below source in all five clusters. Adding P to VR loses another 3.270830 pp (21 continuous improvements / 38 deteriorations). Removing it recovers score, but new_VR remains 1.100515 pp below source.','',
 '2. Reference relations are not validated by the earlier DINO replacement comparison. Current source+R loses 2.563989 pp, with only 1 threshold rescue and 19 breaks. Adding R to VP changes 25 queries and loses 0.206555 pp. Same-grounder reference localization remains a possible engineering replacement, not a demonstrated independent relation benefit.','',
 '3. Color is not the dominant cause of complete-G failure. New V alone is -0.343624 pp (12 rescues / 22 breaks), so it has not established standalone value on this carrier. However VPR beats PR by 2.506095 pp (50 improved / 15 worsened; 18 rescues / 6 breaks). Removing target appearance is harmful in this fixed context.','',
 '4. Measured color on the old learned-mask path has a small positive standalone value: old_V is +0.872083 pp (39 improved / 24 worsened; 14 rescues / 13 breaks). This repairs the missing zero-appearance comparison from R51. It does not show equivalence to Qwen: the complete external G is 28.344056%, whereas measured-color old_VPR is 23.358855%.','',
 '5. The current evidence is not a hidden positive component defeated only by three-way fusion: every new-carrier single cue and every new-carrier combination is below source. Fixed native scoring therefore gives no basis to train the student on these preferences. The old-path results show that the measurement/association context matters; they do not identify a unique bad clustering algorithm.','',
 '## All factorial additions','',
 '| Family | Add | Before | After | Balanced delta pp | Improve / worsen | Rescue / break25 |',
 '|---|---|---|---|---:|---:|---:|']
 for r in transitions:lines.append(f"| {r['family']} | {r['added_cue']} | {r['before']} | {r['after']} | {r['balanced_gain_pp']:+.6f} | {r['improved']} / {r['worsened']} | {r['rescue25']} / {r['break25']} |")
 lines+=['','These are fixed-context cue interventions on cached scores. They are not a unique causal attribution or Shapley contribution: information inside R and cross-cue redundancy remain.','',
 '## Scene boundary','', '| Scene | n | Source | new V | new P | new R | new VR | new VPR | old V | old P | old R |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in scenes:
  v=r['mIoU_percent'];lines.append('| '+r['scene_alias']+' | '+str(r['n'])+' | '+' | '.join(f'{v[k]:.6f}' for k in ['source','new_V','new_P','new_R','new_VR','new_VPR','old_V','old_P','old_R'])+' |')
 lines+=['','No frame-independence confidence interval or broad-domain significance claim is made from five exposed clusters. All split-level results and all scene/arm values are in the public scalar tables.','',
 '## Scale and measurement interpretation','',
 f"Median native score range is {native_stats['native_source_range_median']:.6f}; median source Top1/Top2 gap is {native_stats['native_top_two_gap_median']:.6f}. The median range of the weighted new-P term is {native_stats['weighted_new_P_range_median']:.6f}. Source probabilities were verified in [0,1].",
 '',
 'The inherited common multiplier is therefore not an equal influence across cues. This supports a scale-dominance risk, but does not prove that rescaling would recover accuracy: P is derived from a partial-visible carrier and a fixed image-coordinate predicate, and P alone already harms all five scenes. The data do not separate wrong spatial interpretation, wrong target support and numerical dominance. No learning-rate, coefficient or gate search was performed.','',
 'The carrier uses visible points, not complete-object boundaries or a certified semantic instance. The negative findings are compatible with a source-box-to-background feedback loop, but this diagnostic does not independently certify instance identity, camera calibration or the complete-object center. Do not state that any one of these is proven as the sole cause.','',
 '## Decision and next discriminating change','',
 'Stop the current fixed point-carrier V/P/R rule recipe; do not connect its feedback to MLP or Q/V-LoRA TTA. Preserve the R50 external G as the stronger direct-reranking research baseline. Do not promote new_VR or old_VR as a final method just because they rank higher among these development diagnostics.','',
 'If a subsequent repair is authorized, target the position/association interface first: use the same measured color and relation machinery, compare the actual current position readout with an explicit description-frame/instance-position measurement and a prespecified scale control, retaining a no-position control. It must establish overall net value before any parameter update. Do not retrain source or add TTA regularizers to compensate for currently harmful supervision.','',
 'For reference localization, retain multiple source hypotheses and UNKNOWN as in R51, but require a matched direct source+relation comparison to become positive. The prior nearly lossless replacement of DINO did not establish that its relation evidence itself was useful.','',
 '## Evidence and accounting','',
 'All 5,376 task cells are retained: 1,680 exact reused baseline cells and 3,696 newly scored cells in one post-seal CPU scoring pass. All 10,907 directed negative comparisons are exported. Independent prediction audit checks all 5,376 candidate-score vectors and canonical source-box/readout bytes. Independent saved-scalar aggregation checks all cases, scene/split aggregates, reused identity joins and negatives; its maximal overall error is 7.11e-15 pp.','',
 'The decomposition uses existing numeric caches only. Predictions were sealed before target GT scoring. No source training, VLM/segmentation/detector calls, RGB/point decoding, TTA, holdout access, full-target rerun or automation was performed. This report does not add its development evidence to R51 as an independent replication.','',
 'Reused reference study: [R51 immutable report](https://github.com/Zonglin-He/A/blob/720adf8c976b846c72ee5fa0b63d94ab4392610c/studies/3eed_single_grounder_evidence_replacement_20261102/REPORT.md). Public tables contain anonymized scalar outcomes only; data, descriptions, coordinates, original identifiers, candidate indices, features, weights and third-party snapshots are excluded.','']
 # Numeric prose assertions prevent hand-transcription of the principal claims.
 assert abs(by['new_P']['gain_vs_source_pp']+6.243781465444377)<1e-10
 assert abs(by['old_V']['gain_vs_source_pp']-.872083)<1e-6
 (H/'REPORT.md').write_text('\n'.join(lines))
 dump(H/'DECISION.json',{'state':'R52_COMPLETED_CURRENT_RULE_EVIDENCE_NOT_ADOPTED','single_model_G_adopted':False,'TTA_executed':False,'new_method_selected':False,'next_experiment_scheduled':False,'position_largest_observed_loss':True,'all_new_cue_subsets_below_source':True,'target_color_not_main_complete_G_damage':True,'reference_independent_increment_established':False,'prior_strong_external_G_preserved':True,'development_only':True,'source_balanced_mIoU_percent':by['source']['balanced_mIoU_percent'],'complete_new_balanced_mIoU_percent':by['new_VPR']['balanced_mIoU_percent'],'new_without_P_balanced_mIoU_percent':by['new_VR']['balanced_mIoU_percent'],'new_without_target_V_balanced_mIoU_percent':by['new_PR']['balanced_mIoU_percent']})
 print(json.dumps(native_stats))
if __name__=='__main__':main()
