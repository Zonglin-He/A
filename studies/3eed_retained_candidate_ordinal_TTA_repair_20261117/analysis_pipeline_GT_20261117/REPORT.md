# R134 pipeline-versus-GT analysis — cached evidence only

No optimization launched and no method changed. R135 does not exist; no GPU research process was running. This analysis only regroups sealed predictions and already-computed R131 A256 IoU vectors. No raw media/annotation reads, new geometric scoring, model forward, training or FINAL access. All1798 historically exposed rows remain; primary metric uses equal six directions and original within-direction design weights. Raw counts do not equal design-weighted rates.

## Main finding

The current TTA path improves over its own native-foundation zero state, but is below the intact source-prepared R91 frozen-A model. The previous +0.2435pp statement was conditional on that weaker starting path, not an advantage over the strongest existing frozen system. Source-prepared frozen A13.394463 vsR13412.873988: -0.520475pp. Raw macro also negative(13.403041 vs12.911249). Same descriptions, point indices, source checkpoints, annotation indices and A256 boxes are independently matched.

## Actual frozen/initialization/fusion/update stages

| State | mIoU% | Acc25% | Acc50% |
|---|---|---|---|
| original_source | 12.392140 | 24.608060 | 4.595003 |
| prepared_frozen_A | 13.394463 | 26.263306 | 4.599297 |
| foundation_A | 12.710810 | 25.411569 | 4.915220 |
| foundation_B | 12.614012 | 25.028169 | 5.267026 |
| frozen_FAB | 12.630518 | 25.405773 | 5.054341 |
| LOG3 | 12.793107 | 25.694609 | 5.170967 |
| R134 | 12.873988 | 25.646258 | 5.170967 |
| A256_oracle | 34.800680 | 80.882020 | 17.204307 |

The intact prepared-A scorer uses trained point/RGB fusion and live R91 competition interface. start123 zeros point_adapter.up.weight/bias and gate_P before fitting. body127 disables the live route/interface, then uses current query/text representations plus fixed source-A late_RGB and residual. Thus prepared-A→foundation-A changes initialization and readout together; -0.683653pp is a bundled observed loss, not proof that one zeroed tensor alone caused it. Foundation-A→registered-FAB is a matched-input fusion change at the same reader and gives -0.080292pp overall. TTA recovers +0.243470pp, leaving the complete system -0.520475pp below intact prepared-A. These are stage-accounting comparisons, not a completed causal ablation of each reset/readout operation.

Subgroup contributions use the original full-cohort denominator, not renormalized subset means:

```json
[
  {
    "B_available": false,
    "n": 1034,
    "design_weighted_mass_percent": 59.090802167291315,
    "prepared_to_foundation_A_contribution_pp": -0.40084757212460315,
    "foundation_A_to_FAB_contribution_pp": 0.0,
    "FAB_to_R134_contribution_pp": 0.0,
    "R134_minus_full_prepared_contribution_pp": -0.40084757212460315
  },
  {
    "B_available": true,
    "n": 764,
    "design_weighted_mass_percent": 40.90919783270762,
    "prepared_to_foundation_A_contribution_pp": -0.2828058477950892,
    "foundation_A_to_FAB_contribution_pp": -0.08029161685761298,
    "FAB_to_R134_contribution_pp": 0.24346976735078854,
    "R134_minus_full_prepared_contribution_pp": -0.11962769730191397
  }
]
```

## Fixed-box geometric ceiling and selection loss

The exact cached-IoU identity is 1-IoU(final) = [1-maxIoU(A256)] + [maxIoU(A256)-IoU(final)]. In aggregate, geometric candidate-quality ceiling gap65.199320pp and selection gap21.926692pp sum to total87.126012pp. This is an oracle accounting identity, not causal proof that box regression alone repairs the first term.

| GT-only diagnostic | mIoU% | Acc25% | Acc50% |
|---|---|---|---|
| A256_oracle | 34.800680 | 80.882020 | 17.204307 |
| R134 | 12.873988 | 25.646258 | 5.170967 |
| selected_group_oracle | 12.976302 | 26.112887 | 5.228610 |
| S_oracle | 11.336084 | 26.615331 | 3.717973 |
| C_oracle | 9.145990 | 21.387771 | 2.772506 |

A256 has an Acc25-good box for1437 inputs, but the final decision passes442;995 final Acc25 failures therefore have an available passing box. At Acc50,1518 inputs have no passing A256 box. Selecting among fixed boxes can address the former ranking gap but cannot make the latter1518 cross that threshold. Geometric duplicate grouping is only a proxy: within-selected-group oracle gains0.102314pp, while between-geometric-group regret21.824378pp. This does not certify true object identity.

## Where final Acc25 failures lose supervision

| Category | Raw count | Weighted cohort mass% | Selection-regret contribution pp |
|---|---|---|---|
| final_pass | 442 | 25.6463 | 1.5923 |
| A256_no_passing_box | 361 | 19.1180 | 2.6689 |
| passing_box_but_no_usable_B | 568 | 31.9590 | 10.7914 |
| B_available_but_passing_box_outside_S | 75 | 4.7571 | 1.7609 |
| passing_box_in_S_but_outside_C | 96 | 5.2276 | 1.6597 |
| passing_box_in_C_but_final_wrong | 256 | 13.2921 | 3.4535 |

S/C are supervision qualification/retention sets, not hard filters of the final A256 output. Consequently S/C global oracles can be lower than actual model mIoU; do not add A256→S→C differences as if every final selection were restricted to C. Empty support has oracle0 only for measuring supervision reach. Among996 zero-step Acc25 errors with a passing A256 box:428 have usableB,355 have a good box inS,259 inC,7 have a retained recoverable candidate with the R134 explicit order against the old top. This shows an observation/supervision reach problem in addition to internal ordering.

## TTA beneficial changes versus damage

| Baseline | Better | Worse | Same | Positive pp | Negative pp | Net pp | GT-only no-harm upper mIoU |
|---|---|---|---|---|---|---|---|
| original_source | 197 | 157 | 1444 | +1.743506 | -1.261658 | +0.481848 | 14.135646 |
| prepared_frozen_A | 325 | 346 | 1127 | +2.567890 | -3.088365 | -0.520475 | 15.962353 |
| foundation_A | 86 | 71 | 1641 | +0.612411 | -0.449233 | +0.163178 | 13.323221 |
| frozen_FAB | 54 | 31 | 1713 | +0.447537 | -0.204067 | +0.243470 | 13.078055 |
| LOG3 | 27 | 25 | 1746 | +0.193846 | -0.112964 | +0.080881 | 12.986952 |

Against same-information frozen-FAB, R134 has54 improvements31 harms: +0.447537pp beneficial contribution and -0.204067pp damage. Even an unavailable GT selector that removed every one of those harms reaches only13.078055%, below intact prepared-A13.394463%. Reducing adaptation damage alone therefore cannot close the current whole-system gap. This oracle is descriptive only and is never a deployment selector.

## Signal qualification and text readout boundaries

R134 freezes7157 pair constraints, but their fixed initial-FAB representative-box IoU directions are1097better/819worse/5241tied. The non-tied better fraction is57.25%; this is a box-quality proxy, not verified instance-label precision. Only12 inputs have a pair explicitly opposing initial top, and7 are zero-error/recoverable/retained. Sparse decisive signal explains why decreasing ordinal loss does not imply broad grounding correction.

Final native matching still uses firstnoun_mask from model51, with all-content fallback; complete text enters the encoders. Therefore attributes/references may affect contextual embeddings but are not guaranteed separately aligned by the final positive-token readout. This is an implementation observation, not GT proof that text-span failure causes the dominant loss in this cohort. No new span oracle or text intervention was run.

## Analysis conclusion and next decision boundary

First restore/compare against intact source-prepared frozen behavior and isolate reset/readout/fusion losses before optimizing R134 step/loss weights. For fixed A256, target selection remains a large available Acc25 opportunity; for Acc50, candidate quality is a hard ceiling. Missing usableB and missing supported/retained good candidates limit where current TTA can act. Teacher ordering quality and shared-update damage are subsequent limits, not the whole gap. Do not use the508 unchanged-update subset as the denominator for all pipeline failures. This turn remains analysis-only: no R135, no new TTA/fusion/teacher change and no future run queued.

## Verification

```json
{
  "status": "PASS_ALL1798_CAPTION_SAMPLE_CHECKPOINT_ANNOTATION_INDEX_BOX_BRIDGES_AND_ORACLE_ARITHMETIC",
  "n": 1798,
  "new_model_or_training_runs": 0,
  "new_predictions": 0,
  "raw_GT_or_media_reads": 0,
  "geometric_IoU_computations": 0,
  "method_changes": false,
  "FINAL_access": false
}
```
