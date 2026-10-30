# R8 · Real cross-domain RGB/text/point-cloud pseudo regression

**Completed, sealed, independently scored and audited. Decision: do not adopt this G or regression adaptation configuration.** Real pseudo supervision and nonzero parameter updates are verified, but the new association harms both target domains; the tiny regression changes are reproduced by the same-information output-space control.

## Full target results

All metric columns below are percentages. These are fresh, matched strictFP32 source outputs with nativeyaw0, GT-free puretext BBF readout, no PCA, and all requested expressions retained.

| Target | Method | n | mIoU % | Acc25 % | Acc50 % | Equal-scene mIoU % |
|---|---|---:|---:|---:|---:|---:|
| drone | Frozen source q0 | 2984 | 8.668259 | 18.766756 | 3.384718 | 15.419374 |
| drone | RGB association q1 only | 2984 | 7.033995 | 10.924933 | 1.742627 | 7.491449 |
| drone | Direct pseudo, no update | 2984 | 7.033814 | 10.857909 | 1.742627 | 7.509618 |
| drone | Regression update1 | 2984 | 7.034361 | 10.924933 | 1.709115 | 7.492718 |
| drone | Regression update3 | 2984 | 7.034845 | 10.924933 | 1.709115 | 7.494818 |
| quad | Frozen source q0 | 2928 | 18.022213 | 32.855191 | 15.846995 | 29.299674 |
| quad | RGB association q1 only | 2928 | 7.911287 | 12.534153 | 4.474044 | 11.910392 |
| quad | Direct pseudo, no update | 2928 | 7.945201 | 12.602459 | 4.474044 | 11.975913 |
| quad | Regression update1 | 2928 | 7.912894 | 12.568306 | 4.474044 | 11.912555 |
| quad | Regression update3 | 2928 | 7.915799 | 12.568306 | 4.474044 | 11.916569 |

Source-only is genuine Waymo-onlyepoch100:100source epochs/15300updates, random visual/grounding weights plus generic frozen RoBERTa, without the released union-task checkpoint or Drone/Quad supervision. Targets are Drone2984 and Quad2928. Their VAL data were previously researched in R5/R7, so this is cross-domain evaluation relative to source training, not fresh research confirmation. Only2Drone and3Quad scenes exist.

R7 used other precision settings, so its cached outputs and numeric scores are not paired baselines for R8. Every R8 source prediction was recomputed.

## Association, geometry and parameter effects

Differences below are percentage points of mIoU, not relative percent.

| Contrast | Drone delta pp | Quad delta pp |
|---|---:|---:|
| Association − source | -1.6342636308 | -10.1109258583 |
| Direct pseudo − association | -0.0001807086 | 0.0339136070 |
| Parameter3 − association | 0.0008500528 | 0.0045124047 |
| Parameter3 − direct pseudo | 0.0010307614 | -0.0294012023 |
| Parameter3 − matched output-space3 | 0.0000000323 | 0.0000000861 |
| RGB association − PC-only association | 3.1572806787 | 5.6733602038 |
| RGB association − semantic-shuffle association | 2.7759002640 | 4.0118330855 |

The harmful change is already present in association-only q1. Geometry and adaptation cannot explain or repair a−1.634/−10.111pp association loss when their own differences are only thousandths/hundredths of a point. RGB outperforms PC-only/shuffle controls, showing that matched RGB/text evidence has some value within this weak selector, but it remains worse than the original source decision. This is not a sufficient adoption result.

At Acc25, association rescues130 but breaks364 original successes on Drone; it rescues124 but breaks719 on Quad. At Acc50, the gained/lost counts are17/66 and21/354. All five scenes have negative association mIoU deltas; scene equal-weight association deltas are−7.927925pp and−17.389282pp. These are descriptive results with2/3clusters, not thousands of independent scenes.

Parameter3 is only+.000850pp/+.004512pp above association. Parameter3−directG is+.001031pp on Drone by micro mean but−.014800pp by scene mean; on Quad it is−.029401pp micro/−.059343pp scene. There is no robust task gain over direct use. Its mean IoU differences from exact output-space3 are3.23e-10/8.61e-10, with identical Acc25/50 decisions. The last-layer update is real, but the matched output-space mechanism explains it.

## What was changed in G

The color-at-the-start gate is removed. Actual RGB tight/context crops are encoded against the literal complete description by frozen officialCLIP ViT-B/32. Observed point support and complete-link box/support overlap prepare spatial proposal groups;≤48 proposals reduce to groups and≤16 RGB regions. q0 stays the original source choice; q1 is an actual original proposal chosen by the original text score within the selected group. The final group score is CLIP similarity plus.02PCquality; within the association stage, original grounder confidence limits the pool/ROI shortlist and chooses the within-group anchor. It also orders geometric representatives and weights group-member IRLS consensus later. Its numeric score does not arbitrate the final cross-group decision against q0.

That design overrides source choices on2624/2984Drone and2572/2928Quad cases. q1 is assigned before r_global/axis eligibility, so zero geometric trust does not itself protect the original association choice. Under the controlled association-only comparison, this selector is harmful. The omission of source-aware arbitration is a concrete competing explanation supported by code and measured selection losses; no source-aware replacement was tested here, so it is not a demonstrated fix.

Geometrically distinct source proposals in the selected group supply robust center consensus. Source q1 size prior remains; size weights and nativeyaw arezero. The observed point support constrains eligibility and axes; correlated source boxes are not independent geometric truth. Global strength shrinks the same final pseudo target for directG, parameter and kernel arms, instead of cancelling inside a normalized weighted loss or scaling LR twice.

The19arms comprise sourceq0 and six arms in each of RGB/fulltext, PC-only and fixed semantic-shuffle: association-onlyq1, directG, parameter1/3 and exactoutputspace1/3. Only last4regression tensors/1734parameters are eligible, with867center parameters active because size loss iszero. Frozen feature/kernel controls, fixed q1/feature-norm LR, beta.1m, alpha.25 and source restoration make the attribution explicit. No EMA, 3Dteacher or backbone update is used; frozen CLIP produces similarity and no boxes.

## Real signal and nonzero updates

| Target | All cases | q0→q1 changes | Actual pseudo + parameter + selected output change | No parameter update |
|---|---:|---:|---:|---:|
| Drone |2984|2624|2329|655|
| Quad |2928|2572|2181|747|

These counts verify actual float32 target displacement, real gradient steps, changed parameter tensors and changed selected boxes, not just positive weights or an eligibility flag. PC-only updates2816/2623, semantic-shuffle1922/1798. Zero cases remain in denominators with explicit reasons. Across all queries, parameter/kernel maximum coordinate error is1.52587890625e-5m (<fixed5e-5m), interval excess0, and every episode restores the source model while CLIP remains frozen.

The preregistered hash-fixed40case smoke contains16Drone/24Quad across5scenes, excluding oldsmoke20, with no color or GT filtering. It produced13/16 and17/24 actual RGB updates. Root viewed all10 fixed unlabelled RGB/point/ROI overlays before full execution. Corresponding objects are visible, but same objects can remain in multiple groups; neighboring cars, background and fragments contaminate some crops. These cases were not removed or used to tune thresholds.

Smoke has48→median18groups→median16RGBROIs. The16ROI cap is reached25/40times. Median raw center consensus112.16mm shrinks to median actual pseudo4.998mm; parameter3 displacement median1.615mm among changed cases. These are q1-relative displacements, separate from discrete association jumps. CLIP77token truncation affects38Drone/28Quad full descriptions and is recorded rather than hidden.

## Offline coverage and retained failures

GT is used only for sealed offline scoring and coverage; none of the following numbers chooses online groups or updates. The staircase categories describe G-path coverage, not physical instance identities or actual failure labels. In particular, a source-q0 fallback may already hit despite a missing G-path group.

| IoU>.25 pathway category | Drone | Quad |
|---|---:|---:|
| No source256 candidate hit | 833 | 705 |
| Source256 hit, prepared pool miss | 722 | 208 |
| Prepared hit, RGB ROI pool miss | 11 | 29 |
| RGB ROI pool hit, selected group miss | 997 | 1530 |
| Selected group hit, q1 anchor miss | 95 | 89 |
| q1 anchor hit | 326 | 367 |

Selection is a larger measured bottleneck than the within-group anchor or16ROI cap in this registered pipeline. Source256 oracle Acc25 counts2151/2984 and2223/2928 show substantial proposal coverage, while sourceTop10 counts1033/1703 offer narrower ranking headroom. The full256 oracle is GT-assisted and not an available online selector. Selected-group best-IoU oracle means8.1616%/8.6306% remain below source8.6683%/18.0222%; simply perfecting the anchor inside the currently selected group cannot recover the mean source performance.

Association has681gains/716harms/1587ties on Drone and456/1108/1364 on Quad at absoluteIoU tolerance1e-8. Parameter3 relative to its q1 anchor still has475/379/2130 and428/248/2252. All signed case results remain local and all112328anonymous arm scalars are included in public export;17736G scalar rows preserve zero/rejected cases. Positive/negative private tails are sign-filtered rather than padded with ties. The fixed10gallery uses its own smoke predictions/scores, never full-process query indices.

The prelocked global-strength bins do not establish a reliable acceptance threshold. On Quad even r>.5 has148cases and association mean−3.501788pp. Raising the confidence cutoff is therefore not a validated repair.

## Measurement, official-interface audit and implementation history

Both smoke/full plans were locked before any R8 task score. Full execution waited for the real-smoke physical/nonzero gate. All5912×19 and40×19 predictions were sealed and their independent no-GT audits passed before GT release; no target labels changed G, prompts, steps or thresholds. Every unique selected output uses independent general3D versus batched yaw-IoU; full maximum discrepancy8.881784197001252e-16, strictthreshold disagreements0. Full256 coverage uses batchgeometry and is not claimed to have256 independent3D checks per case. Nativeyaw0, float64 IoU from actual float32 boxes, dimensionclamp1e-6 and strictIoU>.25/.5 remain matched.

Independent reduction verifies2544aggregatecells per stage with error≤5.55e-17; final review checks1008paired scene intervals per stage using seed801/2000draws. All112328full arm IoUs arefinite; failed-arm status0. Five scene clusters limit inference, and all reported intervals remain descriptive.

Official last288D regression features,64D L2 retrieval projection, temperature.07, complete token denominator, camera frames and internal point normalization were reviewed. GTclassspan is intentionally excluded, and BBS is loaded but not an alternate frozen ranking control in this study. This does not exclude every possible training/decoding explanation. Pre-GPU continuousFOV/floorpixel fixes and pre-GT diagnostic sign/macro corrections retain original hashes and history; obsolete drafts were never used for real R8 scored predictions.

Full study pipeline walltime is2187.603s (36.46min), including19arms, per-case snapshots, diagnostics and serialsource/CLIP/heads. Peak torchallocated memory is1294439936bytes (1.205541GiB). This is a research-pipeline cost, not a matched deployed single-arm latency benchmark. CPU scoring32.221s; independent full no-GT audit17.80s. All model processes ended and the GPU lease is released.

## Decision and next discriminating requirement

**Do not adopt R8 G, its re-association, direct pseudo output or regression TTA. Keep the frozen Waymo-only source as this study baseline.** Engineering feasibility of genuine multimodal feedback and nonzero updates is supported; target-quality improvement and unique parameter-adaptation benefit are contradicted for this configuration.

Do not try to compensate for harmful association by increasing update LR or steps. The next mechanism to test is source-preserving multimodal arbitration: retain source support across physical groups and use RGB/text evidence to reject or resolve a competing target, rather than unconditionally replacing most source choices. First demonstrate improved association and pseudo accuracy, including baseline-correct cases, with the same-information no-update control. Residual grouping, relation-aware evidence and candidate recall remain supporting conditions. This is a hypothesis for a newly registered study, not a measured fix here. No further GPU sweep or source retraining is queued.

The completed code/protocol/scalars, negative findings, exact source/CLIP provenance and independent audit receipts are exported under the established public-study convention; raw weights, captions, media, points, original identifiers, coordinates and embeddings remain local.
