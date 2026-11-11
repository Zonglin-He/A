# R74 — Expert-free fixed-region correspondence TTA

## Decision

**Do not adopt this contrastive adaptation recipe.** It produces real, monotone accepted updates but has not established a useful incremental grounding benefit over stronger same-information frozen readout. Keep the expert-free research constraint; do not restart CLIP feedback, source training, layer search, or step sweeps on the basis of this result.

The complete 2,304-input result is +0.014107 percentage points over frozen A/B probability averaging, −0.177635 pp versus frozen A/B representation fusion, and −0.213292 pp versus the three-view R59 reference. The largest single positive contribution accounts for +0.014012 pp; the other 2,303 inputs contribute +0.000095 pp. Correct correspondence beats the shuffled control by +0.037459 pp overall, but loses to it on TRAIN1024. This is a limited descriptive signal, not an established adaptation mechanism.

## Scope and ordering

Waymo-only epoch100; original A256 final boxes and frozen text, point backbone, decoder, text projection. The source input is XYZ plus language, not RGB. Only the existing 184,960-parameter query matching MLP is updated. No new pretrained network, no external semantic feedback, no source-domain retraining, no EMA. CLIP is attached only after all fits as an external-knowledge comparison; it never enters sample selection, region construction, loss, acceptance, or hyperparameters. No FINAL input, media, prediction or label was opened.

The initial mechanism pilot was 64 inputs selected by a fixed hash with 16 per VAL/TRAIN × Drone/Quad and recording quotas, without source-error, eligibility or CLIP filtering. Its tiny task gains arose from single cases while its held-out correspondence diagnostic was negative. A separate **post-pilot** lock expanded the unchanged kernel to all remaining 2,240 existing R69 inputs. The 64 fits were not rerun. No step, temperature, keep strength, grouping gate or learning rate was tuned. This expansion was conditional development validation, not an originally registered blind confirmatory experiment.

VAL1280 contains 640 descriptions per target platform across five recording groups; TRAIN1024 contains 512 per platform across four other recording groups. TRAIN is used for unlabeled adaptation and post-seal scoring, not source fitting. All these target cohorts have historical development exposure. Recording separation does not certify geographically independent environments or distinct wall-clock capture. The combined set has nine recording groups and 2,057 unique point/image frame pairs. This round is neither DEV336 nor ALL5912, and must not be numerically compared with those as if the denominators matched.

## Algorithm and task connection

1. Construct up to 16 fixed spatial regions from the existing geometric A-box/point-support groups. Choose a minimum-index member per group and geometry-only farthest-center coverage. Require at least four common **original point indices** between A/B and query anchors in each fixed box. There are 36,854 retained regions. Raw point/index coordinates and the Drone z shift were independently checked; do not trust backbone index comments.
2. Region representations are uniform pools of existing language-conditioned query vectors anchored in each box, normalized in the trained 64-D query/text matching space. These are global-context query proxies for spatial regions, **not** a trained RoI embedding head, certified instance masks, or physical-object identities.
3. A negative pair requires box IoU <0.05, shared raw-point fraction ≤0.05, at least four visible projected points per region, and shared surface-component fraction ≤0.05 at both scales. The calibrated global RGB/LiDAR surface graph is built before candidate association. RGB supplies local boundary continuity only. No color names or language-target identity rules appear in this loss.
4. Symmetric masked region InfoNCE, temperature 0.07, plus weight 1.0 source normalized-query representation preservation on all 256 queries in both views. Candidate duplicates/overlaps are not blanket negatives. A fixed one-cycle derangement of B region keys is the corruption control; it corrupts positive and negative bindings together, preserving the loss mask/pair budget, and is not a pure positive-only ablation.
5. Per input and arm, restore source; one Adam proposal at 0.0005, gradient clipping 1; transactional backtracking on fixed scales 1 through 1/256. Accept only an unlabeled objective decrease of max(1e−12, 1e−6×|old loss|). No task label or C feature controls acceptance.
6. Recompute native normalized matching, full text-span log-odds, source-probability bridge and fixed R69 B→A candidate transport including UNKNOWN mass, then average A/B probabilities. All outputs select original A256 boxes. Different sampled forwards never reuse a query number as an object identity.

Across three arms, 6,909 actual proposals were accepted and independently replayed, with zero accepted objective increases. One input has no admissible negatives and stays in the denominator with an unchanged head in all three arms (6,912 stored states total). The inability to demonstrate task gain is not explained by zero gradients, failing to write parameters, or an unchanged optimizer. Objective improvement itself is not evidence of correct target identity.

## Frozen direct-use controls

Frozen A/B probability mean uses identical two-view features, B transport and unknown mass. A stronger fixed A/B unit-representation fusion uses those same views, without C or parameter fitting; its coefficient is fixed at one. A second fixed direct-use reader uses the **same RGB-qualified spatial correspondences**: for uniquely assigned query features it shifts toward the other-view regional centroid by one half, preserving within-region residuals; ambiguous membership is unchanged. It performs no fitting. This additional control was locked after pilot scoring but before remaining-population scoring; it does not revise any existing readout or fit. It is a concrete same-region evidence control, not a claim of optimal direct use.

The historical R59 reader uses A/B/C and is therefore a stronger information-budget reference, not the matched two-view control. CLIP is an explicitly different external-knowledge budget.

## All 2,304 inputs

| Method | Drone mIoU % | Quad mIoU % | Equal-platform mIoU % | Acc@25 % | Acc@50 % |
|---|---:|---:|---:|---:|---:|
| Source A | 13.286650 | 22.872610 | 18.079630 | 33.897569 | 16.666667 |
| Frozen A/B probability mean | 13.565061 | 23.091415 | 18.328238 | 34.548611 | 16.970486 |
| Frozen A/B unit-representation fusion | 13.751189 | 23.288770 | 18.519979 | 34.635417 | 17.057292 |
| Frozen same-region centroid fusion | 13.679051 | 22.785187 | 18.232119 | 34.375000 | 16.796875 |
| R59 frozen A/B/C representation fusion | 13.859905 | 23.251369 | 18.555637 | 34.765625 | 17.144097 |
| Correct-region contrastive TTA + RGB qualification | 13.579967 | 23.104722 | 18.342345 | 34.592014 | 16.927083 |
| Shuffled-region contrastive TTA | 13.615591 | 22.994180 | 18.304885 | 34.418403 | 16.970486 |
| Correct-region TTA, no RGB boundary | 13.579967 | 23.104722 | 18.342345 | 34.592014 | 16.927083 |
| CLIP_reference | 15.313386 | 25.686915 | 20.500151 | 37.369792 | 17.751736 |

Relative to A/B mean, correct-region TTA changes the selected query on 83 inputs, improves IoU on 18 and worsens it on 12; the remaining query changes do not change saved task IoU. Acc@25 gains only one net hit and Acc@50 loses one net hit. Both platform means gain about 0.01 pp, but the recording pattern is two positive, three negative and four unchanged groups. No per-recording non-degradation rule was used to choose this decision.

## VAL1280

| Method | Drone mIoU % | Quad mIoU % | Equal-platform mIoU % | Acc@25 % | Acc@50 % |
|---|---:|---:|---:|---:|---:|
| Source A | 10.712182 | 25.584854 | 18.148518 | 33.125000 | 16.484375 |
| Frozen A/B probability mean | 11.128113 | 25.858545 | 18.493329 | 33.906250 | 16.953125 |
| Frozen A/B unit-representation fusion | 11.270153 | 26.354564 | 18.812358 | 34.218750 | 17.265625 |
| Frozen same-region centroid fusion | 11.194454 | 25.710130 | 18.452292 | 33.828125 | 16.953125 |
| R59 frozen A/B/C representation fusion | 11.483505 | 26.517059 | 19.000282 | 34.687500 | 17.500000 |
| Correct-region contrastive TTA + RGB qualification | 11.161077 | 25.956326 | 18.558701 | 33.984375 | 16.953125 |
| Shuffled-region contrastive TTA | 11.158151 | 25.649914 | 18.404032 | 33.593750 | 17.031250 |
| Correct-region TTA, no RGB boundary | 11.161077 | 25.956326 | 18.558701 | 33.984375 | 16.953125 |
| CLIP_reference | 13.320815 | 27.695114 | 20.507964 | 36.484375 | 17.656250 |

## TRAIN1024

| Method | Drone mIoU % | Quad mIoU % | Equal-platform mIoU % | Acc@25 % | Acc@50 % |
|---|---:|---:|---:|---:|---:|
| Source A | 16.504735 | 19.482304 | 17.993519 | 34.863281 | 16.894531 |
| Frozen A/B probability mean | 16.611245 | 19.632502 | 18.121874 | 35.351562 | 16.992188 |
| Frozen A/B unit-representation fusion | 16.852484 | 19.456528 | 18.154506 | 35.156250 | 16.796875 |
| Frozen same-region centroid fusion | 16.784798 | 19.129009 | 17.956903 | 35.058594 | 16.601562 |
| R59 frozen A/B/C representation fusion | 16.830404 | 19.169256 | 17.999830 | 34.863281 | 16.699219 |
| Correct-region contrastive TTA + RGB qualification | 16.603579 | 19.540218 | 18.071898 | 35.351562 | 16.894531 |
| Shuffled-region contrastive TTA | 16.687390 | 19.674513 | 18.180951 | 35.449219 | 16.894531 |
| Correct-region TTA, no RGB boundary | 16.603579 | 19.540218 | 18.071898 | 35.351562 | 16.894531 |
| CLIP_reference | 17.804101 | 23.176665 | 20.490383 | 38.476562 | 17.871094 |

These are separate evaluation cohorts; combined means do not turn them into a blind test. Correct-region TTA is +0.065372 pp versus A/B mean on VAL and −0.049975 pp on TRAIN. Frame-balanced descriptive means preserve that direction: VAL +0.072411 pp, TRAIN −0.049975 pp. Scene/frame scalar tables accompany all task negatives; descriptions from the same frame or recording are not independent trials. No i.i.d. confidence interval is asserted from these nine recording groups.

## Pilot versus the remaining population

Pilot64: frozen A/B 21.980001%, correct-region TTA 22.230416%, a +0.250415 pp increment entirely explained by one Quad VAL input. Held-out C has a different one-case gain. On the remaining 2,240, A/B is 18.223902%, TTA 18.231257%, only +0.007355 pp; stronger frozen A/B representation fusion is 18.430478%. Thus the pilot signal does not justify a robust overall adaptation claim. Pilot task values in the separate table are included in the 2,304 total and must not be counted twice.

## Held-out C: auxiliary and grounding transfer

C representations were first accessed only after the complete A/B fit stage was sealed. The same fixed A regions require C anchor coverage and ≥4 shared A/C original points. All arms use the **same geometry-only admissible retrieval negatives**, avoiding method-dependent evaluation filtering. Each direction evaluates 28,894 eligible regional rows and records 7,960 unknown rows. Unknowns remain in all task denominators.

| Representation | A/B→C retrieval % | C→A/B retrieval % | Mean % |
|---|---:|---:|---:|
| source | 81.497889 | 81.608638 | 81.553264 |
| corr_RGB | 81.227937 | 81.044508 | 81.136222 |
| corr_shuffled | 82.387347 | 82.550010 | 82.468679 |
| corr_noRGB | 81.238319 | 81.061812 | 81.150066 |

The mean retrieval accuracy falls from 81.553264% to 81.136222%, while shuffled correspondence reaches 82.468679%. The mean positive-versus-negative cosine gap nevertheless grows; its improvement does not override the retrieval failure. This specifically rejects treating contrastive loss or gap as a sufficient proxy for grounding.

Held-out C grounding: frozen C 18.274305%, correct-region adapted C 18.232853% (−0.041452 pp), shuffled C 18.343122%. The correct-region update improves IoU on 11 inputs, harms 16, and changes 65 query selections. C was not a fitting input, but this is another sampling of the same physical observation, not a new camera view, scene, or proven cross-input transfer. It does not establish that adaptation transfers to an unseen observation.

## RGB necessity and limitations

Removing RGB line-boundary qualification changes the negative matrices on 224/2,304 inputs, yet **all 2,304 main and all 2,304 held-out C target selections are identical** between correct RGB and no-RGB branches. Their weights/features are not all identical, and the auxiliary retrieval numbers differ slightly. Therefore the RGB ablation is implemented, but this recipe provides no task-level evidence for useful RGB-specific supervision. No-RGB still uses the same calibrated camera visibility and LiDAR geometry; it removes RGB boundary information, not the camera model.

The known correspondence identifies a fixed spatial region, not necessarily a pure object. The trained query features include context and language; pooling by anchor location can blend different semantics. Uniform preservation of source representations limits drift but does not align this regional auxiliary task with the text-grounding decision. These are competing explanations consistent with the result, not separately established causes.

## Evidence, reproducibility and costs

All thirteen readouts ×2,304 inputs (29,952 task values) were sealed before the complete saved-IoU join. Original twelve readout vectors are preserved byte-for-byte during union assembly; the additional direct reader was independently reproduced. Predictions retain original A-box geometry. Scoring reuses previously sealed per-A-candidate IoUs: zero new raw GT/geometry scoring calls. There are 12,393 ordered negative comparisons on VAL and 9,218 on TRAIN, plus the explicitly overlapping pilot table. UNKNOWN, inactive, fallback and every negative comparison are exported.

Independent checks replay all 6,912 stored heads, objective/acceptance, source point/anchor and region pooling relationships, task readout/transport, C region embeddings, direct-use probabilities, all scalar joins, negatives and scene/frame summaries. Maximum independent native probability discrepancy is 1.063e−6; the independent same-region direct reader differs by <8.4e−16. No source-A winner differs between the independently replayed initial native head and cached source.

This is a cached-prefix experiment, valid because only the matching MLP is trainable. It uses the actual existing independent A/B/C point-sampling forwards and original RGB surface graph, not synthetic random feature vectors. New full 3EED, CLIP, detector, or visual-model forward calls this round are zero. It does not measure cold-start full-pipeline throughput. Incremental CPU evidence preparation: 358.735 s; all three adaptation arms: 225.129 s; revised stage readouts: 160.876 s. These totals exclude original A/B/C capture, graph construction, initial superseded pilot reader, independent auditing, final assembly/direct-control bookkeeping and publication. Costs are reported rather than presented as deployment latency.

One analysis bookkeeping attempt failed due to a mismatched optimizer-record field name; it was repaired to read the actual schema. It did not change any prepared evidence, fitted state, prediction, gate or task value. The superseded pilot readout attempt is preserved; adding a stronger direct comparator before pilot scoring left its original eleven readouts unchanged. All stage attempts and receipts remain local.

## Relation to prior methods and stop boundary

DARTH uses a source-trained tracking embedding head, teacher detections, known transformed region correspondences, patch contrastive association and an EMA teacher within a complete detector/tracker adaptation system. We borrow the correspondence/contrastive mechanism, **not** the trained association head, detections, or full DARTH protocol. Its results cannot certify that a frozen 3EED query/text matching space is region-instance compatible. [DARTH paper](https://arxiv.org/html/2310.01926v1).

Point-TTA explicitly prepares auxiliary and main-task relationships through source-stage training/meta-learning; that guarantee is not available under this experiment's no-source-retraining constraint. [Point-TTA paper](https://arxiv.org/abs/2308.16481).

R74 supplies a real negative test of this specific expert-free correspondence objective and update scope, not a disproof of all expert-free TTA or 3EED. Keep source and frozen multi-sampling as baselines. Stop this recipe without enlarging its step count, region count, negative thresholds or learning-rate search. Further work requires an independently motivated auxiliary-to-grounding link or a change in evidence/preparation, not merely more successful fitting of this loss. CLIP remains comparison-only; no next experiment is queued.
