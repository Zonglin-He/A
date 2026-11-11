# R71 — Fixed frozen-CLIP expansion and projection-matched visual control

The fixed CLIP evidence exceeds source, frozen-representation R59 and its projection-matched visual permutation on both expanded cohorts. No adapter, gradient update or new TTA loss is tested. No parameter-adaptation advantage is claimed.

R70 selected the projection-only CLIP direct reader on exposed DEV336. R71 applies that exact reader to existing R69 VAL1280 and TRAIN1024, with the same source A256, frozen CLIP ViT-B/32/revision, literal appearance-prefix parser, tight + 1.25x context crops, native CLIP logit scale, projection-area weights and score centering. No strength, parser, region-pool or update-position search. R59 is a separate comparator; CLIP is added to source, not to R59. R59 uses three point samplings while this CLIP reader uses A plus extra RGB/pretrained visual knowledge, so that is a system comparison rather than equal information or parameter-TTA attribution. The CLIP/permutation pair has identical information and inference budget.

## Separate cohort results

| Cohort | Method | Drone mIoU % | Quad mIoU % | Equal-platform mIoU % | Acc@25 % | Acc@50 % |
|---|---|---:|---:|---:|---:|---:|
| VAL1280 | source | 10.7122 | 25.5849 | 18.1485 | 33.1250 | 16.4844 |
| VAL1280 | frozen_ABC | 11.2556 | 25.9656 | 18.6106 | 34.3750 | 17.0312 |
| VAL1280 | R59 | 11.4835 | 26.5171 | 19.0003 | 34.6875 | 17.5000 |
| VAL1280 | CLIP_direct | 13.3208 | 27.6951 | 20.5080 | 36.4844 | 17.6562 |
| VAL1280 | CLIP_mismatch | 9.3564 | 20.2580 | 14.8072 | 25.5469 | 13.4375 |
| TRAIN1024 | source | 16.5047 | 19.4823 | 17.9935 | 34.8633 | 16.8945 |
| TRAIN1024 | frozen_ABC | 16.6737 | 19.4682 | 18.0709 | 35.4492 | 16.8945 |
| TRAIN1024 | R59 | 16.8304 | 19.1693 | 17.9998 | 34.8633 | 16.6992 |
| TRAIN1024 | CLIP_direct | 17.8041 | 23.1767 | 20.4904 | 38.4766 | 17.8711 |
| TRAIN1024 | CLIP_mismatch | 13.2896 | 15.7137 | 14.5016 | 26.9531 | 13.8672 |
| DEV336_DIAGNOSTIC | source | 15.5399 | 28.1208 | 21.8303 | 40.7738 | 21.1310 |
| DEV336_DIAGNOSTIC | frozen_ABC | 15.7363 | 28.8686 | 22.3025 | 42.2619 | 21.4286 |
| DEV336_DIAGNOSTIC | R59 | 16.3856 | 28.8367 | 22.6111 | 43.1548 | 21.7262 |
| DEV336_DIAGNOSTIC | CLIP_direct | 18.6357 | 30.8080 | 24.7219 | 44.0476 | 22.0238 |
| DEV336_DIAGNOSTIC | CLIP_mismatch | 13.5405 | 23.9474 | 18.7439 | 33.0357 | 19.0476 |

VAL and TRAIN are different fixed expression distributions; do not compare their absolute scores as identical populations or choose the favorable cohort. DEV remains a diagnostic that supplements the missing matched control, not a new confirmation set.

## Overall gains and positive/negative tails

| Cohort | CLIP−source pp | CLIP−R59 pp | CLIP−matched mismatch pp | Better / worse / unchanged vs source | Positive scene means / total | Largest positive contribution pp | Gain after removing largest contribution pp |
|---|---:|---:|---:|---|---:|---:|---:|
| VAL1280 | +2.3594 | +1.5077 | +5.7008 | 238 / 137 / 905 | 5 / 5 | 0.0616 | +2.2978 |
| TRAIN1024 | +2.4969 | +2.4906 | +5.9888 | 204 / 128 / 692 | 4 / 4 | 0.0735 | +2.4233 |
| DEV336_DIAGNOSTIC | +2.8915 | +2.1108 | +5.9779 | 61 / 27 / 248 | 5 / 5 | 0.2074 | +2.6841 |

The final column retains the original cohort denominator; it is an influence diagnostic, not a GT-based sample filter or an alternative official result. Every negative observation remains in the main score and public paired tables. No requirement that each scene improve was used to adopt or reject a configuration.

## Same-weight visual permutation

Only projection-valid group similarities are permuted. The original unknown mask, source scores, group-to-query propagation, clipped-area weights, text, two crops, image encodings and native scaling stay fixed. The eligible-group similarity multiset is preserved. Weighted centering is recomputed after assignment. Invalid/invisible groups remain unknown rather than receiving an artificial zero-embedding negative. Permutation is deterministic with the existing R70 salt, never chosen by IoU. Trivial/fixed permutation entries are retained and counted. This supplies the same-projection-weight control that R70 did not have; it is not directly comparable to interpreting R70’s differently weighted mismatch as a control for the best branch.

## Scene results

| Cohort | Platform / anonymous scene | n | Source | ABC | R59 | CLIP direct | CLIP mismatch |
|---|---|---:|---:|---:|---:|---:|---:|
| VAL1280 | drone D_S01 | 543 | 8.3095 | 8.9440 | 9.1458 | 11.0343 | 7.6199 |
| VAL1280 | drone D_S02 | 97 | 24.1621 | 24.1961 | 24.5696 | 26.1208 | 19.0773 |
| VAL1280 | quad Q_S01 | 338 | 12.3204 | 12.7276 | 12.9544 | 13.5048 | 7.7208 |
| VAL1280 | quad Q_S02 | 115 | 38.9402 | 39.3140 | 40.8434 | 39.3351 | 31.8390 |
| VAL1280 | quad Q_S03 | 187 | 41.3470 | 41.6842 | 42.2210 | 46.1856 | 35.7966 |
| TRAIN1024 | drone D_S01 | 360 | 14.5721 | 14.7812 | 15.1895 | 16.1355 | 11.5165 |
| TRAIN1024 | drone D_S02 | 152 | 21.0819 | 21.1560 | 20.7167 | 21.7561 | 17.4888 |
| TRAIN1024 | quad Q_S01 | 299 | 23.6654 | 23.1768 | 23.1416 | 27.2448 | 21.0996 |
| TRAIN1024 | quad Q_S02 | 213 | 13.6103 | 14.2620 | 13.5930 | 17.4660 | 8.1532 |
| DEV336_DIAGNOSTIC | drone D_S01 | 106 | 10.5850 | 10.7117 | 11.7616 | 13.0738 | 9.6139 |
| DEV336_DIAGNOSTIC | drone D_S02 | 62 | 24.0112 | 24.3268 | 24.2910 | 28.1447 | 20.2537 |
| DEV336_DIAGNOSTIC | quad Q_S01 | 62 | 10.1175 | 10.5063 | 9.6321 | 11.9167 | 6.3643 |
| DEV336_DIAGNOSTIC | quad Q_S02 | 49 | 32.9989 | 35.0709 | 36.0676 | 36.5471 | 31.7108 |
| DEV336_DIAGNOSTIC | quad Q_S03 | 57 | 43.5098 | 43.5098 | 43.5098 | 46.4229 | 36.3992 |

## Repeated media and data exposure

Primary scores average expressions equally within platform, then platforms equally. Some VAL expressions share raw frames. Secondary scores first average descriptions per identical image+point frame, then frames, then the two platforms:

| Cohort | Method | Drone frames | Quad frames | Equal-frame then equal-platform mIoU % |
|---|---|---:|---:|---:|
| VAL1280 | source | 428 | 605 | 18.7394 |
| VAL1280 | frozen_ABC | 428 | 605 | 19.3650 |
| VAL1280 | R59 | 428 | 605 | 19.7042 |
| VAL1280 | CLIP_direct | 428 | 605 | 21.3960 |
| VAL1280 | CLIP_mismatch | 428 | 605 | 15.3740 |
| TRAIN1024 | source | 512 | 512 | 17.9935 |
| TRAIN1024 | frozen_ABC | 512 | 512 | 18.0709 |
| TRAIN1024 | R59 | 512 | 512 | 17.9998 |
| TRAIN1024 | CLIP_direct | 512 | 512 | 20.4904 |
| TRAIN1024 | CLIP_mismatch | 512 | 512 | 14.5016 |
| DEV336_DIAGNOSTIC | source | 168 | 168 | 21.8303 |
| DEV336_DIAGNOSTIC | frozen_ABC | 168 | 168 | 22.3025 |
| DEV336_DIAGNOSTIC | R59 | 168 | 168 | 22.6111 |
| DEV336_DIAGNOSTIC | CLIP_direct | 168 | 168 | 24.7219 |
| DEV336_DIAGNOSTIC | CLIP_mismatch | 168 | 168 | 18.7439 |

R69 VAL1280 contains 1,033 unique media frames in the same five recordings used by prior target VAL development; historical full-VAL exposure was already disclosed. TRAIN1024 contains 1,024 unique media frames from four additional official recordings/place-name families selected in R69. It is not a source-training stage, and target labels do not enter evidence construction. Nominal recording/place-family separation does not certify a global geographical or absolute wall-clock holdout. Both cohorts already received R69 evaluation, so this is fixed-configuration expanded evaluation, not blind independent testing. FINAL files are neither decoded nor inferred nor scored, and no FINAL exclusion manifests are newly read.

The same first-lexicon-noun/all-content-fallback source token readout from R69 remains unchanged; this is not a fresh reproduction of all official annotated-span protocols. Literal CLIP subject parsing may also retain wrong roles or lose attributes. Neither is repaired by test labels in this round. Long CLIP phrases retain R70 token coverage through at-most-75-token chunks plus special tokens and normalized weighted feature averaging; composition across chunks is not preserved. Tokenizer length warnings do not mean an oversized sequence is passed to the model.

## Input equivalence, audits and costs

The field-minimized projection preparer omits B’s unused RGB patches and complex reliability calculations. All 336 cached DEV direct readouts are bit-exact to R70, with raw group/projection reconstruction on a preselected subset before expanded inference. Four engineering inputs cover VAL/TRAIN × Drone/Quad. Main CPU preparation and the single CLIP GPU reader run as a streaming pipeline; every consumed preparation/case file is matched to the final input seal before CLIP evidence is sealed. This changes scheduling, not evidence math.

Main encoding uses 986,946 crops across 2,304 inputs, 499,780 candidate groups and 493,473 projected groups. Preparation 367.71s; CLIP encoding 2368.42s; max allocated CLIP GPU memory 0.663 GiB. No new source forward, optimizer step, adapter fit, external detector, Qwen, SAM or DINO output is used. Timings include relevant cache/I/O and are marginal to existing source captures; they are not full deployment latency. CPU preparation overlaps encoding, so do not sum those times as wall-clock time.

All 2,640 observations × five arms are sealed before saved candidate-IoU lookup. CPU audit verifies every source A256/readout, original R69/R70 baseline, strict valid-group permutation and unknown pattern. Independent float64 probability arithmetic is bounded by the softmax Jacobian roundoff bound while native float32 replay is exact; maximum independent error 5.34e-06. Expanded media hashes match the original R69 selected inputs. GPU audit reconstructs raw point-support groups and camera projections for 18 fixed scene-stratified observations across 9 clusters, reencodes 144 crops and checks literal text embeddings; maximum image feature error 1.34e-06. Independent scalar audit verifies all 13,200 saved-IoU task values and 6,754 ordered negative comparisons. A separate audit checks every derived method/frame/scene/decision table.

All 2,304 expanded input records omit explicit distortion coefficients; the declared zero-distortion assumption is inherited unchanged from R70, not estimated from task labels. Camera projections and rectangular crops remain imperfect instance correspondence: background and neighboring instances may enter a crop; no true RGB instance mask or dense depth is introduced. The fixed CLIP model supplies external pretrained semantic knowledge. Direct-use improvement therefore supports useful visual evidence, not an innovation or parameter-TTA increment. With few scene clusters and prior development exposure, frame/scene tables describe variability; no expression-independent significance or certified new-domain generalization is asserted.

## Decision and next boundary

The fixed CLIP evidence exceeds source, frozen-representation R59 and its projection-matched visual permutation on both expanded cohorts.

Random RGB training and the R70 score-distillation adapter remain paused. No new update loss, adapter step budget, trainable layer, source retraining or FINAL evaluation is queued. If parameter adaptation is pursued later, its zero-step state must preserve the strong direct reader and its supervision must contribute information beyond copying that reader. It must beat a same-information direct-use control; this round does not test or establish that mechanism.

Public artifacts include original owned code, frozen configuration, input/dependency/prediction hashes, anonymous per-observation and scene/frame scalars, all negative comparisons and audit receipts. Raw media, descriptions, camera coordinates, sample/scene identities, features, actual weights, secrets and third-party snapshots are excluded. Full reproduction requires the official data and exact source checkpoint; this package is not a standalone dataset/model distribution.
