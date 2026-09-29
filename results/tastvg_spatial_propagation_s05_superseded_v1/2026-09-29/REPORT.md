> Superseded by the latest teacher-independent native parameter rollout route. This completed result is preserved as history.

# S0.5 — Sparse-to-dense spatial evidence propagation

Completed / audited candidate-support diagnostic on exposed development data. No selector, persistent adapter or OPD was run.

| Diagnostic | S0 corruption | S0.5 corruption | S0 clean | S0.5 clean |
|---|---:|---:|---:|---:|
| Unobserved gain, matched12, pp | 0.3145 | 1.1513 | 0.2826 | 1.3792 |
| Observed gain, matched12, pp | 11.9481 | 4.3608 | 12.1516 | 4.5908 |
| Whole-tube oracle gain,16 sources, pp | 0.8993 | 1.5800 | 0.8949 | 1.7370 |
| Six-layer union gain,16 sources, pp | 1.3886 | 2.1408 | 1.4280 | 2.3223 |
| Unobserved gain, all16, pp | 0.3654 | 1.3595 | 0.3439 | 1.5135 |
| Fixed step3 whole-tube gain, pp | 0.4344 | 0.7917 | 0.4181 | 0.9369 |

## Paired inference and negative cases

corruption: matched12 unobserved gain 1.1513pp CI[0.4845, 1.9040]; change vs S0 0.8368pp CI[0.2752, 1.4881]. Whole-tube change vs S0 0.6807pp CI[0.0491, 1.3896]; union change 0.7522pp CI[0.1471, 1.4383].

Fixed unobserved gains at steps1/2/3, all16: 0.2906 / 0.4115 / 0.5177pp. Unobserved-only oracle upper bound: 1.3676pp (different selection rule; not the primary diagnostic).

clean: matched12 unobserved gain 1.3792pp CI[0.4855, 2.4884]; change vs S0 1.0967pp CI[0.2886, 2.1584]. Whole-tube change vs S0 0.8422pp CI[0.0850, 1.7040]; union change 0.8943pp CI[0.1582, 1.7372].

Fixed unobserved gains at steps1/2/3, all16: 0.2408 / 0.3586 / 0.6829pp. Unobserved-only oracle upper bound: 1.5222pp (different selection rule; not the primary diagnostic).

| Source | S0.5 whole-tube oracle gain pp | Change vs S0 pp | Union gain pp | Matched unobserved gain pp |
|---|---:|---:|---:|---:|
| Q01 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Q02 | 0.0000 | -0.3978 | 0.6318 | 0.0000 |
| Q03 | 0.4727 | -0.5949 | 2.9365 | 0.2633 |
| Q04 | 1.1756 | 0.3548 | 1.1756 | 1.0662 |
| Q05 | 4.0217 | 3.7418 | 4.0217 | NA |
| Q06 | 0.5002 | 0.0000 | 0.5002 | -0.1134 |
| Q07 | 1.4384 | 0.0000 | 1.4384 | 0.8146 |
| Q08 | 0.0000 | -0.9433 | 0.1389 | NA |
| Q09 | 3.0669 | 0.0538 | 3.0669 | 2.8453 |
| Q10 | 3.9502 | 2.8758 | 3.9502 | 3.8967 |
| Q11 | 0.0135 | 0.0000 | 2.2683 | NA |
| Q12 | 2.2152 | 0.0000 | 2.2152 | 0.8246 |
| Q13 | 3.9002 | 3.0652 | 3.9002 | NA |
| Q14 | 3.0340 | 2.0440 | 3.0340 | 2.7043 |
| Q15 | 0.0000 | 0.0000 | 3.4698 | 0.0000 |
| Q16 | 1.4914 | 0.6914 | 1.5049 | 1.5145 |

## Actual configuration and interpretation limits

Original16 C3 VidSTG sources, one query each, same Vid-source TA checkpoint and clean/five source-hash seed0 transient5% corruptions.96 cells. Same frozen H and five Sa2VA masks, zero new expert calls. Fixed .5 global foreground/background cosine contrast + .5 maximum foreground-reference affinity on H_app. Area-projected reference masks (>=.5), fixed reference-class-mean midpoint E threshold, grid-mask enclosing boxes on unobserved frames; original sparse boxes retained at references. This E-to-box readout was specified before scoring to complete the attachment. It is a heuristic; local self matches affect threshold calibration. No trained memory/HTR reproduction claim.

Reuse the unchanged S0 equal-valid-frame L1+GIoU loss, three .004-original-visual-norm H_app-only steps and dynamic native suffix. Adding dense frames also changes the fraction of loss contributed by references. Frozen text/motion/weights. Native B0 retained, so oracle non-harm is structural and does not establish online selector safety. The experiment tests this complete propagation/readout implementation, not only the mathematical affinity formula.

After all96 predictions/evidence sealed, retain only the same16 old GT entries for scoring. sIoU uses all GT-valid sampled frames; choose one entire tube by sIoU and then read its observed/unobserved metrics. Matched12 reproduces S0 denominator; all16 is separate. Five corruption conditions averaged inside each source, then source macro;10000 source-bootstrap draws seed20260929. Descriptive CIs on repeatedly exposed development sources, no untouched-generalization claim.

Propagation coverage: {'missing_foreground_or_background_after_projection': 30, 'propagated': 66}; newly supervised unobserved frame-cells 6022; original valid sparse frame-cells 391; dense total 6413. Loss 3.891277 → 2.941122, decreases in 87/96 cells. This loss magnitude uses a different frame denominator from S0.

corruption: direct propagated target-box sIoU 49.5142% vs native 48.7243% on the identical GT-valid, dense-target-valid unobserved frames, source count 11. Diagnostic target quality, not a new inference arm; excludes unavailable target frames.
clean: direct propagated target-box sIoU 49.7927% vs native 50.0021% on the identical GT-valid, dense-target-valid unobserved frames, source count 11. Diagnostic target quality, not a new inference arm; excludes unavailable target frames.

## Verification and resources

GPU process 71.55s including initialization,288 suffix backward calls,2 genuinely edited full-native reinsertion checks,0 new expert calls,0 new H captures. Six CPU tests. AUDIT.json:960 dual metrics and576 independent loss checks. PROPAGATION_AUDIT.json independently reconstructs area projection, affinities, thresholded masks and propagated boxes in NumPy. PUBLIC_AUDIT.json reconstructs saved scalar summaries. Prior model-constructor warnings are unchanged; loaded source state hash and exact B0 are checked. No failed cells silently dropped.

S1 remains a future proposal. Leave-one-reference-out generation creates fold-specific tubes: scores cannot be averaged as if identical step indices were identical candidates without defining that contract. No S1 implementation, additional hyperparameter search, temporal experiment or production promotion.
