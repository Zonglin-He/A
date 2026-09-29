> Superseded by the latest teacher-independent native parameter rollout route. This completed result is preserved as history.

# S0.5 v2 — Symmetric affinity probability and soft-moment boxes

Completed / audited candidate-support diagnostic on exposed development data. User-specified v2 replaces completed threshold v1, preserved separately as superseded; v1 had already been scored, so this is revised development, not independent confirmation. The changes include background local affinity, moment readout and /T loss normalization. No selector, persistent adapter or OPD was run.

| Diagnostic | S0 corruption | S0.5 corruption | S0 clean | S0.5 clean |
|---|---:|---:|---:|---:|
| Unobserved gain, matched12, pp | 0.3145 | 1.0152 | 0.2826 | 0.8186 |
| Observed gain, matched12, pp | 11.9481 | 4.9524 | 12.1516 | 4.9876 |
| Whole-tube oracle gain,16 sources, pp | 0.8993 | 1.5401 | 0.8949 | 1.4566 |
| Six-layer union gain,16 sources, pp | 1.3886 | 2.0765 | 1.4280 | 2.0561 |
| Unobserved gain, all16, pp | 0.3654 | 1.2618 | 0.3439 | 1.1523 |
| Fixed step3 whole-tube gain, pp | 0.4344 | 0.7886 | 0.4181 | 0.7706 |

## Paired inference and negative cases

corruption: matched12 unobserved gain 1.0152pp CI[0.2622, 1.9253]; change vs S0 0.7006pp CI[0.0537, 1.4815]. Whole-tube change vs S0 0.6408pp CI[0.1406, 1.1984]; union change 0.6879pp CI[0.2124, 1.2229].

Fixed unobserved gains at steps1/2/3, all16: 0.2689 / 0.2250 / 0.4697pp. Unobserved-only oracle upper bound: 1.2993pp (different selection rule; not the primary diagnostic).

clean: matched12 unobserved gain 0.8186pp CI[0.1197, 1.6464]; change vs S0 0.5360pp CI[-0.0254, 1.2193]. Whole-tube change vs S0 0.5617pp CI[0.0665, 1.1081]; union change 0.6281pp CI[0.1644, 1.1464].

Fixed unobserved gains at steps1/2/3, all16: 0.2464 / 0.3329 / 0.4668pp. Unobserved-only oracle upper bound: 1.2204pp (different selection rule; not the primary diagnostic).

| Source | S0.5 whole-tube oracle gain pp | Change vs S0 pp | Union gain pp | Matched unobserved gain pp |
|---|---:|---:|---:|---:|
| Q01 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Q02 | 0.0000 | -0.3978 | 0.6318 | 0.0000 |
| Q03 | 0.7340 | -0.3336 | 2.9365 | -0.2692 |
| Q04 | 0.4846 | -0.3362 | 0.4846 | 0.4674 |
| Q05 | 2.7902 | 2.5103 | 2.7902 | NA |
| Q06 | 0.5002 | 0.0000 | 0.5002 | -0.1134 |
| Q07 | 1.4384 | 0.0000 | 1.4384 | 0.8146 |
| Q08 | 1.9773 | 1.0340 | 1.9773 | NA |
| Q09 | 4.5230 | 1.5099 | 4.5230 | 4.2181 |
| Q10 | 4.0418 | 2.9674 | 4.0418 | 3.9738 |
| Q11 | 0.0135 | -0.0000 | 2.2683 | NA |
| Q12 | 2.2152 | 0.0000 | 2.2152 | 0.8246 |
| Q13 | 3.2252 | 2.3902 | 3.2252 | NA |
| Q14 | 0.9414 | -0.0486 | 0.9639 | 0.4723 |
| Q15 | 0.0000 | 0.0000 | 3.4698 | 0.0000 |
| Q16 | 1.7573 | 0.9573 | 1.7573 | 1.7937 |

## Actual configuration and interpretation limits

Original16 C3 VidSTG sources, one query each, same Vid-source TA checkpoint and clean/five source-hash seed0 transient5% corruptions.96 cells. Same frozen H and five Sa2VA masks, zero new expert calls. For each of foreground and background: .5 cosine-to-prototype + .5 maximum token-bank cosine. Temperature1 softmax yields P_fg; spatially normalized P on token cell centers yields mean and variance, width/height=sqrt(12*variance); clip xyxy corners, convert to cxcywh. Original Sa2VA reference boxes retained. No probability threshold or tuned sharpening. Area>=.5 reference mask-to-token labels remain fixed. This is the explicit new user definition, not the superseded v1 threshold readout or an HTR reproduction.

Reuse native S0 L1/GIoU terms and coefficients; sum valid frame losses / T exactly as requested, three .004-original-visual-norm H_app-only steps and dynamic native suffix. Adding dense frames also changes the fraction of loss contributed by references. Frozen text/motion/weights. Native B0 retained, so oracle non-harm is structural and does not establish online selector safety. The experiment tests this complete propagation/readout implementation, not only the mathematical affinity formula.

After all96 predictions/evidence sealed, retain only the same16 old GT entries for scoring. sIoU uses all GT-valid sampled frames; choose one entire tube by sIoU and then read its observed/unobserved metrics. Matched12 reproduces S0 denominator; all16 is separate. Five corruption conditions averaged inside each source, then source macro;10000 source-bootstrap draws seed20260929. Descriptive CIs on repeatedly exposed development sources, no untouched-generalization claim.

Propagation coverage: {'missing_foreground_or_background_after_projection': 30, 'propagated': 66}; newly supervised unobserved frame-cells 6672; original valid sparse frame-cells 391; dense total 7063. Loss 5.912022 → 5.447999, decreases in 87/96 cells. This loss magnitude uses a different frame denominator from S0.

corruption: direct propagated target-box sIoU 21.4875% vs native 48.6722% on the identical GT-valid, dense-target-valid unobserved frames, source count 11. Diagnostic target quality, not a new inference arm; excludes unavailable target frames.
clean: direct propagated target-box sIoU 21.5144% vs native 49.4188% on the identical GT-valid, dense-target-valid unobserved frames, source count 11. Diagnostic target quality, not a new inference arm; excludes unavailable target frames.

## Verification and resources

GPU process 71.03s including initialization,288 suffix backward calls,2 genuinely edited full-native reinsertion checks,0 new expert calls,0 new H captures. Eleven CPU tests across S0, v1 and v2 (five new v2 contracts). AUDIT.json:960 dual metrics and576 independent loss checks. PROPAGATION_AUDIT.json independently reconstructs area projection, both affinity classes, temperature1 probability and soft-moment boxes in NumPy. PUBLIC_AUDIT.json reconstructs saved scalar summaries. Prior model-constructor warnings are unchanged; loaded source state hash and exact B0 are checked. No failed cells silently dropped.

S1 remains a future proposal. Leave-one-reference-out generation creates fold-specific tubes: scores cannot be averaged as if identical step indices were identical candidates without defining that contract. No S1 implementation, additional hyperparameter search, temporal experiment or production promotion.
