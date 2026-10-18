# Content-source attention exclusion: original32 matched experiment

Same high native target-score content, exact same fixed24 student references, original4 observations, image/text/native900, .35 legality and score rules. Only the excluded native-attention slot changes: A geometry-nearest; B content-source. Six-layer readout boxes/scores, actual propagated references/hidden and native initial900 references saved privately. Last layer is fixed; no layer chosen by GT.

Both conditions actually perform mu_new=mu+sum24(y-a)/24 from E, preserving exact no-feedback no-op. This is one external Gaussian-mean correction, not network-weight training. E, Source, MBR and old predictions unchanged. All32 complete outputs sealed before new GT; original exposed HC16/Vid16, not independent confirmation. Four unsupported Vid queries retained.

| Domain | Source vIoU% | E vIoU% | A unaligned once% | B aligned once% |
|---|---:|---:|---:|---:|
| hc2 | 26.399499 | 26.861045 | 26.833244 | 26.642224 |
| vidstg | 21.380083 | 28.187972 | 28.158499 | 28.140723 |

| Domain | Comparison | Full vIoU pp [95% query CI] | Unobserved actual Decoder sIoU pp [95% CI] | Loss >5pp queries |
|---|---|---:|---:|---:|
| hc2 | A_unaligned_once_minus_E_MBR | -0.027801 [-0.076025,+0.016603] | -0.009766 [-0.058081,+0.037455] | 0/16 |
| hc2 | B_aligned_once_minus_E_MBR | -0.218821 [-0.480469,-0.004661] | -0.288805 [-0.634208,+0.029727] | 0/16 |
| hc2 | B_aligned_once_minus_A_unaligned_once | -0.191020 [-0.434527,+0.031376] | -0.279039 [-0.610586,+0.040977] | 0/16 |
| vidstg | A_unaligned_once_minus_E_MBR | -0.029474 [-0.080689,+0.011311] | -0.061365 [-0.185431,+0.067628] | 0/16 |
| vidstg | B_aligned_once_minus_E_MBR | -0.047250 [-0.108855,+0.006473] | -0.250204 [-0.422344,-0.100760] | 0/16 |
| vidstg | B_aligned_once_minus_A_unaligned_once | -0.017776 [-0.088403,+0.043169] | -0.188840 [-0.415029,+0.000599] | 0/16 |

## Feedback coverage

| Domain | Interface | Admitted / returned anchors | Candidates with feedback | Queries with feedback |
|---|---|---:|---:|---:|
| hc2 | A_unaligned | 9/1224 | 8 | 5/16 |
| hc2 | B_aligned | 175/1224 | 87 | 14/16 |
| vidstg | A_unaligned | 121/1104 | 86 | 12/16 |
| vidstg | B_aligned | 630/1104 | 239 | 12/16 |

## Where improvement is used or lost

All-position gated accounting retains illegal/no-feedback samples as zero correction and query as unit. Accepted-only quantities additionally report their available query counts in JSON. Unknown GT is excluded, not scored as failure. Teacher-vs-E evaluates whether feedback contains information beyond recovery of exploration noise. Compression compares the SAME accepted local boxes with the shared4 target they induced. Actual final once update is separately evaluated over complete tube and unobserved frames. These are different supports and not an additive proof of causality.

| Domain | Interface | Local teacher - sampled input pp | Local teacher - E pp | Shared4 - E pp | Shared4 - local teacher pp |
|---|---|---:|---:|---:|---:|
| hc2 | A_unaligned | +0.030505 [+0.007572,+0.060493] | +0.002462 [-0.032752,+0.031397] | +0.003265 [-0.031949,+0.033005] | +0.000803 [+0.000000,+0.002410] |
| hc2 | B_aligned | +1.244129 [+0.447901,+2.162485] | +0.018568 [-0.240032,+0.229764] | +0.029298 [-0.191211,+0.233696] | +0.010730 [-0.067566,+0.091867] |
| vidstg | A_unaligned | +0.494660 [+0.087727,+1.077227] | +0.090185 [+0.001338,+0.190846] | +0.096409 [+0.017358,+0.179316] | +0.006224 [-0.030727,+0.042338] |
| vidstg | B_aligned | +3.513272 [+1.448317,+5.891429] | +0.668880 [+0.079536,+1.370920] | -0.022052 [-0.278162,+0.263239] | -0.690932 [-1.354750,-0.173425] |

## Layer evidence

| Domain | Interface | Mean target scores at layer1..6 (query means) |
|---|---|---|
| hc2 | A_unaligned | 0.059198, 0.029652, 0.021604, 0.018679, 0.017208, 0.017514 |
| hc2 | B_aligned | 0.059641, 0.062862, 0.074655, 0.097433, 0.117084, 0.130807 |
| vidstg | A_unaligned | 0.287730, 0.163884, 0.105693, 0.090872, 0.088010, 0.087407 |
| vidstg | B_aligned | 0.289481, 0.346830, 0.364229, 0.367120, 0.375480, 0.377138 |

Per-layer head readouts and actual next-layer references are distinct arrays. Native replay and cloned original reference/content/exclusion are qualified across all6 layers. Native initial reference is now measured, not inferred from a postprocessed final box. Layer traces diagnose response; they do not license best-layer selection or replacing the final prediction. Classification is hidden/text matching, not geometric correctness.

## Decision

stop_after_mask_pair_and_once_update. Prelocked continuation requires B coverage>A and B once full vIoU>E in BOTH domains. No target-selected thresholds, layer choice, state dimensions or propagation changes. If qualified, only original32 dynamic3x8 is authorized, with each round correction SUM divided by24, matching explicit cumulative weight to Fixed24. Teacher is reevaluated at new states; no path splicing.

All signed rows, noops, uncertainty and negative tails remain public. No geometric IoU-only identity claims. No new training/backbone/QR/LN/Qwen/SAM/historical replay. Raw media, text/parent identifiers, GT/coordinates, weights and layer tensors remain private. Costs exclude old Source/E and are not complete first online latency.
