# Why native high-score content stays low at student references

Original32 exposed development, cached-only. No model/decoder/processor/weights/media/GT access, no threshold or roster changes, no new predictions. The previous matched intervention held references and masks fixed while changing content. This diagnosis varies stored references under the same high content and attention-mask slot, and measures decoder NET input-to-final changes. It does not invent per-layer traces.

## Available evidence

| Domain | Native final box recovered / active anchor frames | Matched references | Admitted | Input/native final IoU median | Output/native final IoU median |
|---|---:|---:|---:|---:|---:|
| hc2 | 59/64 | 1136 | 9 | 0.220396 | 0.159344 |
| vidstg | 40/48 | 912 | 114 | 0.754430 | 0.940933 |

Native box recovered only by matching original native slot plus target score exactly. Fallback placeholder IDs and missing native Top3 slots remain unknown. One original-cohort full900 cache verifies4 matching frames; no new cohort added. Native FINAL candidate box is neither GT nor INITIAL native proposal; the original probe clipped xyxy to image bounds and converted back to cxcywh, so it is not the raw native bbox tensor. The one full900 case reproduces this original postprocessing exactly. Relative offsets below describe distance to this final detection, not exact displacement from the native starting reference.

## Reference offsets and net correction

| Domain | Input overlap bin | References | Admitted | Conditional score median | Output overlap median | Net overlap gain mean pp |
|---|---|---:|---:|---:|---:|---:|
| hc2 | [0,0.25) | 586 | 0 | 0.004828 | 0.000000 | +0.865708 |
| hc2 | [0.25,0.5) | 246 | 0 | 0.005478 | 0.474797 | +6.959895 |
| hc2 | [0.5,0.75) | 271 | 3 | 0.006967 | 0.656347 | +1.112614 |
| hc2 | [0.75,0.9) | 33 | 6 | 0.014833 | 0.984514 | +12.842154 |
| vidstg | [0,0.25) | 188 | 0 | 0.011145 | 0.000000 | +1.062223 |
| vidstg | [0.25,0.5) | 106 | 0 | 0.014140 | 0.354128 | -0.352320 |
| vidstg | [0.5,0.75) | 155 | 11 | 0.007440 | 0.711439 | -5.448527 |
| vidstg | [0.75,0.9) | 338 | 66 | 0.017082 | 0.986702 | +11.479561 |
| vidstg | [0.9,1] | 125 | 37 | 0.026420 | 0.995030 | +6.370224 |

Center offset is normalized by native final width/height; log-size offset is Euclidean norm of log width/height ratios. Separate descriptive bins are in DIAGNOSIS.json. They can covary and are not independent causal percentages. Same-query candidates are not independent statistical samples.

## Fixed content AND mask reference variation

| Domain | Groups with >=2 distinct refs | Groups crossing .35 | Query mean nearer-minus-farther score [95% CI] |
|---|---:|---:|---:|
| hc2 | 199 | 2 | +0.006961 [-0.003330,+0.024517] |
| vidstg | 156 | 1 | +0.006011 [+0.001246,+0.012486] |

Within each query/frame/high-content/mask group, deduplicate FP32 references, compare largest versus smallest native-final input IoU. Aggregate groups within query first; intervals bootstrap queries. This directly observes reference sensitivity with content and mask held fixed; all decoder responses are still full six-layer responses. It cannot allocate that sensitivity to a particular layer.

## Geometric recovery versus score recovery

| Domain | Final overlap >=.75: admitted / refs | Final overlap >=.9: admitted / refs |
|---|---:|---:|
| hc2 | 9/128 | 9/63 |
| vidstg | 114/521 | 114/481 |

These are agreements with a native detector box, not correct-identity or GT claims. Native geometry and conditional classification are different outputs: recovering coordinates need not restore the original hidden/text compatibility. Conversely, final proximity alone cannot establish an early-versus-late decoder failure.

## What source inspection establishes, and what it cannot

The installed decoder recomputes sine positional embeddings from current references; deformable visual attention samples around the reference center with its width/height scaling. Each layer applies a bbox increment in logit coordinates and forwards the new reference into the next layer. Final target score is hidden-state/text dot product followed by sigmoid/target-token averaging. A high-score content ROW is not the native final hidden state. Replacing the reference therefore changes the actual visual and positional processing, rather than attaching a fixed confidence to a content vector. These are source-code facts, not measured attention maps.

`.detach()` affects backward derivatives, not forward coordinate refinement. It cannot by itself explain missing feedback in this no-gradient detector evaluation. Prior native replay/clone qualification checks the wrapper on native states; it does not qualify arbitrary student references.

The saved pilot contains only final conditional boxes/scores, selected slots, actions, control hashes and native top scores. Native initial references, DINO six-layer boxes/logits/hidden states/sampling points are absent. Other cached six-layer Student boxes are not DINO traces. We cannot tell whether score was low at layer1, decayed cumulatively, or briefly recovered then fell. No new forward was run to fill this gap.

Decision: retain the previous stop before OPD. This diagnosis narrows geometry and interface mechanisms but does not authorize lowering .35, tuning a layer or starting another experiment. A layerwise counterfactual remains unmeasured. Missing native frames/GT stay unknown. Current registry and all prior predictions preserved.
