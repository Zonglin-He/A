# Original32 cached reference-offset and decoder-path diagnosis

Scope: original HC16/Vid16 content-matched pilot only. No new model/decoder,
processor, video, weights, official annotation or GT scoring. No threshold,
parameter, mask/content rule, roster or prediction changes. CPU600s cap.

Reconstruct identical fixed24 student references from saved Source/actions/scales.
Use same-frame high-content conditional final boxes/scores and native final
high-slot scores. Recover native final high-slot BOX only with an exact native
slot ID plus exact target-score match from the original context probe. Fallback
placeholder IDs are not native slots. The one metadata/input-exact full900 cache
is checked as a provenance positive control, not another cohort. Unmatched frames
are unknown; do not fill them using another candidate or nearest final box.

Native FINAL candidate box retains original image-boundary clipping/coordinate
roundtrip. It is a geometry proxy, not the raw native bbox tensor, GT or native
INITIAL proposal/reference. Measure center displacement relative to native final
width/height and log-size ratios separately, and input/output overlap. Report
fixed descriptive IoU bins [0,.25,.5,.75,.9,1], center-error bins [0,.5,1,2,inf]
and log-size-error bins [0,.25,.5,1,inf]; no threshold/temperature sweep or new
accepted output. Native score>=.35 subsets are descriptive controls, not selection
for deployment. Keep all unsupported/illegal/missing-native-box cases explicit.

For position sensitivity hold query/frame/target-high content AND attention-mask
slot fixed, deduplicate FP32 student references, compare nearest/farthest by
native-final IoU within each such group. Average group effects within query,
then report parent/query bootstrap (10000,seed20261018). This measures responses
to different stored references under fixed other inputs. It does not identify
where in the six-layer path the effect occurred or estimate independent causal
percentages of center/size/attention. Center and size associations can covary.

For decoder behavior measure ONLY net input-to-final correction: toward/away
from the native final geometry, residual distance, final score, and geometric
recovery without score recovery. Inspect the local frozen implementation for
reference-dependent positional embeddings, deformable attention, iterative box
updates and final hidden-state/text scoring. Source-code mechanisms are distinct
from measured layerwise trajectories. Inventory cache keys for initial native
reference, intermediate boxes, hidden states, logits, sampling locations and
attention weights. If absent, layer-specific score drop, cumulative vs first-layer
failure, overshoot, and clamped/frozen-reference counterfactuals remain unknown.
Do not run a model to fill those gaps. .detach stops gradients, not forward box
refinement; this study has no gradient update.

Publish anonymous scalar rows, configuration, code, negative findings and
explicit limits through established export; exclude raw coordinates, text,
parent IDs, media, model tensors and private histories. CURRENT unchanged.
