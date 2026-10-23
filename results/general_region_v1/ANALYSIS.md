# Class-free target-content prototype feedback

The fixed candidate removes the **additional ASA word-class readout** from temporal
feedback and uses the original full-caption multimodal visual memory instead.
On the previously exposed128-parent panel per direction, it improves both time and
complete-tube means over Source, Global6 and K16. HC gains over Global6 are
**+2.6009pp tIoU / +1.3590pp complete vIoU**; Vid gains are
**+.4971pp / +.3352pp**. It does **not** pass its original strict qualification:
severe losses increase, and target-location-specific benefits are unestablished.
These outcomes are development evidence, not cross-backbone or MLLM validation.

## Fixed method and comparison

Read only the actual archived TimeDecoder memory, visual suffix after its text
prefix, before temporal decoding. Original full caption and all physical frames
remain. Pool fractional grid overlap with the unchanged full-time complete-box
trajectory expanded by1.5. Other-offset unconditional centering and L2 normalization
remove the common feature direction; compare frame representations through
`K=(1+cosine)/2`. Positive/negative kernel means use native Global6 soft membership,
shrunk with16 pseudo-frames to the fitting fold's common kernel mean. One log-ratio
feedback update uses the original prior; physical MBR and spatial DP are unchanged.

This is a compatibility score, not a calibrated event probability. No word axes,
ASA cross-attention/query, class probability, trained projection, optimizer or new
expert enters the primary feedback. The original Source architecture/training is
unchanged and may internally use its own ASA; this is not an ablation of that model.
Only TA-STVG was executed. A feature adapter of another model remains unvalidated.

Original same-input Source, Global6 and K16 posteriors/intervals/boxes are exact
aliases on all256 inputs. Constant features give bitwise native output. Six CPU
tests cover scalar-reference math, feature basis/translation/scale invariance,
constant no-op, synthetic event separation, spatial pool/reflection/fallback and
opposite-offset fitting weights. Cross-offset samples are dependent and their
separation is not independent validation.

## All128 parents per direction

Numbers are query means in percent; each query belongs to a different parent.

|Direction|Method|tIoU|Complete vIoU|
|---|---|---:|---:|
|Vid→HC2|Source|47.6781|24.5281|
|Vid→HC2|Global6|48.6807|28.2435|
|Vid→HC2|K16|48.9401|28.3906|
|Vid→HC2|RegionKernel|51.2816|29.6025|
|HC2→Vid|Source|36.4970|13.7207|
|HC2→Vid|Global6|37.9187|18.0909|
|HC2→Vid|K16|38.2376|18.3480|
|HC2→Vid|RegionKernel|38.4158|18.4261|

HC paired95% parent intervals versus Global6:
time[+1.0414,+4.1583]pp, complete-v[+.3997,+2.3210]pp.
Vid intervals: time[-.9053,+1.8947]pp, complete-v[-.3064,+1.1199]pp.
Versus K16, HC time+2.3415pp[+.9886,+3.7156], v+1.2119pp[+.3804,+2.0283];
Vid time+.1782pp[-.9541,+1.3609], v+.0781pp[-.6264,+.8694].

|Direction|Short<=5s time gain vsSource|Long>5s time gain vsSource|Time improvements/harms/ties vsG6|>5pp time harms K16→new|>5pp v harms K16→new|
|---|---:|---:|---:|---:|---:|
|Vid→HC2|+3.7043|+3.5655|75/36/17|10→22|5→11|
|HC2→Vid|+.7145|+3.2414|50/44/34|9→16|3→4|

Time/v harm counts describe different endpoints and must not be added as distinct
queries. Larger mean gains do not remove the negative tail.

## What the matched controls establish

Primary minus whole-grid time is only+.0528pp on HC and-.1459pp on Vid; both paired
intervals crosszero. Primary minus reflected-location time is+.1678/+.1393pp,
also inconclusive. Correct-target feedback and whole-grid feedback correlations
have medians.9667/.9321; target/reflected correlations.9565/.9164. Reflection
overlaps the target area by30.48%/38.79% on average and is not a known-negative object.

Exact-L2-matched K16 feedback is weaker than primary on HC by1.7052tpp
[+.5469,+2.8416], but stronger on Vid by.3034tpp, with a crossing interval.
Matched-L2 errors are audited separately. Wrong-time feedback damages HC strongly,
supporting time-placement sensitivity there; the Vid paired interval crosseszero.

Matched-candidate evidence ranks are55.24%/56.82% for primary versus
54.77%/58.60% for whole-grid and56.05%/57.21% for reflection. Consequently,
the data support a promising **native-representation temporal correction** on HC;
they do not establish that correct target position supplies extra event information.

## Concrete gains and failures

- HC114: Global6t27.43→61.65, but whole-grid63.83 and reflection59.47. A rescue
  exists, but cannot be assigned uniquely to correct target support.
- HC068: Global6t88.11→54.23; whole-grid also54.23. The predicted interval shortens
  by5.32s. This failure is not simply over-expansion or wrong ROI selection.
- HC103: Global6t97.24→79.25 while the interval expands by.918s. K16 retains97.24.
- Vid011: Global6t18.87→56.00, larger than whole-grid24.26/reflection25.67, but
  matched-strength K16 reaches73.37. This case alone does not identify a new mechanism.
- Vid118: Global6t93.19→46.71; whole-grid also46.71. The interval shortens6.607s.
  K16 also harms this case, to55.16. Shared weak pseudo-labels/state similarity remain
  plausible causes; these observations do not prove a unique root cause.

No query was skipped or protected with a GT/confidence veto. Empty-region fallback
to the valid full grid exactly matches original K16 on every sampled frame:
HC11/8320 and Vid783/13639 frames. This is preserved protocol behavior, not a newly
introduced fallback or a basis for rejecting queries.

## Execution, scope and user-directed continuation

All256 predictions sealed before an independent CPU labels process. There are6144
official-versus-dense metric checks; maximum error1.55e-15. Prediction55.98s and
scoring21.47s include CPU controls, spatial DP and I/O; neither is deployment latency.
There are zero new model/GPU/RGB/text/fusion/word-head/expert/observation/training calls.
Private feature arrays, media, captions, labels and box coordinates are not published.

The user then explicitly accepts small performance declines and prefers the general
method direction. The original failed qualification is **retained**, not rewritten.
The exact candidate is user-selected for an additional fixed32-parent-per-direction
recheck under a separately predeclared engineering tradeoff. No kernel, feature,
shrinkage, strength, layer or iteration is retuned, and no control is promoted.
K16 full remains paused at HC320/Vid0. New full is conditional on the recheck,
not on the ability to write a stronger story.

[Protocol](protocol.json) · [all metrics and paired intervals](report.json) ·
[signed outcomes](signed.tsv) · [all severe outcomes](severe.tsv) ·
[original qualification](QUALIFICATION.json) · [postseal audit](POSTSEAL_AUDIT.json) ·
[fallback compatibility](FALLBACK_COMPATIBILITY.json).
