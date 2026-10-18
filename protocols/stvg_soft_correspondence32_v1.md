# Frozen visual-correspondence feedback qualification32

New explicit user authorization. First phase only reward qualification, no TTA,
training, new correspondence head, diffusion, Qwen, SAM, extra DINO or GT routing.
Same exposed rollback32 (HC16/Vid16), four planned original DINO slots; only
original admitted Top1 boxes produce maps. Keep all queries, empty references
produce uniform candidate weights/no feedback. Native time is fixed. Visual
features come from frozen original Source appearance input-projection maps BEFORE
text fusion, captured as a read-only hook during original Source forward. No new
visual encoder. Reuse same frames/pixel hashes; Source output must replay exact.

Each query samples M32 original antithetic logit-cxcywh Gaussians with original
domain sigma/tau and stable original query/round0 seed on all actual decoder
frames, centered at original online Before output. Existing observed round0
samples are copied exactly at admitted support. All feedback conditions share
identical detached samples. Not calibrated posterior probabilities.

For each anchor: L2-normalized local features, target-to-anchor row softmax of
cosine similarity /0.07, over valid anchor patches. This fixed temperature is a
single interface definition, not target-GT calibration or a sweep. All valid
anchor patches may match all target patches; no motion-radius/Native-box gate.
The anchor mask is fractional cell overlap with its box, not a segmentation.
Transfer mask A*m; target invalid padding gets zero. Equal mean of four/admitted
anchor soft-IoU scores, not averaging identities or constructing a teacher tube.
Soft IoU integrates the fractional target box/cell overlap; area includes the
whole original box, even outside image. Keep separate per-anchor maps/scores and
disagreement. Zero valid anchors => zero reward, uniform weights.

Controls: (1) real frozen correspondence; (2) one fixed cyclic derangement of
unobserved frame features, anchors fixed; (3) original boxes interpolated on
physical frame IDs, constant nearest anchor outside range, same IoU reward.
Both admitted and rejected planned DINO slots are excluded from primary
unobserved support. Far=physical distance to nearest planned slot >=25% of input
span. Fixed threshold before scoring. No sparse-frame GT is invented.

Seal all32 maps/samples/rewards/weights before GT rank scoring. Offline GT only:
weighted candidate true IoU minus uniform mean, query means then parent means,
observed/unobserved/far/native-decoder-only separate, unknown frames counted and
excluded only from evaluable support. Common full-cohort size/motion bins and
query type, all signed effects. Model-assisted old wrong-reference labels may
be joined only for offline attribution, never online eligibility.

Gate for phase2: both domains unobserved parent mean rank increment positive
with parent-bootstrap CI95 lower>0; real-minus-shuffled and real-minus-interp
paired parent CI95 lower>0; far-support mean real increment>0 and real exceeds
both controls. Need >=8 evaluable independent parents/domain for unobserved and
far. No post-hoc threshold change. These are finite research allocation gates,
not universal impossibility or evidence of calibrated rewards.

Only pass triggers ONE three-arm matched online32: original sparse OPD, sparse+
soft-feedback OPD, same-feedback direct actual-parameter optimization; same
initial state/order/LN carry/1792/steps, reward maps frozen, actual decoder outputs.
Current and inherited effects, harm and cost mandatory. Not GT regression.
No pass => stop this feature interface, no training or scope expansion.

Phase1 cap900 GPU-lease sec /2h wall; shared GPU lease,16/20GiB memory,swap1GiB,
host6GiB/GPU4GiB/disk30GiB. Actual model/correspondence/CPU times separate. Raw
maps/media/text/GT coordinates/weights private. Publish code, exact protocol,
all anonymous signed scalars/unknown/negative reports with remote/archive checks.
