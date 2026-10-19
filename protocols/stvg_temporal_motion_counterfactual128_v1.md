# R22 temporal-mean motion counterfactual, frozen HC128 capability

R21 early ROI hard+.056pp/expected-.074pp did not establish advantage and is
closed. Keep spatial implementation/config/all E/Live41/42/43 coordinates fixed,
same Source Vid checkpoint, original65 positions and original text/subject.
No new expert/checkpoint/model/head training, optimizer or student fit.

Question: does removing time variation from the existing motion field expose a
component of the native interval preference that fails to depend on this motion
variation? This is a capability probe, not OPD and not a causal decomposition.
Old R5 noun-caption contrast used the same quotient and failed (-4.936pp versus
Native, -4.603 versus PriorSquare); preserve that negative evidence. Here ONLY
the visual motion slice changes, not caption. VCD (Leng et al., arXiv2311.16922,
original §3) motivates contrasting original/distorted-input distributions, but
its LVLM Gaussian-image-noise and token plausibility rule are NOT transferred.
The unit coefficient below is a fixed experimental design, not an assertion of
official VCD reproduction or a calibrated likelihood ratio for event times.

Use R21 Source-exact cached INPUT to ground_encoder.encoder. For EACH of original
33/32 offsets separately and EACH spatial cell/channel, take FP32 mean along T
of lastL motion tokens and broadcast to the same T. Only valid motion cells
change; original padding pattern must be time-constant or stop. Appearance,
text, original dtype(FP32), position tensors/masks/durations unchanged. No joint
65-frame mean, norm compensation, scaling, caption edits, offset changes, noise
or alternate erasure. Constant motion is exact identity. Re-run original early
fusion in original AMP, capture pre-norm prefix, then qualified native FP32
full two-pass query/gate decoder replay. Native input/prefix/time exact controls
every parent; Source RGB capture reused, no new RGB/vision encoding. Fusion
encoder replays are real computation and counted (max512), full native prefix
replays max256. New Source spatial boxes discarded; frozen spatial arrays only.

MeanMotion is NOT a static video/pure language or visibility prior: original
motion features already encode temporal windows/actions, appearance varies,
positions remain. Mean can shrink norms and mix moving foreground/background
into out-of-distribution ghost features. Save norm ratios and scalar changes;
do not hide this alternative explanation or claim newly proven teacher ability.

Original legal65/2080 decision grid and physical two-offset CDF/envelope/MBR
unchanged. Build qMeanMotion using original rules. On EXACT p0 positive support,
Contrast=normalize(exp(2log p0-log qMeanMotion)). Nonfinite or unequal support
is engineering failure; no epsilon/clamp or mass expansion. qMeanMotion==p0
returns p0 exact, uniform denominator equals normalized p0². Save/report Native,
MeanMotion raw, PriorSquare, Contrast, HiResAuxMix. Contrast ONLY eligible; no
coefficient, mean-region, norm, posterior mix or temperature scan, no per-parent
gate. PriorSquare controls generic sharpening, strong HiResAuxMix retained.

All128 old-exposed RC parents/onequery each, no TC109 selection; three frozen
spatial seeds mean WITHIN parent. All128 predictions and counterfactual prefixes
globally seal before newGT. Independent per-cell FP64 scalar means compared to
FP32 accumulation: absolute bound (T+2)*eps32*mean(abs(x))+eps32*abs(mean64),
with tiny floor 1e-30 only for reporting scaled error, NOT changing features.
This forward-error bound is prelocked; Native exact is never relaxed. Audit
nonmotion/padding/pos exact, unchanged support, quotient algebra, independent
CDF/MBR, official/Dense outputs, full-support expected candidate GTquality,
tails/LOO/all parents. GT used only offline after sealing, never online gates.

Expected Contrast GTfullv gain>=2pp with positive pairedCI and exceeds both
PriorSquare and HiResAuxMix with positiveCI qualifies ONE actual controlled
transfer, not system gain. Hard fullv>=2pp with CIpositive and exceeds both
controls but no distribution advantage qualifies ONLY static confirmation.
Otherwise close this mean-motion/quotient config without coefficient/norm rescue.
Raw MeanMotion cannot become a posthoc winner. Final method needs independent
parents plus Direct/fresh Fixed/Live; no CURRENT/MAIN/PAPER or spatial promotion.

GPU<=900s including load/IO/3s cleanup reserve; first2 noGT maxcase*128+load+60
estimate must fit. Original nonblocking gpu.lock; host6/disk30/freeGPU4 GiB,
allocator.8 and5GiB reserve, never signal other services. Save partials/failures
and actual calls/cost, no auto budget reset/restart or old queues/automations.
Publish explicit allowlisted code/protocol/all signed unknown/negative results,
actual remote byte/hash/tree receipts; no raw text/IDs/GT/cache/media/weights.
