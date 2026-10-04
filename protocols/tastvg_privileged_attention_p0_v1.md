# Event-scoped privileged spatial attention: P0 v1

User attachment f7175e26 authorizes a staged route. This stage asks whether the
same frozen TA-STVG obtains a spatial advantage from Grounding-DINO evidence.
No gradient, optimizer, OPD target, persistent state, temporal expert, candidate
probe, RGB crop, feature masking, or method promotion is permitted in P0.

## Frozen inputs and model

VidSTG and HC-STVG-v2 use their official same-domain EMA checkpoints, the original
Paper48 merged two-offset sample grid and exact cached corruption pixels.
Deterministically take original ordinals 0..7 (development) and 32..39 (historically
exposed confirmation) from the already locked C1 same-domain cohort. There are 16
sources/queries per dataset and clean plus frame-drop, frame-freeze, motion-blur,
occlusion and exposure at 5%, hence 96 cells per dataset, 192 total. There is no
stream/order effect because all parameters are frozen. Confirmation is source
disjoint within this panel, not fresh/unseen. No source is selected using GT or
outcomes. Checkpoint, input, cached H, parsing and implementation hashes are sealed
in the private runtime lock before new inference.

The native interval I0 and native frozen ordinary tube are the baseline; this is
not A8/UVTG, not the prior online C1 output, and not a 25% expert-budget evaluation.
Each query may request four original sampled frames inside I0 using the existing
DeCoTA uniform_positions rule. Fewer than four available samples are recorded,
not filled from outside I0. Every P0 cell is eligible for expert observations.

## One predeclared evidence field and one attention intervention

Use the existing local Grounding-DINO tiny checkpoint, FP32 official processor.
Use DeCoTA's cached conservative visual referent phrase/entity extraction;
direction/action words are not newly interpreted. Keep all finite, positive-area
raw proposals with the existing phrase score >= .35, after clipping coordinates
to [0,1]. No top-K, NMS, top-1 winner, margin gate, box-coordinate target or tuning.
Raw proposals, duplicates, scores, token/input receipts and selected indices are
retained privately. Empty phrase/evidence yields an exact no-op.

For each valid spatial token center (x,y), using normalized cxcywh boxes:

    G = exp(-.5*((x-cx)/(width/2))**2 - .5*((y-cy)/(height/2))**2)
    weights = softmax(phrase_score / 1.0)
    M = sum(weights * G)

Thus M is in [0,1]. Gaussian half-box standard deviations, tau_E=1, alpha=1 and
epsilon=1e-6 are implementation defaults fixed before evaluation, not selected
from results. There is no per-frame max normalization. Original-coordinate token
centers account for the cached spatial padding mask. Padded spatial keys and text
keys receive zero bias; their native masking behavior is not changed.

Apply alpha*log(M+epsilon) only on the observed frame's valid image keys in all
six blocks of the final (second) native PosDecoder. The first decoder pass,
native gates, queries entering the final decoder, all encoded features, temporal
decoder/head, and native I0 are held fixed. There is no change to the RGB or token
values. Image-vs-text attention mass may also change: this is the literal additive
image-key prior, not a claim of preserved cross-modal mass. Absent evidence/alpha0
does not add a mask and must exactly reproduce ordinary inference.

Native FP32 decoder/cache factorization is validated against real full inference
on the first fixed clean query of each dataset. Privileged reinsertion is also
validated on these two queries. All module hooks are exception-safe. Hook records
save raw per-head QK logits, actual additive bias, and returned average attention;
an independent CPU implementation must reproduce the attention math. Cross-frame
self attention can spread the intervention to unobserved frames; we measure this
and do not claim hard scope invariance.

## Seal, scoring and conditional continuation

All 192 ordinary/privileged prediction pairs and evidence must be globally sealed
before GT scoring. GT is only an offline diagnostic. Use the existing official
dense scorers and a separately implemented interpolation/IoU cross-check, paired
source macro means and 10000 bootstrap draws (seed 20261004). Report development,
confirmation, pooled, clean, each corruption; vIoU, full-GT dense sIoU and invariant
tIoU; gross gain/loss, >5/20pp harms, correct-to-wrong at .3/.5; native event support,
empty evidence, observed/unobserved changes and input/forward costs. Preserve
negative cases. Worker wall time includes loading and IO, not pure kernel time.

P0 advantage requires positive corrupt vIoU means on both development datasets
with the same positive direction in both confirmation datasets; CI and clean
negative tails determine evidential strength and are reported without hiding
inconclusive results. A nonpositive panel stops this implementation before OPD.
This gate is fixed before GT. No broad parameter search or Sa2VA swap is authorized
by this P0. If the gate passes, implement and freeze a separate episodic residual
distillation stage before it runs; only after that succeeds consider LN1/16.

Public export includes code, exact defaults, anonymous metrics, successful and
harmful cases, limitations and costs. Private media, annotation, proposal boxes,
attention/prefix tensors, weights and personal attachment text remain excluded.
Archive check/snapshot/check and verified GitHub synchronization are required.
