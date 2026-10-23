# BoxFixed with class-free target-region prototype feedback

The frozen spatial method uses one ReferDINO and eight original observations,
shared geometric initialization, complete-box propagation and original path DP.
Observed=.5, guard=.1, link=.25; no spatial parameter changes.

The temporal prior is the original Global6 two native224 offsets/six layers with
old hull mean and original physical-time MBR. Construct a full-time BoxFixed
auxiliary trajectory independently of the predicted event interval. Expand each
frame box width/height by1.5, and fractionally area-average the original fused motion
visual memory grid before TimeDecoder. The actual TA adapter outputs256dimensions.
The full caption is already fused; text tokens are excluded from spatial pooling.
No RGB crop/re-encoding, extra classifier, query fitting or network training.

For heldout offset, fit the other offset mean, center and unit-normalize frame
vectors. Kernel K=(1+cos)/2. Originalq0 soft membership weights event/background
kernel means; shrink both toward unconditional fitting-fold kernel mean with
kappa16. Use their log ratio as a compatibility score, inclusive interval sum,
ONE q0-prior update and original MBR. Offsets are dependent; this is not independent
validation or calibrated event probability. Zero contrast gives exactq0.

This extra feedback does not require a predefined word/action classifier or class
vocabulary. The originalSource model still runs its native ASA/TTS architecture.
Only TA-STVG is implemented/tested. Structural portability to other STVG/MLLM feature
interfaces does not establish their empirical effectiveness. Correct targetlocation
has NOT shown a reliable advantage over fullgrid/reflected controls; no spatial
causality or reliable event-recognition claim is established.

The exact additional-parent recheck was not qualified: Vid relative toK16 loses
1.0688pp time and.5542pp completev,8timewins17harms and4severe vs1. The user explicitly
accepts these exact differences and selects the frozen method for full evaluation.
Both failed development/recheck qualifications remain intact; this selection does
not relabel previously exposed data or constitute statistical equivalence.
