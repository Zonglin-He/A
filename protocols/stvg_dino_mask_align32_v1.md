# Original32 content/mask alignment and one shared4 update

Original exposed HC16/Vid16, original4 image positions, the exact fixed24 actions
from the repaired pilot, same Source/E/scales/final-MBR and high-content slot.
A excludes geometry-nearest native slot. B excludes the high-content SOURCE slot.
References, text/images, native900 context, learned content, legality and .35
threshold are identical; only the external query's native-attention exclusion
changes. Native queries cannot read extras; extras cannot read one another.

One ordinary native image/text forward per original active anchor, shared between
A/B. Save native initial900 reference and six-layer boxes, references and target
scores, plus appended-query six-layer boxes/references/scores/hidden states.
Use canonical per-layer bbox heads with that layer's input reference and normalized
hidden; actual propagated references are separately stored. Native replay and
clone reference+content+same-slot mask qualifications cover all six layers and
include high native slot, middle and last. Same-mask controls match; A's final
boxes/scores/admission reproduce the prior high-content unaligned pilot.
No best-layer selection. Highest native target slot is first argmax, without GT.

For each condition compute teacher action y_k as the mean accepted anchor
correction in (cx,cy,log w,log h), normalized by the original source scale.
No feedback/illegal whole-candidate input means y_k=a_k. Each condition makes
ONE external-state update mu'=mu+sum(y_k-a_k)/24 from fixed E mu. All24 samples
remain in the denominator; no random-mean drift. Four unsupported Vid queries
stay no-op and in full32 metrics. No network parameter update or new training.

Seal both complete output trajectories and all feedback on all32 BEFORE new GT.
GT used only offline on its exact known support. Diagnose teacher versus sampled
input, teacher versus current E, local teacher versus compressed shared4 action,
and actual once-updated whole tube versus E on complete vIoU/unobserved actual
Decoder-frame sIoU. Report accepted-only and all-position/noop-denominator views,
query-level 10000 bootstrap(seed20261018), signed tails, no-feedback and unknowns.
Native geometry agreement is not GT or identity accuracy.

Prelocked continuation: B coverage exceeds A in BOTH domains AND B one-update
complete vIoU exceeds E in BOTH domains. No extra CI or target-tuned gate. If met,
only a bounded same32 aligned dynamic3x8 comparison is authorized. Its per-round
update uses sum8(y-a)/24, so explicit cumulative weight matches Fixed24. Same24
noise chunks/seed, E start, scales, .35, original4 support, frozen model and MBR.
Teacher responses recomputed at new dynamic references; cannot splice old paths.
Compare dynamic vs Fixed24/B-once and E. No automatic enlargement or tuning.

Total GPU budget900s for stage1+conditional dynamic, total wall2400s from initial
lock, no reset. Shared GPU lease and original host/GPU/disk guards, 16/20GiB
MemoryHigh/Max, Swap1GiB. First original case/domain times and qualification.
Publish actual implementation/configuration, anonymous scalar results, layer
summaries and negatives, excluding images/text/GT/coordinates/weights/raw tensors.
CURRENT and prior predictions unchanged. No old queues, Qwen, new critic, SAM,
backbone, QR/LN adaptation or extra evidence frames.
