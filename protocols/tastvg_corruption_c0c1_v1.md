# TA-STVG corruption C0/C1: frozen anatomy and student support

Registered before new inference or this round's GT scoring, 2026-09-29.
The current user attachments authorize C0 and C1 as the immediate experiments.
The longer critic/OPD/reliability roadmap remains conditional and unexecuted.

## Question and scope

Observed: Round2 native H supports correction, but ASA vulnerability does not
predict gain and naive preservation is not task safety. Historical controlled
corruptions did not show stable mean frozen video harm. Test the existing
corruption panel and native candidate support, without changing severity.

Use the original ordered 32 development parents in
`artifacts/c1_controlled_corruption_v1/INPUTS.json`, one VidSTG-test query per
parent. All are historically exposed; this is not fresh confirmation.
Primary only: VidSTG-source TA-STVG on VidSTG. HC2-source is reserved for later
transfer, not run or used for selection. Same official checkpoint, two offsets,
unchanged per-parent physical sampling (20–200 frames; first smoke has48), input resize224 and FP16 prefix/FP32 suffix contract
as the historical frozen artifacts. Report actual frame counts if different.
Use unchanged clean/noise_light/noise_medium/defocus_light/defocus_medium/
jpeg_light/jpeg_medium transformations and deterministic per-frame seeds.
These are custom severity levels, not ImageNet-C certification.

C0:32 x7=224 frozen conditions. Reuse historical native predictions after SHA
verification. Supplement missing evidence with a full native frozen forward;
require pixel hash, preprocessed pixel hash, boxes and both endpoint logits to
match exactly. Capture both TTS branches, both stages of ASA and Qs/Qt, exact
selection lists. No adaptation, gradients, new expert calls or H attacks.

C1:the first16 parents in the unchanged input order, x clean/noise_medium/
defocus_medium/jpeg_medium=64 cells. Selection uses ordinal only, no outcomes.
Capture final-routing-pass decoder outputs at the six existing layers. These
are deterministic native intermediate hypotheses, not stochastic policy samples.
No query noise, augmentation, extra rollout, checkpoint or severity sweep.

## Locked candidates

Spatial:whole-video box tubes from final-pass layers L6,L5,...,L1 in that order.
Remove exact tensor duplicates; retain native L6 as candidate0. Never select a
different layer per frame. Other branch (temporal interval) remains native.

Temporal:at most8 unique physical intervals. Candidate0 is native final-layer
two-offset envelope. Then add envelopes of L5,...,L1, exact interval deduplication.
Fill remaining slots by highest joint final-layer legal-span log score across
the two offsets: each offset requires start<end; pair score is the sum of offset
log probabilities, readout is the same min-start/max-end envelope as native.
Order ties by row-major offset span indices, then Cartesian row-major order.
For each physical interval retain its first origin and list layer origins in the
saved raw layer predictions. No GT used in candidate generation or deduplication.
Use logit/log-softmax arithmetic matching FP32 native MAP. Enumerate top pairs
with a heap, independently audit against exhaustive small cases and actual
NumPy complete Cartesian enumeration until8 distinct envelopes are recovered.
If fewer than8 legal unique intervals exist, retain all and report the count.
Other branch (boxes) remains native. No probabilistic OPD claim follows merely
from deterministic candidate headroom.

## Readout and decisions

Seal all224 captures and64 candidate sets before opening this round's GT subset.
GT from the existing label container:stream and retain only the32 authorized
keys, then evaluate. Existing history already exposed these labels; this new
seal rule is about computation dependencies, not claiming never-seen GT.
Validate physical grids/event masks/box-valid masks against saved input grids.
Use existing dual-implementation s/t/v scoring. Native sIoU uses all GT-valid
frames; vIoU uses the established sampled support/temporal-envelope denominator.

C0 per-condition paired corrupted-minus-clean s/t/v,10000 parent bootstraps,
seed20260929. Track >5pp losses, baseline>=.5 retention, success and failure
examples by deterministic largest-loss / smallest-absolute-change ranks.
Signed losses dS=cleanS-corruptS,dT=cleanT-corruptT. Four exhaustive descriptive
groups with delta=.05: robust/no-material-S/T-loss if max(dS,dT)<=delta;
S-dominant if dS>max(dT,0)+delta; T-dominant symmetrically; otherwise joint/mixed.
Do not call robust cases absolutely correct:report clean accuracy separately.
Internal drifts:TTS sigmoid-normalized JSD; ASA cosine/top20 support IoU on
common selected frames only with coverage and selection Jaccard; Q cosine and
relative norm. These are associations, not causal failure attribution.

C1:GT selects one whole temporal candidate by tIoU and one whole spatial
candidate by sIoU (ties first/native). Report native, T-oracle,S-oracle,combined
oracle and actual vIoU including negative tails; component maxima do not imply
video improvement. Also uniform-candidate mean as a support-only control, unique
counts, chosen-layer/origin, >5pp opportunity fraction, good/bad-case details.
Report clean separately and each medium condition separately; primary corrupted
summary averages the three medium cells within each parent before bootstrap.
Separate deterministic layer-only headroom from extra final-logit top spans.

Finite resource gate for the next critic qualification, separately by branch:
corrupted parent-macro target gain>=.02, bootstrap lower bound>0, and>=4/16
parents have mean corrupted target gain>.05. Passing only supports testing a
critic on this candidate set; no critic/adaptation is automatically run now.
Failure prioritizes improving that branch's candidate generation, not stronger
experts. Uncertain CIs remain inconclusive, not route impossibility.

## Resources and invariants

Serial5090, cumulative GPU-process wall cap3600s including failures/loading;
new artifacts<=8GiB and free disk>=8GiB. CPU analysis excluded from GPU allocation.
At most224 new two-offset frozen captures; smoke output reused. No backbone
optimization steps, model updates, new downloads, external expert inference,
OPD, gates, preservation, PCGrad, low-rank adaptation, online memory or PTD work.
Constructor annotation opens denied, query parsing reused from input-only cache.
Keep all failures and source revisions. Protect CURRENT registrations and old
artifacts. Finish with independent readback, archive update, public anonymous
code/results export and remote byte/SHA verification.
