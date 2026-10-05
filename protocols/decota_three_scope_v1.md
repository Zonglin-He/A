# DeCoTA three scopes: finite acquisition, boundary and compatibility test

New authorization reopens only native TTS acquisition, dual-offset consistency,
and pre-update correction compatibility. It does not revive historical queues.
The deployed CURRENT_METHOD registry stays unchanged. Prior global rank-8
transfer did not qualify; this does not prove all global or conditional memory
impossible. The text supplied by the user is a scientific proposal, not results.

## Fixed inputs and spatial algorithm

Use the previous locked 32 development plus 16 confirmation sources per dataset,
one query per source, two fixed orders, clean plus frame_drop_5, frame_freeze_5,
motion_blur_5, occlusion_5, exposure_5. These sources have historical exposure;
confirmation means disjoint from this development set, not a fresh test. Official
VidSTG and HC-STVG-v2 same-domain EMA checkpoints and original sampled frames
are preserved. Private PLAN retains exact query/pixel/source/cache hashes.

Exactly one frozen Grounding DINO-T, original context parsing, admission .35
and distinct margin .05, old NMS .5 and top-three support construction. Admitted
per-frame Top1 singleton energy is negative box IoU; it is not ungated raw Top1
or an identity correctness guarantee. Joint query256 plus LN1536, Adam .03,
betas(.9,.999), eps1e-8, ten updates, first minimum own loss among steps0..10.
Query and Adam reset each arrival; Native temporal output is fixed; actual LN
writeback is selected delta times 1/16. No new spatial expert, loss, optimizer,
backbone or temporal expert. No parameter search or total deadline.

## Round 1: TTS-Stratified4 vs Uniform4

Official code uses (sigmoid(appearance)+sigmoid(motion))/2, delta=.5. Frozen
heads are evaluated on the existing normalized H; no video backbone is run.
For each offset map the actual score to its original frame; interleave into the
existing merged grid. Partition Native interval into min(4,available frames)
equal physical-time bins. Pick the greatest TTS score in each nonempty bin,
earliest frame on ties. Empty bins use the original Uniform4 farthest uncovered
physical-frame fallback, earliest on ties. No duplicate frame or outside-interval
fill. This preserves at most four observations and the previous frame support.

New positions require DINO forward on the identical corruption pixels. Reuse
old detections only at the identical frame, RGB and text/context receipt. No
new GT enters selection or prediction. Max 2304 new DINO observations over
576 unique query-condition inputs; actual overlap reuse and counts are reported.
Run both per-query episodic and independent 100%-expert online streams. Each
split/condition/order starts at source LN. All 2304 logical outputs are sealed
before scoring. Episodic order2 can reuse the identical query/source state with
a receipt. Uniform4 references are the already sealed actual episodic/online100
Top1 streams; Frozen is the same saved Native. A changed acquisition stream
never borrows old post-update state. This also supplies the unconditional
Frozen / S-only / Tscope+S integrated comparison in Round5.

Report source-macro official dense vIoU, tIoU, sIoU, paired 10000 source-bootstrap,
clean/corruption/order/condition, Before vs Frozen, current correction and net
matched memory. Record >5/>20pp tails against Frozen and Uniform4, observation
valid/admitted rates, event/scored-frame coverage and admitted Top1 GT IoU.
No GT-support observation is not scored as correct; missing evidence is explicit.
Observation IoU is a referent proxy, not an identity-label measurement. An
acquisition benefit is established only with both confirmation corruption paired
vIoU lower CIs >0 versus Uniform4; positive point gains or safety tradeoffs alone
remain limited/inconclusive, never silently promoted.

## Round 2: boundary consistency P0, CPU, no backward

Use cached Native start/end logits for the two actual original offsets. Softmax
each in float64. Linearly interpolate probability VALUES to the existing merged
physical sampled-frame grid, zero outside each offset support; floor1e-12 then
renormalize each discrete endpoint vector. This is not a calibrated continuous
time density, and the correlated offsets are not independent experts. Geometric
mean and normalize for separate start/end. Strict legal i<j; earliest lexicographic
MAP ties; readout [frame_i,frame_j+1). Same fixed spatial Top1 trajectory and
Native temporal baseline; no GT time enters decoder. Predictions, probabilities
and selections are sealed before diagnostic GT score. Report all 1152 old
online100 arrivals, all 48 sources each dataset, clean and corruption separately;
JS start/end/mean vs Native temporal error with source-bootstrap uncertainty.

Prelocked Round3 GO: both dataset confirmation-corruption paired tIoU and vIoU
95% CI lower >0, and both development-corruption means >0 for both metrics.
Failure/inconclusive means no temporal gradient. If GO, keep last temporal head
layer514, SGD .01 K3, Gaussian-free consistency teacher, stopgrad reference,
forward endpoint KL plus source-native anchor coefficient1, reset per query.
Align both predictions to the same physical grid. Do not expand parameters or
tune learning rate; separately lock the actual loss/interface before running.

## Round 4: pre-update compatibility P0

Reuse all 939 previously sealed, deduplicated exact single-write pairs. Utility
is the actual 1/16 donor write, source-initialized recipient Before minus Frozen,
before any second nonzero write; it is not full donor-delta utility or cumulative
long-stream improvement. Zero recipient-correction pairs remain in the target.
Use only raw-query frozen native RoBERTa masked-mean768, source-initialized
final spatial decoder latent mean256, and Native tube/time/TTS summaries. No
recipient delta, expert GT quality, gradient, selected step, utility or future
state enters a key. Native suffix replay and frozen text forward are recorded
costs; they are not described as pure CPU if run on GPU.

Label-free similarities: query cosine and spatial latent cosine. Fixed supervised
diagnostic: L2-normalize each embedding block, concatenate donor key, recipient
key, absolute differences and these two cosines; ridge alpha1 with intercept.
Standardization fits only training pairs. No tuning alpha, complex network or
search on confirmation. Development leave BOTH test donor/recipient source
nodes out of fitting, across all conditions and roles; confirmation fitting
uses development sources only, disjoint for BOTH roles. Labels are existing
GT-derived utilities, disclosed as supervised diagnostic development, not an
unlabeled deployable universal readout. Compare a training-mean constant null;
undefined correlation/class support stays missing, not zero or positive.

GO for conditional retrieval: at least the SAME predeclared signal in both
confirmation-corruption panels has source-node-bootstrap corr lower >0, or
help/harm AUC point>.65 with CI lower>.5. Exact zero utility omitted only for
binary AUC, retained for correlation/regression. Joint node bootstrap resamples
one common source count in donor and recipient roles, recipient-source-equal
base weights, 10000 draws. Development honest predictions and all failures
are reported, no in-sample claims. Multiple tested signals are exploratory;
qualification is for a subsequent locked matched memory trial, not deployment.
If no GO, retrieval is not implemented, baseline 1/16 remains.

## Round 5 and closure

Use actual independent Round1 streams for Frozen, S-only, Tscope+S comparisons.
Only qualified temporal or memory additions get a separate runtime and actual
independent online integration; no failed conditional arm is invented or called
completed. Preserve all negative and severe cases. Production is not promoted
from these development/confirmation results. Each material stage updates the
archive check/snapshot/check. Completion requires independent math/state/dense
audit, inspectable anonymous results, report/plots visual inspection, and exact
remote code/protocol/results verification at Zonglin-He/A. Private media, query
text, annotations, model weights, raw H/state/gradient caches and chat attachments
remain excluded. Engineering failures are preserved with a pinned revision;
scientific rules are not relaxed to pass. No recurring automation is created.
