# Prior-preserving temporal increment screen, HC128

2026-10-19 user authorized cache feedback reweighting, fixed space, no new expert
calls/training/modules. This is a capability screen, not temporal OPD. All128
existing RC parents are development data. Primary spatial predictions are the
unchanged current Live seeds41/42/43, metrics averaged within parent; unchanged E
coordinates are secondary. Never choose a spatial seed or feedback per outcome.

p0 must be exactly the distribution accepted by current final_mbr: native FP32
endpoint logits -> per-offset full legal endpoint softmax -> original min-start,
max-end physical envelope mass_table. Reuse exact compatible cached logits;
when absent, export frozen Source only, no DINO/ReferDINO, optimizer or student
updates. Source output must match original Source boxes exactly and p0 MBR must
recover the saved original time exactly for every parent. Save raw logits/q.
GPU lease cap900s including load/decoding while leased; shared gpu.lock, device
free4GiB, host6GiB, disk30GiB; finite4h orchestration deadline; no other services
signalled and no old queues/automations resumed.

Do not project p0 onto the preceding experiment's coarse Voronoi interval pool.
Keep all65 physical endpoints, all strict-index start<end hypotheses and all
original strict-index MBR decisions, including zero-mass decisions. No new
candidate pool, pruning, smoothing epsilon or MAP readout. Actual frame intervals
are [ids[a],ids[b]+1). Zero prior mass remains zero. lambda=0 returns q bitexact,
and uses the identical existing final_mbr code. Missing/flat feedback returns p0.

Feedback is exactly the existing EventLive curve for each primary spatial seed,
or EventE for E: same eight observations, word-balanced premax event logits,
same globally associated teacher track, same Voronoi duration inside-minus-outside
contrast. Evaluate that same formula on the original MBR's interval support.
Standardize within each video/feedback curve using unweighted population mean
and standard deviation of R across all legal intervals with defined outside
contrast, including zero-mass decisions; no GT, clipping or learned calibration.
Full actual video has undefined outside contrast and standardized increment0.
If actual video extent is not on the original endpoint support, do not add it.
sigma<=1e-12 or flat curve leaves p0 unchanged. Eight-frame duration integration
is not new dense visual evidence, and intervals with no independent outside
observation are not reliable event-negative labels.

Only lambda in {0,.5,1,2}. p_lambda proportional p0*exp(lambda*standardized_R),
stable log-space calculation on positive mass. Identical native MBR utility and
deterministic physical row-major tie rule. Prelock all128 readouts before new GT.
Select one common lambda by full-vIoU mean over128 parents after averaging all3
seed metrics within parent, tolerance1e-12 ties smaller lambda including0. No
per-parent/error-type/seed parameter choice; no more grid values or R changes.

Primary full-vIoU gain at fixed space; secondary tIoU, absolute start/end errors
in physical frames, severe declines, all signed parent/seed results and posterior
KL/L1 diagnostics. Same spatial predictions imply actual sIoU invariant. Paired
10000 parent bootstrap, not treating seeds, queries or intervals as independent.
Recompute old reported mean gain/loss contributions with the original denominator
as an arithmetic check, not a new candidate or selection mechanism.

Postseal GT-only error classes fixed from original interval: zero overlap ->
wrong segment; proper contained prediction -> missing part; proper containing
prediction -> excessive outside time; remaining mixed boundary offsets or exact
match -> other (exact count disclosed). For each class report original temporal
loss relative to exact GT-time readout at the same fixed spatial trajectory;
maximum recovery over original MBR decision support; actual selected-lambda
recovery. All three use original group parent denominator and original spatial
coordinates, not cross-baseline subtraction or GT online gates.

Confirmation: the current validation split has237 parents, only109 outside RC128.
Prelock all109 identities before development selection; old RH/RF/NH overlap and
earlier whole-Source/old-method exposure explicitly disclosed. This is below the
requested minimum128, not an automatic relaxation: user scope answer is pending.
No token caches currently identified for them; under current no-new-expert scope
do not fabricate confirmation or relabel development data. Separate user queries
ask whether to allow109 independent parents and whether to retain no-new-expert
restriction or authorize only missing confirmation evidence. No such calls before explicit answer. If lambda0 wins, it is the
identity baseline, not a discovered temporal component; report no positive screen
and do not claim independent confirmation or launch costly teacher runs merely
to confirm an algebraic identity. If a nonzero lambda wins, freeze it before any
confirmation predictions/GT and execute only authorized evidence collection.

lambda0 mass/interval exactness, independent brute expected-tIoU probe decisions,
feedback integral recomputation, official-vs-DenseTube metrics and source/input
SHA checks required. Publish code/protocol/all actual outputs' signed scalar
measurements/negative findings and exposure/confirmation limits to Zonglin-He/A;
raw images, query text, parent identifiers, weights, logits/posterior caches and
GT coordinates remain private. CURRENT/MAIN/PAPER and frozen spatial package unchanged.
