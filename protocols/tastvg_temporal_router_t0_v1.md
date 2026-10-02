# T0: Temporal evidence router and consensus-weighted critic

Authorized 2026-10-02 after the H-lite A/H comparison (`5d6da2b`).
Question: does UniversalVTG proposal agreement improve event support and native
candidate selection, with the successful spatial A route frozen?

## Actual scope and data

Reuse the original 32 development sources / one query per source of each
dataset, two orders and clean + five transient 5% corruption conditions.
VidSTG uses its official same-domain TA-STVG checkpoint; HC-STVG-v2 uses its
official same-domain checkpoint. Original Paper48 input pixels, frame sampling,
physical coordinates, offsets, cached specialist inputs and 25% availability
stay fixed. All sources were historically exposed. This is development GT
diagnosis, not a fresh holdout or a new full-dataset experiment.

The sealed H-lite control A supplies 384 pre-update outputs per dataset, 768
in total. Only the 96 scheduled expert arrivals per dataset have a temporal
specialist. Distinct scheduled sources are 16 Vid and 14 HC; 80 corruption
expert arrivals per dataset are the primary temporal-router diagnostic.
Nonexpert arrivals retain their output exactly. No expert is imputed there.

Spatial A remains rank-RKL / SGD, 1792 parameters, nine probes, rho=.05,
student_temperature=1, D=4. Vid: lr=.033761698432507946, teacher_temperature=
.34902548789596055, K=1. HC: lr=.006097133675874025,
teacher_temperature=1, K=8. No changes to spatial targets or execution.

## Frozen formulas, before diagnostic GT

S: equal votes over actual deduplicated native student temporal candidates,
including native, at most eight. All interval membership is half-open in the
already cached physical frame coordinates. Existing proposal endpoints have
already been converted from seconds to physical frames; do not convert twice.

E: retain every cached UniversalVTG proposal and its confidence. For M>1,
a_m = sum_{n!=m} tIoU(J_m,J_n)/(M-1), q_m=c_m*a_m.
E(frame) = sum_m q_m*1[frame in J_m]/sum_m q_m.
SE is exactly (S+E)/2. No learned fusion, NMS, proposal deduplication, confidence
threshold or new candidate is introduced. Agreement is a heuristic, not a
calibrated estimate of true localization quality. Correlated or duplicate
errors can be amplified and must remain visible.

Empty proposals, a singleton proposal, mutually disjoint proposals, or zero
confidence can produce zero consensus evidence. In those cases E is the zero
map, E quantiles and normalized mass are undefined, explicitly counted, never
replaced by uniform evidence or raw confidence. SE retains exact half S and
half zero. Nonfinite, negative-confidence or nonpositive-duration inputs fail
the contract instead of being silently filtered. Actual input counts are
reported before GT use. Native-first argmax breaks ties, including all-zero QC.

Current score: max_m c_m*tIoU(I_k,J_m).
QC score: max_m q_m*tIoU(I_k,J_m). Final interval stays in the original student
candidate set. Current-score parity is verified against the sealed A output.

## Support and task metrics

Discrete temporal support uses the original sampled grid, equal sample weight;
no new dense temporal interpolation or duration weighting is introduced.
Primary event recall, normalized GT mass and soft IoU use the physical event
span under the existing dense evaluator's half-open convention. A separate
scored-frame mask (sampled frame has an actual GT box) reproduces H-lite's
47.36%/58.54% diagnostic. These two denominators are not conflated. HC metadata's
inclusive last box versus the predecessor's end-coordinate convention is
preserved, and any sampled-endpoint discrepancy is counted, not repaired here.

Soft IoU = sum_GT w / (sum w + number_of_GT_samples - sum_GT w).
Reference coverage: form the discrete support CDF and select the first sampled
index reaching 10%,30%,50%,70%,90%. Keep repeated positions; report hits/5,
number of distinct positions and any-hit rate. Also report actual uniform-five
reference positions, valid cached references, and useful references lost to
zero weighted mass. No new frame decoding or Sa2VA call. Frame-location recall
does not guarantee that a segmentation specialist would identify the object.

For Current and QC report native-candidate tIoU/vIoU, oracle selection regret,
native-correct destroyed, missed correct candidates and correct/wrong changes
at the existing strict >.3 and >.5 boundaries. A bad winning contributor is
defined offline as winner proposal tIoU<=.3 while another cached proposal has
tIoU>.5. This is diagnosis, never an online threshold. Also report continuous
best-proposal minus winning-proposal tIoU regret, score changes, candidate
counts, unique proposals, mean agreement and duplicate counts.

Aggregate conditions within source/order then orders within source, with a
paired 10000-source bootstrap, seed20261001. Publish cell means/counts as well
as source macro and CI. Multiple corruptions/orders are not independent videos.
Report clean and corruption separately; report expert-only versus full-stream
readout, with nonexpert difference exactly zero by construction.

## Conditional T1 and exact CPU equivalence

Current temporal selection affects output indices only. Spatial A's reward,
targets, backward, K-step loop and persistent state never consume that choice.
Therefore a changed temporal readout over the completely sealed A trajectory
is exactly the corresponding fixed-spatial online output, without new model
execution. Root verifies all bindings, state links, input pixel matches,
current scores and output parity before accessing GT. It is not a fresh GPU
TTA trial and cannot create transfer to future nonexpert arrivals.

T0 necessarily calculates candidate utilities for the requested correct/wrong
diagnosis. As a predeclared development signal, qualify T1 if in at least one
dataset E or SE increases both mean GT mass and hypothetical quantile coverage,
does not reduce event recall, QC increases mean tIoU, and QC does not increase
native-correct destroyed at either threshold for tIoU or vIoU. No threshold,
mixture, proposal subset or config is selected using these labels. If qualified,
release the paired Current/QC exact CPU readout as T1; if not, leave those
numbers as the negative T0 candidate-selection audit, with T1 not triggered.
This is a finite follow-up decision, not a learned or GT-dependent online gate.

Do not run H-full, new event-conditioned Sa2VA, memory/prototypes, temporal
parameter adaptation, new parameter search or old paused full-query jobs.
Support/location/critic gains alone never establish online spatial benefit.

## Completion

New namespace `artifacts/tastvg_temporal_router_t0_v1`. Bind all inputs and code;
produce new maps/decisions and global seal before GT interpretation. Independently
check scalar formulas, current-output parity, cached physical-coordinate overlap,
state-chain equivalence and source bootstrap. Publish code, frozen protocol,
anonymous scores/counts, negative results, figures and limitations to Zonglin-He/A,
verify remote bytes and scalar audit, then update research history check/snapshot/check.
Keep raw boxes/intervals, label files, media, weights, states and personal records
private. Do not promote the research result to CURRENT_METHOD.
