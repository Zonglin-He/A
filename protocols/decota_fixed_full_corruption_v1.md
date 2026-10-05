# Fixed DeCoTA full same-domain and cross-domain corruption evaluation

## Authorization and question

2026-10-05: the user explicitly requested the complete official evaluation
cohorts for the already determined method, under both same-domain and
cross-domain corruption. This is a new isolated evaluation, not a continuation
of a historical paused full-query job or a new method-selection experiment.

The frozen research working point is Native-WHEN + Uniform4 + one frozen
Grounding DINO + admitted Frame-Top1 energy + joint spatial query/LN Adam .03
and persistent LN delta / 16. Its selection is the archived hard-identity
experiment's SEARCH_SELECTION.json (top1). Production registration is separate.

## Complete cohort and matrix

* VidSTG official test: all 10,303 queries from 732 source videos.
* HC-STVG-v2 official validation: all 3,482 queries from 237 source videos.
* Same domain: Vid-trained -> Vid, HC2-trained -> HC2.
* Cross domain: HC2-trained -> Vid, Vid-trained -> HC2.
* Conditions: clean, frame_drop_5, frame_freeze_5, motion_blur_5,
  occlusion_5, exposure_5. Existing original-frame sampling, source-keyed
  burst corruption and exact dataset-specific decoding are retained.
* Two source-blocked, query-complete, prediction-independent orders copied
  from the previously locked full metadata cohort. No source/query is chosen
  or removed using GT, scores, expert admission or current results.
* 123,636 online arrivals for each Vid-target checkpoint and 41,784 for
  each HC2-target checkpoint; 330,840 total, 165,420 unique
  checkpoint/query/condition inputs. All arrivals have the expert available;
  admission can legitimately yield no spatial evidence.
* Multiple queries per source, historical exposure and development-source
  membership must be reported. Outside-development does not mean fresh test.

## Exact method and state

Temporal parameters and interval readout are native and frozen. Four physical
uniform observations come from each input's zero-update native event interval.
Use the existing query parser, contextual DINO view and existing admission
rules. Top1 is the highest-scoring retained valid proposal on an admitted
frame, not an unfiltered raw detection or ground-truth pseudo-box.

Spatial loss and all update rules are the exact top1 arm of
vg_tta/decota_identity_commitment_v1.py: 1,792 parameters, Adam lr .03,
betas (.9,.999), eps 1e-8, no weight decay, 10 steps, earliest minimum own
label-free loss among steps 0..10. Empty evidence is a no-op. Query residual
and Adam are reset per query. Last-block spatial LN is inherited; commit
initial + (selected-initial)/16, with zero query residual. LN resets at each
checkpoint/target/condition/order boundary; it does not reset between sources.

Frozen, inherited Before and corrected After all use the same native temporal
interval. Every order has its own actual continuous online state chain. Frozen
input evidence may be reused only under an exact input/parameter/hash receipt.
There is no full EP, budget sweep, parameter search or temporal adaptation in
this task; Before-Frozen and After-Before give its stage attribution.

## Engineering, locks and storage

The single-GPU finite controller runs the four matrix jobs serially. Query-only
parsing is CPU preparation. No GT or score file is accessible to the inference
worker. Code, configuration, predecessor choice and input metadata are pinned.
Source checkpoints and DINO weights are hashed; model/source weights must be
unchanged at each worker exit. The first two official clean queries per matrix
job verify raw frozen-prefix replay versus the previous normalized interface,
all fit states/gradients/boxes, inherited-state parity and LN write arithmetic.

Store lossless float32 step boxes and initial/selected/committed states, admitted
and rejected expert probes, input/model hashes and source/frame metadata.
An independent NumPy loss/Adam/state arithmetic auditor checks every full
in-memory fit before compaction; durable receipts retain all checks, errors,
gradient/update hashes and norms. All raw gradient/state trajectories are
retained for smoke and the first input of each 100-arrival block. Full offline
GT diagnosis uses every step's boxes. Do not describe hashed gradient receipts
as a full offline decoder-Jacobian replay.

Encoder features and decoded frames use bounded transient RAM only (no unbounded
H disk cache). Exact frozen observation receipts are durable and shared between
orders. Compact output is atomic/hash-receipted; unreceipted partial files and
engineering failures are preserved and require explicit root recovery. Resume
only verified prefixes, including their committed state; no skipped query,
silent fallback, total deadline or result-dependent method revision.
Stop on resource failure with evidence if free disk falls below 8 GiB.

## Seal, scoring, diagnostics and completion

All four jobs and all 330,840 online predictions must be sealed under a global
prediction barrier before diagnostic labels are read. CPU scoring uses the
audited official dense physical-frame scorer with the established per-dataset
coordinate/end-point conventions, independently checked against DenseTube.
Report tIoU, vIoU, dense sIoU, vIoU > .3/.5; query macro and source macro;
paired 10,000 source-cluster bootstrap, both orders, clean/corruption/families,
development-source membership, gain/loss and >5/>20 pp tails.

GT pipeline diagnosis is required: Frozen -> inherited Before -> current After,
correct-to-wrong/recovered at .3/.5; empty/rejected/admitted wrong evidence;
event support coverage, admitted expert localization quality; loss descent vs
task harm; per-step and selected-step harm, observed vs unobserved frames;
LN inheritance vs current correction. Native temporal headroom is a readout-only
diagnostic, never GT fed to a decoder or target-based configuration selection.

Completion means actual full coverage, independent root audit, reports and
figures, RESEARCH_HISTORY check/snapshot/check, public code/configuration/all
anonymous results and negative findings on Zonglin-He/A, exact remote file
verification and FINAL_COMPLETION. The controller ending pending_root is not
final completion. Private media, labels, weights, features and raw caches remain
excluded. No paused historical queue is resumed or scored.
