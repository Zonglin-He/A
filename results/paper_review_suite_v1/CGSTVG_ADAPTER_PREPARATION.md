# CG-STVG adapter preparation — not a completed backbone experiment

The second backbone remains the **last GPU stage**, after the TA photometric
TTA, observation-budget and controlled-shift studies. No CG model was
constructed or evaluated by this preparation. The main TA method and every
previously frozen full run are unchanged.

## Verified upstream interfaces

The [official repository](https://github.com/HengLan/CGSTVG) is pinned to
`f062d84b72aee8be4425a4a5ab20aa5b077ca142`. The CPU audit compared all 58 local
code/config/text files with the Git blob IDs in that immutable upstream tree.
Both 3,548,827,981-byte checkpoint files match the official hashes recorded in
the prior source audit. Cross-domain binding is target HC2 ← VidSTG checkpoint,
and target VidSTG ← HC-STVG2 checkpoint. Loading a target-trained checkpoint is
not permitted.

The original CG forward already exposes the projected **pre-text appearance
grid** (`input_proj`) and the post-fusion appearance-token prefix. The first is
the Native2D pooling representation; the latter supports spatial matching.
Neither interface requires a word-classification head, an independent motion
representation, or aggregation of multiple decoder layers.

CG's official evaluation uses a **single frame grid**, not TA's two offset
passes. HC test sampling doubles 3.2 FPS to 6.4 FPS and preserves the official
last-frame convention; Vid test sampling caps the input at 128 frames with
the official `linspace`/integer-index rule. The published checkpoint uses
420-pixel input height. The new preparation does not reuse the archived CG
224/448 multi-pass adapter, its temporary optimization states, or its six-layer
posterior fusion.

The supplied CG postprocessor reads its last-layer FP32 endpoint head and
chooses a legal joint MAP interval. The adapter therefore needs two explicit
baseline readouts: **original native MAP** for reproduction, and **native
posterior + MBR** for comparison with SharedCorrection/Space/Full under the
same decision rule. Any MAP→MBR improvement must be reported separately from
Native2D feedback. The posterior itself is formed on CG's one native grid,
without manufacturing a second offset or a TA posterior.

## Label isolation

Upstream `PosDecoder.forward` unconditionally invokes `gt_info(targets)` even
when `iteration_rate=-1`; its return values are used only by the training
branch. The instance-scoped adapter bypasses precisely this dead read, rejects
training/nonnegative iteration rates, accepts an object that forbids every
target read, and restores the original function on exit. Vendor files remain
unchanged. Executing the actual official function with synthetic operators
and synthetic targets verifies exact output equality before and after this
bypass. This is a CPU interface proof, **not** real-checkpoint no-op evidence.

## Remaining release gates and fixed comparison

1. Finish all TA GPU stages. Construct the official CG model in a fresh
   interpreter with the matching source-domain config and strict EMA state
   coverage. Preserve intrinsic CG context-generation/actionness operations;
   none becomes a new external feedback signal.
2. Verify real RGB/native output, endpoint logits, original postprocessor,
   native grids and observer no-op before any feedback. Also verify cold versus
   captured feature pooling. Resource failure is an engineering outcome to
   preserve, not permission to silently change resolution, precision or frames.
3. Freeze an independent runtime plan/code snapshot before screening. Reuse the
   suite's fixed 64 screening and 128 confirmation parents/domain without
   target-GT parameter selection. Use original MAP, Source-MBR,
   SharedCorrection-MBR, Space-MBR and Full as explicit arms. Apply the same
   SimplePath and all-video prototype equations, κ64/β1/context1.5.
4. Exact eight-frame ReferDINO evidence may be reused only with matching RGB,
   caption, complete observation context and runtime hashes. The CG frame grid
   differs from TA, so existing TA evidence cannot be assumed compatible.
   Any new calls use the same frozen expert in the new study cache.
5. If unchanged TA source-calibration constants are transferred to CG, label
   that transfer explicitly. They are **not** CG source-training calibration;
   gamma also lacks fully established historical source-only selection
   provenance. Do not repair that description by using target labels.
6. Seal each complete stage before independent CPU official scoring. Keep all
   signed outcomes, severe union counted once, duration groups, cluster CIs,
   resource failures and actual acquisition costs. Screening detects interface
   or resource failure; it does not select CG target parameters.

The executable pure interfaces are in
`studies/paper_review_suite_v1/cgstvg_interfaces.py`. Six CPU audit groups cover
upstream bytes, both weight hashes, ten exact official sampling fixtures,
three exact native MAP fixtures and a posterior numerical reference,
single-grid pooling exact equality, and the label-read guard/restoration.
The full receipt is `cgstvg_cpu_interface_audit.json`. Real-checkpoint
construction/no-op, inference, scoring and cross-backbone effectiveness remain
**pending**.


## Native runtime implementation now frozen, model execution still pending

The independent CG plan is now frozen at
`f767b3424b3a2a11f2f5358a36bbc614b62a2fc3af3dce29be8828052faece5d`
([protocol](cgstvg_protocol.json)). Native loading/capture, generic feedback,
separate CPU official-CG scoring and the two-stage controller are implemented
in `cgstvg_runtime.py`, `cgstvg_score.py` and `cgstvg_study.py`. The CG loader
refuses even model import/construction until TA consistency, budget and shift
confirmation receipts/seals are complete and their owned workers have exited.
No waiting CG model worker has been launched.

Four additional CPU contract groups passed: the hard last-stage gate before
CG import, both official dataset resize+test-normalization operations byte-exact
on synthetic RGB, both domains' zero-correction/no-information feedback
identity, and six pinned official-CG evaluator versus independent dense checks
with maximum error zero. No real video, annotation, checkpoint model or expert
was used in these tests. Actual learned-model observer/no-GT/cold-cache bridges
remain pending (first two screening queries/domain, four extra native passes).

Two initial unfrozen CPU preflight issues are preserved: unnecessary
valid-box coordinate conversion introduced rounding under a zero update; the
new clipping helper preserves unaffected rows exactly. The original CG Vid
evaluator also requires an unused `pred_conf` argument, now supplied as an
empty dictionary only for that signature. Both repairs preceded plan freezing
and all CG model execution; no metric formula or selected data changed.

The plan budgets at most388 native Source forwards and384 same-expert8 joint
queries, with fixed wall limits. Stage elapsed wall includes model construction,
loading and instrumentation. Four native no-op queries include an extra Source
pass and CPU coupled replay; their audit-inclusive query timings must not be
represented as an independently measured deployment benchmark. Original TA
production, method parameters, old roots and Fig1 are unchanged.


### Dependency-binding revision after budget resource failure

Before any CG model execution, the budget queue hit its fixed exact-search
resource limit on one K32 configuration. Its independent recovery uses a new
namespace and receipts. Accordingly, the final prepared CG entrypoint is now
`cgstvg_study_v2.py`, plan
`160febd9f0962e09c30c7dc8cd348dda1e0b45b12e5a47ce55a85dc6759e2621`
([v2 protocol](cgstvg_v2_protocol.json)). The v1 prepared source/plan remains
unchanged and unused. V2 changes dependency receipt/owner checks and output
namespace only; native sampling, precision, model/weight loading, feedback,
scoring, budgets and cases stay the same. The four CPU contract groups passed
again, without constructing a CG model or using any real RGB/annotation.
The new gate checks the recovered budget's own seal/score binding and the
actual new queue identities; it never fabricates an old-plan completion file.
