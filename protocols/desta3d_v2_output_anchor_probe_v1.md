# Differentiable native-output anchor: source-only interface probe

The fixed 8-parent TTA screen shows positive noise adaptation but clean tube
damage despite lower feature loss. Test the missing differentiable readout
interface before changing its objective. This is not a target efficacy run.

Use frozen PTD4B and fixed B1 B155, the lexical-first existing source-train
query key10016, its same 32 clean frames, and the existing brightness1.05 /
contrast.95 view. No labels or target inputs. Source metadata and actual pixels
are hashed. No new backbone, source training or target evaluation.

Record the exact native cached-v3 event and fresh-spatial probe schedules,
including generated prefix, copied query token, RoPE starts, context limits,
and cache lengths. Capture actual 2x32 temporal and frames x4x1001 coordinate
logits. Replay each schedule with fresh multimodal prefill, graph-connected
branch residual and the official undecorated cached probe helper under SDPA.
Do not reuse an inference-mode cache. Teacher tokens define support, not GT.

Same-state event and coordinate logits must agree with native inference to
absolute 1e-5 and categorical KL absolute 1e-7. Preserve raw tensors, errors,
and cache audits before checking. A failed interface is not a negative method
result and cannot be bypassed by scoring target cases.

Then perturb only the 66,816 FiLM/LN parameters using the existing fixed
three-step calibration+alignment update (AdamW1e-5 wd0 clip1, alignment.01).
Record actual step counters and changes; this is an ephemeral source-only
diagnostic, not a new fitted checkpoint. On that state measure separately:
current pre-gate total gradient; mean parameter-anchor gradient; actual time
KL and coordinate KL gradients toward the same-observation B1 teacher.
Normalize each output KL by mean endpoint or frame-coordinate, sum category.
Save raw gradients in the same ordered 66,816-dimensional space. Require finite
nonzero output-anchor gradients, frozen gates/backbone and exact final reset.
At the source initial state the parameter anchor is expected to have zero
gradient; a small scalar loss alone is not a gradient-scale claim.

All steps, loading, failed attempts and wrapper overhead enter the cumulative
ledger (cap null). Single GPU lease, 8GiB reserve, 900-second engineering phase.
Save all raw evidence before terminal assertions; keep any failed version and
register a new isolated attempt after a minimal correction. No target launch
until this interface is valid; no target-GT choice of coefficient. Afterward,
use this source gradient evidence to register one matched output-anchor factor.


Public review note (2026-09-28): this is the original stage protocol, not an instruction to run it. Superseded and failed versions are retained for provenance. See REVIEW_START_HERE.md for current status. Referenced local data, weights and artifacts are not bundled.
