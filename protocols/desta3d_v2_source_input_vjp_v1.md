# Source input VJP v1

Question: at the predeclared source28199 B1 state, does the actual postcast
merger-input task gradient respond differently to saved FP32 endpoint changes
and saved BF16 endpoint jumps? This is an interface diagnostic, not an optimizer
or precision intervention and not a proof of finite task effects.

Keep official frozen PTD4B, B1 adapter4a2ef2cf..., first fixed source parent
5624461612/32 frames, GT teacher-forced support and original event+spatial CE.
All PTD and adapter parameters frozen, no optimizer. Use source training labels
only for this explicitly supervised diagnostic. No target inputs/labels, no new
native predictions or saved-state selection. Existing native and task results
remain unchanged. No alpha/LR/lambda/steps/precision sweep.

Register postcast_leaf AFTER the unmodified branch_injection merger hook. Its
return is output.detach().requires_grad_(True), with exact value/dtype/shape.
Each branch independently runs its original multimodal prefill; no event KV reuse.
Keep model.train/vision.eval/adapter.eval and original joint_loss with use_cache
False and chunks32. Deterministic algorithms/cudnn, CUBLAS_WORKSPACE_CONFIG=:4096:8,
seed20260927. Compare actual merger endpoint hashes and all81 target-token
logits/logsumexp/CE with existing cast probe B1. Detaching a leaf is not assumed
forward-equivalent without those checks. Remove lm_head observation hook before
backward recomputes checkpointed output chunks; visual merger must execute once.

autograd.grad(CE,postcast_leaf) yields complete per-branch input gradients.
Save actual dtype, full shape and all gradient elements, source support hashes,
81 per-token evidence and worker summary. Check parameter hashes unchanged,
all parameter grads None, live CE equal old CE. Do not substitute parameter
gradient or STE simulation for this actual input derivative.

Use previously sealed source_cast_probe_v1/RAW_ENDPOINTS.pt without regenerating
either endpoint: gV dot FP32 delta and gV dot actual sparse BF16 delta. Compare
separately per branch and in sum with existing gTheta dot exact three-step
parameter difference and measured finite CE change. FP32 delta is itself a
finite-precision endpoint difference, not an exact real-valued function.

CPU controls: synthetic BF16 post-injection leaf preserves values/loss and exactly
matches direct-leaf derivative with frozen parameters; sparse/dense dot equality
and support mismatch rejection. Independent NumPy full-vector norm/dot readback
after seal. Full gradients expected36.7MB BF16 or73.4MB FP32, actual dtype recorded;
lock new artifacts<=100MB including1MB metadata buffer, 8GiB free disk floor.
GPU serial lease,600s engineering stage, cumulative cap=null; all real failure,
load and nonoverlapping wrapper times counted. Preserve any partial or failure,
no silent reduced support or sample replacement, original pins unmodified.

Interpretation: agreement of gV dot FP32 delta with the parameter prediction
would weaken adapter-local linearization as the source of the current mismatch.
A postcast dot much different from the precast dot exposes conversion's local
effect, but does not establish that it explains finite CE. If postcast prediction
still misses finite CE, retain downstream LM/logit quantization/nonlinearity and
GT-prefix/native-readout explanations. No automatic new target/64 run, precision
change, extra training or production promotion follows. Both outputs/signs kept.


Public review note (2026-09-28): this is the original stage protocol, not an instruction to run it. Superseded and failed versions are retained for provenance. See REVIEW_START_HERE.md for current status. Referenced local data, weights and artifacts are not bundled.
