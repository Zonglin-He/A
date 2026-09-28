# Source cast endpoint probe v1

Predeclared source-only numerical diagnostic; zero optimizer steps. Fixed first
query vidstg_source_query:28199, parent 5624461612, from the existing 16-parent
source task-control panel. Compare the exact B1 state with its already saved
supervised3 state; no interpolation, learning-rate/alpha/lambda search or selection.

Official frozen PTD4B, hidden128 dual adapter, existing FiLM/LN66816 scope,
deterministic algorithms/cudnn and CUBLAS workspace :4096:8, seed20260927.
Source GT is explicitly used only for the original task CE diagnostic. No target
inputs or labels. The GT-prefix/use_cache=False task path is not asserted to be
equivalent to free native cached decoding. No new native predictions are needed.

Reuse the original branch_injection context unchanged. It computes adapter
updated_tokens in FP32 then casts to the merger token dtype BF16. Capture those
actual endpoints for event and spatial at both saved states. Store the full FP32
endpoint delta, every sparse changed postcast index with BF16 before/after values
and FP32 before/after values at those indices, full shape and endpoint hashes.
Full endpoint-to-cast equality is checked live; unchanged endpoint values are not
stored, so an independent reader cannot re-execute that entire equality from the
delta alone. It can independently recompute all delta norms/support/counts and
rounding at every changed postcast position. No support truncation is allowed.

Observe the original joint_loss lm_head outputs through a non-mutating hook,
preserving its actual chunks of at most32 and label masks. Save all target token
positions/IDs/NTP flags, target logits, logsumexp and per-token cross entropy,
with physical pixel/query/grid/time/teacher-prefix/position/context hashes. The
per-branch CE must match the existing completed mode-probe CE exactly. Compare
the finite CE change with the already saved initial gradient dotted with the
exact saved three-step parameter difference in ordered66816 coordinates.

CPU controls: BF16 erasure/crossing and exact sparse reconstruction; actual chunk
hook preserves outputs and CE algebra; storage rejection before writes. These
synthetic controls are separate from GPU evidence. Original files/pins unchanged.

Engineering allocation600s, GPU serial lease, cumulative cap=null. Actual worker
and nonoverlapping launch/import/finalization overhead, including failures, are
receipted. Expected two dense deltas73,400,320 bytes for32x8x14x2560. All new output
storage<=90,000,000 bytes with 8GiB free disk after writes; actual serialized bytes
checked before writing and1MB metadata buffer retained. If support/storage fails,
preserve failure and partials, do not drop samples/indices or delete research data.

Seal all raw files then independently recompute with CPU NumPy. One source point
and two existing endpoints diagnose numerical transport only. Erasure/jumps,
CE changes and large finite/linear ratios do not prove that BF16 causes task
failure or that increasing residuals is beneficial. Official finetune_video.sh
also enables BF16 and gradient checkpointing; tiny gated residuals differ from
that training setup. No automatic extra steps, target experiment,64-source run,
method promotion or push follows this diagnostic.


Public review note (2026-09-28): this is the original stage protocol, not an instruction to run it. Superseded and failed versions are retained for provenance. See REVIEW_START_HERE.md for current status. Referenced local data, weights and artifacts are not bundled.
