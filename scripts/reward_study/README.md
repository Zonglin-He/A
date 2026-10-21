# Reward study: CPU audit code

These are portable copies of the CPU analyses used for the exposed HC128 reward study. They inspect existing predictions, traces, teacher features, and native masks. They do not implement a new adaptation method, select a new configuration, or establish new task performance.

`source_export_cpu.json` records the original local source SHA256, exported SHA256, and the root-path/import changes. Export verification parsed and compiled the code without importing or executing it, checked unchanged function/class ASTs, and verified that reversing only the documented transformations restores the original source bytes. Absolute private roots are redacted in the descriptions.

The project root is read from `STVG_PROJECT_ROOT`; its default is this repository root. The copies preserve the original private artifact layout under `outputs/layout_reward_dev/`. Required inputs include the sealed HC128 predictions and receipts, per-query student caches and traces, development manifests and labels, and the separately sealed native mask export for mask analyses. Earlier audit JSON files are dependencies of some summaries. Missing inputs cause failure; no data or predictions are generated as a substitute.

| Script | Existing evidence inspected |
|---|---|
| `data_audit.py` | Candidate quality, GT box scale, teacher alignment, and saved spatial paths |
| `data_groups.py` | Existing scored-query groups and reward/propagation summaries |
| `gradient_signal.py` | Initial layout and OPD directions in normalized box coordinates |
| `matched_state_sampling.py` | Matched initial states, saved Adam steps, and CPU ROI sampling stability |
| `algorithm_trace_audit.py` | Actual saved rewards, gradients, updates, and prediction displacements |
| `candidate_support_audit.py` | Saved ROI grid support and candidate weights |
| `mask_completeness.py` | Sealed native mask range and fixed candidate diagnostics |
| `mask_same_candidates.py` | Native mask scores on the identical saved initial 32 candidates and exact coordinate gradients |

The CPU environment requires Python, NumPy, PyTorch, and SciPy, together with the repository's `stvg` package and existing `scripts/layout_reward` helpers. Scripts that use PyTorch force CUDA visibility off and use one PyTorch thread; the NumPy-only summaries do not start models. ROI operations and derivatives reuse frozen cached features or masks; they do not run the video encoder, ReferDINO, or student decoder. Coordinate-only directions are diagnostics, not actual content-state adaptation results.

The mask analyses require complete native-mask sealing before GT access. After those private inputs have been prepared and verified, an example invocation is:

```bash
STVG_PROJECT_ROOT=/path/to/stvg-opd python -B scripts/reward_study/mask_completeness.py --run-after-seal
```

Several scripts perform their audit at module top level; run them explicitly rather than importing them to inspect their APIs. Their output belongs in the private `outputs/` directory and can contain query identifiers or GT-derived values. It must not be published directly. Public reports use separately reviewed aggregate exports.

Publishing these scripts does not publish the dataset, annotations, checkpoints, raw teacher outputs, caches, or complete execution assets. The repository alone is insufficient to rerun these audits without the corresponding sealed local artifacts.

## GPU evidence and derivative diagnostics

Four additional files preserve the executed GPU audit computations. They are **diagnostics, not a new reward method, adaptation setting, or task benchmark**. `source_export_gpu.json` records original/exported SHA256 values, every root-path and private-ID lookup change, and per-function/class AST checks. The exported scripts were parsed and compiled without importing or running them. The original private scripts were left unchanged.

The original private derivative script was `decoder_audit/derivative_validation/run.py`; its public name is `run_derivative_validation.py`, so all four public files have the same repository-root default `Path(__file__).resolve().parents[2]`. Fixed mechanism cases are located through the **1-based HC rows 36, 84 and 125** of the sealed development manifest, rather than publishing private query identifiers. The source/export audit verified that these lookups select exactly the original three cases in that manifest. Row 125 alone is used by the layer probe. Native evidence export uses the existing complete HC128 scope.

| Script | Executed diagnostic scope and required private inputs |
|---|---|
| `export_native_masks.py` | Export all native boxes, scores and masks from the same frozen ReferDINO and the same eight existing observation positions; verify selected boxes against the existing teacher cache. This script decodes existing RGB and performs the already authorized teacher evidence forward, while running no student adaptation or Source encoder. |
| `run_decoder_audit.py` | Reconstruct the saved LayoutOPD states and Adam buffers at steps 0 and 20 for the three fixed cases; compare OPD and Direct at identical state/history, then obtain the initial content-to-output derivative matrices and map the existing five-seed box gradients. Uses frozen cached decoder inputs only. Default behavior describes the plan; `--run` executes it. |
| `probe_layers.py` | One cached initial Replay for HC row 125, with read-only LayerNorm, query and box-reference hooks. Checks forward equality and makes no state update. |
| `run_derivative_validation.py` | Fixed two-direction, four-step-size central differences on the same three initial states, compared with the native truncated backward graph and a reference-detach-free graph bound only to an isolated decoder instance. Forward values must remain identical; no persistent source change or adaptation result is produced. |

These scripts retain the shared `runtime/gpu.lock`, checkpoint/configuration checks, cached-input hashes, exact forward checks and diagnostic output layout. Decoder-only scripts do not read RGB, run Source encoders or call a teacher. Native mask export is the exception described explicitly above. No diagnostic script is permitted to resume the paused formal evaluation or to alter the registered method.

**Keep sealed outputs immutable.** Public copies have different source fingerprints from the originally executed private scripts. Do not run them over the original `outputs/layout_reward_dev/research/native_masks/`, `decoder_audit/` or `derivative_validation/` output files, and do not weaken fingerprint checks to reuse a differently signed receipt. Prepare a separate private artifact tree with the required original inputs and a fresh destination for the particular audit stage. Preserve failures, seals, raw tensors, and previously generated results. Existing outputs remain evidence of their original source fingerprints; publishing a portable source copy does not retroactively re-sign them.

The layer probe and derivative-validation script perform work at module top level. **Do not import them to inspect code.** Review or compile them without execution; run only explicitly within an authorized GPU diagnostic budget. Their private outputs may contain query IDs and model activations and must not be uploaded directly. The public summaries are separately reviewed exports.
