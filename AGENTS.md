# STVG project

Only one active method is maintained: single ReferDINO initialization + spatial OPD + Strong temporal inference. `configs/default.json` is authoritative. The public entry is `python -m stvg`; do not restore historical methods, override parameters, add experts or run new experiments without a user request.

Before changes, read `docs/history.md`, `docs/method.md` and the saved results. Preserve dataset-specific decoding, all numerical configurations, one-expert evidence reuse, per-query reset, physical MBR and offline-only scoring. Inference must never read GT. Respect shared `runtime/gpu.lock` and never signal unrelated services.

For engineering work, run `python -B tools/check.py`; use the two existing-query smoke checks only when an inference change needs numerical regression. Do not confuse these checks with a new experiment or claim improved task performance from them.

Update `docs/history.md` after material changes. Run `.conda/tubedetr/bin/python -B scripts/research_archive.py check`, then `snapshot`, then `check`. This validates current files and saved evidence; it is not model inference.

Publish code, configuration, required reproducibility material and all signed final results to Zonglin-He/A. Never upload data, annotations, weights, raw caches, environments, private queries or credentials. Preserve Git history, verify the remote commit/tree/bytes and record the synchronization receipt. Historical worktree removal is allowed only within the user's explicit cleanup scope; unrelated user changes must be preserved.
