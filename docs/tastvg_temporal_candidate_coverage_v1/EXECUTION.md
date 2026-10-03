# Matched CPU candidate coverage execution

1. `.conda/tubedetr/bin/python -B scripts/test_tastvg_temporal_coverage_v1.py`
2. Prepare exact old A/capture/expert hashes without label reads:
   `scripts/prepare_tastvg_temporal_coverage_v1.py`.
3. Run label-free `scripts/generate_tastvg_temporal_coverage_v1.py`.
   Exactly288expert pools, all1152readouts and a global prediction barrier.
4. Only after the global barrier, run CPU
   `scripts/score_tastvg_temporal_coverage_v1.py`, then independent root audit.
5. Report paired effects and limits, render/inspect plots, run the standalone
   anonymous public audit, public-export allowlist, push/verify Zonglin-He/A,
   save FINAL_COMPLETION and research archive check/snapshot/check.

All scripts run on CPU with CUDA_VISIBLE_DEVICES empty; no controller or GPU
queue is needed. A completed stage is not completed publication. Preparation,
generation, scoring and root audit have separate receipts. Engineering repairs
must preserve failed source/outputs and explicit pin overrides; never revise
the locked scientific allocation after scoring. Frozen original temporal
critic uses cached first view only, not the two-view rule. Old queues stay
paused and the existing monitor remains paused.
