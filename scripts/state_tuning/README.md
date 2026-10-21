# HC64 full-state OPD

See docs/state_tuning.md for the fixed scope and label-free calibration rule. This independent experiment does not replace the registered method or resume formal evaluation. Existing Source inputs, native ReferDINO masks and baseline predictions are required private assets.

Run sequentially from the repository root:

```bash
python -B scripts/state_tuning/capacity.py
python -B scripts/state_tuning/score.py --trial outputs/state_capacity_hc64 --output results/state_capacity_hc64 --names Full FullR4 PCA4 PCA4R4
python -B scripts/state_tuning/calibrate.py
python -B scripts/state_tuning/run.py
python -B scripts/state_tuning/score.py --trial outputs/state_tuning_hc64 --output results/state_tuning_hc64 --names StateOPD Direct DirectMC DirectFullRef
```

Scoring is CPU-only and refuses to read labels before the complete64 seal and hashes are verified. The capacity intervention replays original geometric adaptation exactly; it is a reference-state diagnostic, not a new mask-reward method. New adaptation uses the unchanged mask-projection/trajectory-protection objective. All four new arms share the full state and calibrated sigma/tau, with the same forward allowance and separately reported actual costs.

Do not overwrite completed outputs. Source/code/configuration and input fingerprints bind each run. Cached costs do not represent end-to-end deployment latency.
