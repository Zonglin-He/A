# Reproduce native spatial rollout support

Requires the private original C3 roster/native candidates, S0 exact H caches/barriers, original Vid-source TA-STVG checkpoint/environment and same16 historical GT entries for scoring. Raw media, captions, source identifiers, labels, H and model weights are not published. No Sa2VA installation/masks are needed by this prediction worker.

Fresh write-once run, from project root:

```bash
.conda/tubedetr/bin/python -B -m unittest discover -s tests -p 'test_tastvg_native_spatial_rollout*' -v
.conda/tubedetr/bin/python -B scripts/run_tastvg_native_spatial_rollout_s05_v1.py prepare
bash scripts/with_local_cuda.sh .conda/tubedetr/bin/python -B scripts/run_tastvg_native_spatial_rollout_s05_v1.py run
.conda/tubedetr/bin/python -B scripts/score_tastvg_native_spatial_rollout_s05_v1.py
.conda/tubedetr/bin/python -B scripts/report_tastvg_native_spatial_rollout_s05_v1.py
```

Public scalar-only reproduction (Python + NumPy, no private caches):

```bash
python scripts/audit_tastvg_native_spatial_rollout_public_v1.py results/tastvg_native_spatial_rollout_s05/2026-09-29 results/tastvg_spatial_expansion_s0/2026-09-29
```

This verifies864 candidate scalar oracles and source aggregation/CI, not original model inference. Prediction process rejects expert/propagation/GT files, and all864 candidates are sealed before scoring. No S1/Reverse-KL or persistent online learning is implemented here.
