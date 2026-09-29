> Superseded by the latest teacher-independent native parameter rollout route. This completed result is preserved as history.

# Reproduce superseded soft-moments S0.5 v2

Private prerequisites: the original S0 LOCK, H_BARRIER, EXPERT_BARRIER, 96 capture files, 96 expert masks, original C3/deployment caches and the same 16 historically exposed labels; TA checkpoint and its established environment. Large/private caches are deliberately not published. Original source paths are resolved from the local roster. Model/source hashes and protocol are in PROVENANCE.json.

From the project root, with a fresh output directory (write-once outputs):

```bash
.conda/tubedetr/bin/python -B -m unittest discover -s tests -p 'test_tastvg_spatial*' -v
.conda/tubedetr/bin/python -B scripts/run_tastvg_spatial_propagation_s05_v2.py prepare
bash scripts/with_local_cuda.sh .conda/tubedetr/bin/python -B scripts/run_tastvg_spatial_propagation_s05_v2.py run
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .conda/tubedetr/bin/python -B scripts/audit_tastvg_spatial_propagation_s05_v2.py
.conda/tubedetr/bin/python -B scripts/score_tastvg_spatial_propagation_s05_v2.py
.conda/tubedetr/bin/python -B scripts/report_tastvg_spatial_propagation_s05_v2.py
```

Public scalar results can be checked without media/models/GT (Python + NumPy):

```bash
python scripts/audit_tastvg_spatial_propagation_public_v1.py results/tastvg_spatial_propagation_s05_superseded_v2/2026-09-29 results/tastvg_spatial_expansion_s0/2026-09-29
```

This independently checks saved scalar aggregation and paired bootstrap, not inference or the original dataset metrics. The root PAIRED_PUBLIC_AUDIT.json records this exact audit against local rows, and publication readback verifies the exported bytes. S1 and future OPD proposals are not implemented by these commands.
