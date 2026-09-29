# Reproduction

Run from the repository root with the authorized original private models/caches and existing environment. Public artifacts omit media/query IDs/GT boxes/H/masks/prediction tubes/gradients/weights.

```bash
.conda/tubedetr/bin/python -B -m pytest -q tests/test_tastvg_joint_j0_v1.py
.conda/tubedetr/bin/python -B scripts/run_tastvg_joint_j0_v1.py prepare
bash scripts/with_local_cuda.sh .conda/tubedetr/bin/python -B scripts/run_tastvg_joint_j0_v1.py run
.conda/tubedetr/bin/python -B scripts/score_tastvg_joint_j0_v1.py
.conda/tubedetr/bin/python -B scripts/report_tastvg_joint_j0_v1.py
```

Fresh runs lock the corrected runner directly. Existing original LOCK and initial failure are retained with IMPLEMENTATION_REVISION. Artifacts are write-once; use a new run directory for independent repetition. Do not overwrite saved state/labels/results. CPU public-scalar verification, no private models or data:

```bash
python scripts/audit_tastvg_joint_j0_public_v1.py results/tastvg_joint_j0/2026-09-29
```

The package OnlineMethod accepts native post-encoder cached data, fixed probe deltas and lazy temporal/spatial evidence providers; use reset only between independent streams and close to restore source parameters. H caching is valid because encoder parameters never change. Freeze covers recipe, not arbitrary untested runtime/cohort compatibility.
