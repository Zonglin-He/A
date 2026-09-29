# Reproduction

From repository root with the original authorized private caches/model assets:

```bash
.conda/tubedetr/bin/python -B -m pytest -q tests/test_tastvg_spatial_rank_s11_v1.py
.conda/tubedetr/bin/python -B scripts/run_tastvg_spatial_rank_s11_v1.py prepare rank
.conda/tubedetr/bin/python -B scripts/run_tastvg_spatial_rank_s11_v1.py prepare norm
bash scripts/with_local_cuda.sh .conda/tubedetr/bin/python -B scripts/run_tastvg_spatial_rank_s11_v1.py run rank
bash scripts/with_local_cuda.sh .conda/tubedetr/bin/python -B scripts/run_tastvg_spatial_rank_s11_v1.py run norm
.conda/tubedetr/bin/python -B scripts/score_tastvg_spatial_rank_s11_v1.py score rank
.conda/tubedetr/bin/python -B scripts/score_tastvg_spatial_rank_s11_v1.py score norm
.conda/tubedetr/bin/python -B scripts/analyze_tastvg_spatial_rank_s11_v1.py
.conda/tubedetr/bin/python -B scripts/report_tastvg_spatial_rank_s11_v1.py
```

Artifacts are write-once; preserve completed/failed runs and use a new destination for independent reproduction. Both new predictions must seal before scoring. Raw S1 remains read-only. Public-only scalar verification needs Python/NumPy, no model, media or labels:

```bash
python scripts/audit_tastvg_spatial_rank_s11_public_v1.py results/tastvg_spatial_rank_s11/2026-09-29
```

Public export contains scalar results and code, not private tensors, masks, query text/IDs, GT coordinates or weights. Full inference reproduction requires authorized local inputs; scalar reproduction does not.
