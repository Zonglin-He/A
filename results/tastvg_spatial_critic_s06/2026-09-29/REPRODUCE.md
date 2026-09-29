# Cached critic qualification reproduction

Private prerequisites: original S0 expert masks/barrier and native S0.5 sealed candidate tubes; existing native scalar metrics are only opened after reward seal. No GPU or new expert inference.

```bash
.conda/tubedetr/bin/python -B -m unittest discover -s tests -p 'test_tastvg_spatial_critic*' -v
.conda/tubedetr/bin/python -B scripts/run_tastvg_spatial_critic_s06_v1.py prepare
.conda/tubedetr/bin/python -B scripts/run_tastvg_spatial_critic_s06_v1.py run
.conda/tubedetr/bin/python -B scripts/analyze_tastvg_spatial_critic_s06_v1.py
```

Public scalar reconstruction (Python+NumPy):

```bash
python scripts/audit_tastvg_spatial_s06_s1_public_v1.py critic results/tastvg_spatial_critic_s06/2026-09-29
```

Masks, raw tubes, captions/source IDs and GT coordinates remain private. Public ROWS includes reward and9 candidate sIoU scalars, sufficient to reconstruct all ordering; PAIRS and SUMMARY preserve coverage/denominators. A new run needs a fresh output directory because evidence files are write-once.
