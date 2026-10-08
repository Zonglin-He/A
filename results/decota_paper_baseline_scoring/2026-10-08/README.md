# Baseline scoring of sealed predictions

[Report](REPORT.md), [source and query table](TABLE.csv), [paired comparisons](PAIRED_COMPARISONS.json), [recorded cost](RECORDED_COST.json), and the three PNG/PDF figures are actual post-seal results. EATA is only the completed HC2 direction; the other direction remains paused. OPD has no full-roster result here.

Reproduce statistics without private assets:

```bash
python -B scripts/audit_decota_paper_baseline_scoring_public_v1.py results/decota_paper_baseline_scoring/2026-10-08
```

Dependencies: Python 3 and NumPy. Anonymous compressed JSONL retains all 217221 logical rows, including negative results and three complete orders. Large JSON files are losslessly compressed as described in PUBLIC_DATA_FORMAT.json; use scripts/decota_public_result_io_v1.py. No video, caption, GT annotation, trained weight, optimizer, parameter vector or raw cache is distributed. The raw evaluator requires separately obtained official/private inputs and preserved stage receipts; the public scalar audit does not claim to reconstruct unavailable production gradients or model Jacobians.
