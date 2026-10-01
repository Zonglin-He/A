# Selected-configuration matched Frozen/Ours result

All 8808 arrivals and posthoc GT pipeline diagnosis completed and independently root audited. This is the fixed historically exposed Paper48 P1/P5 one-query/source roster, not the official all-query benchmark.

- [Result and diagnosis review](../../../docs/TA_BEST_QUICK_REVIEW.md)
- [Full diagnosis tables](../../../docs/TA_BEST_QUICK_UPDATE.md)
- [Precompute cache dependency boundaries](../../../docs/TA_FULL_CACHE_REUSE.md)

Dataset directories contain all anonymous scalar rows, all spatial step rows, paired previous-panel rows, full summaries, positive and negative cases, pipeline diagnosis, compute/wall receipts and root audits. No raw media, annotations/GT boxes, captions/source identities, weights, states or raw tube predictions are published. Configuration and code were frozen before this run; no winner reselection or production promotion. The prior all-query job remains preserved and paused.

The additional diagnosis auditor needs only Python standard library:

```bash
python -B scripts/audit_tastvg_best_quick_diagnosis_v1.py results/tastvg_best_quick/2026-10-01/vidstg
python -B scripts/audit_tastvg_best_quick_diagnosis_v1.py results/tastvg_best_quick/2026-10-01/hc2
```

It reconstructs all diagnostic tables and verifies multi-step telescoping utility using anonymous scalars, without models, GT or GPUs. This command refreshes its local readback timestamp; it does not change predictions.
