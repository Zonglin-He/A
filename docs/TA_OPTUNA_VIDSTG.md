# VidSTG small-cohort Optuna search and fixed confirmation

VidSTG's 48 attempted trials and the preselected/default confirmation are complete. HC-STVG-v2 is still running separately; this is not closure of the two-dataset search. The published Paper48 recipe and production registration remain unchanged.

The selected trial 5 uses **lr 0.005, rho 0.05, teacher temperature 0.05**. Only temperature differs from the default (1.0). Search selected it before confirmation, maximizing source-macro corrupt future/nonexpert dense vIoU improvement over Frozen. Search gain was +0.786177 pp versus default +0.017387 pp; these are selection-biased development scores.

| Fixed confirmation endpoint | Default delta vs Frozen (pp) | Selected delta vs Frozen (pp) |
|---|---:|---:|
| Corrupt future/nonexpert | +0.017697 [0.004217, 0.032117] | +0.166143 [-0.011859, 0.365932] |
| Corrupt all arrivals | +0.319611 [0.035672, 0.694200] | +0.442998 [0.081842, 0.894393] |
| Clean future/nonexpert | +0.027967 [0.003474, 0.061834] | +0.280405 [0.016921, 0.606837] |
| Clean all arrivals | -0.193998 [-0.832298, 0.228158] | +0.045873 [-0.583581, 0.555119] |

Brackets are 95% paired-source bootstrap intervals (10,000 replicates). The direct selected-minus-default corrupt future contrast is **+0.148446 pp [-0.022045, 0.338146]**. Its positive mean does not establish a reliable improvement. Both confirmation arms have zero corrupt cells with vIoU harm exceeding 5 pp; clean all-arrival results each contain one such cell. Source-average harms exceeding 5 pp are zero in both arms. The clean control also improves, so these results do not establish a corruption-specific benefit. The all-arrival corrupt-minus-clean excess is +0.513609 pp [-0.013240, 1.314250] for default and +0.397125 pp [-0.285550, 1.240363] for selected.

The cohort has 32 search sources and 16 source-disjoint confirmation sources, one query per source, two fixed orders, and clean plus five transient 5% corruption conditions. All sources have historical project exposure; confirmation is independent of this round's selection only, not a fresh test set. Each search trial has 384 arrivals; each confirmation arm has 192. The source checkpoint is the same VidSTG-trained TA-STVG checkpoint used in Paper48 P1. K=1, 1,792 update parameters, 25% expert arrivals, online prediction before update, and native temporal reranking are fixed.

Log search ranges are lr [1e-5, 0.5], rho [0.001, 0.5], teacher temperature [0.05, 20], with default, six single-coordinate extremes, eight corners, and seeded Optuna TPE without pruning. Of 48 attempts, 47 completed and one failed numerically: trial 11 (lr 0.5, rho 0.001, temperature 0.05) encountered nonfinite rank KL after 116 arrivals. The failed configuration remains FAIL, counts toward the budget, has no assigned score, and was not retried. All attempted configurations and completed scalar outcomes are published.

Root verification rehashed all 18,432 completed prediction payloads and checked their sealed state receipt chains, source split and ordering, configuration pins, and selection-before-confirmation timestamps. Independent scalar/source/order/bootstrap/harm checks passed (199,354 root checks). Recorded run audits cover 4,326 SGD updates, 4,608 temporal expert checks, 36,864 dual metric evaluations with zero discrepancy, and 72 exact full-forward reinsertion checks. This closure does not rerun model inference or claim a new independent ground-truth evaluation.

Reproduce public scalar aggregation with:

```bash
python scripts/audit_tastvg_optuna_public_v1.py results/tastvg_optuna/2026-10-01/vidstg
```

Code and locked configuration are included. Public scalar CSVs contain anonymous parent indices and task metrics; private media, labels, tubes, H caches, weights, and personal conversation records are excluded. [Trial outcomes and confirmation](../results/tastvg_optuna/2026-10-01/vidstg/REPORT.md), [all trials](../results/tastvg_optuna/2026-10-01/vidstg/TRIALS.csv), [protocol](../protocols/tastvg_optuna_v1.md), [root audit](../results/tastvg_optuna/2026-10-01/vidstg/ROOT_READBACK.json).

HC2 continues on the original locked queue. No automatic full-dataset run, new parameter selection from confirmation, or production promotion follows these results. Fig1's separate frozen-model diagnosis waits until both datasets finish and are reviewed and published.
