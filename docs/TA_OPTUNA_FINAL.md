# Independent Optuna tuning: two-dataset closure

Both finite searches and fixed confirmations are complete. Each dataset used 32 search sources and 16 disjoint-source confirmation sources, one query/source, two orders, clean plus five 5% corruptions. All have historical project exposure; this is supervised small-cohort development, not a fresh full-dataset evaluation.

The selected VidSTG parameters are lr=0.005, rho=0.05, teacher temperature=0.05. HC-STVG-v2 selected trial 44: `{"lr": 0.10966593490131663, "rho": 0.19757784966059425, "teacher_temperature": 9.528864583977343}`. Each selection was sealed before confirmation labels and results.

| Dataset / confirmation endpoint | Default delta vs Frozen (pp, 95% source CI) | Selected delta vs Frozen (pp, 95% source CI) |
|---|---:|---:|
| vidstg / corruption nonexpert | +0.017697 [+0.004217, +0.032117] | +0.166143 [-0.011859, +0.365932] |
| vidstg / corruption all | +0.319611 [+0.035672, +0.694200] | +0.442998 [+0.081842, +0.894393] |
| vidstg / clean nonexpert | +0.027967 [+0.003474, +0.061834] | +0.280405 [+0.016921, +0.606837] |
| vidstg / clean all | -0.193998 [-0.832298, +0.228158] | +0.045873 [-0.583581, +0.555119] |
| hc2 / corruption nonexpert | -0.017878 [-0.030982, -0.004334] | -0.014562 [-0.044606, +0.011469] |
| hc2 / corruption all | +0.512984 [-0.172744, +1.646907] | +0.517082 [-0.174651, +1.648584] |
| hc2 / clean nonexpert | -0.017636 [-0.032327, -0.003132] | +0.001216 [-0.020714, +0.021666] |
| hc2 / clean all | +0.473337 [-0.372253, +1.872238] | +0.490015 [-0.355660, +1.888034] |

Intervals use 10,000 paired-source bootstrap draws. The nonexpert endpoint excludes current specialist calls and measures transferred spatial adaptation. HC2 confirmation has 15 eligible sources / 120 corrupt cells because one source is expert-scheduled in both orders; all-arrival endpoints use all 16 sources / 160 corrupt cells. This schedule-defined denominator is retained.

HC2 selected-minus-default corrupt nonexpert delta is +0.003316 pp [-0.019003, +0.022665]. The search objective improved from +1.230171 to +1.319067 pp, but this did not yield a meaningful positive confirmation gain over Frozen. VidSTG's paired confirmation difference is +0.148446 pp [-0.022045, +0.338146]; positive mean with uncertainty crossing zero. Neither result establishes stable superiority of dataset-specific tuning. No automatic promotion, full-panel rerun or confirmation-driven reselection.

HC2 completed all 48 attempts without failure. VidSTG completed 47/48; trial 11 numerical failure is preserved and counted, not scored as zero or retried. This is 96 attempted configurations total, 95 completed searches, four fixed confirmation runs, 37,248 valid arrivals (not independent samples), plus the retained failed prefix. HC2 confirmation has no cells or source averages with vIoU harm exceeding 5 pp in either arm; clean and corrupt controls and all negative findings are retained.

HC2 root verification covered 18,816 payload hashes and state receipt links, 203,500 independent scalar/bootstrap/harm checks, and the saved audits for 4,416 SGD updates, 4,704 teacher checks, 37,632 dual metric evaluations with zero discrepancy and 72 exact full-forward reinsertions. All 48 requested parameter configurations and 15 preset anchors matched the locked search. VidSTG closure was already published in commit 6c3e7be4948fb36910d7b0f48477f02ed7e62d95.

Reproduce each public scalar audit with `python scripts/audit_tastvg_optuna_public_v1.py results/tastvg_optuna/2026-10-01/hc2` (or `vidstg`). No model or private labels are needed. Public files exclude media, annotation identifiers, raw tubes, H caches, model weights and personal conversations.

[HC2 all trials](../results/tastvg_optuna/2026-10-01/hc2/TRIALS.csv), [HC2 report](../results/tastvg_optuna/2026-10-01/hc2/REPORT.md), [VidSTG report](TA_OPTUNA_VIDSTG.md), [protocol](../protocols/tastvg_optuna_v1.md).

The original Paper48 results and production method stay unchanged. The separately authorized frozen-model Fig1 diagnostic is the next task; it is not part of this tuning result.
