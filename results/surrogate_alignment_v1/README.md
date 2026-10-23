# Fig. 1: task alignment of surrogate signals in STVG

These are empirical statistics, not the illustrative numbers in the requested layout. Each figure is a single chart with two curves. Both official target domains are provided without selecting the more favorable one. No video, expert or method diagram is included.

- [HC-STVG2.0 vector figure](fig1_hc2.svg): 1,902 queries / 227 parent videos.
- [VidSTG vector figure](fig1_vidstg.svg): 10,303 queries / 732 parent videos.
- [Frozen score definitions and protocol](protocol.json), [verification](verification.json).
- [HC bin data](hc2/bins.csv), [Vid bin data](vidstg/bins.csv).
- Anonymized per-query scalar scores and saved Source vIoU are in each domain's `scores.tsv`.
- Reproducible cache-only extraction and rendering: `../../studies/surrogate_alignment_v1/run.py`. Local PDF/300-dpi PNG exports are retained privately in `outputs/surrogate_alignment_v1/figures/`.

## Definitions

**Temporal confidence:** one minus normalized Shannon entropy of the original native physical-hull interval posterior, `1-H(q0)/log(T(T-1)/2)`. Higher means more concentrated time predictions. The normalization accounts for different candidate counts. This measures the saved native prior; no new temporal network or extra decoder layer is evaluated.

**Spatial consistency:** mean IoU of adjacent original Source boxes over the complete sampled trajectory, after clipping to image coordinates. This is a **trajectory smoothness proxy**, not augmentation-view agreement. Paired augmentation predictions are not saved in these caches, so augmentation consistency has not been measured. Object/camera motion and sampling affect this proxy; it is not equivalent to spatial correctness or the SimplePath path score.

**Quality:** the same query's original frozen **Source** full vIoU from previously completed official scoring. Neither curve is evaluated against the adapted Full result, avoiding a mismatch between a Source signal and a different model output.

Each score independently ranks all queries into five percentile bins, with ties kept together. The ordinate is equal-query mean vIoU. Error bars/shading are pointwise 95% parent-video cluster-bootstrap intervals (5,000 resamples, seed 20261023). Bins stay fixed during resampling. Legend rho is **sample-level Spearman correlation**, not a correlation between five bin means. Both domains and all bins are retained. These sets were already exposed; this is descriptive analysis, not fresh method validation or parameter selection.

## Measured results

| Target domain | Temporal confidence rho | Spatial trajectory consistency rho |
|---|---:|---:|
| HC-STVG2.0 | -0.000778 | 0.242044 |
| VidSTG | -0.038550 | 0.204643 |

Time confidence has negligible monotonic correlation with joint grounding quality here. Spatial trajectory consistency carries a positive association, so the figures do **not** justify claiming that both signals are uninformative. The neutral title is retained. Neither sample-level correlation nor bin means establish how optimizing a TTA objective would affect grounding.

## Suggested caption

**Figure 1. Task alignment of surrogate signals in STVG.** Queries are independently grouped into quintiles of temporal confidence (normalized native interval-posterior concentration) and spatial consistency (mean adjacent-box IoU). Curves show mean Source vIoU, with 95% parent-video bootstrap confidence intervals. Legend rho denotes query-level Spearman correlation. Temporal confidence is weakly aligned with joint grounding quality, while spatial trajectory consistency shows a positive but incomplete association.

## Verification and scope

All 12,205 source captures were checked against their completed full-run SHA-256 bindings. Scores were saved/sealed before joining previously saved quality. Published Source vIoU means were reproduced. Existing metrics, seals, production source and parameters were not changed. Zero new model/expert/GPU calls, predictions or GT scoring. Raw videos, captions, annotation coordinates, features and model weights remain private.
