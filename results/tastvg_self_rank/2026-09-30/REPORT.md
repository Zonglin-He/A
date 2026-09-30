# Self-Rank: student spatial preference control

16 exposed VidSTG sources, five locked orders, clean and five 5% conditions. Frozen Final unchanged. Each row is mean ± sample SD over the five shared-source orders.

| Arm | Whole Δv pp | Future Δs pp | Future Δv pp | Future arm−Final pp |
|---|---:|---:|---:|---:|
| Frozen | +0.000000 ± 0.000000 | +0.000000 ± 0.000000 | +0.000000 ± 0.000000 | -0.025505 ± 0.020209 |
| Fast-only | +1.496239 ± 0.863637 | +0.000000 ± 0.000000 | +0.000000 ± 0.000000 | -0.025505 ± 0.020209 |
| Slow-only | +0.021456 ± 0.015196 | +0.094390 ± 0.071792 | +0.025505 ± 0.020209 | +0.000000 ± 0.000000 |
| Final | +1.517681 ± 0.859943 | +0.094390 ± 0.071792 | +0.025505 ± 0.020209 | +0.000000 ± 0.000000 |
| random_rank | +1.494774 ± 0.859934 | -0.009502 ± 0.047416 | -0.004011 ± 0.016776 | -0.029515 ± 0.033549 |
| self_rank | +1.482004 ± 0.854531 | -0.053115 ± 0.032443 | -0.017059 ± 0.013361 | -0.042563 ± 0.020366 |

Self-Rank uses descending ranks of the detached student probability; the nine candidate boxes and rank teacher remain detached. Temporal reranking, nine probes, 1792 parameters, SGD .005 and the 25% schedule are unchanged. No Sa2VA output was read by this arm.

Admission caveat (declared before execution): Self-Rank attempts all 120 scheduled updates, whereas Final and Random-Rank made 102 updates because Sa2VA had no valid mask at 18 positions. Therefore this is a spatial-teacher-free control, not a perfectly rank-only causal intervention. No extra teacher-derived validity gate was introduced.

Metrics reproduce the sampled-grid development endpoint. The five orders share the same 16 exposed sources and do not establish performance on fresh sources. All order-level values, clean results, harms and update diagnostics are retained. A result here cannot prove external knowledge is universally necessary or identify the fraction of gain due to knowledge transfer.
