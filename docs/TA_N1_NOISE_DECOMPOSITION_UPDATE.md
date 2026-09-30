# N1: teacher preference noise decomposition

N1 is complete. Under its fixed pairwise directional loss, filtering to directionally correct teacher signals gave a small mean improvement over All, while correct positive guidance outperformed correct negative guidance in every one of five shared-source orders. Noisy-only had a small, order-sensitive positive mean. This does not support a negative-suppression-dominated explanation for this configuration.

| Signal subset | Future ΔsIoU pp | Future ΔvIoU pp |
|---|---:|---:|
| All: pairwise reference | +0.023387 ± 0.012347 | +0.006063 ± 0.003893 |
| Useful-only | +0.033835 ± 0.013469 | +0.008615 ± 0.004104 |
| Noisy-only | +0.003345 ± 0.012331 | +0.001029 ± 0.003076 |
| Useful-Positive | +0.039819 ± 0.013959 | +0.010084 ± 0.004261 |
| Useful-Negative | +0.018733 ± 0.011373 | +0.004773 ± 0.003496 |
| Useful-Matched | +0.020960 ± 0.014136 | +0.005162 ± 0.003477 |
| Noisy-Matched | +0.003041 ± 0.012133 | +0.001041 ± 0.003027 |

These are corruption source-macro changes relative to Frozen on future nonexpert arrivals; ± is sample SD over five orders of the same 16 exposed sources. “Useful” means correct relative to the current query's GT sIoU direction, not a guarantee of benefit on future queries. Candidates are four antithetic parameter directions plus the center, so the eight probe comparisons are correlated.

Useful minus All future ΔvIoU was +0.002553 pp on average, positive in four of five paired orders; the first order was −0.0000145 pp. Useful-Positive minus Useful-Negative was +0.005311 pp, positive in all five orders. Noisy-only was below All in all five orders, but its own delta versus Frozen was positive in three and negative in two. Its +0.001029 pp mean is insufficient to establish a meaningful noise-induced regularization benefit.

The supplementary pair used exactly equal signal counts at every scheduled arrival. In the corruption streams, each received 135 signals over 51 updated arrivals; Useful-Matched minus Noisy-Matched averaged +0.004120 pp, positive in four of five orders (first order −0.0000522 pp). This is a conservative overlap-eligibility comparison: when one current pool is smaller, both are reduced to its size. Noisy-Matched is distinct from the full Noisy-only arm (176 signals).

Counts and update scales matter. Useful-Positive made 79 corruption updates with mean gradient norm 1.0914; Useful-Negative made 69 with mean norm 0.6179. Thus the observed positive-over-negative result is configuration-specific, not an isolated proof of intrinsic superiority of positive signals. The count-matched pair had 51 updates each and mean gradient norms 1.2133 versus 1.1846; norms were measured but not matched or tuned. All and Useful each had 79 corruption updates, with respective mean gradient norms 0.6099 and 0.9797. Clean future deltas showed the same broad ordering: All +0.006871, Useful +0.008607, Noisy +0.001733, Useful-Positive +0.010003 and Useful-Negative +0.004890 pp. All seven arms had zero >5 pp source/cell future vIoU harms in these development streams.

The historical S0.6 Tier-0 claim was independently reproduced: 576 corruption center/probe comparisons, 568 decisive, 208 correct-positive, 208 correct-negative, 78 noisy-positive and 74 noisy-negative. Noise is 152/568 = 26.7606%; clean is 25%. Those fixed-source supports differ from the newly adapted, scheduled supports and should not be conflated.

N1 uses GT to filter Sa2VA's original signs, not to replace them with a GT-best teacher. It is explicitly a GT-assisted mechanism oracle, not deployable TTA. It completed 3360 arrivals: five primary arms plus one pair of exact-count supplementary trajectories, 16 sources, five orders, clean and five 5% transient corruptions. The fixed Vid-trained TA-STVG checkpoint, 1792 parameters, nine on-policy probes, SGD .005, one persistent step and pre-update output are unchanged. Temporal reranking is unchanged. All predictions were sealed before final metric evaluation, but GT had already influenced the filtered trajectories; sealing does not remove that exposure.

Pairwise softplus replaces the frozen method's coupled reverse KL to make individual signals removable. N1 therefore supports conclusions about the specified pairwise intervention. It does not establish whether RKL absorbs noise, prove that original RKL is positive- or negative-driven, or demonstrate a GT-free noise filter. The historical RKL-Final future delta is +0.025505 pp, versus pairwise All +0.006063 pp; no objective or method is promoted from this analysis. No OPSA reproduction is claimed.

Validation: 3360 state links, 541 pairwise updates, 969472 SGD-coordinate checks, 6426 independent teacher rewards, 7560 GT quality values, 13440 temporal critic scores, 28 full spatial and 14 full temporal reinsertions. Three mathematical CPU tests passed. The independent public audit passed 30666 checks on 5280 anonymous scalar rows (3360 new and 1920 reused controls). GPU runtime was 1360.115 seconds (22.67 minutes), zero failures and zero new specialist inference calls.

Paper48 remains saved and paused at spatial 2658/2658 and temporal 1039/2658. Its monitoring automation is paused. Existing budget files and every completed receipt are preserved; no automatic restart, old queue, baseline or method change was made.

See [protocol](../protocols/tastvg_noise_decomposition_n1_v1.md), [full report](../results/tastvg_noise_decomposition_n1/2026-09-30/REPORT.md), [all paired contrasts](../results/tastvg_noise_decomposition_n1/2026-09-30/PAIRED_CONTRASTS.json), [signal counts](../results/tastvg_noise_decomposition_n1/2026-09-30/SIGNAL_SUMMARY.json), and [figure](../results/tastvg_noise_decomposition_n1/2026-09-30/N1_ANALYSIS.png).
