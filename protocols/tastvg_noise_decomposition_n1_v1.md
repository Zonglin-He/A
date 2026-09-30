# N1: GT-audited teacher preference noise decomposition

## Question and authority

User attachment a68f14d2, 2026-09-30, requests N1 and saving/pausing Paper48. Paper48 is stopped at spatial 2658 and temporal 1039 receipts; its hourly automation is paused. Do not restart it automatically after N1. Existing deadline metadata is preserved, not silently extended. No old B1, baseline, Self-Rank rerun or method promotion.

Does removing Sa2VA's directionally incorrect signals change subsequent nonexpert performance, and does useful negative guidance act differently from useful positive guidance? This is an explicitly GT-assisted offline mechanism oracle, not a deployable or label-free TTA method. GT chooses which teacher signals to retain; it never changes the sign of a teacher label or creates a GT-best ranking.

## Fixed panel and implementation

Same already-exposed 16 VidSTG sources, one query each, five J0.1 orders, clean plus five 5% transient corruptions. Same Vid-trained TA-STVG checkpoint SHA256 5ab12c86363ef0ce0ee006c00fd11c6b659c3a9b2cb01a4f2c613efe22a2aa83, 1792 spatial parameters, nine current-state probes, frozen L1/GIoU coefficients, SGD .005, one step, persistent per-stream state, and outputs emitted before update. UniversalVTG temporal reranking and scheduled positions [0,4,8,12] are unchanged. Use cached H/expert outputs with their existing hash/pixel checks and exact downstream replay. Reset only between streams/arms.

For each scheduled arrival, generate the nine detached candidate tubes around that arm's current state. Recompute Sa2VA rewards on this support and fixed-GT-frame sIoU for classification. Let yE=sign(Rk-R0), yG=sign(Gk-G0), with absolute tie tolerance 1e-12. Four decisive classes: correct positive (+,+), correct negative (-,-), noisy positive (+,-), noisy negative (-,+). Teacher/GT ties and invalid specialist evidence are counted separately. Historical 26.76% noisy is a frozen-source support statistic; do not assume it holds along every adapted trajectory.

## Five main arms

- All: every teacher-decisive center/probe comparison, including GT ties; GT does not filter this arm.
- Useful: correct positive plus correct negative.
- Noisy: noisy positive plus noisy negative.
- Useful-Positive: correct positive only.
- Useful-Negative: correct negative only.

For selected indices S, z_k=-D(B_current, detached B_k). L=mean_{k in S} softplus[-yE_k (z_k-z_0)]. Empty S means no update. Invalid Sa2VA means no update for every arm. Labels, selected indices and candidate targets are detached/fixed during each update. No LR search, norm matching, entropy mechanisms or reliability learner.

This switches from the deployed research candidate's reverse KL to a decomposable pairwise objective. All is the within-N1 reference; old RKL Final is contextual only. Findings about this loss cannot establish why the original RKL works, absorbs noise, or is driven by negative guidance. Selected-signal means change gradient weighting/norm; these are measured, not silently corrected.

## One exact-count supplementary comparison

Use two additional independent persistent trajectories: Useful-Matched and Noisy-Matched. At each scheduled arrival, prepare each trajectory's own support and eligible pool before either update; set m=min(available useful, available noisy). Deterministically hash-sort probe IDs with SHA256('N1-count-v1|order|parent|probe') and retain m in each trajectory. Both skip when m=0. Thus signal counts and update opportunities are exactly equal at every paired arrival. This conservative check subsamples both pools when needed, and is not mislabeled as the unfiltered Noisy arm. Arm states and eligibility naturally diverge; this is not a same-state gradient decomposition. The quoted historical count 152 is not imposed on the new on-policy streams.

Total: five main trajectories plus one paired supplementary check = seven arms × 480 arrivals = 3360, 210 resets. GPU wall-time cap including failures: 3600 seconds; free disk floor 8GiB. All endpoints, arms and count rule fixed before execution. No outcome-based continuation or selection.

## Evidence and evaluation

First reproduce S0.6's saved 576/568 comparisons and four class counts using sealed scalar rows (no model, no new GT). Only retain the 16 authorized GT keys from the old label container for online oracle filtering, and log this access before updates. Unlike ordinary TTA, GT explicitly influences filtered training trajectories; sealing predictions before the final scoring pass does not erase this exposure.

After all 3360 predictions are sealed, score the same historical sampled-grid sIoU/tIoU/vIoU endpoints, with independent implementations, fixed source-macro averaging and five-order mean/sample SD. Reuse Frozen/Fast/RKL Final scalars after checking matched identities; do not rerun or tune Final. Report each order, clean, corruption, future nonexpert results, >5pp harms, signal class/selected counts, update frequency/norm, exact count equality and paired differences versus All. Orders share sources, so do not treat them as independent test cohorts.

Audit every state link, SGD coordinate, teacher reward, GT direction, eligible and selected index, pairwise term and mean; independently verify count matching and temporal critic scores. Run full spatial/temporal reinsertions on fixed order/condition points for every arm. Preserve failed attempts. Publish implementation, actual protocol, anonymous scalar results/diagnostics and limitations to Zonglin-He/A; exclude media, annotation coordinates, raw predictions/states, weights and conversations. Do not claim reproduction of OPSA from this analogy.
