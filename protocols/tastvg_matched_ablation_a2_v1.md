# A2: Raw-RKL and pairwise ranking, five matched orders

Authorized 2026-09-30 as part of the frozen-method paper matrix. Planned; no effect claim until sealed execution and independent scoring.

Reuse the A1/J0.1 exposed 16 VidSTG sources and 16 queries, five prelocked orders, clean + five 5% transient corruptions, Vid-trained TA-STVG checkpoint and the exact frozen J0.1 recipe. Two new arms × 5 × 6 × 16 = 960 arrivals, 60 resets. Same 25% specialist schedule, detached nine current-policy candidates, 1792 parameters, SGD .005, one step, current output before spatial update and inherited future state. No new expert/capture except bounded full reinsertion checks; frozen Final/J01 controls reused.

Raw-RKL uses q=softmax(raw IoU reward), temperature 1, exactly the earlier S1 raw loss. Pairwise uses mean softplus(d_i-d_j) over each pair whose expert reward rank i is strictly better than j; tied pairs omitted, no comparable pairs yields differentiable zero. d is the unchanged L1+GIoU geometry distance. No temperature, margin, loss-scale tuning or norm matching. Hence comparison is within this matched configuration, not universal dominance of an objective family. Final remains Rank-RKL regardless of results.

Seal every prediction before rereading only the existing 16 labels. Report all/future source-mean s/t/v, five-order mean and sample SD, paired Final differences, clean, >5pp harm, q spread, gradient/step norms; preserve every order. Audit reward ranks, raw softmax or pairwise scalar independently, SGD state chain, 8 spatial + 4 temporal full reinsertions. GPU cap 2400 seconds including failed attempts, disk floor 8 GiB. Run serially after B1; never compete for its GPU.

Entrypoints: scripts/run_tastvg_matched_ablation_a2_v1.py prepare / run; scripts/score_tastvg_matched_ablation_a2_v1.py; scripts/report_tastvg_matched_ablation_a2_v1.py. Parent code, locks and results unchanged.
