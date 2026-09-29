# Frozen TA-STVG paper evidence matrix

Authorized 2026-09-30. Question: does the frozen rank-calibrated specialist preference + persistent parameter adaptation improve STVG under deployment corruption? Current B1 is running, A1 completed, new comparisons unmeasured. No new mechanism, no winner promotion, no on-policy-superiority claim. This program does not change CURRENT_METHOD or any frozen B1 file.

## Execution order and current status

| Priority | Evidence | Concrete scope | Status |
|---|---|---|---|
| 1 | Main VidSTG result | Existing B1, all retained 9411 queries / 670 sources, 16 conditions × 3 orders; Frozen/Fast/Slow/Final shared execution | running; no full metrics yet |
| 2 | External STVG ports | ViTTA, RoTTA, CoTTA, TENT, SAR; same source checkpoint, inputs, order, corruption, stream reset; no target GT or specialists | official sources pinned; TENT/SAR/ViTTA persistent optimization cores CPU-tested, backbone smoke/full runners pending; CoTTA/RoTTA port unresolved |
| 3 | A2 matched ablations | Raw-RKL and Pairwise Rank; 16 exposed dev sources × 5 J01 orders × 6 conditions; 960 new arrivals; unchanged Final reused | locked, CPU-tested; serial launch after B1 completion |
| 4 | External full benchmark | Qualified ports × the exact B1 cohort/conditions/orders | planned, requires validated ports and per-port finite runtime estimate |
| 5 | Budget | 0, 1/8, 1/4, 1/2, 1 availability; 64 sources, one fixed query/source, 3 hash orders, clean + five 5% conditions | planned; deterministic post-B1 panel, no selection/tuning from its scores; overlap with main disclosed |
| 6 | Mixed dynamic stream | Frozen/ViTTA/RoTTA/Final; 3 orders of existing 670 sources, 7 equal-as-possible source blocks: clean/drop5/blur10/clean/occlusion5/exposure10/freeze5; no internal reset | schedule builder CPU-tested; GPU execution pending ports |
| 7 | Second dataset | HC-STVG-v2 first, clean + five5% conditions, 3 locked orders; same-domain source package if checkpoint available; HC1 secondary | planned; verify official split/media/checkpoint before roster lock; never silently label cross-domain as same-domain |
| 8 | Main offline analysis | first-source vs later-source nonexpert; quartiles; order variation; severity/dose/harm/excess; query-type breakdown; parameter drift | readout code CPU-tested; waits for full prediction/scoring barriers |
| 9 | Efficiency | 100 fixed exposed queries, genuine uncached Frozen/qualified baselines/Final; complete expert calls, latency tails, peak memory,1792 params | planned; exclusive GPU, include model switching/loading assumptions; cached research runtime is not deployment latency |
| 10 | Optional extension | Vid→HC / HC2→Vid clean or5%; URPA temporal related-work audit; TubeDETR small transfer | backlog after required matrix; no automatic second-backbone launch |

Current sole GPU remains B1. A2 cap2400 GPU seconds,8GiB free floor. No added unbounded jobs or concurrent CUDA tests. Full external jobs may be long; estimate and set finite per-stage caps after actual smoke throughput, not an invented multiweek ETA. Maintain negative/no-op/failure receipts. Completed evidence requires public export, remote byte/hash verification, archive receipt.

## Metric and claim contracts

1. The retained B1 cohort is a **development-source-excluded official-test subset**; 10303 official queries minus892 queries/62 current-route sources gives9411. It has historical project exposure and is not globally untouched or the entire official split.
2. Keep original B1 source-macro vIoU-corrected/sIoU/tIoU endpoints. Add query-macro m_tIoU/m_vIoU and strict vIoU>0.3/>0.5, with declarative/interrogative. But **sampled-grid B1 vIoU is not automatically the official dense interpolated evaluator**. Need a separately audited dense interpolation/official scoring sidecar before labelling main-table values official. Original scores and predictions remain unchanged; GT only after global barrier.
3. Source bootstrap clusters videos; orders reuse sources. Average conditions and orders transparently. Threshold at each query before averaging, never threshold the mean.
4. First-source nonexpert vs later-source nonexpert are disjoint. First-source means first occurrence in the inference stream; it does not erase historical project exposure.
5. State drift is actual saved pre/post parameter norm from the same source initialization. Corruption-clean excess is paired within source. A separate clean stream does **not** measure retention of a corruption-adapted final state. Reserve a no-update clean replay of sealed final states for retention; report it pending until executed.
6. Nominal burst and observed dose are GT-independent; correlation of dose with ΔvIoU is a post-barrier, label-dependent analysis. Do not call the entire accuracy-dose association GT-free. Do not strengthen severity to improve the story.
7. Report >5pp harm and define catastrophic drop descriptively as >20pp vIoU loss, fixed before full metrics. No outcomes chosen or omitted based on these bins.
8. Full benchmark never tunes Final. A2 tests only fixed loss configurations. A1 Off-Policy was slightly above Final, so no on-policy-superiority novelty claim. Corruption-specific recovery requires measured positive excess; otherwise use adaptation under corruption.

## External-port fidelity checklist (implementation work, not another scientific gate)

- TENT/SAR: endpoint categorical entropy + mean native Bernoulli actionness entropy, decoder LayerNorm scope, live full forward, persistent optimizer/parameters. Report gate admissions and exact no-op cases; no silently relaxed SAR gate.
- ViTTA: source-train statistics from the matching checkpoint only, persistent target-statistics EMA, temporal-view consistency, persistent SGD; existing episodic momentum=1 path is not the online baseline. Source-training-statistics access must be disclosed separately from source-free baselines.
- CoTTA: preserve source anchor, EMA teacher, confidence-triggered augmentation averaging and stochastic parameter restore. Spatial transforms must map boxes back; output/readout timing must be explicit.
- RoTTA: preserve teacher, uncertainty/age-aware class-balanced memory, timeliness weighting and robust BN where the architecture has suitable BN. STVG has no fixed classification label space, and TA-STVG uses frozenBN/LN: select and justify a structured-output grouping/statistics mapping before claiming a faithful port. A generic replay buffer alone is not RoTTA.
- Bind to identical data/stream/reset; use native baseline inference timing and report a matched before-update transfer readout in addition, rather than hide current-query adaptation differences.
- Test original/port formulas, no-update equality, persistence/reset/resume, finite gradients, and two old real fixtures under exclusive GPU. No new whole-benchmark scores until these engineering contracts work.

## Primary sources inspected

- TA-STVG official code: https://github.com/HengLan/TA-STVG ; local `datasets/evaluation/vidstg_eval.py`, `engine/evaluate.py`.
- ViTTA: https://github.com/wlin-at/ViTTA (c8e01fa63f8a821a2ebdf1f1272872a867b78cdb).
- RoTTA: https://github.com/BIT-DA/RoTTA (67e34c900cdd355fc07e55edd4c577ea7b8ebcc9).
- CoTTA: https://github.com/qinenergy/cotta (c212a204b32be4005092e4323105a24a29ad2952).
- TENT: https://github.com/DequanWang/tent (e9e926a668d85244c66a6d5c006efbd2b82e83e8).
- SAR: https://github.com/mr-eggplant/SAR (20f6e24b17525f34503510afccedc0629b67b7c4).

Downloaded source files and URL/SHA receipts are under artifacts/tastvg_paper_matrix_v1/official. No third-party install scripts executed, no additional weights downloaded.
