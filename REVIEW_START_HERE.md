# 代码与研究审阅入口

## Latest TA-STVG: frozen-method matched ablations; full test registered

[Matched ablations](results/tastvg_matched_ablation_a1/2026-09-29/REPORT.md), [Chinese interpretation](docs/TA_MATCHED_ABLATION_A1_UPDATE.md), [full test protocol](protocols/tastvg_full_b1_v1.md). Five-order future vIoU: Final +0.025505 pp, Random-Rank −0.004011, Off-Policy +0.026227, Direct PL +0.003409. Preference beats this random control and fixed PL; **on-policy superiority is not established**. Frozen recipe unchanged. Full official test registration retains all9411queries/670sources after excluding62current-route development sources;3orders×16conditions. Historical project exposure disclosed; full-run results pending, not claimed complete.

## Latest TA-STVG: J0.1 online schedule robustness

[Chinese results](docs/TA_SCHEDULE_J01_UPDATE.md), [five-order report](results/tastvg_schedule_j01/2026-09-29/REPORT.md), [recipe freeze](methods/tastvg_dual_evidence_j0_v1/FREEZE_J01.json). The exact J0 method was tested on five prelocked source-hash orders with 25% specialist availability. Whole-stream Fast and Final improve in 5/5 orders: Final gain +1.5177 pp, sample SD 0.8599 pp. Future spatial transfer averages +0.0255 pp but is positive in 4/5 orders, negative in one. Original J0 negative result retained separately; all six also summarized. Freeze the existing research recipe for subsequent evaluation; no favorable-order selection, new gate or production promotion. Same 16 exposed sources, not five independent cohorts.

## Latest TA-STVG: J0 integrated sparse-expert online method

[Chinese interpretation](docs/TA_JOINT_J0_UPDATE.md), [four-arm report](results/tastvg_joint_j0/2026-09-29/REPORT.md), [fixed recipe](methods/tastvg_dual_evidence_j0_v1/config.json). Current-policy temporal reranking + persistent spatial Rank-RKL executes correctly. Future Final−Fast vIoU+0.0472pp reproduces S1.1, but whole-stream Final−Frozen is−0.0363pp (CI crosses0): the positive final-method criterion is not met. Fixed25%expert availability selects four sources with negative temporal reranking gain, despite a positive prior full-availability reference. No favorable-schedule selection, new module, parameter-space OPD or production promotion. Components and implementation archived; mechanism optimization stopped.

## Latest TA-STVG: S1.1 rank preference and normalized spatial updates

[Chinese interpretation](docs/TA_SPATIAL_RANK_S11_UPDATE.md), [three-arm report](results/tastvg_spatial_rank_s11/2026-09-29/REPORT.md), [fixed protocol](protocols/tastvg_spatial_rank_s11_v1.md). Raw-RKL reused; two new on-policy persistent1792D arms,192arrivals/36updates. Rank preference gives small positive future-nonexpert transfer; normalized1%-probe steps do not improve over Rank-SGD. No sweep, parameter-space OPD, joint run or production promotion.

## Latest TA-STVG: spatial critic qualification and persistent Reverse-KL

[Chinese interpretation](docs/TA_SPATIAL_CRITIC_S06_ONLINE_S1_UPDATE.md), [S0.6 critic report](results/tastvg_spatial_critic_s06/2026-09-29/REPORT.md), [S1 online report](results/tastvg_spatial_online_opd_s1/2026-09-29/REPORT.md). Cached critic antithetic ordering is informative, so the user-authorized persistent1792D spatial RKL was run.18 actual SGD updates across six16-arrival streams, current-policy probes regenerated, no expert regression target. Fixed tau1/SGD.005 yields negligible practical future-nonexpert gains. Both positive critic evidence and weak online outcome retained; no joint run or production promotion.

## Latest TA-STVG: S0.5 Native Spatial Rollout Support Test

[Latest report](results/tastvg_native_spatial_rollout_s05/2026-09-29/REPORT.md), [Chinese interpretation](docs/TA_SPATIAL_PROPAGATION_S05_UPDATE.md), [protocol](protocols/tastvg_native_spatial_rollout_s05_v1.md). Student-only1792D parameter neighborhood,4 orthogonal antithetic pairs plus native, one5% radius,96cells/864candidates. No experts or GT during generation; post-seal oracle support remains limited. No S1/OPD/online learning or production promotion. Completed [threshold v1](results/tastvg_spatial_propagation_s05_superseded_v1/2026-09-29/REPORT.md) and [soft-moments v2](results/tastvg_spatial_propagation_s05_superseded_v2/2026-09-29/REPORT.md) are preserved and explicitly superseded by the latest user route.

## Latest TA-STVG: S0 RVOS-guided spatial expansion

[Report](results/tastvg_spatial_expansion_s0/2026-09-29/REPORT.md), [Chinese interpretation](docs/TA_SPATIAL_EXPANSION_S0_UPDATE.md), [protocol](protocols/tastvg_spatial_expansion_s0_v1.md). Real Sa2VA-4B sparse-video masks guide three fixed appearance-only native trajectories on16 previously exposed sources × clean/five existing5% corruptions. Native B0 always retained; whole-tube spatial oracle compared against matched six-layer support. This measures candidate headroom, not a deployed selector or online-TTA benefit. Temporal research frozen at native candidates + UniversalVTG reranking; prior OPD/KNN reports remain historical. No S1 or production change.

## Latest TA-STVG: O3 Conditional Online OPD completed

[Report](results/tastvg_conditional_opd_o3/2026-09-29/REPORT.md), [Chinese interpretation](docs/TA_CONDITIONAL_OPD_O3_UPDATE.md), [protocol](protocols/tastvg_conditional_opd_o3_v1.md). Matched768->128->1 Pairwise/Reverse-KL students on original O2 streams:40 verified CPU SGD updates, but both preserve all60 nonexpert choices and yield zero t/v gain. Reverse-KL remains an implementation candidate per user tie preference, not a validated benefit or production promotion. [Earlier KNN measurements](results/tastvg_preference_memory_superseded/2026-09-29/REPORT.md) completed before steering and are separately archived as superseded. S0 deferred; no spatial inference. Earlier latest/running entries below are historical.

## Latest TA-STVG: O2 regime-coherent online transfer completed

[Report](results/tastvg_regime_online_o2/2026-09-29/REPORT.md), [Chinese interpretation](docs/TA_REGIME_ONLINE_O2_UPDATE.md), [protocol](protocols/tastvg_regime_online_o2_v1.md). Five fixed16-source streams, four specialist writes each; nonexpert macro gains +0.1293pp tIoU / +0.0454pp vIoU, driven by two of60 cells. Four choices change; three regimes have no change. Conditional intervals include zero, so retain local positive cases without claiming coherence solves transfer. No method tuning or follow-on experiment. Earlier latest/running statements below are historical snapshots.

## Latest TA-STVG: O1.1 fixed-state scale readout completed

[Report](results/tastvg_o11_scale_readout/2026-09-29/REPORT.md), [Chinese interpretation](docs/TA_O11_SCALE_READOUT_UPDATE.md), [protocol](protocols/tastvg_o11_scale_readout_v1.md). CPU-only alpha1/8/16/32 on the saved O1 arrival states: many choices move, but no amplified scale improves mean vIoU. No retraining, new expert calls, online trajectory or normalization rerun. Earlier latest/running entries below are historical snapshots.

## Latest TA-STVG: O1 sparse-critic online transfer completed

[Report](results/tastvg_sparse_online_o1/2026-09-29/REPORT.md), [Chinese interpretation](docs/TA_SPARSE_ONLINE_O1_UPDATE.md), [protocol](protocols/tastvg_sparse_online_o1_v1.md), [persistent ranker](vg_tta/tastvg_sparse_online_v1.py). One32-source stream, eight specialist writes; primary future non-expert t/v gain is exactly zero in this fixed first implementation. State and chronology are verified; no added mechanisms or production promotion. Earlier latest/running statements below are historical snapshots.

## Latest TA-STVG: minimal temporal four-arm experiment completed

[Results](results/tastvg_temporal_fourarm/2026-09-29/REPORT.md), [current protocol](protocols/tastvg_temporal_fourarm_v1.md), [one-step implementation](vg_tta/tastvg_temporal_fourarm_v1.py), [anonymous per-cell results](results/tastvg_temporal_fourarm/2026-09-29/ROWS.json). Frozen/Rerank/Hard/OPD on the unchanged transient setting plus clean, only motion-H updates. Earlier C2.5 gates and C0.6 multiseed plans are superseded; no production change. Older running/next statements below are historical snapshots.

## 先读代码

手动调参入口是 [`docs/desta3d/README.md`](docs/desta3d/README.md)。核心 adapter 保留在 [`vg_tta/desta3d_v2.py`](vg_tta/desta3d_v2.py)，新可编辑工作区在 [`desta3d/`](desta3d/)，参数集中于 [`configs/desta3d/`](configs/desta3d/)。

建议阅读顺序：

1. [计算图与文件地图](docs/desta3d/CODE_MAP.md)：分清 adapter、Joint mixer、DirectionMixer 和 pixel evidence。
2. [参数表](docs/desta3d/HYPERPARAMETERS.md)：尤其输入 feature_dim 与 hidden_dim、radius 与 loss 的关系。
3. [`models.py`](desta3d/models.py) 的 `DirectionMixer.forward` 和 `direction_loss`。
4. [`train_cached.py`](desta3d/train_cached.py) 的 `_fit`：等 query 梯度累积、Adam counters、末态保存。
5. [`test_desta3d_workbench.py`](tests/test_desta3d_workbench.py)：CPU 等价与输入/输出合同。

新入口只在明确执行 `fit-cache` 时训练小 predictor，没有自动 PTD native、external teacher、OPD、target 或后续队列。Local/Global/R16 preset 保留原实验起点供人工审阅，不代表恢复已停止的调参链。

## 当前科学状态（2026-09-29）

最后一次 external evidence→same-PTD Dev16 qualification 已完成并独立审计。B1 的 vIoU 为23.041295%，TS privileged policy 为19.079401%，Δv为−3.961893pp，CI跨0；按用户预锁 gate 停止该 privileged-correction/OPD 主线投入。此结论不撤回 Full/R16 oracle 的正证据，也不泛化成所有 latent correction 不可能。

详细指标、正负尾、两个工程修复与不确定性见 [最终报告](https://github.com/Zonglin-He/A/blob/6b0cce3e4defd982ab63407107b9cba159429ab9/results/desta3d_v3/2026-09-29/EXTERNAL_POLICY_GATE.md)。代码整理没有新增 GPU 研究测量。

## 历史与复现

旧审阅页完整快照在 [REVIEW_START_HERE_HISTORY_20260929.md](REVIEW_START_HERE_HISTORY_20260929.md)。旧 `scripts/` 和 `protocols/` 仍是各自已锁实验的原件；不要把其中早期“running / next”文本当成当前队列状态。

公开仓库只含代码、协议和匿名结果；媒体、标签、cache、权重、预测 raw 留在本地。本地最高优先级状态入口为 `docs/RESEARCH_HISTORY.md`，生产方法由 `methods/CURRENT_METHOD.json` 决定。

## New authorized DESTA experiment (running; results pending)

Two independent specialists → native pseudo-targets → R16 latent adaptation. [Readable method and hyperparameters](docs/desta3d/DUAL_EXPERT_NATIVE.md), [27 → 6 → 1 protocol](protocols/desta_dual_expert_native_v1.md). Previous negative results remain; this is not yet an efficacy claim.

## TA-STVG Round1 evidence sensitivity — completed

[Full64 report](results/tastvg_evidence/2026-09-29/REPORT.md): 25/64 parents preserve their native tube while passing a prelocked strong-evidence-drift screen; new extension48 contributes16 cases. Signal concentrates in ASA; native correctness and TTA benefit are unmeasured. [Protocol](protocols/tastvg_evidence_vulnerability_v1.md), [cached capture](vg_tta/tastvg_evidence_capture_v1.py), [attack](vg_tta/tastvg_evidence_attack_v1.py), and [all192 anonymous scalar rows](results/tastvg_evidence/2026-09-29/ROWS_INDEX.json). Round2/3 have not started.

## TA-STVG Round2 — completed causal and oracle audit

[Full report](results/tastvg_causal_round2/2026-09-29/REPORT.md), [protocol](protocols/tastvg_causal_correctability_round2_v1.md), [native oracle and causal implementation](vg_tta/tastvg_causal_round2_v1.py), and [all anonymous scalar rows](results/tastvg_causal_round2/2026-09-29/ROWS_INDEX.json). Official detach semantics retained; seven-arm GT oracle on original64 exposed development parents, not an unlabeled TTA result. Round3 remains unstarted.

## TA-STVG corruption C0/C1 — frozen anatomy and student support

[Full report](results/tastvg_corruption_c0c1/2026-09-29/REPORT.md), [protocol](protocols/tastvg_corruption_c0c1_v1.md), [anonymous rows](results/tastvg_corruption_c0c1/2026-09-29/ROWS_INDEX.json). Same-domain Vid-source primary;32 parents x7 frozen conditions and first16 x4 candidate sets. No experts, gradients or adaptation; C2 remains unstarted.
