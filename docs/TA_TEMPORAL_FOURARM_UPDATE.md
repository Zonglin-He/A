# 2026-09-29 — Minimal temporal four-arm experiment completed

主问题：在固定 transient deployment corruption 下，adaptation 是否比直接 expert rerank 更好？

完成原序前16个历史曝光VidSTG父源/16query，5类full-video query/GT-independent random burst ×1/5/10%=240个corrupted输入，另16 clean。VidSTG-source官方TA-STVG checkpoint 5ab12c86…；UniversalVTG best+PE对同一学生候选排序。Frozen/Rerank/Hard/OPD完整四臂，后两臂各一次固定 .004×联合visual H范数步，仅H_motion、官方detach、正常动态reroute；Hard sigma2/TEMP_COEF2，OPD所有严格排序pair的logistic loss。没有backtracking/保护/低秩/可靠性/多seed，旧C2.5/C0.6计划被最新用户指令取消。GT仅输出封存后评分。

先在每父源内平均15条件，再bootstrap16父源10000次。Transient Frozen/Rerank/Hard/OPD tIoU=35.3431/42.6600/36.8558/36.6961%；vIoU=16.0677/19.3343/16.8038/16.2843%。

- Rerank−Frozen：Δt +7.3170pp CI[2.5797,12.6979]，Δv +3.2666pp CI[.6539,6.4638]。
- Hard−Frozen：Δt +1.5127pp CI[.1974,2.9823]，Δv +.7361pp CI[.2114,1.3313]。
- OPD−Frozen：Δt +1.3531pp CI[−1.1532,3.6807]，Δv +.2167pp CI[−.9966,1.2301]。
- OPD−Rerank：Δt −5.9639pp CI[−10.3589,−2.3079]，Δv −3.0499pp CI[−6.0893,−.8172]。
- Hard−Rerank Δv −2.5304pp CI[−5.7493,−.0039]；OPD−Hard差异CI跨0。

本次固定一步不支持gradient adaptation超越Rerank，后续保留更简单的rerank作为当前实验参照；生产登记不变。不是四臂都无效，也不是所有OPD路线不可能。Hard对Frozen有小正效应，OPD本次平均改善未确立。Hard/OPD在全部256输入各自loss均下降；独立NumPy重算1024个前后loss最大误差Hard6.13e−8/OPD5.31e−7，因此不能把本次结果简单归因于反向未生效，但此证据也不能排除目标、步长或离散解码问题。

Clean Frozen/Rerank/Hard/OPD vIoU=16.7440/19.6060/16.9554/16.5893%。Rerank的corrupt-minus-clean v增益+.4045pp CI[−.6671,1.5420]，更符合一般refinement，尚未建立corruption-specific优势。Transient >5pp v损害：Rerank19cell/4父源，Hard0，OPD17cell/2父源；240cell并非240独立源。Rerank平均最佳不代表逐例安全。

保留正负案例：Q14 frame_drop_10的v为Frozen0/Rerank29.022/Hard0/OPD0%；Q09 frame_drop_5为5.175/4.616/5.171/18.039%，OPD仍有局部成功。Q10 frame_drop_10 Rerank由28.014降至16.892%，OPD28.421%。完整匿名逐条结果与各条件汇总、正负案例均公开，不用极端案例替代主均值。

256原始H replay、4完整回插、全部非motion冻结/模型hash、3072双实现任务评分、256独立critic重建、31无实际像素变化输入四臂与clean精确相同、3CPU测试、519公开scalar bootstrap核验通过。无mapping collision、无运行失败。总GPU进程807.944586s（13.47min），包括capture smoke/加载/CPU解码和验证，不含CPU开发评分及旧clean缓存创建。已完成512次单步适应，未逐step重跑backbone。

仅此前曝光开发16源与现有单次burst实现；不作为fresh full benchmark或corruption专属性证据。本轮不自动追加任何实验，spatial和其余想法留backlog。

Results: [report](../results/tastvg_temporal_fourarm/2026-09-29/REPORT.md), [rows](../results/tastvg_temporal_fourarm/2026-09-29/ROWS.json), [protocol](../protocols/tastvg_temporal_fourarm_v1.md).
