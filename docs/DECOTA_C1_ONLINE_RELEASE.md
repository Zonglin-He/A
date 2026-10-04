# DeCoTA 最新锁定 online 研究版本：C1–Scale06

当前研究锁定版本是 `c1_final_research_v1` 的 **C1–Scale06 online DeCoTA**。这里补齐其正式配置与方法说明；本次公开没有重新选参、改算法或改变生产注册。

| 入口 | 内容 |
|---|---|
| [空间正式配置](../methods/C1_FINAL_RESEARCH_CONFIG.json) | 原始锁定设置的公开副本，仅将工作区绝对路径改成相对路径 |
| [时间最终研究状态](../methods/C1_TEMPORAL_RESEARCH_STATUS.json) | 732来源规模验证后保留原 NLL＋hinge，NLL-only 未选中 |
| [完整方法与复现参数](C1_METHOD_CORE.md) | 两分支适应、当前输出与未来 LN 写回 |
| [online 控制器](../scripts/continue_tastvg_decota_c1_same_domain_v1.py) | 有界串行执行与预测封存入口 |
| [空间 Scale06 fit](../vg_tta/c1_luna_tricks_v1.py) | 原始 Adam、损失、0–10 最优状态选择 |
| [空间 LN 继承](../vg_tta/spatial_consolidation_v1.py) | query 清零，选中 LN 位移按 1/16 写回 |
| [原时间实现与缓存适配](../scripts/run_tastvg_decota_c1_same_domain_v1.py) | 原 NLL＋hinge、末态收缩、逐 query 时间重置 |
| [同域 corruption 已完成评估](TA_DECOTA_C1_SAME_DOMAIN_REVIEW.md) | Vid/HC2 的完整指标、负例、成本和独立审计 |

每个 query 的 256 维残差及 Adam 重置；1,536 个空间 LN 参数持续继承。空间使用四帧原 DINO 参考、原接纳、5 L1＋2 GIoU/planned4、Adam .03 十步；当前 query 使用参考损失最小状态，之后 LN 位移只写回 1/16。时间分支仍是原 66,306 参数头的 NLL＋hinge，AdamW 五步、末态向 source 收缩 .25；时间头与优化器逐 query 恢复。原 Vid 工作点是 lr .1 / center .5；最新 HC 同域评估固定沿用原 HC lr .001 / center1，空间 Scale06 没有为 HC 新选参。

`methods/decota_final_simplified_v1/DeCoTAPredictor` 是更早的 **episodic** 精简实现，每次 episode 恢复 source，不能用它代替本页的 online LN 继承。`methods/CURRENT_METHOD.json` 与 `CURRENT_WORKING_METHOD.json` 是各自历史登记；这次发布没有改写它们，也没有把研究胜者静默晋升。

同域 C1 评估已公开在提交 `a9eece98d36cf8173b283784ceaaaf395f410d07`。其各 32 开发＋16 历史曝光确认来源、一 query/源、双序、clean＋五种 5% corruption，属于小面板 online 评估。它不是 fresh-test 或全 query 全量成绩。

现在新增的 Spatial-DeCoTA Direct/critic P0 是独立研究对照：去掉时间适应与 LN 继承，逐 query source reset，只测当前空间纠错。它不替换以上正式锁定版本；其结果和决策见 [P0 报告](TA_DECOTA_CRITIC_P0_REVIEW.md)。

仓库不包含私有媒体、标注、权重、原始专家框、特征缓存或本地环境。配置中的历史证据路径与哈希保留为 provenance，不能据此声称私有 payload 也已公开。代码和配置入口公开不等于下载仓库即可无数据运行。
