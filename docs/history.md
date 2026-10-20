# 项目历史与当前状态

2026-10-20，用户明确将“单 ReferDINO 初始化＋空间 OPD＋Strong 时间分支”及简化参数作为唯一主方法，并授权删除废弃开发产物。该整理替代以前的多方法登记；唯一配置为 `configs/default.json`，唯一运行入口为 `python -m stvg`。

当前 HC 参数为 lr0.014、tau0.14、40步、初始化gamma0.5、输出alpha1；Vid为0.01、0.5、30步、gamma1、alpha1。权重冻结，seed41。保留原256条正式结果、所有正负样本和严重尾损说明。本轮只是依赖迁移、清理与工程回归，没有改变算法、重新选参或运行新实验。

清理前的完整公开证据版本：[7a0a813](https://github.com/Zonglin-He/A/commit/7a0a8131733d124c9262db0678260c46f095ead4)。旧方案、完整参数矩阵及诊断可通过正常 Git 历史恢复，例如：

```bash
git show 7a0a8131733d124c9262db0678260c46f095ead4:results/stvg_error_state/2026-10-20/single_referdino_parameter_search256/rounding/REPORT.md
```

清理前的私有研究总档案已在项目外单独保存；不把私人记录上传 GitHub。其他任务留下的未跟踪文件也在项目外保留，未作为废弃代码删除。

工程回归：两条已有查询从视频到最终输出均与旧封存结果逐位一致；原256条预测的五行指标全部复核，最大绝对误差约1.6e−15。核心policy、共识初始化、时间后验及MBR等23个运算函数的 AST 一致。检查只验证重构兼容性，不产生新科学成绩。结果分别见 results/migration_checks.json、results/score_regression.json 和最终 cleanup_checks.json。

历史结论保留：本方法的现有结果属于已曝光开发面板，Vid的适应均值增量尚不稳定；BV030等负例未删除。工作树只保留当前可运行方案不代表此前负结果被反驳，也不代表已经完成全量泛化验证。

2026-10-20 整理更新：主方法迁入独立 `stvg-opd/`，共享外层必需资产，不改变冻结配置。按用户追加要求并行审计推理标签隔离、开发数据曝光和公开导出；最终审计见 docs/data_integrity.md 与 results/data_integrity.json。Git历史保留，因此本轮公开内容检查限定于新提交树，不能视为删除历史公开记录。

泄漏审计结论：未发现当前推理读取GT；128条离线选参合法，不计为数据泄漏。已修复封存校验/标签读取顺序、解码前schema检查及注释软链接防护，14项CPU检查通过。新正式评估按用户最新指令暂停，本轮不安排新实验。迁移到独立目录后的两条原查询完整回归仍逐位一致。

GitHub同步完成：实现提交 [87294995](https://github.com/Zonglin-He/A/commit/87294995eea16161e3edb88289ee80463c87cb13)，tree `b42ffc2848e8e76e1db2c48189686a1bf7177686`。main与原research分支以fresh expected-SHA、非强制快进更新。远端重新fetch后，248个文件／2,023,178字节全部与本地导出SHA256一致，旧7a0a813提交仍是祖先；详见 docs/publication.json。本条及回执在随后的档案提交中保存，不代表新实验或新方法晋升。

本地Git迁入独立目录后，补核Git忽略规则对资产软链接同样生效（目录专用尾斜杠改为兼容文件/链接的根路径规则）。公开导出始终使用明确allowlist，私有链接未进入任何本轮提交。最终远端核验回执另存本地 outputs/maintenance/publication.json。
