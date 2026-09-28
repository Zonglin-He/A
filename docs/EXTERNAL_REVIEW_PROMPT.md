# 可以复制给另一位 GPT 的审查任务

请审阅 https://github.com/Zonglin-He/A 中的 DESTA-3D v2。先读 `REVIEW_START_HERE.md`、`docs/DESTA3D_V2_ARCHITECTURE.md`、`docs/DESTA3D_V2_CORRECTIONS.md` 和 `results/desta3d_v2/2026-09-28/README.md`，再沿入口查实际函数，不能只复述文档。

我们还没有建立可靠目标TTA收益。请把问题分成实际实现缺陷、接口/目标不匹配、数值/优化条件、假设证据不足；每项给代码文件和函数、已有支持与反证、最小验证办法。特别审查：66816 FiLM/LN的无标签pre-gate目标与最终native解码联系；teacher-forced GT-prefix CE与native预测anchors支持差异；NTP/MTP真实分母；小残差的BF16注入和离散endpoint；optimizer恢复后的live参数绑定；冻结和checkpoint重算作用域。

已试过辅助回传三臂、输出双anchor、time-only、identity-view、source moments估计量替换诊断、源16监督正控及最终head FP32监督正控。请保留现有正负和失败更正，不重复推荐已经否定/削弱的相同设置。不能把CE下降或梯度连通称tube改善；目标8多次开发曝光，不能当held-out。不要直接给LR/lambda/gate/步数/精度/teacher大网格或更换backbone。

最终给出最值得优先修复的具体问题，或承认暂无足够实现证据；推荐一项能区分竞争解释的最小源端对照，明确控制、过程读出、native指标、失败分支及成本。仓库没有视频/权重/逐样本raw，请明确哪些判断不能由当前公开材料验证。不要把未执行的native-endpoint协议草案当已完成结果。
