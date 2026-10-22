# TubeDETR 新方法双方向完整全量结果

对齐GitHub main 5bd213f7，固定128独立父来源分别完成21配置后锁参，先Vid10303/732后HC1902/227，全部预测封存后独立CPU评分。无OPD、448、优化器或训练。开发曝光与全量重叠，不作为独立确认；全量未反向选参。

| 方向 | Source vIoU | Spatial vIoU | Temporal vIoU | Full vIoU | Full−Temporal | dense空间差 | 未观察空间差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HC2 → VidSTG | 13.1169 | 12.8396 | 16.2644 | 15.9698 | -0.2946 | -0.6958 | -0.7499 |
| VidSTG → HC2 | 15.4251 | 15.4017 | 25.7844 | 25.8211 | +0.0367 | +0.1221 | +0.0881 |

## 判断与边界

Global6时间分支是两方向主要收益来源。Full相对Source的总改善不能全部归于空间传播。相同Global6下，Vid空间模块整体负增益；HC只有很小的正向均值。

- vidstg: Full−Temporal -0.294599pp，父聚类bootstrap95%区间[-0.443523,-0.152847]；wins/losses/ties=4670/3834/1799。这只是已选配置描述，未校正选参偏差。
- hc2: Full−Temporal +0.036692pp，父聚类bootstrap95%区间[-0.232113,0.308264]；wins/losses/ties=929/914/59。这只是已选配置描述，未校正选参偏差。

保留正负结果，不删损伤query，不因全量成绩再改参数或复活旧数值/空间/OPD队列。当前不能宣称空间传播有稳定跨数据集收益。

## 统计范围与执行

原full_metrics的mean_admitted_frames/mean_changed_from_ERef_frames只来源各域复用的开发128 receipt，不能称10303或1902总体均值。两域metric_scope_audit分别澄清，原seal/report/prediction保持不可变。所有总体v/s/t与观察/未观察空间从完整评分明细重新汇总核对，所有receipt hash核验，未再次扫描NPZ载荷。

execution_audit给出真实Source/教师调用计数和新full条目的观测时长。包含原生时间空间消融以及缓存效果、排除进程初始化和复用128此前工作，不能冒充无缓存端到端部署效率。

原final_seal中的publication_pending是封存时状态；最终是否发布按outputs/maintenance/boxfixed_publication.json中晚于本终态的同步回执判断。所有终态及本报告发布核验后停止半小时监控，不保留空跑任务。
