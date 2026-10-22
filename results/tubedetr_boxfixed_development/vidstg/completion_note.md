# HC2 → VidSTG：BoxFixed + Global6 完整10303结果说明

完整10303 query / 732父视频已预测封存、独立CPU评分；所选配置来自事前固定128，gamma=.0125、准入=.4、保护=.4、连接=.5，全量不反馈选参。开发曝光与全量重叠，不称独立确认。

| 方法 | m_vIoU | dense sIoU | m_tIoU | 未观察帧 sIoU |
|---|---:|---:|---:|---:|
| Source | 13.1169 | 41.4379 | 28.6740 | 41.4304 |
| Spatial | 12.8396 | 40.8588 | 28.6740 | 40.8055 |
| Temporal | 16.2644 | 41.4379 | 36.5756 | 41.4304 |
| Full | 15.9698 | 40.7421 | 36.5756 | 40.6805 |

Full−Source m_vIoU +2.8529pp，父视频bootstrap区间[2.4023,3.3258]；这是完整Full的总变化，包含时间模块贡献，不证明空间传播有效。相同Global6下Full−Temporal为−0.2946pp、dense−0.6958pp、未观察帧−0.7499pp；观察帧空间+1.9468pp。Spatial−Source m_vIoU−0.2773pp。结果支持时间聚合改善，空间传播整体仍受损；观察处改善不能代替未观察/全轨迹收益。

原始full_metrics.json中mean_admitted_frames=34.375、mean_changed_from_ERef_frames=13.234375仅来自复用开发128的receipt；其余10175条receipt未保存这两个计数，不能称全量均值。metric_scope_audit.json逐项记录此范围澄清；全部10303定位指标由完整details重新核对，所有receipt hash已核验，不修改任何原seal、报告或预测，不影响HC正在运行的代码/配置。

当前HC2完整1902仍在串行运行；本文件不是两方向全部完成报告。VidSTG新终态与本范围澄清一起发布，新回执scope区分Vid已完成和HC待完成。
