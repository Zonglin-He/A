# 3EED 跨平台 TTA 探索汇总（截止 2026-11-17 09:00 北京时间）

本目录汇总已完成、已独立审计并公开的实验。探索已截止；本次提交只整理和发布已有结果，没有启动模型、训练或新评分。

**结论：未建立稳定优于同信息零更新对照的六方向无专家 TTA。R123、R125、R126 不采纳。** R124 的源域 GT 受控诊断显示点适配器 Up 的4,896个参数有任务修正能力；这项能力诊断不属于无标签目标域 TTA。

- [完整汇总与机制分析](REPORT.md)
- [六方向扩展集结果 CSV](SIX_DIRECTION_EXP.csv)
- [主要方法对比 CSV](MAIN_COMPARISON.csv)
- [R113–R126 原报告与不可变提交入口](PUBLICATION_INDEX.json)
- [截止状态与研究边界](CLOSEOUT_STATUS.json)

## 最新原始研究报告

- [R123：真实点对应适配器 TTA](../3eed_registered_point_correspondence_adapter_TTA_20261117/REPORT.md)
- [R124：源域 GT 受控能力诊断](../3eed_source_controlled_point_adapter_capacity_20261117/REPORT.md)
- [R125：原生语义监督适配器 TTA](../3eed_native_semantic_registered_point_adapter_TTA_20261117/REPORT.md)
- [R126：完整读出验收与拟合／迁移／融合分解](../3eed_full_readout_validation_of_point_adapter_updates_20261117/REPORT.md)

各原研究目录保留实现、协议锁定、全部匿名逐例标量、错配／随机／直接使用对照、场景结果及审计。R125/R126 共169个文件11,806,757字节，已于更新主分支前完成不可变远端字节验证。

## 边界

DEV768 与扩展集1536均有历史开发曝光，不能称为全新独立测试。C是同一原始观测的另一采样，未参与监督构造和拟合，不代表新场景。原始A256框保持不变，FINAL未访问。后验读出分解只用于解释，未据此选择新方法或超参数。

18页原图／点云案例图册保留在私有工作区。公开仓库不包含原始媒体、原文描述、数据身份、框坐标、特征、权重、梯度或密钥；所有匿名正负结果仍保留于原研究目录。
