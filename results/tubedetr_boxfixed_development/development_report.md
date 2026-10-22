# TubeDETR BoxFixed + Global6：固定128开发选择

本轮对齐GitHub main 5bd213f7bdb4241d4d759ab4fb8b20f8f7117b57，采用完整教师框传播、固定评分与精确DP，加六层原生时间后验Global6和原物理MBR。没有OPD、448分支、优化器或训练。仅TubeDETR接口移植；原始空间实现仅变更import命名空间。

两域沿用既有seed41输入哈希固定128 query / 128独立父来源，分别完成21个完整profile，全部预测先封存再独立CPU评分。两域选择都锁定后才启动Vid10303，随后HC1902；全量不用于再选参。开发集已曝光且与全量重叠，不称独立测试。

| 方向 | gamma | 准入 | 保护 | 连接 | 新方法原默认Full m_vIoU | 所选Full | Temporal-only | Full−Temporal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HC2 → VidSTG | 0.0125 | 0.4 | 0.4 | 0.5 | 17.5190 | 19.6015 | 19.6738 | -0.0722 |
| VidSTG → HC2 | 0.025 | 0.4 | 0.4 | 0.5 | 24.3132 | 27.3153 | 27.6495 | -0.3342 |

## 结果解释

减弱初始化、提高准入并加强保护，使两域在开发集上相对新方法原默认分别提高2.0825pp与3.0022pp。准入与保护阶段同时减少改动帧数和相对Temporal损失超过5pp的query数量；当前证据支持更保守的传播设置。数值干预和损伤计数是开发集描述，不证明传播错误的具体因果机制。

所选Full仍未超过相同Global6的Temporal-only：Vid差−0.0722pp，HC差−0.3342pp。dense空间差分别−0.9287pp/−0.7594pp，未观察帧差−0.8405pp/−0.8046pp；不能把相对差的默认配置提升当成空间传播已带来正收益。父来源bootstrap仅描述已选配置，未校正选择偏差。详见各域selection_diagnosis.json。

准入/保护/连接选值位于预先网格上边界。当前不事后扩大网格；全部候选和负结果在decisions公开。gamma0与小gamma差值很小，不能声称数值最优具有独立稳定性。

新全量按已锁参数保留Source、Spatial（BoxFixed/Native）、Temporal（Source/Global6）、Full（BoxFixed/Global6）四项，完整总体预测seal后再独立评分；部分全量不作为主表。旧主实验、旧五行消融、旧OPD结果及其他TA checkout保持。

## 工程边界

86工程检查通过；上游默认DP与快速gamma完整几何重建逐位一致，真实TubeDETR双域Source/Native、八观察与教师框逐位核对。v1曾在推理guard前hash标签文件但未解析，已归档并全新v2重做预测；v2只有完整seal后的CPUscore能打开标签。详见engineering_recovery.json。

当前文件只报告已完成128开发；不代表10303或1902全量完成，未来全量结果与发布回执单独核验。
