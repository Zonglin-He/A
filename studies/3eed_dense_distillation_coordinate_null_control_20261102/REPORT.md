# R43：原 G 稠密蒸馏的学生候选坐标与 nullΔG 控制

## 裁决与总体任务结果

**停止当前固定 z 配置。** 在未用于拟合但已暴露的原 V64 上，仅换 teacher KL 学生坐标，mIoU 从原 probability control 的 28.9723% 降至 **1.2516%**。nullG 的相同坐标改变也从 27.7364% 降至 **1.6046%**。严重损害在两个平台均出现，不能归结为少数负例，也不能仅归因为完整 G 中的错误反馈。

| 原 V64，mIoU % | Drone | Quad | 两平台平均 |
|---|---:|---:|---:|
| Frozen source | 24.5126 | 31.6144 | 28.0635 |
| Full G，原 probability | 26.2069 | 31.7376 | **28.9723** |
| Full G，token-log-odds | 0.5534 | 1.9498 | **1.2516** |
| Null G，probability | 24.5126 | 30.9601 | 27.7364 |
| Null G，token-log-odds | 0.7154 | 2.4939 | **1.6046** |
| 同 G 直接重排，不更新 | 26.7235 | 35.6344 | 31.1790 |
| 原 probability＋同当前 G | 26.1778 | 37.6370 | 31.9074 |
| token-log-odds＋同当前 G | 3.2267 | 5.5883 | 4.4075 |
| 原 full-G A-memory，不更新 | 18.7400 | 33.3346 | 26.0373 |
| Null 的零增量 memory | 24.5126 | 31.6144 | 28.0635 |

- 坐标改变在 full G 内为 **−27.7207 pp**，在 null G 内为 **−26.1317 pp**。
- 两种坐标中 full−null 的差上差为 **−1.5889 pp**。这是此固定干预的描述性效应，不是一般因果保证。
- 原 probability 的 **+0.9088 pp** source 增益保留；叠同当前 G 比直接 G 高 **0.7284 pp**，但这是旧控制复现和暴露开发结果，不包装为本轮新方法突破。
- 原 full-G memory 是完整 G 的同信息控制，不是 null 的同信息基线；null 的无更新对照为 source/零增量 memory。

z-full 和 z-null 在 V64 上都改变全部 64 条 query，分别相对 source 有 5 条改善／39 条恶化、5 条改善／40 条恶化。A208 上也严重退化（full-z 平均 2.1742%，null-z 1.8938%），所以不足不是只在跨观测验证时出现。全部逐样本值、负比较和场景表完整保留，没有删掉不利反馈。

原表达族限制与本轮失败可以同时成立：更宽的学生候选坐标不等于更好的任务监督。null 同样崩坏表明，无需外部 G 错误就能产生这一失败。初始化 posterior 重整形、梯度传递和既有 source-KL/hinge 交互仍是解释，不把其中单一因素未经额外干预当成唯一根因。停止当前配置，不继续加步、扫温度或扩大更新层，也不泛化否定全部 logits-TTA。

## 执行状态

原 R24 D208/A208→V64 的 2×2 已真实执行：两种 teacher（完整旧 G／nullΔG）×两种 teacher-KL 学生坐标（native 概率／pre-probability token-log-odds），两个平台各自 source 初始化，共 8 个匹配 MLP 状态、128 个优化步。训练与保存约 4.04 秒，峰值 GPU 分配约 0.46 GB；GPU 已释放。没有新 G/VLM 调用、源训练、RGB 分支、框更新或 FINAL/U/T 读取。

272 条 A/V 的全部原框、完整候选分数、各训练头的 token-log-odds、teacher 与 source posterior、mask/padding 合同和损失记录已先封存。独立 CPU 代数审计与总体任务评分已完成并通过，全部负例保留；结果见上表。

## 配对控制先复现

两个平台的 `full_probability` 先执行并通过原 `query_mlp_lr1_s16` 复现，才训练其余 6 个状态。两平台 MLP 参数都逐位相同，最大差为 **0**；各平台 A104＋V32 的 **136/136 query 精确一致**。没有按任务标签选择容差。

source 特征与投影参数来自原 R24 `capture336_attempt001`，完整旧 G 仍为原 bank272 的 `s0+2vision+2position+2relation`，先按原运算顺序得到 F64，再 cast 为实际 teacher F32。没有使用后来的 source 重捕捉结果或被 fit29 import hook 改写的 loader。所有使用到的 A/V source 与 G 文件字节哈希均与旧 seal 对照。

## 唯一主改动与 null 因子

只有完整 teacher KL 的学生候选分数由 sθ 换成 zθ；source KL、hinge、参数锚定、权重、AdamW5e−4、weight_decay0、16 步和最终 native 概率读出都保持原 R24。z 来自同一前向中 LSE(positive-token logits)−LSE(complement)，保持原 token mask、温度 0.07、归一化、sentinel 和实际有效 token。

null teacher 在两条坐标臂中都固定为 source-prob posterior，不能给 z-null 偷换 source-z target。null 中 qG 是 source 第一名，原 hinge 仍可能非零，因此两个 null 都实际优化，没有拿 frozen 替代 probability-null。

此前无标签前提发现 94/208 条 F 范围超过 native 表达族、20/208 条 teacher 峰值越过上界；同时 109/208 条 ΔG 为零，而这些行换 z 后的源初始化 teacher KL 平均 22.7983。新配置既可能改变 teacher 的可表达范围，也可能在无 G 信号时引起分布校准。任务比较必须同时保留 full/null，不能把 teacher 不变等同于收益必然来自 G 信息。

## 拟合记录不代替任务收益

更新后最后一步各项均保存，以下为 teacher KL：

| 配置 | Drone | Quad |
|---|---:|---:|
| Full probability | 0.3376 | 0.5099 |
| Full token-log-odds | 0.6197 | 1.0411 |
| Null probability | 0.000146 | 0.000349 |
| Null token-log-odds | 0.3114 | 0.5473 |

这些绝对 loss 处于不同学生坐标和初始化，不能仅按数值大小选择方法。完整初末 teacher KL、source KL、hinge、参数锚定、梯度范数和参数变化见 `FIT_RECEIPTS.json`。z-full 与 z-null 的参数变化都约 1.9；这是保存的状态变化，不是外部 G 知识已成功迁移的证据。

## 同信息控制与范围

- Full-G 的无更新控制为当前同 G 直接重排，以及由同一 A 历史 G 构造的原 V64 memory（α1）。原 memory 已封 query 和 box 单独引用，旧全分数未保存一事明确披露；独立审计可按原公式重算而不替换原 query。
- Null-G 的无更新控制为 source/零增量 memory；完整 G 的控制拥有更多信息，不能称为 null 的公平同信息对手。
- 两个 full 模型另保存叠加同一当前 ΔG 的结果，与 frozen 直接 G 比较。它含当前 G 的信息，必须和无更新直接 G 区分，不能把总提升全部算作参数适应。
- R30 是其他监督机制的系统参照，不是本项的因果 baseline。未把 22 条偏好或 U189 保持损失混进本项。

V64 是原 R24 使用过的暴露开发集。本轮没有选参网格，也不能将它再称为新确认。全部框保持原 source 6D 表示与原评估约定；没有混入 yaw/PCA 或几何后处理。

## 封存与后续裁决

预测 seal SHA：`3ad0dc3865fb980ada8b53190a53f0922fa82b3d951fa2b2a6452d7a31c8739a`。原 source/G 控制、所有新正负尾、null 效应和主体候选均保留，预测后才允许任务评分。

独立评分已同时报告 z-full−prob-full、z-null−prob-null、各坐标 full−null，以及差上差。按预先边界，当前配置没有总体增益，且 null 也严重退化，因此停止，不增加步数、温度搜索或更新层。

本轮仅缓存头执行，还没有新权重的 full-grounder closure。即使暴露 V64 有收益，也不是已部署的三维 grounding 方法；若值得保留，必须另锁闭环验证和同信息比较，不直接使用独立 FINAL。

## 独立审计与证据入口

独立审计通过全部 8 个头、1088 个完整 native 分数向量及 token-log-odds、最终 loss 各项、原控制状态和原 memory：CPU64 native 最大误差约 1.36e−6，z 最大误差约 7.38e−6，最终 loss 最大误差约 6.6e−8，CPU64 argmax 差异为 0。原 R24 两平台状态逐位一致，272 条 A/V query 和对应历史任务值精确复现。原 memory 的全部 64 条 V64 query 由原 CPU32 规则独立重算一致，没有拿审计精度的另一个赢家替换旧预测。

2512 个任务 IoU 和全部 2465 条负参考比较均留存；负比较不是 2465 个独立样本。预测 seal、代码锁和 root 授权先于任务评分，审计没有触碰 FINAL/U/T。

- `TASK_SUMMARY.json`、`FACTOR_CONTRASTS.json`、`SCENE_ROWS.json`：全部总体和配对因素结果。
- `ALL_ROWS_PRIVATE.json`、`ALL_NEGATIVES.json`：逐条任务结果和所有不利比较。
- `DECISION.json`：固定 z 配置停止，旧 probability 阳性结果保留。
- `../independent_audit/`：独立代数、原控制和 memory 重放证据。
- `../premise/`：无标签表达前提和 null 初始化分布差异。
- `../review/`：Hinton/RLCF 原文核对、范围和静态审查。

所有新状态仅完成缓存头实验，没有新增 full-grounder closure；本轮不采用或部署新方法。
