# 模型修改前可复用的全量缓存

本说明来自 2026-10-01 当前 quick/full runner、输入构造、encoder capture、动态 suffix 和专家接口的代码核对。只做 CPU 资产清点与依赖分析，没有启动全量预计算或恢复暂停作业。缓存能否复用由实际依赖决定，不能承诺任何模型改动都不影响它。

## 可以优先保存

| 缓存 | 改学生 loss/critic/decoder 后 | 必须保持的依赖 | 当前情况 |
|---|---|---|---|
| 原媒体与 metadata、媒体 SHA、query/source 映射、原始 frame IDs、fps/尺寸 | 可复用 | 原文件与 cohort 不变 | 已有全量本地媒体和锁定 metadata |
| corruption 物理 burst seed/区间/参数/recipe | 可复用 | source、frame count、corruption 实现/severity 不变 | 不使用 query 或 GT；改采样仍可沿用物理 recipe，观察帧须重构 |
| subject 解析、tokenization | 可复用 | query、parser/tokenizer 版本/规则不变 | 暂停全量 Vid 的 10303 query subject 已封存；HC 全量尚未生成解析封存 |
| UniversalVTG proposals、confidence、physical-time mapping | 可复用 | 同像素/网格/query，独立 teacher 权重/代码/预处理/精度 | 不依赖学生候选或参数，已有多轮匹配缓存 |
| Sa2VA reference boxes、valid、positions | 可复用 | 同 corrupted 观察/query/prompt/5 帧采样，teacher 权重/代码/精度 | 缓存保存框/有效性，没有保存原始 masks |
| 解码 RGB、固定 resize/normalize 输入 | 可复用 | 同媒体/物理帧、RGB/解码库、resize/normalize、corruption | 按精确观察共享，不能按 source 合并不同 clip/grid；全保存可能很大 |

换学生甚至换 backbone，只要这些专家的输入和实现不变，独立专家输出仍可复用。若新方法让专家看学生 crop、预测区间或改写 prompt，专家输出就依赖学生，不能直接沿用旧缓存。

UniversalVTG 的 CLIP 视觉特征可在共享同一观察的多 query 间复用，文本 embedding 可按 caption 复用，值得单独持久化。当前 worker 会计算它们，但未单独保存可复用 feature cache；这是后续接口建议，不是已完成的缓存。Sa2VA raw masks 也不在当前缓存里；未来若改 mask-to-box 或使用 mask reward，需额外保存原始 mask。

## H 的边界

当前 H 是 post-multimodal-encoder 表示，含 appearance/text/motion 两 offset views，依赖 ResNet101、VideoSwin、RoBERTa、multimodal encoder 及其 norm、query/subject、像素/位置、采样和精度。当前更新的 1792 参数全部在 H 后：decoder 最后层三组 LayerNorm weight/bias（1536）与第二 pass query residual（256）。

修改 lr/rho/K/temperature、reward/critic、decoder 后续适应逻辑时，原始 H 可以作为输入，重新执行 TTS/ASA/actionness/dynamic selection/decoder。改上游权重、encoder/subject 输入、帧采样、resize/normalize 或精度则 H 须重建。改 decoder 架构还需检查 feature layout、位置和 replay adapter 兼容性，不能把旧 route/decoder metadata 当新模型输出。

当前 scratch 包含原 Frozen prediction，key 包含整个 model-state hash，不是已实现的独立 encoder cache。未来应将 H 与 Frozen prediction 分开：H key 使用上游计算闭包的代码/权重 hash + 输入身份 + dtype/layout；Frozen 另按完整 frozen 模型版本保存。这个拆分是建议，本轮没有改已 pin writer。

旧 Frozen predictions 仅对原 frozen 模型/输入有效；只改适应方法可以继续比较同一 Frozen。改变 baseline checkpoint、native decoder 或输入时，新的 Frozen 必须重新推理。旧在线候选、奖励/ranking、soft distribution、梯度/optimizer/state、post-update prediction 和 Ours 结果依赖新模型及在线历史，不能当新版本结果复用。

## 容量与已有资产

CPU 清点在 `artifacts/tastvg_full_cache_inventory_cpu_v1/INVENTORY.json`。暂停全量 Vid spatial 有 10122 份收据，这是文件数，尚未逐份验证与未来输入/teacher/cache 的匹配覆盖，不能称全部保证可直接使用。

官方全 query 六条件共 82710 个 query-condition 观察（Vid 61818、HC 20892）；双序 165420 到达不需重复保存相同 H，每份 H 已含两个 native offsets。当前每数据集 scratch 约 1 GiB；retained tail 平均单份 Vid 22.6 MiB、HC 14.6 MiB，据此外推全部 H 约 1.62 TiB。该 tail 不是全 cohort 形状普查，只是量级警示，不是实测全量体积。磁盘约 83 GiB，不能直接保存全部 H。

优先次序：metadata/recipe/subject → 独立专家输出 → 可共享的冻结视觉/文本特征 → 所需观察的 bounded H cache。FP32 H 应无损压缩或受控逐出；直接 FP16/量化会改变数值合同，不能称 exact replay。

## 复用合同

每层只包含实际依赖：媒体/pixel SHA、physical frame grid、query/subject 或 prompt、provider/encoder checkpoint 与代码版本、preprocessing、dtype/layout、输出 cache SHA。只按 filename/source ID 不充分；query-independent 视觉特征不应把无关 caption 放进 key。当前专家 input digest 没显式编码所有 provider 版本，版本由已锁 namespace/runtime 保证；跨版本共用存储前须补明确合同。

新模型使用旧 feature cache 前，在固定无 GT 输入核验 native/replay 预测及 metadata，保存缓存命中计数。原预测、诊断日志、负例和封存状态继续保存；GT 诊断在新预测全封存后进行，不能写进在线 cache 的选择规则。
