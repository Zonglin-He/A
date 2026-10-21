# 候选评分审计与跨锚点探针

本轮只完成固定 v0 的评分审计及一个缓存评分探针，没有执行 OPD、FixedKD、Direct 的新适应。主方法、默认参数、旧结果及正式 Vid/CG 暂停状态保持不变。HC64 是已曝光的开发面板，固定使用 HC128 原顺序前64父来源、每父1条查询。

## 固定候选审计

[fixed.md](../results/score_audit/fixed.md) 与 [fixed.json](../results/score_audit/fixed.json) 使用旧 v0 的完整封存候选和真实保存路径，不重新生成候选、不执行新 rollout。实际非 E 替换是主要分母；同状态分解保留 support、incoming link、memory、guard，固定整路径分数的单替换差同时包括 incoming/outgoing。不能把单替换差相加当作整路径差。

观测节点的原生教师框全部纳入，包括因 E-NMS 或 inactive 未进入候选的框；组件仅按原来严格大于自身峰值一半、四连通规则从保存 mask 恢复。GT 最佳组件明确仅是离线诊断，不用于任何在线选择。

## 严格下游留出

每父来源做八折留一，共512折；统计先在父来源内平均，再对64父来源 bootstrap，不能把512折或重复帧当作独立视频。

每折的 proposal 输入只有原始 Source 轨迹、Source 全视频特征及七处教师输出。原生 ReferDINO query 原先由全八帧平均分选择，因此现在先切出七行 logits，再重选一个跨帧 query；七帧框和 mask 均使用该 query。slot0、几何保护和回退全部绑定融合前 Source，不读取 ERef、旧 all-eight 支持、旧 memory 或旧候选。

候选规则沿用 v0：最多三个槽、相对峰值0.5连通组件、NMS 0.7、cycle置信门0.1；不扫参数。全部512折 proposal 及候选向留出帧传播的预测图先封存，之后才允许读取留出帧、同一七帧选定 query 的 mask。最后才打开 GT 评价。留出节点自身禁止自匹配并保持 Source。

固定一次跨锚点反馈为：候选框分数占据图，通过 Source 特征空间原 top20 / temperature0.1 对应传到留出帧，再对留出的原生 mask 计算二维 soft Dice。比较同一新候选库上的 `Projection − 0.1G` 和 `HeldDice − 0.1G`。两者都没有 memory/link 项，正分数差大于1e−12才替换，规则在读取新 GT 结果前锁定。这是定位评分的一次探针，不是原完整 v0 的单因素方法复验；新库基线是 Source，不能把它与旧 ERef 方法均值直接相减归因。

留出的是教师输出信息：原缓存 ReferDINO 具有 tracking memory 和无时间遮蔽的 temporal self-attention，七行输出仍含八帧视觉上下文。本轮没有重新运行独立七帧教师，因此不能宣称独立教师观测泛化。整体评分读出使用“七处构建＋第八处反馈”的既有八处教师信息；这也不是 GT oracle。

## 执行入口

```bash
# 所有操作 CPU-only；不读取RGB，不运行重模型、专家或decoder。
CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B scripts/score_audit/fixed.py --run
CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B scripts/score_audit/run.py proposals
CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B scripts/score_audit/run.py feedback
CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B scripts/score_audit/evaluate.py
```

旧输入缓存仅本地保留，需遵循原封存协议获得；公开仓库不上传视频、标注、mask/特征/轨迹张量或私有ID。已封存阶段拒绝覆盖。配置在 [config.json](../scripts/score_audit/config.json)，公开聚合、匿名全部正负和证明见 [结果分析](../results/score_audit/REPORT.md)。

新反馈仍有明确反例：Source 描述子无法区分位置时，对应退化为均匀平均，二维 Dice 仍可能偏爱全图框。测试保留该反例，不以改成二维或增加一个留出步骤预先认定反馈可信。
