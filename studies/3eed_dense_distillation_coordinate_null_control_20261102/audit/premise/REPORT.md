# R43：无任务标签前提的独立数值审计

结论：PASS。原208行、每行256候选全部复算；未修改作者任何数值、公式或文件，未读取任务GT、真实媒体或FINAL，未实例化模型、训练或使用GPU。

源输入及作者封存hash通过；原R24 `teacher` 的运算顺序、F64→F32 cast、三个evidence分量与正/负/零计数逐值一致。独立使用SciPy logsumexp重算分布、熵、KL与缓存token-log-odds；CPU float32 Torch后验与作者封存逐位相同。source-z最大差约1.07e−14，teacher posterior最大差约1.78e−15；全部208行标量、平台计数、quantile与均值在1e−10以内复现。

| 检查 | Drone104 | Quad104 | 全208 |
|---|---:|---:|---:|
| F32跨度>1 |38|56|94|
| teacher峰值>理想native上限 |5|15|20|
| ΔG全零 |62|47|109|
| ΔG常量 |62|47|109|

理想native posterior上限独立推导/计算为0.17635166756422543。源缓存全部概率确在[0,1]内，最小约5.28e−9、最大0.99620819，没有浮点越界成分。将F32存储值作为实数计算的精确跨度，与作者float32减法产生的94行阈值计数相同；这里没有边界舍入改变结论。原rank_adapter的无界线性残差反例也复现，不能将范围界推广到它。

关键混淆同样复现：null teacher的 `KL(source-prob posterior || source-z posterior)` 全体均值25.786637 nats；109条零ΔG行中teacher→source-z初始KL均值22.798292，而teacher→source-prob为0。旧native null hinge全体均值0.02027456，也不是零更新控制。因而任何未来teacher KL坐标p→z的收益，都需要full/null配对来区分外部G信息与分布坐标重整形，不能仅与frozen比较。

source-z复算保留实际1–3个正token及29–86个全部token，包含实际缓存中的特殊/参照/其他token；没有删除分母候选或按熵/峰值/GT筛选。数值zero不重新标成语义UNKNOWN。

有限结论：这些结果证明一个表达能力必要界及初始化坐标差异，不证明teacher选对主体、表达限制是指标瓶颈，也不保证换z后总体会提升。共享MLP/冻结特征对可达分布仍有限制；原GPU float32与当前CPU64不是逐位相同的测量合同。当前审计没有评价后续训练任务结果。

执行记录：attempt001在全部数值断言通过后，写JSON时因NumPy int64序列化失败；attempt002已写完正确结果，但终端print仍有同类序列化失败。仅修复本审计的输出序列化，保留两份错误日志及第一次未完整写出的文件；attempt003成功退出。没有为了PASS更改数值容差、科学门、输入或作者输出。

证据：`AUDIT_RESULT.json`、`INDEPENDENT_ROWS_PRIVATE.json`、`INDEPENDENT_REDUCTIONS.json`、`audit_attempt003.log`。CPU guard已经释放，R42下载未被干预。
