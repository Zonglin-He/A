# R42 独立执行代理的封存后 CPU 草稿静态复核

状态：**草稿静态复核通过，未运行；需全体预测 seal 和 root 的 POSTSEAL_EVALUATION_AUTHORIZATION 后才能执行。** 本次不读取未封存预测、任务 GT 或 FINAL，不操作 GPU。

核查 `audit42_draft.py`、`score42_draft.py`、DRAFT_HANDOFF、ROOT_STATIC_REVIEW 和实际 PLAN/run42 的静态合同。没有修改这两个草稿代码。

独立解析器只读取 JSON；重复键、非有限数、布尔坐标、多候选、额外字段及不能唯一确定 schema 的结构拒绝。兼容单 dict/长度为 1 的 dict list、三种既定 point 键及可选惰性字符串 label；这些结构与 R38 固定兼容规则一致。原回答作为数据解析，不执行其中字符串。未知/歧义结果回到 source，不成为正监督。

从实际 grid×14、绝对 resized 像素、原图比例及 floor 命中规则重新得到唯一 mask；按封存支持矩阵的原阈值和唯一组规则核全组，oldsubset 同键必须一致，再以原 source 分数最大、下标并列合同选择原框。这里只重构封存几何支持矩阵后的映射，没有重新计算点投影或更改 source 框。

输入审计用本地 processor/tokenizer 重构 216 条像素与 token 输入；不加载生成模型。3B processor 文件哈希与固定 prompt/hash/grid 绑定，216 条 raw、token IDs、input-binding、call 对照及持久化 mtime 顺序明确；预算 ledger 必须唯一覆盖 8 synthetic＋208 dataset calls。实际 runner 的单次 greedy、beam1、96 tokens、repetition penalty1.05 固定，无答案重试。导入 run37.messages 不运行其 main，顶层类导入没有模型实例化。

synthetic 的真实蓝块命中与绝对/1000 反事实误差只作定位能力诊断，不能以高命中作为结果成立门，更不能反选单位。严格加载、权重来源和冻结证明复用已封下载/loader 绑定；审计没有声称额外独立复读 16.6GB 权重。

原 24 条身份注释只作封存后 join，18 明确、6 歧义，哈希固定；不新增人工看图或重标注。CPU 读取原图是输入重构，不能表述成人工语义身份审查。身份记录仍有历史暴露/非专家限制，不按该子集选方法。

任务程序只在同一 prediction seal 的独立审计 PASS 和 root 明确 task-GT 授权后加载 A208 对应 GT。完整 208×8=1664 IoU，旧六臂要求逐值复现；保留每平台、等权总体、source/同接口 3B 的得失、全部有序负比较和少场景边界。identity join 不改变反馈或任务评分。没有 U/T/FINAL 专属索引或 GT 路径读取入口。

没有发现需要在封存前修改的阻塞问题。未来若实际运行因字段/环境/技术条件失败，应保留失败并区分测量问题；不得为通过审计放松 schema、门槛或科学比较。
