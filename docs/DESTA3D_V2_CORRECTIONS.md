# 历史更正和版本边界

## Optimizer 恢复事故

旧`cpu_copy`递归复制把字典的整数key转成字符串。Adam state_dict的整数parameter ID因此在load后成为orphan state，下一次更新为live Parameter重新创建momentum。旧测试只比较serialized内容，没检验恢复后的两次真实更新与连续控制，漏掉了这个问题。

受影响的原sourcefit002/003及contrast002的**连续Adam/继承momentum声明撤回**。旧panel001原梯度和实际位移保留，但继承原momentum的解释失效。旧预测/正负报告和no-update/P0测量不因此全部作废。历史文件没有删除，2026-09-27发布目录应结合本更正阅读。

有效COMMON_A来自连续618query/95Vid父源/155A步。旧B344权重被失败续跑覆盖，不能声称可恢复它。新recovery_v2从同A、fresh AdamW和新seed重新开始B；不是原B的RNG逐位重放。新`optimizer_checkpoint.py`保留整数keys，拒绝orphans，检查live Parameter绑定、state精确一致；每allocation保存不可变入口恢复点。CPU真实hidden128三臂恢复后两次更新逐位对照和真实GPUcounter连续性均已验收。CUDA RNG在CPU测试中mock，与实际GPU验证分开。

## 其他必须保留的工程失败

- target8 v1错误地要求token格式合法时所有geometry均有效；官方可能产生合法token对应无效零框。隔离v2修正结构合同并保留零框，未按成绩筛样本。首失败native在assert前未存，这一缺项没有伪称补回。
- 可微output-anchor的source/target OOM、CPU reserve及stride失败都保留且计费；后续memory修复不改loss/support。只有MLP checkpoint，attention/KV不checkpoint；offload/物理padding不等于改变数值支持。
- source task-control v1遇到相同状态下反传重复性不满足逐位检查，触发的第二raw未保存。后续明确开启deterministic algorithms、cudnn deterministic与CUBLAS配置后逐位重复通过；未定位具体kernel，不能据此声称全部旧梯度无效。
- FP32源16 CPU scorer v1在读评分标签前因局部变量遮蔽helper失败；独立v2仅改helper名/输出目录，预测和指标不变，无GPU重跑。prefix-support CPU v1字段合同失败，隔离v2修复；缺少完整实际support的15源没有自动补GPU。

## 哪些结论不应扩大

CPU合同通过、真实梯度非零、输出logits改变、CE下降、一步一阶task·Adamdelta为负，分别是不同层次的证据，都不是native tube或目标TTA成功的充分条件。FP32对照的源负结果也不是整条dual路线或FP32普遍无效的证明。

目标8父源是反复开发曝光集合；CI为单seed小panel描述统计，未做多重比较校正。源训练Vid-only，HC是跨数据集迁移。全部结果保留格式失败/无效几何/负尾，不用loss或GT挑最终state。

正在公开的脚本含历史runner和诊断依赖；请使用`REVIEW_START_HERE.md`的状态映射。最新native-endpoint草案只有proposed状态，未构成新的实验结论或自动运行授权。
