# STVG: Query–Caption Temporal + Direct Spatial OPD

当前唯一研究入口是 [`main_method`](main_method/)；冻结配置见
[`methods/MAIN_METHOD.json`](methods/MAIN_METHOD.json)，执行说明见
[`README_MAIN_METHOD.md`](README_MAIN_METHOD.md)。

一个冻结 Qwen3.5-9B 同时提供双假设 Query–Caption 相对反馈和四锚 Direct 几何参考。
Temporal 使用原 512 参数合并区间 KL 适应与 Marginal MAP；Spatial 使用原 1792 参数
query/LN episodic Gaussian OPD、完整 response/验收与回滚。域参数和原算法固定，
不使用外部 Actionness、DINO、跨 Query 继承或新的参数搜索。

正式评价计划是 HC-STVG-v2 validation 3,482 queries / 237 videos 与
VidSTG test 10,303 queries / 732 videos。新的联动 Full 在实际完整预测封存、
评分和远端公开之前不称已经完成；原有开发负结果保留。

- [冻结执行协议](protocols/stvg_main_qwen_v1.md)
- [代码/存储清理及 CPU 工程收据](results/stvg_main_qwen/2026-10-17/cleanup/REPORT.md)
- [历史实验及负结果](results/)

旧方法源码从当前公开代码树退出，仍可从 Git 父提交恢复。历史结果、公开报告和
CURRENT/PAPER 历史注册保留；它们不构成启动旧队列的授权。权重、媒体、标注、
实际 Query/Caption、原始响应和私有缓存均不包含在公开代码中。
