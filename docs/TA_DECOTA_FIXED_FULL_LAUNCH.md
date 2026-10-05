# Scope replaced on 2026-10-05

The previously launched four-direction all-query corruption queue was stopped
at 880 preserved partial arrivals when the user authorized the formal paper
scope. No partial scoring or automatic resume. The text below is the saved
historical launch record; the current task is
[paper experiments](DECOTA_PAPER_EXPERIMENTS_EXECUTION.md).

# 固定 DeCoTA：全量同域与跨域 corruption 已启动

2026-10-05 登记的是实际运行中的完整评估，当前没有本轮 GT 指标。
固定工作点为 Native-WHEN、Uniform4、单个冻结 Grounding DINO、已准入
Frame-Top1 energy、joint 1792 空间参数、Adam .03、10 步、自身无标签
loss 最早最小值选步，以及 selected LN delta / 16 持续写回。
query residual 与 Adam 逐 query 重置；LN 在每个完整条件／顺序流内继承，
不按视频重置。时间参数与 Native 区间冻结。

| checkpoint → target | 官方完整 query／来源 | 六条件、双序到达 |
|---|---:|---:|
| Vid → Vid | 10,303／732 | 123,636 |
| HC2 → HC2 | 3,482／237 | 41,784 |
| HC2 → Vid | 10,303／732 | 123,636 |
| Vid → HC2 | 3,482／237 | 41,784 |

总计 330,840 在线到达、165,420 个不同 checkpoint/query/condition 输入。
条件为 clean 加五类 5% corruption：drop、freeze、motion blur、occlusion、
exposure。全部官方 query 保留，同一来源的多个 query 均参与。
专家可用率固定 100%；现有 admission 仍可能给出空证据。
名单具有历史曝光，outside-development 分组不称为 fresh test。

八个真实 clean smoke 覆盖四种 checkpoint/target 配对，每种前两个 query。
旧 normalized-prefix 与当前 raw-prefix replay 的全部状态、梯度、loss、
box 逐值相同，包含继承状态和 LN 写回。保存结果的独立 CPU 算术复核
336,996 项通过；另有 38 项 CPU 合同检查通过。这些属于执行验证，
不是完整评估指标或方法收益。

唯一有限 GPU 控制器按表中顺序串行运行。独立 CPU 接续已启动，等待
四个任务全部预测封存后才读取 GT、评分和诊断。对照读出为同输入的
Frozen、继承 Before 和当前纠正 After；以官方 dense scorer 检查全部
读出，报告 source macro／query macro、双序、clean／corrupt、负向尾部，
以及 10,000 次配对 source bootstrap。

本次 GT pipeline 诊断包括正确变错／纠正、空证据与错误准入、正确
proposal 漏选、事件覆盖、loss 下降但定位受损、每步及选中步损害、
观测／未观测帧、LN 继承与当前纠正的分解。GT 不进入推理或配置选择。
完成仍要求完整覆盖、根审计、报告和图、全部匿名结果公开及远端逐文件核验。

最初 subject 记录存成字典导致第一次 smoke 在预测前失败；失败原件
保留，revision001 只恢复相同 subject 的字符串类型。当前八个 smoke
已通过，不将旧故障解释为本轮在线失败。原暂停全 query 队列未恢复。
生产方法登记保持独立，研究工作点没有被静默晋升。

用户授权的磁盘清理同时完成：删除 311 个闲置权重／临时缓存文件，
实际净释放约 30.36 GiB。当前模型依赖、官方数据、封存预测、训练
checkpoint、正负结果和失败记录保留。退休 Sa2VA／PTD 权重的旧实验
精确重放需要先恢复对应权重。

参见 [协议](../protocols/decota_fixed_full_corruption_v1.md)、
[固定配置](../methods/DECOTA_FIXED_FULL_RESEARCH_CONFIG.json)、
[执行与收尾](decota_fixed_full_corruption_v1/EXECUTION.md)、
[清理记录](STORAGE_CLEANUP_20261005.md)。
