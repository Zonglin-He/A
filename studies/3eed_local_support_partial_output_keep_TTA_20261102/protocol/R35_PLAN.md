# R35：只补齐反馈观测中未判断候选的输出保持

状态：方案与代码已准备，纯 CPU mask 合同检查通过；未运行 GPU、未训练、未读取 U128/FINAL384 的样本、媒体或标签。精确合同见 `experiment/PLAN_LOCK.json`。

R30 只对 189 条无接受反馈的 A 观测施加输出保持；19 条有反馈的观测中，未参加任何接受偏好的 query 也可自由改变。本轮检验这一保持范围空洞是否值得补齐。A_drone_034 的独立身份复核是机制线索，不能因为修好该例就采用方法；裁决应以预登记总体 DEV 与新观测结果为依据。

唯一新配置保留 R30 全 query 匹配 MLP、22 条偏好、原分母10/12、exact log-odds、source q0、softplus tau=0.1、AdamW5e-4、weight_decay0、16步、beta=1。只将 keep 从原 U 整行改为所有 A query，去掉每个观测接受的全部 P/N query 的并集。重复端点只释放一次，未知第三候选不获得新负标签。

令 source q0 固定，d_theta(x,i)=z_theta(x,i)-z_theta(x,q0)，掩码 M(x,i) 在该 query 不属于本观测任何接受 P/N 组时为1。损失为原偏好损失加：

`sum_{M=1}(d_theta-d_source)^2 / count(M=1)`。

| 平台 | 原 U 整行坐标 | 新保留坐标 | 与原 U 重叠 | 新增反馈行坐标 | 已接受端点并集释放数 |
|---|---:|---:|---:|---:|---:|
| Drone | 24,576 | 26,042 | 24,576 | 1,466 | 582 |
| Quad | 23,808 | 26,484 | 23,808 | 2,676 | 140 |

每个平台实际 A 矩阵为104×256。原 U 全部保留，标量 mean 按实际保留坐标计数，q0 的零残差若在 mask 中仍计入分母，与旧 R30 的处理一致。此归一化也使原 U 单坐标权重分别变为旧值的约94.37%和89.90%；因此加入一个事先固定的名义强度对照：保留原U掩码，但将平方残差sum除以新mask坐标数。比例直接由计数计算，无GT和调参。该对照与主方案的每个原U坐标权重完全相同，两者唯一损失差别是反馈行未判断坐标的新增保持项。它只隔离名义监督强度，不能声称匹配了梯度范数或Adam实际步长。

必须先复现原 `R30_U_rows`，再运行 `U_rows_matched_denominator` 名义强度对照和固定主方案 `partial_output_keep`。三配置、两个平台各16步，共6个状态96次更新；这不是beta扫描，也不从强度对照中另选更优beta。任一source重放或原R30状态复现失败则停止，作为工程问题保留日志，不继续解释新方法。A仍用原group-max/source代表框读出，DEV用原raw argmax，候选框与backbone都冻结。

先封存全部A208/DEV64预测，再在已暴露DEV64按原至少5/6 B方向、4/6 B最终正组的可行性和平台平均mIoU检查。root会在R34/R35全部方案及DEV选择后锁定主配置，再一次性封存新U预测；U不参与选法，FINAL384不读。DEV和U都报告该名义强度对照；主方案仍固定为partial mask，对照不产生待调beta。旧frozen、同反馈直接使用、不更新参数和QPmemory等强对照继续保留。不能仅超过frozen就声称TTA带来额外价值。

该保持项是 source 决策先验。q0 锚定与释放 P/N 可间接促进正组超过未知 C，但不证明 B>C；不允许将 C 自动标负。错误 source 排序也可能被保守约束保留下来，sourceq0的共享梯度还会把不同坐标耦合。本轮不改变监督真假、输入模态或RGB缺失，不更换adapter、不扫beta/lr/steps。

## 运行与审计

准备者只执行了：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  crossplatform/env/bin/python research/round35/cpu_checks/check_mask35.py
```

它检查重复端点并集、坐标归一、全部原U重叠、固定q0相对分数对共同偏移不变、名义强度对照与新增坐标项的精确可加分解，以及真实22条偏好的mask计数；不读取任务GT或媒体。结果 `cpu_checks/CPU_CHECK_RESULT.json` 为PASS。

以下命令只由主代理在锁定合同、检查GPU资源与截止时间后运行，本准备阶段没有执行：

```bash
crossplatform/env/bin/python tools/night_guard.py --resource gpu -- \
  crossplatform/env/bin/python crossplatform/round35/partial_keep35.py \
  > research/round35/fit.log 2>&1
```

输出为 `crossplatform/round35/partial_keep_attempt001/`，包括每平台每配置的六数组状态、A/DEV完整分数和query、固定mask、坐标计数、初始和末态损失、偏好落实与最终选择、以及封存回执。目录已存在会拒绝覆盖。审计应复核原R30状态、分数、选择、全部负例和DEV表；未通过工程复现不得动用新U确认。
