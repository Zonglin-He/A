# R35 封存后的 A/DEV 结果与独立CPU审计

状态：训练由 root 完成并封存；本工具随后完成 CPU 独立重放和评分。没有访问 U128 / FINAL384。本轮没有按个别案例更改反馈、mask或参数预算。

## 总体结果

| 配置 | DEV Drone mIoU% | DEV Quad mIoU% | DEV 平均mIoU% | 相对R30平均pp | A208平均mIoU% |
|---|---:|---:|---:|---:|---:|
| Frozen source | 20.653307 | 27.883355 | 24.268331 | −0.822573 | 19.663563 |
| R30 原U189行保持 | 22.225026 | 27.956782 | 25.090904 | 0 | 20.599567 |
| 原U行／新mask分母控制 | 19.364333 | 27.956782 | 23.660557 | −1.430347 | 20.580434 |
| 新partial-output保持 | 18.195411 | 27.956782 | 23.076096 | −2.014808 | 20.485230 |

partial 相对同分母控制为 −0.584461 pp。三个配置均实现全部22条偏好、6/6条要求换目标的B方向及6/6次B最终正组选择，均通过原可行性门。DEV更差不能归因于偏好没有拟合；也不能把A上的个别修复当作总体改善。三个配置Quad任务IoU相同，损害来自Drone。

## 保护目标是否实际起作用

独立CPU64分解表明新增保持确实降低了反馈行中未被P/N判断坐标的漂移：

| 平台 | R30新保护区MSE | 同分母控制 | partial保持 |
|---|---:|---:|---:|
| Drone | 3.291900 | 3.439175 | 2.473765 |
| Quad | 0.331000 | 0.353520 | 0.287816 |

因此“对未判断坐标的保护没有施加上去”不是当前解释。更小的源相对分数漂移并未转化为更好的DEV选择；源排序先验本身可能阻碍部分有益变化，且共享头的影响仍不等于逐候选独立更新。此结果限于既定mask、beta1、16步、同22偏好，不否定所有局部保持策略。

分母控制也很关键：按新mask坐标总数归一，原U坐标权重变为Drone的0.943706、Quad的0.898958倍。该自动强度变化本身已经造成大部分DEV均值下降；增加新坐标后仍有额外退化。不能把全部变化都归因于新增保护坐标。

## 独立审计

`audit35.py` 未导入fit或mask实现，独立重构旧／partial两个mask，NumPy64重放6个封存head的816个完整256维概率向量、66条偏好、272个R30控制向量。概率／exact-logodds gap／最终pref与keep reduction、分母均在预设数值容差内通过；CPU64 argmax与官方GPU预测没有差异。R30控制权重与原封存b1一致。没有运行优化器、autograd或GPU。

评分只使用已封存官方GPU预测，全部A208/DEV64样本、场景汇总、pair执行行和相对各控制的全部负例均已保存。脚本不选方法；跨round主候选由root按预先规则在DEV上锁定，U不得重新选法。

证据入口：ADEV_SUMMARY.json、DEV_FEASIBILITY_TABLE.json、ADEV_ALL_ROWS_PRIVATE.json、ADEV_ALL_NEGATIVES.json、ADEV_SCENE_ROWS.json、ADEV_PAIRED_COMPARISONS.json、A_ALL_PAIR_EXECUTION_ROWS.json、CPU_AUDIT.json、CPU_HEAD_AUDIT_ROWS.json、A_PRESERVATION_DRIFT_DECOMPOSITION.json。
