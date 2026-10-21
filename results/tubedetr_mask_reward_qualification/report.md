# TubeDETR 原生指代mask奖励资格检查

仅同一Vid128/同八帧/单ReferDINO；原框与RGB逐位一致后补存原生mask。无学生适应、无参数搜索、无新全量。完整奖励封存后独立GT诊断。

| 奖励 | 成对排序准确率 | Spearman | Top1 IoU regret（越低越好） |
|---|---:|---:|---:|
| Geometry | 0.749952 | 0.589878 | 0.121997 |
| Layout3 | 0.746771 | 0.578428 | 0.126525 |
| MaskDice | 0.744730 | 0.561890 | 0.147697 |
| MaskContrast | 0.616452 | 0.257461 | 0.231499 |

资格：{"MaskDice": false, "MaskContrast": false}
逐帧文本分数事件支持：{"mean_query_AUC": 0.6512272192827747, "eligible_queries": 81, "AUC_minus_chance_pp": {"queries": 81, "parents": 81, "mean_pp": 15.122721928277482, "CI95_pp": [7.715038947677835, 22.187977660199873], "equal_parent_mean_pp": 15.122721928277485, "wins": 59, "losses": 20, "ties": 2, "below_minus5": 20}, "not_used_as_a_gate": true}
开发资格不证明OPD优势。软mask复制错误教师的风险仍在，替代query不是已知错误身份，事件外缺少GT不能当遮挡。两奖励均有同样的Direct连续梯度。
