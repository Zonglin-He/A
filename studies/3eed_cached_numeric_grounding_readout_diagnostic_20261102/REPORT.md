# R41：source/R30 × native probability/token-log-odds 缓存读出诊断

## 裁决

**固定 U128 中，两种权重各自的 native 概率与稳定 token-log-odds 读出都没有改变任何 query。** source 为 0/128，R30 为 0/128；source 同次投影重放概率与历史存储概率的桥接也为 0/128。根据预登记机制停止条件，结束本数值读出分支，不扩 GPU、不扩数据范围、不新优化、不改当前方法。

这不表示 source 和 R30 预测相同：R30 相对 source 仍改变 Drone 的 38/64 条、Quad 的 11/64 条。改变的是原历史参数适应；本轮替换末端读出没有增加或减少这些变化。

U128 已暴露，只是机制诊断，不是新确认。全部 128×5 臂完整分数、query 和原框在任何新任务分析前封存，随后 CPU 算术重放通过。没有新增 VLM、GPU、全模型前向、优化或源训练，没有读取 FINAL，也没有新读取任务 GT 或计算几何 IoU。

## 主对照与桥接

| 权重 | native 臂 | stable 臂 |
|---|---|---|
| source | R35 同次投影保存的 `source_probability_projection_replay` | 同次 `exact_token_logodds[0]` |
| R30 beta=1 | `probability_scores[R30_U_rows]` | `exact_token_logodds[R30_U_rows]` |

第五臂保留历史 source `probability_scores[0]`，单独检查 source 重放算术桥接，避免把其变化错误算成读出或 TTA 收益。R35 的 `R30_U_rows` 六个 MLP 参数数组与原 R30 beta=1 状态逐位相等；没有重新优化。

两种分数来自相同 GPU float32 前向的既有缓存，保持同一 token mask、有效 token、温度 0.07、特殊 token、sentinel 和归一化。所谓 `exact_token_logodds` 是闭式表达式 \(\operatorname{LSE}(v_P)-\operatorname{LSE}(v_N)\)，仍是 GPU float32；不是从舍入后的概率取 logit，也不是任意精度真值。稳定 log-mass 仅由 \(-\logaddexp(0,-z)\) 计算，并验证最高集合相等、排序单调，不另做可以按 GT 择优的臂。

## 数值结果

| 诊断 | Drone（64） | Quad（64） |
|---|---:|---:|
| source：native→stable query 改变 | 0 | 0 |
| R30：native→stable query 改变 | 0 | 0 |
| source replay-p→历史 stored-p query 改变 | 0 | 0 |
| source native 最高分精确并列观测 | 0 | 0 |
| R30 native 最高分精确并列观测 | 0 | 0 |
| source/R30 native 候选概率等于 1 | 0／0 | 0／0 |
| source/R30 stable 最高分并列观测 | 0／0 | 0／0 |
| 原 native tie 被 stable 打破 | 0 | 0 |
| 非并列数值反转 | 0 | 0 |
| R30 相对 source 的 query 改变 | 38 | 11 |

全部 640 个所选框与原 source 框库按 query 索引精确一致。独立审计已按 alias、query、原框精确关联 R35 封存 IoU 640 个值；同一权重 native/stable 的 mIoU、Acc@25/50 及所有逐样本得失完全相同，2×2 读出交互项为 0。没有读取新任务 GT 容器或重新运行几何评分。

| 方法 | Drone mIoU% | Quad mIoU% | 两平台等权平均% |
|---|---:|---:|---:|
| source 同次投影 native | 11.956924 | 17.964424 | 14.960674 |
| source token log-odds | 11.956924 | 17.964424 | 14.960674 |
| R30 native | 13.342329 | 19.284087 | 16.313208 |
| R30 token log-odds | 13.342329 | 19.284087 | 16.313208 |
| 历史 source native 桥接 | 11.956924 | 17.964424 | 14.960674 |

两种读出下的 R30 相对 source 增量均为 +1.352534pp。这个数重述既有 U128 的参数适应结果，不是本轮新收益。相对 source，R30 在 Drone 14 条改善、5 条恶化，在 Quad 4 条改善、3 条恶化；许多 query 改变仍为零 IoU。5 臂对固定 3 个参考臂的全部 86 条负比较保留，重复相同分数臂的负例不是新增独立证据。Drone 仅 1 个场景、Quad 3 个场景，不能把 128 条当作 128 个独立场景证据，也不计算不可靠的场景区间。

独立审计状态为 `PASS_R41_CACHED_ARITHMETIC_AND_EXACT_OLD_IOU_JOIN`：640 个完整分数向量独立重构、640 个原框一致，原 R30 与 R35 复用的六个头参数数组逐位相等；概率与 sigmoid(z) 的最大数值差为 2.202135e-7。完整任务标量、全部负比较、4 个匿名场景汇总和逐观测数值状态均保留。当前任务表为旧预测值的精确关联，不是新的 640 次模型前向或几何评分。

这组证据削弱“U128 上稳定 log-odds 训练获得的收益主要被 float32 noun 概率饱和或最高分并列掩盖”的解释。结论限于这批固定输入和两个既有状态，不将它推广成所有历史观测、所有模型精度或全部 TTA 优化都不受数值问题影响。R29 曾出现的概率损失梯度/尺度问题仍是独立的训练侧事实，没有被这轮推理读出阴性结果否定。

## 已有覆盖与本轮增量

R29/R30 用 log-odds 修复训练损失，但最终仍按概率读出；R32/R34 的 `ell_eps(p)` 刻意保留原概率并列。旧 CPU64 重放已多次检查小误差和 Top-1 稳定，不能说问题此前未检查。本轮的增量是：从同次既有 GPU 前向的成对概率/log-odds 构成明确 2×2，保留 source 自身对照和重放桥接，并逐条检查最高分集合，而不混入新训练或不同投影精度。

官方源码保留 L2 投影归一化和温度 0.07 的 token softmax mass；官方论文包含 not-mentioned 增强。本轮没有发现应通过再加归一化、改温度或删除 sentinel 才成立的缺失 trick。原 first-lexicon-noun token mask 的语义局限也没有在本项中被改变；不能把数值读出诊断说成已解决主体/关系理解。

## 封存、成本与使用边界

- `ROOT_EXECUTION_AUTHORIZATION.json`：root 在实现和执行前授权固定 cached-only 范围。
- `IMPLEMENTATION_LOCK.json`：所有实际 runner 和草案代码、原封存输入 hash。
- 预测 seal SHA：`3032c45c51d20028226c2ad6d900518a2ab756e41067b1ddc648e06ba88be2da`。
- `CPU_ARITHMETIC_AUDIT.json`：128 条、640 个分数向量和框、所有原概率/log-odds、两组状态与稳定 log-mass 检查。
- `MECHANISM_DECISION.json`：`STOP_NO_WITHIN_WEIGHT_READOUT_QUERY_EFFECT`。
- `../independent_audit/AUDIT_RESULT.json` 和 `TASK_JOIN_RECEIPT.json`：独立算术审计及旧 IoU 关联通过，预测先封存、后任务诊断。
- 只读检索和官方引用见 `../review/REVIEW.md`、`HISTORY_EVIDENCE.json`、`SOURCES.json`。

所有调用为 CPU 缓存提取和算术重放；新增模型推理、教师调用、优化步数为 0。当前结果是数值读出诊断，既不是新 TTA 方法，也不是稳定 z-winner 的新增全模型部署验证。旧 full-model closure 覆盖原 probability/boxes；本轮没有为一个不存在的新赢家扩大 closure，也没有换坐标重做无参数 memory。精度默认和独立 FINAL 保留。
