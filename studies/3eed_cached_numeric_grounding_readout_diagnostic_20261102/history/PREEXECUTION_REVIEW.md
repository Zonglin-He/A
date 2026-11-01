# R41 只读竞争解释审计：训练 log-odds 与推理 noun probability

## 当前判断

**尚未找到统一的 source/R30 权重 × native float32 noun mass / token-log-odds 的 2×2 任务结果表，但相关数值审计已经做过，不能说此问题从未检查。** 尤其 R35 的 U128 已保存同一 GPU 前向中的完整概率和 token-log-odds，足以先做一个零新 GPU 的缓存读出对照，不应立即重跑 grounder 或重新优化。

旧 CPU 审计在 R35 A/DEV 的 816 个头×观测向量、R32 T128 的 source/R30、R35 U128 的 512 个头×观测向量均未发现其 CPU64 概率重放与已封 GPU 概率 Top-1 不同。这削弱“大量总体收益被概率饱和并列吞掉”的预期，仍未直接排除 GPU32 概率与同次 GPU32 log-odds 排序的差异。审计误差小也不能代替明确检查最高分并列和最终离散读出。

本次只读核查代码、历史封存记录和官方资料，**没有执行新的模型前向、优化、缓存评分或任务 GT 评分，没有读取 FINAL**。

## 历史已覆盖和未覆盖的范围

| 证据 | 已做 | 尚缺 |
|---|---|---|
| R29 `fit29.py`、`finalize29.py` | 比较概率损失与精确表达式 log-odds 损失；最终仍用 float32 概率选 query | 两套训练状态不同，不能当同权重的读出 2×2 |
| R30 `fit30.py` | log-odds 偏好和相对输出保持；最终 `scores()` 仍是概率读出 | 不能由偏好满足率推出最终 Top-1 在稳定读出下的收益 |
| R31 独立审计 | CPU64 重放 source/各头和 log-odds，核查损失与分数误差；stored query 按 stored probability 读出验证 | 未把 CPU64 log-odds 的 query 当新臂评分；它的 400 条重放不能直接写成稳定读出已系统测过 |
| R32/R34 `ell_eps` | 对已保存的 float32 概率作严格单调转换，保留原并列；两端 source/R30 排名不变 | 量化已经丢失的信息无法由 `ell_eps` 恢复，它不是从原 token logits 重新取得 log-odds |
| R32 source-head T128 审计 | source 与 R30 的 CPU64 概率重放各 128 条，argmax 全相同 | 没有同 GPU logits 的稳定读出任务臂 |
| R35 A/DEV 审计 | 6 头、816 个完整概率向量；CPU64 argmax 差异总数为 0，其中 R30 原控制 272 条 | 没有 source/R30×两种读出的共同任务表；不能因此证明所有读出表达式完全等价 |
| R35 U128 `predictU35.py` | 保存 4×256 概率和 `exact_token_logodds`；source 另存同次投影重放概率；source/R30 控制状态已绑定 | 最终 `queries=prob.argmax`，保存 log-odds 未用于新读出评价；此缓存可直接补最小 2×2 |
| R36 | native `logmass + alpha*rank`，架构及源监督权重不同 | 不能充当当前 XYZ source/R30 权重的读出消融 |

详细路径与哈希见 `HISTORY_EVIDENCE.json`。

## 官方源码与论文核对

官方 `bdetr.py` 对 query 和 token 的投影都做 L2 normalization；`grounding_evaluator.py` 对 dot product 使用固定温度 0.07，再对所有实际 token 做 softmax，最后对 positive-map 指定的 token 概率求和。官方论文描述 temperature-scaled cosine contrastive alignment 和 not-mentioned 增强。当前代码已经保留这些机制，没有证据支持再补一个新的归一化或调温度来修该读出疑点。[官方模型](https://github.com/worldbench/3EED/blob/main/models/bdetr.py)、[官方评分](https://github.com/worldbench/3EED/blob/main/src/grounding_evaluator.py)、[论文 B.1/B.3](https://arxiv.org/html/2511.01755v1)。

需要区分三处合同：

1. **当前 target-span 代理是已缓存的 first-lexicon-noun mask**；找不到词才回退 all-content。它不等于自动掌握完整文字的主体与关系。本项保持原 mask，不能把改 mask 混进数值读出实验。
2. 特殊 token 和附加的 ` . not mentioned` 仍在实际 token 分母中。缓存全模型逐条捕捉，后来头的批处理只排除人为添加的 batch padding，不能把原 sentinel 或特殊 token 当 padding 删除。
3. 官方 evaluator 用 descending `argsort`，现有复现实验通过 `argmax`/固定组代表实现自己的历史并列合同；两者在精确并列时不保证相同。本项 native 臂保留既有封存行为，不顺手改成另一种排序或稳定排序。

因此，若有读出变化，首先应解释为数值/接口效应，不直接称新的 TTA 机制。

## 数学与数值边界

令有效 token logits 为 \(v\)，原正 token 集为 \(P\)，补集为 \(N\)：

\[
a=\operatorname{LSE}(v_P),\quad b=\operatorname{LSE}(v_N),\quad
p=\frac{e^a}{e^a+e^b},\quad z=a-b.
\]

在实数精确运算中，\(p\)、\(z\) 和 \(\log p\) 的候选排序相同。有限精度的 `softmax(...).sum(P)` 可能先量化为完全相同的概率，特别接近 1 时；再对这些概率取 log、clamp 或 `ell_eps` 无法恢复丢失差异。

**直接写 float32 `LSE(P)−LSE(all)` 也可能把很小的负 log-mass 消减成 0**。若需要 log-mass，应从 token-log-odds 用：

\[
\log p=-\operatorname{softplus}(-z)=-\logaddexp(0,-z).
\]

不能用 `log(sigmoid(z))` 再引入概率饱和。PyTorch 官方说明融合的 log-softmax 和稳定 logsumexp 适用于数值稳定计算；但某个分组求和/相减是否丢精度仍需具体核对，不能仅因为调用了 logsumexp 就认定整个表达式无消减。[log-softmax 文档](https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.log_softmax.html)、[logsumexp 文档](https://docs.pytorch.org/docs/2.9/generated/torch.logsumexp.html)。

历史字段 `exact_token_logodds` 指闭式表达式，不代表任意精度真值；它本身仍是 GPU float32。CPU64 从头重投影还会改变 MLP、归一化、dot product 的舍入，不能把它与 GPU32 概率的差异都归因于最后 softmax。最小主对照优先使用**同一次 GPU 前向已保存的 probability/log-odds 对**。

## 建议的最小 R41：先零 GPU U128 缓存 2×2

该项仅为待批准提案，`PROPOSED_PLAN.json` 不是执行锁。U128 现在已暴露，只能诊断，不能改称新确认或挑选最终方法。

使用 R35 已封的 `predictionsU128_attempt001`，原 source 和 `R30_U_rows` 两个权重（后者与 R30 beta=1 状态精确复现）。四个主臂：

| 权重 | native 概率 | token-log-odds |
|---|---|---|
| source | 同次重放 `source_probability_projection_replay` | `exact_token_logodds[0]` |
| R30 | `probability_scores[R30_U_rows]` | `exact_token_logodds[R30_U_rows]` |

source 已封历史概率 `probability_scores[0]` 额外保留为**算术桥接控制**。若同次重放 source 与历史 source 出现 query 变化，分别报告，不把桥接变化算成 TTA 或稳定读出收益。不新增训练，22 条反馈、U189、16 步历史预算和所有已训练状态保持原样。

共同记录完整 256 分数、最高概率精确并列数、所有 top-set、stable winner 是否属于 native top-set、非并列 numerical reversal、概率到 1 的饱和数量，以及 raw argmax 和历史选择。log-mass 用上式作单调一致性诊断，不能再把它当另一个可按分数择优的臂。

所有新读出 query 和原框先封存，随后独立 CPU 审计，再读暴露任务 GT 统一报告两个平台的 mIoU、Acc@25/50、所有正负尾和场景表。关键效应为：

\[
\Delta_{\rm native}=M(R30,native)-M(source,native),
\]
\[
\Delta_{\rm stable}=M(R30,stable)-M(source,stable),
\]
\[
\text{interaction}=\Delta_{\rm stable}-\Delta_{\rm native}.
\]

只比较 R30-stable 与旧 source-native 会混合两个变化，不能回答 TTA 增量。精确并列与近并列只作解释，不按 GT 选择 query、候选集合或温度。

若两种权重各自的 native/stable query 在全部 128 条均相同，此分支直接结束；不要求 source 与 R30 彼此相同，因为历史 TTA 本来可能改变预测。Root 已接受仅此 U128 缓存方案，不登记或执行 A/DEV/R/S/T 扩展，不因得到零变化再重跑 GPU、换精度或换头。source replay-p 与历史 stored-p 桥接差异如有发生，独立归因，不算入读出或 TTA 效应。

## 交付状态

`readout41_draft.py` 为不自动读取文件、不含训练/评分入口的草案，只定义安全的缓存对照提取、稳定 log-mass 和 tie 分类。尚未执行该草案，也未生成新的预测/任务结果。现有审计已经使“大规模数值读出收益”缺乏强先验证据；本方案的价值是用低成本明确排除或量化这个竞争解释。
