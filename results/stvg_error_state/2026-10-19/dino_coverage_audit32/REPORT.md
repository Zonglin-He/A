# Vid五父视频未观测：DINO调用与四帧预算只读审计

2026-10-19。completed / provenance and control-flow audited。原NV16为主，NH16为预算/拒绝控制；没有新模型、DINO、GPU、训练、候选、学生更新、视频/图像/标注读取或任务重评分。

## 结论

**NV003/NV006/NV009/NV013/NV015均为调用前目标文本gate跳过，实际原始DINO前向0/4。不是DINO检测后拒绝，不是异常中断，也没有发现下游缓存遗漏。**
五条均是what问句；缓存context为eligible=False、reason=unresolved_wh_category，备用visual phrase为空、rule=unknown_queried_entity。源码在每个planned位置的第一条语句即 `if not eligible and not s1["phrase"]: continue`，在读取图像、注册model hook和DINO调用之前跳过。Source和后续query均完成保存；因此“调用前跳过”不能混写成整个query崩溃或运行超时。
Vid16共计划64帧，实际原始DINO观测44帧（68.75%），接纳42、调用后拒绝2、未执行20。四帧名义上限没有用满；原接口本来允许无目标文本时零调用，这不是已经证明可简单删除的错误条件。

## 五条逐项结果

| 匿名父query | what的缓存句法角色 | 计划/原始调用/接纳 | 后续条件DINO图像编码 | 旧GT→E已知计划标签数 | 旧GT→E完整v增量pp |
|---|---|---|---:|---:|---:|
| NV003 | nsubj | 4/0/0 | 0 | 4 | +36.1663 |
| NV006 | obj | 4/0/0 | 0 | 1 | +15.9809 |
| NV009 | nsubj | 4/0/0 | 0 | 3 | +14.7799 |
| NV013 | nsubj | 4/0/0 | 0 | 0 | +0.0000 |
| NV015 | obl | 4/0/0 | 0 | 4 | +42.9643 |

句法角色来自既有解析，不是本轮新增语义GT。三条nsubj、一条obj、一条obl；当前解析器处理who可产生person，但不能因为what就无条件设成person，或拿句中其他名词冒充被询问目标。它们的共同根因是当前teacher接口没有可用的目标类别/短语，不是tokenizer超256触发（context已经False，长度检查短路），也不是观测帧不足（32条均有4个合法计划位置）。
最后两列仅从已公开reference_capacity32标量读取，**没有新GT访问/评分**。NV013没有已知计划GT，0增量表示无该oracle监督、保持E，不能推断其缺覆盖无损或DINO没有价值。其余四条对全Vid16的oracle均值贡献合计6.8682pp。那是完美证据/标注支持的诊断潜力，不是恢复调用后的预测收益。

## 区分四种原因的证据

| 解释 | 本轮核验 | 对五条的结论 |
|---|---|---|
| 没调用/调用前gate | 最早metadata new_DINO=0、observations为空；原runtime绑定源码；准确CPU重放，禁止expert/processor被访问 | **成立：逐slot正常跳过** |
| 异常提前退出、OOM、超时 | 原始NPZ有正常receipt；P1和Source哈希链一致；Source与conditional日志各有完成行；该分支在try/forward前continue | 无支持：不是异常退出；整个query完成 |
| 调用后反馈被拒绝 | 正常拒绝保留probe、receipt与实际forward计数；其他Vid有两个可核拒绝帧 | 五条不属于此类，因前向根本没发生 |
| 缓存序列化/复制遗漏 | 原始NPZ→P1→Source专家payload规范化相同32/32，五条从最早NPZ就为空，Source文件和旧barrier/receipt核hash | 未发现遗漏；缺口源于原始获取流程 |

纯CPU控制重放使用真实缓存parses、frame IDs、native indices，expert/processor是只要被访问就报错的哨兵，placeholder图像在跳过分支根本未读取。5/5精确重现空expert payload、expert touches=0、CUDA未初始化；另用独立AST审计确认skip位于model调用之前。不是用假图像跑了一次新检测。

## 名义预算、历史实际调用和缓存复用计数

| 域（原面板各16） | 计划帧 | 历史原始DINO实际帧 | 接纳帧 | 调用后拒绝帧 | 未执行帧 | 后来条件DINO编码帧 |
|---|---:|---:|---:|---:|---:|---:|
| HC | 64 | 64 | 53 | 11 | 0 | 60 |
| Vid | 64 | 44 | 42 | 2 | 20 | 44 |

Vid另11条均实际调用4帧，其中NV005有1个low_target_score拒绝、NV011有1个ambiguous_distinct_instances拒绝；这两帧预算已消耗。HC有1条实际调用4帧但没有接纳，进一步证明“无可用参考”不等于“没有调用”。
endpoint Source再捕获的new_original_DINO_calls=0，以及P1旧缓存的new_DINO_calls=0，是该阶段复用已有观测的增量成本，不能据此把所有32条都计作未调用。历史实际帧数来自原始观测收据/metadata。后续conditional编码是对相同帧的另一次模型计算，不能算新的物理观测覆盖；其成本也不免费。五条原E support_frames=0，后续 `if active` 整段跳过，所以没有靠center/24-state查询补回覆盖。

## 原因定位

原链条为：缓存依存解析无法生成what答案类别 → context不合格且备用phrase为空 → 仍生成4个planned位置，但每个位置在DINO前continue → 原始expert为空并被正常缓存 → P1/Source忠实复用 → E无参考、保持Source → conditional阶段active=False再次跳过。
重复跑原流程或只恢复文件，不改变目标文本条件，仍会得到同样0/4。把它称为“降低接纳阈值可以修复的覆盖”不成立，因为尚无候选分数。把它称为DINO不擅长这五条也不成立，因为没有真实调用。

## 是否值得作为下一条OPD路线的基础

**值得保留为高优先级的专家接入/覆盖可行性问题，但尚不足以直接确立OPD主线。** 它有真实且可定位的未用预算（20/64位置），旧完美参考干预也显示这批来源有恢复容量。当前缺少的却是“在没有GT目标类别时，怎样把what的目标指代转换为同一冻结DINO可用且可靠的条件”，不是再调局部坐标损失。
后续若授权最小修复，应先锁定不使用GT的目标文本/目标span构造，并处理主语、宾语、介词角色；不能用GT object category、默认person、无条件first noun或删除gate来冒充可靠对象识别。需版本化缓存条件，防止旧空观测负缓存被静默复用。首个证据应是四个既定位置真被执行、参考质量与E完整轨迹实际提高，并保留已支持问句的回归控制。只补调用次数或得到高分不构成成功。
在此基础上才比较同一新增证据、同一学生容量和预算下的直接E/固定监督与Live OPD；若固定方法获得同样收益，应把贡献记为覆盖修复。固定候选池可以支持on-policy，不要求教师每轮提供新知识；仍需实际Live−Fixed增量。
本轮没有实施目标短语修复、补DINO调用、新预测或训练，也没有改pipeline、production或已冻结OPD参数，不自动启动后续实验。

## 审计与归档

metadata/纯CPU路径审计记录约0.400s，独立AST/预算审计0.315s；准备、初始清点和发布未完整计时。新增GPU/Source/DINO/模型训练/学生更新/标注访问/任务评分均0，历史前向与GT容量成本排除。
原runtime-pinned observations.py字节与当前一致，原NPZ文件及runtime/subject/P1/source/conditional receipts逐项核验；source/E/old formal64/current registries保护hash不变。结果公开匿名32条及五条原因、预算、代码证据和以前的GT scalar边界；原问题文本、parse tokens、父ID、媒体、坐标、权重和raw caches不公开。
