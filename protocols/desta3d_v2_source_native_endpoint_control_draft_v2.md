# 源端 native 时间端点监督正控：候选协议草案 v2

状态：**proposed / untested，未 GPU 登记、未运行，不是已通过的方案。** 本草案来自 source_prefix_support16_v2 的已存证据审计，不能覆盖先前混合 task CE 正控、FP32失败或当前生产状态。

## 问题与区别

原固定三步监督在每支全部任务 token 上取均值：事件分支含语义、时间结构、端点与 null；实际端点为4个位置，占9.09%–12.12%。三个状态48个语义 reference 均与GT相同，但 B1/旧监督时间与框支持仅4/16等于GT，新监督3/16。原GT box条件与native预测时间条件不同；31845原本支持完全一致仍被改坏，因此不能将支持差异直接命名为主因。新 FP32 混合CE下降8/16中有5例native v受损，且两个branch方向在7/16相反。

要区分的假设是：**直接监督当前原生时间读出，能否在同一小校准接口和固定三步内建立时间定位正控？** 这是监督目标定义的单一候选改变，不是再次改精度、增步或改残差。端点token占比仅动机，不能作为其梯度弱或新目标有效的证据。

## 固定范围及唯一目标

固定原PANEL16、B1、官方PTD4B原BF16 body/head/native、66816FiLM/LN可更新白名单、fresh AdamW1e-5/wd0/clip1/固定3步；gates、head、out、textpool、PTD冻结。训练标签仅已有16源的GT时间端点，源 Vid-only，不读取target或新增标签池。每episode复位B1；不按loss/GT选态，不改样本、分数、区间后处理或tie规则。

从B1的**实际生成reference**捕获时间probe的原token/cache/position/context及全部合法T个time token（T实际7–32）。每次反传独立event prefill/replay，使用同一观测源视频与锁定的B1 reference。GT只提供两个端点类别，不替换reference，不把GT时间或框喂入当前预测的空间读出。

候选 loss = 两个端点在完整合法T个时间类别上 cross entropy 的均值，系数1；不含语义/格式/null/box/aux/parameter/output anchor，**不自动重新加权或重新等范数**。这是与原完整152775词表混合CE不同的目标定义，不能直接比较loss刻度。范围仍声明66816白名单，但未连通的空间专属张量可以grad=None并保持不变；记录实际非零梯度/更新维数，不伪称全部66816均有效更新。共享LN可能改变空间输出，必须保留完整原生tube读出与负尾。

## 执行前工程门（本轮未执行）

1. 静态核 `desta3d_v2_output_anchor.py` 的 capture/replay 对event全部Ttime支持、原BF16head调用及独立prefill；不得直接把旧GT teacher-forcing CE helper改名复用。
2. CPU synthetic控制：全T类别CE与独立标量实现一致；标签边界/变长T/均值分母；冻结与恢复；未连通参数不更新；原损失入口禁用开关不变。新helper与测试用隔离文件。
3. 另注册固定首源28199、精确B1、0 optimizer、1次B1基线native重放用于capture的真实工程验收：同输入、B1实际捕获的time logits与已保存B1全2×32 logits逐值相等，replay与capture逐值相等；CPU完整time支持CE与GPU CE在预锁dtype容差内，有限非零梯度仅校准白名单，head/PTD/gate无grad、状态精确恢复。不得用GT最佳状态替换B1，不为吻合数值改支持。
4. 工程验收只能在写出单独REGISTRATION/CONFIG/LOCK、资源/存储评估后串行运行；与本次CPU审计区分。先实测GPU空闲、host和8GiB磁盘余量，沿已验内存策略，失败原件与实际秒入账；不为补15个缺失完整prefix而新增GPU重放。

## 接口通过后才另登记16源对照

只增加一个新臂。旧 B1/原混合 BF16监督/FP32监督预测在物理及B1 native完整重放相等后复用，旧文件不改。新臂固定末态原BF16自由native解码；记录每step完整实际端点分布/GT目标/raw梯度/Adamdelta/counters、两个原混合CE只作定义明确的旁路诊断（如资源允许，先锁定，不事后添选态）、真实注入/更新范围，保存全部16的time/box/tube。原生输出不强制GT时间。全seal/hash/state后独立源评分与第二实现汇总；无target。

主要读出：新−B1及新−旧BF16监督的v/s/t父源宏/描述CI、时间端点和完整框支持、native-good保持、>5pp负尾，成功/失败例全部保留。该源正控可有GT，不能声称无标签TTA成绩。工程容差、allocation、实际存储上限需在执行前单独锁定，本草案不是GPU allocation。

若时间目标改善且原生时间/定位也改善，仅提高“信号与读出目标对应”优先级，再回查无标签可提供的合法证据，不能将GT目标带入TTA。若端点CE改善而原生不改善，降低这条接口正控优先级，保留离散读出/空间耦合与优化充分性；若三步目标本身仍不可靠，如实报告，不再追加LR/精度/steps/lambda网格。任何结果均不自动target/64/生产晋升/push。

## v2 草案计数更正（未执行）

静态读 `capture_teacher` 确認它调用 `decode_shared_reference_two_pass`，会重新执行B1的event+spatial原生生成，并保存实际probe schedule。v1草案把未来验收写成“0新自由native”不准确；v2明确锁**一次B1基线重放（两支）**，无新更新态预测、无optimizer，全部生成/加载计时入账。复用capture_teacher时不能把baseline replay统计为0native；也不将既有保存logits误称完整trace已存在。首源工程门需把这次原生重放的完整geometry/text/logits/物理input与旧B1逐值核验；另有可微event replay/反传，分开计数。当前CPU审计仍真实0GPU/0native，本更正仅拟议阶段的调用计数，不修改本轮审计原件。旧草案与COMPLETE原hash保留，后续采用本v2。


Public review note (2026-09-28): this is the original stage protocol, not an instruction to run it. Superseded and failed versions are retained for provenance. See REVIEW_START_HERE.md for current status. Referenced local data, weights and artifacts are not bundled.
