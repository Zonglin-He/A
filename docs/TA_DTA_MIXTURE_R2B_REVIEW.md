# R2b: Equal-Weight Multi-Hypothesis Temporal Teacher

**E-Mix 已实际完成，裁决 NO-GO：全部 raw proposals 等权混合没有建立相对 Native 的正向参数适应收益。** 四个专家 corruption 面板的 vIoU 均值均负；Vid 确认配对95%区间在零以下。HC 开发相对失败 E-Deploy 有明确恢复，两个确认相对 E-Deploy 的区间均跨零，不能把“比失败 baseline 少坏一点”写成有效部署 TTA。原 A、生产 CURRENT 及旧结果保持；未启动 weighting、PoE、purification 或 R3。

## 这次只改变 teacher 聚合

同一 R2 raw UniversalVTG 支持，所有有效 proposal 的端点、重复和顺序原样保留，每条权重 1/M。没有 confidence weighting、top-K、NMS/dedup、gate、PoE、新视图、sigma 调整。新增仅 E-Mix；E-Deploy/E-Oracle/R1/Native/A8 均读取上轮已公开并核验的封存结果，不重新拟合、选参数或挑样本。

保持原 TA-STVG 同域 EMA、原两 offset 采样与整数物理端点读出、514 名义 temporal head 末层参数（两个 bias 在归一化 span 分布下不可辨识）、K3、SGD lr=.01、sigma 一原 offset cell、beta1、KL(q||pθ)+KL(p0||pθ)。每query reset 后适应3步、当前query输出后丢弃；空间 A 完整框/参数轨迹、1792空间参数、Vid K1/HC K8、Uniform专家位置及25%到达率全部固定。研究 A 不等于生产 CURRENT。

等权的精确定义：对每个 offset 和每条 proposal，先用 R1/R2 相同 Gaussian 在原严格 i<j 合法网格独立归一化，再算 q_mix=(1/M)Σq_m；FP64 logsumexp，保留联合分布，**没有把混合的 start/end 边缘再相乘**。先混合未截断密度再整体归一化会按合法网格质量重加权分量，不是本轮等权实验。M=1 测试逐参数、逐梯度复现 R1。

Native student 仍是 start/end additive logits 加严格 i<j 归一化，两个 offset 损失平均，解码沿用 FP32 logsoftmax 首argmax后 envelope。Mixed teacher MAP 单独用联合 q 的 FP64 argmax 再 envelope，仅作离线读出诊断，不替换 student。保持完整 joint teacher 不等于原 student 可表达任意多峰关联，结果不唯一归因 uncertainty 保留或原proposal错质量。

每集32开发+16确认、一query/源、两序、clean+五种5%瞬态corruption；共1152到达，288新query适应/864 backward，864非专家保持缓存 A。专家独立来源开发 Vid16/HC14、确认 Vid8/HC7，均历史曝光，不能称 fresh test。复用235个独立expert缓存、每cell 26–312条原有效proposal；没有新GPU/backbone/专家调用。

prepare 从已核验的R2 export manifest绑定旧指标hash而不读取数值；E-Mix独立worker带GT/scored-result读入guard，288预测全部封存后另一个进程才读取GT dense评分。全目标配置/teacher/预测先seal；GT仅用于离线指标与teacher质量解释，不参与mixture、步骤、参数、选择或筛样本。

## 主结果：E-Mix − zero-update Native

专家 corruption，单位pp；同来源配对、等来源宏平均、10000 source-bootstrap，固定seed20261004。Native为相同A空间输入的原head zero-update readout；不是原始Frozen空间模型。

| 面板 | 独立源 / cells | ΔtIoU [95% CI] | ΔvIoU [95% CI] | >5pp v损害 |
|---|---:|---|---|---:|
| vidstg/search | 16 / 80 | -0.6927 [-3.0161, +0.8329] | -0.7580 [-2.3450, +0.1019] | 5 |
| hc2/search | 14 / 80 | -3.3911 [-11.1652, +2.7349] | -2.2733 [-7.0153, +1.3533] | 19 |
| vidstg/confirm | 8 / 40 | -7.7024 [-17.9903, -0.6789] | -4.2126 [-9.3874, -0.5370] | 9 |
| hc2/confirm | 7 / 40 | -6.4214 [-19.8314, +2.1979] | -4.4424 [-13.2648, +0.7480] | 5 |

Vid确认没有任何corruption专家cell相对Native正增益，20/40条降低vIoU，9条超过5pp；负向source见图及集中性记录。HC确认14/40条有正增益，但17条负、5条严重负尾，净来源宏平均仍负且区间跨零；不能把局部好例抹掉，也不能称两个数据集均统计确定地下降。

## 匹配 teacher 对照及 A8 参照

| 面板 | E-Mix − E-Deploy Δv | E-Mix − A8 Δv | E-Mix − E-Oracle Δv |
|---|---|---|---|
| vidstg/search | -0.5376 [-2.2886, +1.0896] | -3.4196 [-9.9663, +1.4111] | -3.6130 [-7.8984, -0.6354] |
| hc2/search | +9.8974 [+0.2159, +21.7492] | -1.2296 [-6.2358, +3.4748] | -7.6397 [-12.2760, -3.6385] |
| vidstg/confirm | +3.0652 [-1.8214, +11.3624] | -2.6915 [-7.0557, +1.0936] | -10.2396 [-22.7860, -1.6773] |
| hc2/confirm | +1.4591 [-11.9214, +12.7517] | -5.7282 [-18.6671, +2.9215] | -7.3948 [-16.3113, -0.8677] |

只有HC开发 E-Mix−E-Deploy 的配对v区间明确高于零；Vid开发均值反而低于Deploy、区间跨零。两个确认面板相对Deploy有正均值但CI跨零；都未相对Native建立正向。因此不能说本轮确认了“hard top1毁掉uncertainty，而mixture解决了它”。这一版等权目标仅在部分面板/坏例缓解top1错误，没有通过部署收益检验。

| 确认专家corrupt | Native vIoU % | E-Deploy % | E-Mix % | E-Oracle % | R1 GT % |
|---|---:|---:|---:|---:|---:|
| vidstg | 23.5691 | 16.2913 | 19.3565 | 29.5961 | 29.3911 |
| hc2 | 32.4715 | 26.5701 | 28.0292 | 35.4239 | 35.0753 |

GT R1和support Oracle是旧监督诊断，复用正控而非新方法；原R2的Vid Oracle确认Native参照CI跨零且存在严重负尾仍保留。高support oracle不意味着all-proposal平均质量高，也不意味着当前factorized head可以安全利用整个joint mixture。

## 原支持的平均质量与上限分开

以下只在全部E-Mix预测封存后用GT计算，不参与训练或决定。raw指标为每cell全部原proposal的连续物理tIoU，再来源宏平均；offset expectation为各联合网格teacher的期望，不能当成最终envelope tIoU。

| 面板 | raw mean tIoU % | raw best % | raw比例 tIoU≥.5 % | mean teacher entropy (nats) | mean offset expected tIoU % |
|---|---:|---:|---:|---:|---:|
| vidstg/search | 20.0974 | 83.7481 | 13.4673 | 5.9157 | 19.3703 |
| hc2/search | 20.3197 | 88.6057 | 14.4621 | 5.7118 | 20.0654 |
| vidstg/confirm | 21.0121 | 82.2738 | 15.3483 | 5.7263 | 20.4838 |
| hc2/confirm | 22.2503 | 91.6568 | 15.5556 | 5.7176 | 21.9797 |

确认集原raw支持平均tIoU约21.01%/22.25%，支持最高约82.27%/91.66%，达到tIoU≥.5的raw条目仅约15.35%/15.56%。这些测量直接支持“支持里有好条目，但等权经验mass整体质量低”；.5阈值只为报告质量分布，未用于teacher或gate。不是证明重复、diffusion或student投影分别导致损害。

Forward KL到原additive endpoint student，其logit梯度由teacher的start/end边缘决定；多峰joint相关性不能被这个固定student任意表达。本轮没有改变模型族或 loss，也没做joint-vs-marginal新消融。因此“坏mass”“分布过宽”和“teacher投影”仍是竞争解释，不能仅凭本轮唯一定位其中一个。

## clean、顺序、全流和来源集中性

| 确认面板 | Δv vs Native | Δv vs A8 |
|---|---|---|
| vidstg/expert_clean | -5.0293 [-10.9862, -0.5368] | -3.9007 [-9.9822, +1.3030] |
| vidstg/flow_corrupt | -1.0532 [-2.4839, -0.0028] | -0.6729 [-1.8920, +0.2679] |
| hc2/expert_clean | -4.4205 [-12.6642, +0.8825] | -5.7571 [-17.0406, +1.1056] |
| hc2/flow_corrupt | -0.9532 [-2.9422, +0.1797] | -1.2804 [-4.1910, +0.5801] |

Vid clean专家相对Native也有明确负向区间，不能把风险仅归因于corruption。全流是固定状态读出模拟，非专家864行保持A逐值相同；Native全流同样仅在专家位置取消Fast。仅25%位置改变，真实全流source-macro配对重算，不能把expert收益直接当全流收益，更不能宣称未来query迁移。

| 确认 / Native参照 | 正源/总源 | 前两负源份额 | leave-one-source-out净值范围 pp |
|---|---|---:|---|
| vidstg | 0/8 | 86.99% | [-4.814454071700639, -2.13235340228922] |
| hc2 | 3/7 | 94.63% | [-5.612079680844154, -0.20599120276390734] |

SUMMARY完整保留两序独立配对CI、每源数值、gross gain/loss、.3基线正确→错误/纠正及>5pp严重损害；没有按source、corruption或结果丢样本，也没有按确认结果重新选参数。独立source仅8/7，集中性与宽区间需一起读。

## 正负例与执行兑现

- Vid确认source36 exposure/order1：原Deploy 6.0260%，E-Mix恢复至48.4319%，但Native仍为59.1701%；mix loss 7.3403→3.2025。它缓解了一个top1灾难，但没有超过无更新输入。
- HC开发source8 occlusion/order2：原Deploy 5.3568%，E-Mix回到Native 72.0670%；这是明确保留的恢复例，不外推为确认通用收益。
- Vid确认source37 exposure/order2：Native43.2520%→Mix24.4508%，loss8.9280→2.8769；目标被执行但GT受损。
- HC确认source41 frame_drop/order2：Deploy46.2299%、Native44.5656%→Mix14.5194%，loss7.2210→3.2286；等权聚合也能破坏原本好的top1。
- HC确认source42 frame_freeze/order1：Native14.5381%→Mix17.9085%，为保留的真实局部正例；全组净效应依然负。

四个corruption专家面板全部loss下降；其中loss下降但Native vIoU变坏的条目分别Vid开发11/确认20，HC开发35/确认17。冻结K3而不是按GT挑步；每步loss、梯度、位移、native读出与dense指标均公开。CASES对每面板、Native/A8/Deploy三个参照保留最大正例和最大负例。

## 审计、成本与裁决

独立NumPy逐分量Gaussian归一化/等权joint混合、完整288 fits的864次joint-marginal梯度和SGD参数算术、head reset、logits、原生MAP、teacher joint MAP、旧raw支持及复用R2指标、1152个A状态绑定和official dense全部核验通过；根检查4,300,889。公开独立bootstrap/trace/paired算术核验48,024标量和360tail计数通过。8项CPU单元检查覆盖M1逐值parity、重复质量、多峰joint非边缘乘积、guard、归一化等。

本轮新288适应/864 backward，fit阶段CPU 6.776s、dense阶段20.604s；不含phase前hash/加载、开发、审计、绘图。独立根审计93.308s另计。新GPU/backbone/suffix/expert调用、空间更新、时间跨query写入均0；private状态轨迹19,834,592bytes不公开。

**最终保留 A，停止将本轮 raw equal-mixture teacher 作为可部署改进。** 本轮没有证明不确定性保留解决单中心瓶颈；也没有否定 temporal gradient channel、所有mixture或经过净化的expert证据。优先问题若继续，应是“如何获得可信监督mass，并区分student投影代价”；不是在这轮失败后擅自调LR/K、加confidence weighting/PoE/gate或进入R3。没有运行任何后续分支。

## 复现材料

- `protocols/tastvg_dta_mixture_r2b_v1.md`；`docs/tastvg_dta_mixture_r2b_v1/EXECUTION.md`。
- `vg_tta/tastvg_dta_mixture_r2b_v1.py`；有限runner `scripts/run_tastvg_dta_mixture_r2b_v1.py`。
- `scripts/test_tastvg_dta_mixture_r2b_v1.py`；独立审计 `scripts/audit_tastvg_dta_mixture_r2b_v1.py root`或公开结果目录；报告与图生成 `scripts/report_tastvg_dta_mixture_r2b_v1.py`。
- `results/tastvg_dta_mixture_r2b/2026-10-04`：全1152匿名指标行、288三步轨迹、GT后验teacher质量、完整source CI/tails/cases及三PNG/PDF。
- 旧R2对照来自commit324b3357dbd67b435d1fb507ae516a44ecf53961；R1正控来自885762286dd75b806fae34b0bb712583d5088d84；未重跑它们。Private GT、原proposal坐标、hidden/weights、媒体与当前生产注册不公开。
