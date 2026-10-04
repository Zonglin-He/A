# R2: Expert-Teacher Distributional Temporal Adaptation

**本轮已完成两臂实际适应与独立核验：原始专家最高置信度单区间 teacher 未通过，不能进入 R3 持久学习。** 同一专家支持中用GT选中心的E-Oracle可产生接近R1的梯度收益，但Vid确认相对Native的配对区间跨零且有5个严重损害，不把它写成两个确认面板均稳定通过。裁决仅针对预锁单中心转换；没有否定专家支持或native temporal adaptation路线。

## 唯一变量：teacher interval 来源

原TA-STVG同域EMA checkpoint、原32开发+16确认来源/每源一query/双序/clean与五种5%瞬态corruption/25%专家位置、原始输入像素和两offset采样均保持。空间A（1792参数、Vid K1/HC K8 Uniform Rank-RKL）状态与完整框轨迹只读复用；生产CURRENT仍为DeCoTA，研究A不等于生产方法。

本轮沿用R1独立源验证选出的lr=.01，两个数据集相同；只原native `temp_embed.layers.1`的2×256 weight+2 bias，514名义参数、bias在joint归一化后不可辨识。缓存final hidden先经过冻结MLP首层/ReLU。K3普通SGD，sigma一原offset中位相邻网格cell，beta1，forward KL(q||pθ)+KL(p0||pθ)。每query回到原head，最终第3步当前query读出后丢弃；不是未来query更新或online persistence。完全复用R1的fit_query，未调LR/K/subset/先验/解码。

| Arm | Gaussian中心 | GT参与teacher |
|---|---|---|
| R1（复用正控） | 原GT时间 | 是；不重跑 |
| E-Oracle | 现有UniversalVTG raw proposal列表中连续物理tIoU最高的区间 | 是，明确监督支持诊断 |
| E-Deploy | 同一列表中proposal_confidence最高的区间 | 否，部署可获得的raw top1 |

保留原列表全部有效proposal、重复、次序和小数端点；exact tie取首索引。没有NMS/unifier/新增候选/mixture/PoE/score weighting/gate/memory。E-Deploy是缓存raw专家top1，**不是**A8在Old8学生区间上max(confidence×tIoU)的最终读出，也不是新实现另一个expert postprocessor。两臂Gaussian形状和所有优化机制相同。

原288个专家位置，每臂288次，合计576 query-arm适应/1728 backward；专家支持含26–312个原proposal，235个独立缓存文件。其他864到达保持缓存A；总1152是匹配读出模拟，未重跑backbone、suffix、空间专家或完整在线流。确认实际专家独立源Vid8/HC7，开发16/14；均有历史曝光，不能称fresh test。

Deploy在独立worker装入GT-reading guard，所有288预测封存后，另一个Oracle worker才读取目标GT选择中心；其288预测全部封存后才单独dense评分。部署worker禁止GT/Oracle/scored results读入；6项测试验证guard和selector并列等规则。Oracle选择按教师区间tIoU，不按更新后vIoU挑中心/步数。

## 主对照：E-Deploy / E-Oracle − zero-update Native

专家corruption；单位pp，等来源宏平均、10000次配对source-bootstrap。Native是在同A空间输入上的原head输出；优先回答teacher→gradient是否产生定位收益。

| 面板 | Teacher | ΔtIoU [95% CI] | ΔvIoU [95% CI] | >5pp v损害 |
|---|---|---|---|---:|
| vidstg/search (16源/80cells) | EDeploy | +1.5696 [-3.5170, +5.7241] | -0.2203 [-3.5213, +2.2677] | 4 |
| vidstg/search (16源/80cells) | EOracle | +7.0035 [+2.9772, +11.3288] | +2.8550 [+0.6306, +5.6609] | 0 |
| hc2/search (14源/80cells) | EDeploy | -16.6758 [-33.9731, -1.5777] | -12.1707 [-24.0992, -2.3316] | 38 |
| hc2/search (14源/80cells) | EOracle | +11.1035 [+4.0657, +19.2342] | +5.3664 [+2.2096, +8.9319] | 0 |
| vidstg/confirm (8源/40cells) | EDeploy | -12.1333 [-28.2006, +0.0176] | -7.2779 [-17.6543, +0.0467] | 10 |
| vidstg/confirm (8源/40cells) | EOracle | +7.7652 [-2.4123, +22.9332] | +6.0269 [-1.2409, +18.0795] | 5 |
| hc2/confirm (7源/40cells) | EDeploy | -12.4495 [-32.9411, +1.0194] | -5.9014 [-14.4214, +0.2985] | 11 |
| hc2/confirm (7源/40cells) | EOracle | +5.9210 [+1.1000, +11.3320] | +2.9524 [+0.4399, +6.4471] | 0 |

E-Deploy四个面板vIoU均值均负；HC开发t/v两个区间明确为负。确认两集相对Native的v区间跨零，不能把负均值包装成统计确定的普遍下降；但严重负尾与未建立正增量已足以停止此部署teacher版本。

E-Oracle在三个面板相对Native的t/v区间为正；Vid确认均值正但两区间均跨零，5/40条相对Native下降超过5pp，集中在source37。全部面板E-Oracle−R1区间跨零：不能把两者均值接近称为等效性检验通过。其结果仍说明该专家support可以给已验证的参数通道提供有用中心，不能直接认定support完全不足。

## 第二层参照：当前A8与R1

| 面板 | Teacher | ΔvIoU vs A8 | ΔvIoU vs R1 | >5pp损害 vs A8 |
|---|---|---|---|---:|
| vidstg/search | EDeploy | -2.8819 [-10.3300, +1.9505] | -2.7224 [-7.8672, +0.1435] | 6 |
| vidstg/search | EOracle | +0.1934 [-2.4331, +2.6415] | +0.3530 [-0.0182, +0.9955] | 6 |
| hc2/search | EDeploy | -11.1270 [-24.2492, -0.9496] | -17.6561 [-28.6012, -8.4680] | 35 |
| hc2/search | EOracle | +6.4101 [+2.0793, +10.7960] | -0.1190 [-0.4540, +0.2468] | 6 |
| vidstg/confirm | EDeploy | -5.7567 [-16.4360, +1.5636] | -13.0998 [-27.3190, -2.1153] | 10 |
| vidstg/confirm | EOracle | +7.5480 [+0.5309, +19.1329] | +0.2050 [-2.0141, +2.4911] | 0 |
| hc2/confirm | EDeploy | -7.1873 [-15.7686, -1.2442] | -8.5051 [-18.3832, -0.5420] | 18 |
| hc2/confirm | EOracle | +1.6665 [-4.1025, +9.1268] | +0.3487 [-0.3140, +1.2314] | 7 |

E-Deploy相对A8的HC开发/确认v区间均低于零；Vid两面板跨零且均值负。E-Oracle确认Vid相对A8正，HC仍不确定且有7个严重损害；它使用目标GT，本来也不能直接部署替换A8。

## Teacher选择、网格投影与执行分开看

| 面板 | Teacher | 连续teacher tIoU % | 直接teacher vIoU % | Gaussian MAP vIoU % | 3步后 vIoU % |
|---|---|---:|---:|---:|---:|
| vidstg/search | EDeploy | 37.8590 | 12.8309 | 12.8398 | 14.9504 |
| vidstg/search | EOracle | 83.7481 | 25.4987 | 25.8367 | 18.0257 |
| hc2/search | EDeploy | 36.6984 | 17.9606 | 18.0324 | 19.0703 |
| hc2/search | EOracle | 88.6057 | 47.7767 | 47.5551 | 36.6074 |
| vidstg/confirm | EDeploy | 30.2635 | 13.2858 | 13.0050 | 16.2913 |
| vidstg/confirm | EOracle | 82.2738 | 37.9945 | 36.6555 | 29.5961 |
| hc2/confirm | EDeploy | 47.0798 | 25.3432 | 25.6633 | 26.5701 |
| hc2/confirm | EOracle | 91.6568 | 46.3565 | 45.9115 | 35.4239 |

连续teacher tIoU用于proposal选中质量诊断；直接teacher/全部预测的official dense scorer按现有代码对端点整数截断，Gaussian中心仍保留小数。Gaussian MAP沿原两offset严格i<j支持和envelope，无GT直接送进decoder。直接teacher与MAP仅为诊断，不能冒充梯度TTA。

E-Oracle直接区间和Gaussian MAP均明显高于E-Deploy；这条差距在优化之前已存在。E-Deploy跟随的高置信度中心可能远离正确事件，三步KL确实下降，却沿错误目标损害输出。此次没有改变representation、梯度实现或学习率来补偿。

Oracle选中也不是GT中心：Vid确认source37 exposure/order2，Native 43.2520%→E-Oracle 37.5886%，直接teacher 38.7454%、teacher MAP 34.4994%，R1保持Native。存在较好的support不等于选中的近似边界在固定空间下必然好，更不等于每一步参数优化保证vIoU。

## 集中性、顺序、clean与完整缓存流

| 确认集 / teacher | 正源/总源 | 最大正源份额 | 前两负源份额 | leave-one-source-out净增量范围 pp |
|---|---|---:|---:|---|
| vidstg/EDeploy | 2/8 | 76.97% | 99.70% | [-8.347623871291422, -2.403180860614984] |
| vidstg/EOracle | 5/8 | 84.50% | 100.00% | [0.384658684736345, 7.695976773641279] |
| hc2/EDeploy | 2/7 | 62.29% | 90.17% | [-7.251211361049414, -1.9764674508952225] |
| hc2/EOracle | 5/7 | 59.73% | — | [1.3870270285537387, 3.444470613849921] |

保留所有来源和leave-out变化，不删除source33/36/37/43追分。Vid Oracle正收益仍高度集中，Native参照负尾也集中；HC确认专家源仅7个。图中公开每源两臂净变化，SUMMARY保留每个顺序独立CI和gross gain/loss，不能用cells数替代独立source数。

| 确认全corrupt缓存流 | Teacher | ΔvIoU vs A8 | ΔtIoU vs A8 |
|---|---|---|---|
| vidstg (16源/160cells) | EDeploy | -1.4392 [-4.4176, +0.3731] | -2.3220 [-6.7073, +0.7301] |
| vidstg (16源/160cells) | EOracle | +1.8870 [+0.0669, +4.9635] | +2.6526 [+0.1923, +6.5531] |
| hc2 (16源/160cells) | EDeploy | -1.7092 [-3.8778, -0.2037] | -3.2236 [-8.1425, -0.1728] |
| hc2 (16源/160cells) | EOracle | +0.3186 [-0.9521, +1.9603] | +0.9215 [-1.1203, +3.5052] |

这些为全流来源配对重算；仅专家位置变，其余864行逐值A，不能称时间监督向未来非专家位置迁移。Full-flow Native也仅在专家位置取消Fast，其他位置仍A。

| 确认clean专家 | Teacher | ΔvIoU vs Native | ΔvIoU vs A8 |
|---|---|---|---|
| vidstg | EDeploy | -8.7579 [-22.0148, +0.0437] | -7.6293 [-22.2496, +1.8042] |
| vidstg | EOracle | +6.0551 [-1.2745, +18.1372] | +7.1837 [+0.3580, +18.7768] |
| hc2 | EDeploy | -7.8036 [-17.2813, +0.2374] | -9.1402 [-18.7562, -1.4794] |
| hc2 | EOracle | +2.3548 [+0.5947, +4.1305] | +1.0183 [-2.9047, +4.3730] |

## 具体正负例与loss

- Vid确认source36 exposure/order1：Native 59.1701%→E-Deploy 6.0260%，直接teacher和Gaussian MAP为0；loss 13.1850→4.9672。错误center被梯度执行，不是loss不下降。
- HC开发source8 occlusion/order2：Native 72.0670%→E-Deploy 5.3568%，loss 16.6071→4.6945；同support Oracle保留有用中心。
- HC确认source43 exposure/order2：Native 31.3400%→E-Deploy 0%，loss 19.5209→4.4598，baseline-good被破坏。
- Vid确认source33及HC source47的Oracle正例保留在CASES；每个面板/每臂相对Native和A8均公开最大收益及最大损害，不能只挑恢复案例。

所有corruption专家fit的loss下降；loss下降但Native vIoU下降，Deploy开发/确认Vid12/11、HC45/13；Oracle Vid5/5、HC3/1。非负loss不是任务correctness。EXECUTION_DIAGNOSTICS给出所有MAP是否改变及每步轨迹，而非用最终KL替代定位指标。

## 核验、成本与当前决定

根审计独立NumPy重算576个query-arm fit全部1728次joint-marginal梯度、SGD、head reset、logits和原生MAP；两臂teacher首argmax与全部teacher Gaussian/MAP、R1输入/输出保留、1152个A绑定及official dense指标全部通过。根检查8,404,066；公开标量76,416、tail计数480，六CPU测试通过。

阶段内CPU时间Deploy 1.922s、Oracle 1.945s、dense 29.305s；不含phase前hash/加载、开发、审计、报告。根独立核验97.953s另计。新增GPU/backbone/suffix/expert均0；空间更新和时间跨query写入0，private运行轨迹39,636,992bytes不公开。

**结论：本轮E-Deploy NO-GO，暂不进入R3；保留A与native-head梯度路线。** 结果更接近“support提供有用中心，部署top-confidence选择不可靠”，但Vid Oracle确认未稳定通过Native参照，不能硬套成两个数据集完美的E-Oracle✅/E-Deploy❌。

本轮没有验证“hard selection损失有效uncertainty”的因果说法：Oracle−Deploy同时改变了中心correctness；只有另一次matched aggregation实验才能验证多hypothesis保留是否有用。Equal mixture/weighting/PoE/persistence均未运行，不把它们作为已获收益。下一条件问题可以是同机制下无GT证据聚合；需先固定teacher与合法span投影，不能凭本轮就扩大K/LR/参数、换专家或重新扫描gate。

## 可复现文件

- `protocols/tastvg_dta_expert_r2_v1.md`；`docs/tastvg_dta_expert_r2_v1/EXECUTION.md`。
- 有限runner：`scripts/run_tastvg_dta_expert_r2_v1.py`；共用R1 fit：`vg_tta/tastvg_dta_oracle_v1.py`。
- 测试：`scripts/test_tastvg_dta_expert_r2_v1.py`；根/公开审计：`scripts/audit_tastvg_dta_expert_r2_v1.py root`或结果目录。
- 匿名1152行指标、576条三步轨迹、全部teacher诊断/CI/负尾/cases/PNG-PDF：`results/tastvg_dta_expert_r2/2026-10-04`。
- Private support/GT坐标/hidden/head weights/媒体均不公开；公开标量审核无法重建这些被排除资产。
