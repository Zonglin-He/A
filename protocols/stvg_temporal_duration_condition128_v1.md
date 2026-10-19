# R24：保留原长度边际的时间证据融合（预锁）

空间实现/config、E和Live41/42/43数组固定；现有TA-STVG Vid Source同checkpoint，RC128全父query反复曝光开发来源，TC109不选择配置。无新专家、观测、模型、网络训练、学生/空间更新或GPU；本轮是教师能力探针，不叫OPD。每CPU阶段<=900s、own非阻塞run.lock、主机6GiB/磁盘30GiB底线。任何保护失败封存部分、停止，不换分母。

旧结果：R19 raw Action inclusive energy归一化教师完整v−3.435pp，期望GTv−2.830pp；实际Live虽比Fixed+1.144pp仍低于Native。该负结果保留，不再扫温度/先验权重/步数。R18已有同长度方向正确率.7571；新的旧缓存无GT核对显示RawTeacher与Native节点数边际TV均.509951，102/128>.3，期望节点数25.8417→19.6607，平均绝对变动8.8193。这给出不同的因子分解假设：利用同长度位置判断，避免同时覆盖原模型长度判断。不是已经证明长度漂移就是失败原因。

k=j-i+1为65网格的包含节点数（不是精确物理持续时间）。E(I)=sum rawAction[n],n=i..j。唯一候选DurationConditional：T(I)=p0(k)*p0(I|k)exp E(I)/组内Z。结构正支持原样，非正支持保持0，无epsilon。每组严格保留Native边际，rawAction加任意常数c的c*k精确在该组抵消。平坦曲线bitexact返回Native。非平坦曲线可更新位置但不增加候选支持，也可能保留错误长度、双重计入Source信息；并非校准概率或新视觉能力。

控制：Native、旧R19 RawTeacher、GlobalFuse=normalize(p0 expE)、DurationShift32=相同条件化但roll(rawAction,32)、DurationWrongParent（复用R18已封存、受者mean/std匹配的下一个父视频Action曲线，不新增错配选择）、强HiResAuxMix。GlobalFuse隔离一般加入p0与边际保持的差别，不选控制赢家。全部使用同2080候选belief支持、原完整2080 MBR动作（不限制动作边际）；无GT规则、温度/系数/长度菜单。能解释的是belief边际保留，不保证最终单个区间长度不变。

锁前合成检查：flat exact、边际守恒、加常数不变、独立标量公式、与Global不同。全128预测/q/曲线/能量/边际/原空间数组/读出先封存，完成global barrier后才新GT。评分前独立用另一实现重算组内权重及守恒/offset不变、原动作MBR utility；旧Raw/Native/强参照q及区间exact、空间exact；官方Dense对照和所有2080候选质量。固定三空间seed在父内平均，10000父bootstrap；全父含零/无参考/负例，报告完整v/tIoU/边界误差、正负/严重尾/留一，期望候选GTv单独判定。不得按结果切换候选/时空曲线。

唯一候选相对Native完整v>=2pp且CI正、并超GlobalFuse/Shift/WrongParent/强参照才有一次静态独立确认资格；候选期望GTv同样>=2pp且CI正、超GlobalFuse/Shift/WrongParent/强参照才有一次实际Direct/Fixed/Live受控转移资格。硬读出失败不等于所有OPD失败，两种门槛分别裁决。否则关闭这一配置，不按本轮结果扫长度/温度/步数或挑控制晋升。完整2–3pp目标尚未达到，研究goal保持active。

历史区别：旧TrajCal duration prior为bias/temperature菜单、单最佳长度约束P0；F43–45为物理权重/score先验与软目标/参数收缩，未在所核对模块中实施逐组完整p0(k)边际保持。不是论文新颖性证明。

代码/协议/依赖SHA、128R19源缓存/128强参照缓存/128旧wrong缓存均锁定。完成或失败同轮归档，按既有public allowlist公开代码配置全signed结果/负发现，main与research freshCAS及实际HTTPbytes/SHA/blob/tree核验；CURRENT/MAIN/PAPER不晋升。
