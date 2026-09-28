# 裁决与后续门：不把“位置太晚”当已证实根因

本轮完成了附件优先链的自由merger→冻结输出列空间→旧mask比较，并进一步完成同幅度early/late位置对照。保留共享THW/双reader主线；未开始OPD。两源自由/列空间改善属于显式GT监督可达性，不是适应效用。列空间对应latent改变量2698/5310倍，不能与微量旧mask混写成同等干预。

位置对照使用原16训练父源，每例Base及event/spatial×early/late×correct/wrong，共144输出，0optimizer；相同原始pixels/B1，空间missing-neutral/时间负控13可区分保持。early是FiLM后local-reader前，query pooling未改。四个candidate都按各源各支old late-correct实际FP32 merger差norm配平，非按成绩选幅度。

|主量（百分点）|late 正确−错误|early 正确−错误|early−late 的正确性差值|
|---|---:|---:|---:|
|时间tIoU，13可区分父源|+4.304384 [0,+12.876168]|−0.594989 [−6.717236,+4.858300]|−4.899373 [−15.450164,+3.238866]|
|空间sIoU，全部16父源|−0.042080 [−0.321658,+0.211893]|+0.087716 [−0.243841,+0.372558]|+0.129796 [−0.215176,+0.454344]|

不能把late正确−错误的正均值当正确mask提高原点：late正确时间相对Base仍−1.785714pp，early正确−1.829268pp；错误负控退化驱动部分差值。新late-wrong经过预定幅度配平，与旧wrong不是同一数值干预：one outcome-exposed source case仅scale.99597194即[2,27]→[2,12]，相对旧wrong t降55.476190pp，v降34.223381pp，native-good v/t各丢1；全部16的旧/新wrong读出已保存，不按损害筛选。另两例时间区间也改变，BF16 endpoint ties/margins均保留；不能仅由此归因全部变化发生于单一cast层。

early的原始差通常更小：非zero时间正确/错误归一化系数及空间11.88–72.56/9.93–73.10倍见ROOT_NORM_SENSITIVITY_READBACK；这是下游reader/LN等整个链的响应，不单独证明LN根因。范数匹配后BF16改变位置/幅度仍不同，完整raw/scaled差与稀疏postcast变化已封存。未保存所有原FP32base端点，实际addition norm/未变端点是worker核验；root对raw/scaled及稀疏差独立复算，不冒称全部端点二核。

全部144格式合法。432scalar/tensor指标误差2.22e−16；7628GT类别margin独立循环误差0；174父源/CI/保持/尾部和36interaction值独立重算最大1.11e−16。GT只源mask/诊断，训练开发曝光，单seed/描述性未校正CI；原target历史曝光保留，本轮未读target。

这轮不支持把主因简化成late placement，也未否定所有early-conditioning：只测FiLM之后标量衰减、当前小norm、固定B1/readout。不能把“大幅自由latent能改善”写成“correct evidence易学”。上下文支持仍是竞争解释；按附件顺序，下一若检验context，应先固定一个明确operator和幅度，单独改变support及相应wrong-control，不同时换reader/gate/训练量，不据本轮最佳分数选臂。当前没有新context协议/注册/结果，不开OPD、target、64或参数网格。

官方外部LLaVA/SigLIP已下载且核hash；用户另授权的pixel baseline仍待官方loader/decode smoke，这不替代latent正确性门。定时保留单Luna30min，真实GPU均退出，根负责异常修复。
