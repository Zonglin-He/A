# 原生可控性诊断链：完成结果

同官方冻结PTD4B+B1、两个结果选出的源训练query，每例固定30步；source GT用于明确监督正控，未读target。所有原生失败与中间状态保留，不选best。

|控制|时间源 tIoU / vIoU %|空间源 sIoU / vIoU %|
|---|---|---|
|B1原点|29.166667 / 18.672159|59.654986 / 19.884995|
|旧正确mask（0步）|0.595238 / 13.973231|59.654986 / 19.884995|
|旧错误mask（0步）|29.166667 / 18.672159|65.303068 / 21.767689|
|actuation_free002|68.055556 / 50.699382|0.000000 / 0.000000|
|actuation_full001|68.055556 / 50.699382|74.883013 / 24.961004|
|actuation_span001|68.055556 / 50.699382|73.834416 / 24.611472|

受限坐标loss控制的空间0分来自原生格式失败，其event时间未改变。新完整词表控制只改归一化支持，不修生成token。时间命中登记GT token仍受原物理采样量化限制，不等于物理tIoU100%。

## 解释范围

Source actuation controls are not no-update-mask single-factor causal proof. QR absorbs the gate, so equivalent latent changes may be large. No learnability/generalization claim.

新模型梯度步骤150，另24次保存梯度Adam算术重建；所有加载/失败/收尾实测入账，累计41976.213868007秒，cap=null。

旧16源oracle和所有正负保留。before-reader vs after-reader的同norm实验尚未执行；不能提前确认“证据太晚”或增gate收益，也未建立OPD/target效用。

event列空间最终更新对应 ΔZ/Z L2=2697.537088，ΔF相对原F=0.087687，约旧correct mask注入norm的764.013倍。

spatial列空间最终更新对应 ΔZ/Z L2=5309.504268，ΔF相对原F=0.170316，约旧correct mask注入norm的563.981倍。
