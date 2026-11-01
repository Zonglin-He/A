# R44 phase2 封存后独立审计预案

本文件仅记录审计合同，不读取数组或任务标签，不授权 GPU/优化。实际程序与全 A/V 预测 seal 完成后，按 root 授权先无 GT 审计，PASS 后再评分。禁止 U/T/FINAL。

输入与状态：逐文件核原 R24 source capture、原 source projection、R37 完整组与 phase1 封存反馈。E32 必须逐值绑定且每行唯一，D14/Q18；保持集合精确 A\E，D90/Q86。两个查询 MLP 各184960参数、16步，source初始化，q0冻结，z0 source/no_grad，原加载器无 import hook。

独立数值：用独立 NumPy/SciPy CPU64 MLP、L2、dot/0.07、有效 token softmax/LSE 重放全部 p/z。概率容差2e−5、z容差5e−4；只记录 CPU64 argmax 差异，不用它替换封存 GPU 赢家。用封存 GPU p/z 与 source-z0 重算最终 RankNet/keep/total，容差5e−4；用独立 fresh CPU64 输出核相同项，容差1e−3。参数变化从状态重算，输入 source 数组/原框不允许变动。

A 读出从所有原 R37 完整组独立计算 group-max，再用原 source 最大、下标最小的固定代表；未映射 query 作为独立选项。核 source 端点、原始 argmax、最终代表、偏好是否满足、是否第三组胜出；第三组不自动补负。V 始终按封存 native 概率原 argmax。current G 补充读出逐项核原 F64 和相同增量，单独标明当前额外信息。

memory 不只信作者存储索引。独立重建原 CPU32 source query×noun keys、value矩阵、E-only/allA库；核 ±max(.1-gap_p,0)/2 的概率单位、U全零、count平均与alpha1。重算完整CPU32相似度，验证每个记录候选为其A行最大（并列合同）、每组top4的唯一/排序/未选边界合法性、真实索引范围与库映射，再核 CPU32 exp权重、取值、归一化修正与最终 native-p 单位分数。近并列不通过 float64 另选邻居。零修正返回 source 精确一致。

任务：所有新预测先 seal。通过独立无 GT 审计后，仅对原 A208/V64 读取对应 GT，独立 CPU 评分全部 arms/负比较/场景表。A 的 source、unrestrictedG、eligible direct 必须与 phase1 已封任务值复现；旧 R24/R30 结果仅作历史系统参照。保留所有32反馈的正负/零得失，不按任务值筛监督。没有完整grounder closure的缓存结果不称部署验证；是否进一步闭环须另锁。
