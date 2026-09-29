# S1.1：rank preference带来小幅未来迁移，固定归一化没有进一步提高收益

本轮已完成。Raw-RKL引用原S1封存结果；新增Rank-RKL与Rank-RKL+Normalized-Step两臂。每臂同16曝光VidSTG来源/16query、clean及五类5% transient corruption，共96到达、六条独立流、18次有效更新。原TA-STVG Vid-source checkpoint、Sa2VA五帧缓存、1792维接口、四组正反probe与到达顺序保持；每次专家写入围绕当前参数重新产生9个student rollouts。

## 主要结果

主指标是未来nonexpert的pre-update输出，相对Frozen；BudgetedRerank在这些到达没有专家调用，与Frozen相同。五种corruption先源内平均，再12来源macro；每个区间为固定轨迹下的source bootstrap95% CI，单位pp。

| Arm | corruption ΔsIoU | corruption ΔvIoU | clean ΔsIoU | clean ΔvIoU |
|---|---:|---:|---:|---:|
| Raw-RKL | +0.000228 | +0.000089 | +0.000057 | −0.000004 |
| Rank-RKL | **+0.078608** | **+0.047181** | +0.084098 | +0.051306 |
| Rank-RKL+Norm | +0.057182 | +0.031166 | +0.069146 | +0.036195 |

Rank-RKL的corruption Δs CI为[+0.022241,+0.143997]，Δv CI为[+0.007487,+0.095654]。Norm的Δs CI为[+0.018959,+0.100168]，Δv CI为[+0.005766,+0.061403]。这是本曝光小样本上的小幅正迁移，不能写成仍然零收益，也还不足以确立有实用价值的最终方法。

Norm−Rank的Δs为−0.021426pp，CI[−0.047843,+0.001615]；Δv为−0.016015pp，CI[−0.034268,−0.001987]。归一化没有优于原SGD，不继续扫步长、温度或多步。Clean增益相近，当前更支持一般空间refinement，不证明专门恢复corruption。

首个专家arrival0在每流均空，因此每流前三个nonexpert尚无有效历史更新；保留在主12来源分母。另列首个有效更新后9来源：Rank Δs/+Δv为+0.104810/+0.062908pp，Norm为+0.076242/+0.041555pp；这仍不是大幅提升。

## 机制读出

| Arm | teacher q跨度中位数 | 参数步长中位数 | 步长/probe | KL下降 |
|---|---:|---:|---:|---:|
| Raw | 0.002532 | 0.000066831 | 0.004239% | 18/18 |
| Rank | 0.631986 | 0.016562200 | 1.050514% | 18/18 |
| Norm | 0.631986 | 0.015765797 | 1.000000% | 18/18 |

Raw→Rank只改变teacher编码，仍为一步SGD.005，teacher排序权重提高了梯度和实际更新幅度，并产生小幅任务收益。Rank的步长中位数已经与预先规定的1% probe同量级，Norm也会压低原本较大的梯度步，不能把它描述成“在Rank上继续统一放大”。Rank的最大步长为0.091131（probe5.78%），Norm最大约0.015766。各臂不同参数会产生不同后续候选，后续不是完全固定同一支持集的loss因果对照。

## 成功和负例

同来源corruption均值：Rank在Q10 Δs+0.303607pp、Q12 +0.289932pp；Norm分别+0.199977/+0.173257。Q07的Δs为Rank−0.003876、Norm−0.012960pp。Q14的Δv为Rank−0.000434、Norm−0.013584pp。非expert源Δv两臂均7正/1负/4零，三个早期无历史来源包含在零中。全来源效果保存在SOURCE_EFFECTS.json，没有只保留成功样本。

## 执行与验证

新增GPU进程合计96.104968秒，432条current-policy候选、36次反向；0新expert、0新encoder capture。192状态继承/12次流reset、36次独立参数更新和KL重构、64512参数坐标核对、8次学习后完整原生回插、384次双实现metric、3项CPU合同及13086项公开scalar检查通过。最大参数重构误差1.1921e−7、最大KL误差5.5864e−7，在原锁定FP32容差内。两臂全预测封存后才读取原16源GT；nonexpert不访问专家cache。Raw预测不重跑、不重写。

Ranking的并列项取平均名次(tol1e−12)，全部同分得到均匀q；compatibility-RKL对均匀q仍可能通过拉平p产生梯度，不承诺uniform teacher自动no-op。空专家/零梯度才为本轮的no-op。没有GT在线选择、回退或额外门控。

## 本轮决定

保留Rank-RKL作为小幅正迁移候选；不将本结果写成compatibility-RKL已失败，也不因CI为正就宣布spatial主线跑通或晋升生产。固定Norm没有额外收益，停止本轮归一化救援。用户预设的“仍接近零就换parameter-space OPD”分支本轮没有直接触发；parameter-space版本仅留下一候选，不并行开跑。此次只完成S1.1，未启动joint/final evaluation，temporal继续保留既定native候选＋UniversalVTG fast rerank方向，CURRENT_METHOD不改。

所有来源都已多轮曝光，六条短流共享16来源、顺序固定。结果说明本次rank编码能产生小幅未来迁移，不是独立泛化确认，不证明专家权威必须限制在student support的普遍必要性定理。
