# 3EED 第三轮：取消长训练后的朝向与几何诊断

2026-10-30，Asia/Shanghai。仅推进3EED，AVIS/GREAT没有新实验。用户询问训练开销并明确太久就不训练后，正式Waymo-only100epoch训练未启动，GPU短计时也已停止，0个完整训练epoch，没有生成source-only checkpoint，没有自动训练控制器。

**结论。** P0完成资源与协议审计，source-only跨平台实验仍未完成。P1–P3在旧round2 dev18+confirm36的54个缓存父样本上实际执行，属于已曝光历史样本的描述性诊断；不是新确认集，不是官方全量复现，不是source-only迁移证据。未训练新选择器、未采用新TTA。

## P0：权重可用性和训练开销

官方HF baseline ZIP（535,349,344B）中央目录36,719成员，只有一个final_6384 checkpoint，训练配置包含Waymo/Drone/Quad。分别命名的验证脚本仍指向联合模型。当前官方repo、release/tag/branch、project和HF页面未找到Waymo-only权重；[官方issue回复](https://github.com/worldbench/3EED/issues/6#issuecomment-4273059426)说明其他checkpoint因服务器重置丢失。因此不能用联合模型微调伪造source-only。

实际Waymo loader为3,682条源训练表达、3,782条源验证表达。batch24、drop_last对应153步/epoch，100epoch共15,300步。按官方历史混合训练日志比例估计约2.35小时纯训练，非5090实测，另有验证、加载、保存开销。短计时首次因缺distributed初始化失败；使用真实单rank NCCL重试，在没有有效完整步时按用户指令终止。原日志保留，进程/4个loader子进程已回收；无有效实测throughput。

官方协议还依赖GT目标类别辅助caption span，而旧round2是固定纯文本first-noun读出；这两条件不能混称。官方模型仅6D中心尺寸，无yaw。当前科学实验入口不变：联合权重+旧纯文本读出。P0未完成的直接后果是本轮不能回答Vehicle-trained→Drone TTA是否有效。

## 数据、预测和统计边界

54个父样本来自23个场景簇：Waymo18、Drone2、Quad3。两阶段开发/确认合并仅为诊断，旧36帧不当36个独立证据。每平台18帧，但每平台场景数极不相等。主描述性估计量为各平台内scene-macro均值再等平台平均；bootstrap先在平台内抽整场景、再固定等平台权重，10,000次seed20261030。全表同时报告原始帧均值；两者是不同估计量。Drone2/Quad3场景数不足以支持广泛跨场景泛化，区间不能消除历史曝光。

所有新几何方法共享同source query、center/size和AABB crop，1e-7m容差，crop判定沿原float32，之后double几何。P2的double裁剪判定和P3的原dtype判定在54例逐mask相同。orientation-only组min8唯一点且共线回退；P2扰动PCA probe min3另行命名，不混为相同部署策略。P2/P3输出及代码先封存后才打开数值GT评分。GT只在独立CPU离线诊断使用，不进入任何几何目标、接受规则或优化步。

## P1／P3：固定中心尺寸的朝向比较

| 方法 | 帧均值IoU % | Acc>.25 | Acc>.50 |
|---|---:|---:|---:|
| Source | 39.6721 | 41/54 | 19/54 |
| 旧PCA（原长宽比资格） | 49.9296 | 42/54 | 33/54 |
| 标准PCA（轴对应） | 50.3178 | 42/54 | 34/54 |
| 精确最小面积矩形 | 49.6779 | 43/54 | 30/54 |
| L-shape area | 49.6984 | 43/54 | 30/54 |
| L-shape closeness | 48.7164 | 42/54 | 32/54 |
| L-shape variance | 49.1365 | 42/54 | 32/54 |
| 同信息朝向几何优化 | 50.8727 | 42/54 | 36/54 |
| 旧joint adapter3，仅历史参考 | 39.3591 | 41/54 | 19/54 |
| GT yaw注入，仅离线 | 57.8981 | 42/54 | 40/54 |
| 有限朝向搜索，仅离线 | 58.5018 | 44/54 | 40/54 |

GT yaw注入比标准PCA高7.5804pp，有限搜索比PCA高8.1840pp，朝向仍有明显可达空间。有限搜索包括720个0.25°网格、source/GT/各固定方法yaw及前三局部最大附近精化；只称已找到的可行最大值，是真正连续最大值的下界，不是严格上界。另外保留体积与z重叠约束的合法但松的IoU上界。

GT yaw注入41改善、3退步、10不变；固定错误中心/尺寸时，GT yaw不一定使IoU最大。最坏注入退步约5.3863pp，没有删除这些反例。有限搜索46改善、0退步、8不变；8例均仍为零IoU，其中1例的z重叠上界为0，能严格证明仅调yaw无效，其余不把有限搜索失败夸称连续空间不可救。GT yaw的Oracle不选择不同实例，不改center/size，不作TTA成绩。

L-shape主三臂使用原论文Alg3/4/5，0..89°；其整向量选边、variance ties与PythonRobotics参考实现不同，经过独立阅读原PDF确认。d0=.01m来自公开MIT参考默认值而非GT调参；variance空组/单例为0且ddof0，说明未规定处的实现约定。最小面积矩形枚举所有凸包边方向，对GEOS合成检查最大面积误差1.07e-13。全部模块是传统几何后处理，不称神经TTA。

朝向几何优化复用原surface energy，在同crop上仅优化yaw，固定三起点、maxiter20，不使用原teacher资格门控、不换query、不改center/size。它是相同信息的直接优化对照。旧joint adapter允许原历史中心/候选变化，只在独立容量组显示。随后补充了独立容量组的完整框拟合：使用同一已冻结API、同source query/crop，重估XYZ中心、XYZ尺寸及yaw。补充锁发生在原54例诊断评分后，明确为posthoc描述扩展；五种标准算法不做任何GT调参，全部新输出再次先封存后评分，不计为预确认结果。

| 朝向几何优化减去对照 | 帧均值差 pp | 等平台scene-macro差 pp | 平台内scene bootstrap描述区间 pp |
|---|---:|---:|---|
| 标准PCA（轴对应） | +0.5549 | +0.4443 | [-0.8472, +1.7537] |
| 精确最小面积矩形 | +1.1948 | +0.9704 | [-1.0059, +3.1450] |
| L-shape area | +1.1743 | +0.9518 | [-1.0145, +3.1180] |
| L-shape closeness | +2.1563 | +2.1463 | [+0.1003, +4.4962] |
| L-shape variance | +1.7362 | +1.5926 | [-0.8129, +4.2426] |

优化对标准PCA/最小矩形的区间跨零，不足以认定稳定优于普通几何。对closeness一项的描述性区间为正，也不能在多项比较和历史样本上宣称全面优越。Waymo上优化较好、Drone上反而低于PCA，所有平台与正负尾保留。旧神经joint adapter仍39.3591%，不因新几何优化的正均值而获得TTA有效性。

## P3补充：完整框拟合容量组

| 可见点包络完整拟合 | 帧均值IoU % | Acc>.25 | Acc>.50 |
|---|---:|---:|---:|
| fullrefit_PCA_standard_axisaware | 27.7260 | 30/54 | 10/54 |
| fullrefit_minimum_area_exact | 26.8563 | 30/54 | 8/54 |
| fullrefit_Lshape_area | 26.8998 | 30/54 | 8/54 |
| fullrefit_Lshape_closeness | 28.5621 | 31/54 | 12/54 |
| fullrefit_Lshape_variance | 28.4344 | 30/54 | 11/54 |

这些方法每种49/54例改变中心或尺寸，5例按原API退化回退；原9臂与模块不改。原source39.6721%，而这组26.8563%–28.5621%。可见点的紧包络会损失原模型给出的完整物体尺寸，当前结果符合部分可见条件下的收缩偏差；它不是对带形状先验、鲁棒分割或未知表面补全的所有经典方法的否定，也不能拿不同能力组的差距证明神经TTA有效。所有负例与正尾保留。这组另由独立3D凸体评分复核270个IoU，最大误差3.05e-15，见INDEPENDENT_FULL_REFIT_AUDIT.json。各方法有47–49/54框的体积比source小，中位体积比.524–.607，支持可见点包络收缩的描述。

## P2：可靠性关联和受控扰动

预先列出全部16个数值描述符，不按GT翻转AUROC方向、不训练selector、不选最优阈值。PCA标准法的修正为30正、12负、12不变；AUROC仅比较42个改变样本的正负，不变另保留。多数结构特征有51例可用，3例因crop<3无描述符，明确缺失。下表为历史pooled关联，不是heldout预测能力。

| 无标签特征（数值高为正方向） | AUROC 正修正vs负修正 | Spearman vs连续IoU变化 | 可用父样本 |
|---|---:|---:|---:|
| PCA_eigengap | 0.6889 | 0.4236 | 51 |
| XY_hull_area_fraction | 0.3278 | -0.3147 | 51 |
| background_injection_stability | 0.4889 | 0.1593 | 51 |
| bootstrap_stability | 0.6556 | 0.3734 | 51 |
| components_count | 0.2750 | -0.2987 | 51 |
| crop_expansion_stability | 0.5944 | 0.1477 | 51 |
| crop_shift_stability | 0.7194 | 0.3887 | 51 |
| horizontal_normal_support | 0.5931 | 0.0915 | 51 |
| log_point_density | 0.3722 | -0.1561 | 54 |
| nearest_neighbor_components_fraction | 0.6583 | 0.2863 | 51 |
| partial_visibility_stability | 0.7194 | 0.5103 | 51 |
| source_crop_boundary_fraction | 0.3444 | -0.1202 | 51 |
| unique_crop_count | 0.3778 | -0.0675 | 54 |
| unique_input_points | 0.4667 | -0.0378 | 54 |
| vertical_normal_planarity | 0.6250 | 0.1811 | 51 |
| z_span_fraction | 0.3861 | -0.1919 | 51 |

裁剪偏移稳定性和部分可见稳定性pooled AUROC约.719，PCA eigengap约.689，子采样稳定性约.656；点数/密度并未给出可靠的正方向区分。对表中标准PCA这组，只有4个场景同时有正负改变、能定义场景内AUROC，其余场景单类或单帧保留NA；同一特征的场景内方向并不一致，例如部分可见稳定性有场景AUROC.333，也有1.0。不能按pooled结果直接造选择器或把稳定等同正确。

受控观察变化使用封存后的PCA probe yaw，在固定source中心尺寸上独立评分。移除半侧点、上半部点不保证是物理真实遮挡；shell注入来自同一扫描外圈点，不保证全是其他实例；几何GT框occupancy/retention仅离线proxy，不能称真实实例纯度。

| 观察扰动 | 有trace父样本 | 变体数（非独立样本） | 相对未扰动probe平均IoU差 pp |
|---|---:|---:|---:|
| background_yaws | 51 | 102 | -0.0857 |
| bootstrap_yaws | 51 | 816 | -0.1997 |
| expanded_yaws | 51 | 51 | -0.6828 |
| partial_visibility_yaws | 51 | 459 | -3.4948 |
| shifted_yaws | 51 | 306 | -0.1799 |

部分可见半点扰动平均−3.4948pp，存在−51.256pp的单变体负尾；背景注入平均仅−.0857pp但负尾−22.611pp。不能用均值小推断背景污染无害。crop扩张平均−.6828pp，±10%尺寸位移平均−.1799pp，均有正负变化；某些错误裁剪也可能因加入被遗漏目标点而改善。全部父样本/变体的正负尾留档，不把1,734个变体当独立样本做显著性。

## 审计与决定

评分基于float64二维裁剪×高度，54例source对旧已公开独立审计最大差5.3844e-8。独立代理已用3D凸体交集复核594个主IoU及全部扰动评分，最大主IoU差7.11e-15；所有scene-macro/分层bootstrap和P2 AUROC/Spearman分母通过独立复算。具体收据为research/round3/audit/INDEPENDENT_RESULTS_AUDIT.json。

当前决定：不重训，不启动任何后续训练或selector搜索。P0source-only迁移问题保留为未验证；P1确认朝向仍有可达空间但GTyaw非严格上界；P2保留裁剪/部分可见稳定性作为待独立验证假设，不采用阈值；P3没有证据认定几何优化稳定超过PCA/最小矩形。所有结论限定联合模型的54历史父样本与当前固定裁剪。

证据入口：crossplatform/round3/PLAN_DIAGNOSTIC_LOCK.json，EXECUTION_CLARIFICATIONS.json，p0/P0_AUDIT.md，p0/CANCELLED.json，p2/SEALED.json，p3/RUN_LOCK.json，p3/predictions/SEALED.json，diagnostics/SCORER_LOCK.json，diagnostics/RESULTS_PRIVATE.json，diagnostics/PAIRED_COMPARISONS.json，diagnostics/PERTURBATION_SUMMARY.json。
