# R91 — 冻结强基座的源域竞争实例判别与原物理恢复TTA

北京时间 2026-11-13 13:13。全部DEV336为历史暴露开发数据；未访问FINAL。

固定主要方法 **R76_full_disc_physical_bt=22.872394%**。相对自身零步-0.159652pp，相对同信息概率平均-0.101785pp，相对同信息表示融合-0.101785pp。

源域新绑定前提：{'R76': False, 'R89continue': False}。同时超过两个同信息零更新对照：False。没有独立确认或稳健方法采纳。三阶段分别裁决，辅助loss或成对间隔变化不代替最终选框。

## 实际范围与对照

单一Waymo-only epoch100的原3EED权重保持不变；保留R76双向RGB–LiDAR结构，R76e6和R89选定的matching-only continuation作为两份独立初始化。R89基座的融合参数与R76逐项相同，额外source准备历史不同，因此两基座不称完全相同总训练预算。当前每次推理仅加载一个3EED并重用同一冻结骨干，不引入Qwen/CLIP/SAM等专家、新encoder/decoder或meta训练。原A256坐标始终固定。

本轮source阶段仅训练候选接口，原融合、RGB编码器、matching和恢复头全部冻结。因这些表示确实固定，缓存融合后特征这次是有效加速；它与R90需要学习融合却缓存掉梯度路径的情形不同。局部支持采用严格支持内归一化，原生query/text保留全局上下文。值向量本身带上下文、框内点未必纯实例，不能宣称获得真实instance mask。

完整空间接口的最终残差MLP屏蔽直接query输入，query只用于条件化主体/参照路由；缺少角色时残差为0，不偷偷归给query后备路径。query-only是独立对照；no_local撤回local/own值但保留几何/角色，no_ref撤回显式参照。此读出限制与竞争损失同时存在，full_ce对照保留同样限制以隔离新增损失。没有把它称为EG-3DVG/EDA/CORE-3DVG复现。

## 源域监督与负例资格

除原目标框外，本轮明确使用Waymo ground_info/others中已有的非目标物体框，仅用于支持不同同类实例。source CE只用原目标IoU，新pair loss利用这些额外source物体标注；不是相同标签使用方式，不掩盖这项监督资源差异。它们不提供参照角色身份或点级实例标注；目标域从不读取这些标签。

候选对：正框为原A256最高目标IoU且>0.25、至少2个支持点；负框与另一个同类标注物体IoU>0.25、相对其他归属至少0.1间隔、与目标IoU<0.05。同时排除预测框IoU>0.1或点支持Jaccard>0.1的含混重复。其他物体与目标标注框IoU<0.05，去重同位置重复标注。每份冻结基座在优化前，按原生分数为每个物体固定一个最高分候选，最多4个不同竞争实例。不将远框/低分框自动当实例负例；资格不足时pair项为0，样本仍参与CE和总体评价。

损失为原soft-IoU CE＋mean softplus(s_negative−s_positive+0.2)，pair权重1，完整与两遮挡输入各占1与均值1。source512/128、94/82不同场景，4轮/分支、lr1e-4/decay1e-4/clip1，同一固定次序。相同尝试样本预算、有效更新/活跃参数量并不相同；空间头没有角色时不伪造梯度。十分支零更新sourceVAL均值最早最大值选epoch1，在内层更新和目标任务评分前封存。

| 基座 | 范围 | n | 有合格pair | 原Top1是接受负例 | pairs |
|---|---|---:|---:|---:|---:|
| R76 | TRAIN | 512 | 384 | 17 | 780 |
| R76 | VAL | 128 | 95 | 8 | 191 |
| R89continue | TRAIN | 512 | 384 | 16 | 780 |
| R89continue | VAL | 128 | 95 | 7 | 191 |

384/512与95/128的资格覆盖不代表全部是真正决定当前Top1的困难负例；原Top1错误实例被覆盖的数量单独列出，不以大量已满足的成对关系扩大纠错证据。

| 基座/接口 | 源域VAL完整mIoU % | 遮挡均值mIoU % |
|---|---:|---:|
| R76_native_frozen | 47.385454 | not_measured_here |
| R89continue_native_frozen | 47.118085 | not_measured_here |
| R76_query_disc | 47.445664 | 46.228313 |
| R76_full_ce | 46.740137 | 46.323785 |
| R76_full_disc | 46.892290 | 46.432858 |
| R76_no_ref_disc | 46.896928 | 46.707714 |
| R76_no_local_disc | 47.385454 | 46.637159 |
| R89continue_query_disc | 47.114669 | 46.788589 |
| R89continue_full_ce | 46.811467 | 46.616192 |
| R89continue_full_disc | 46.811063 | 46.896613 |
| R89continue_no_ref_disc | 46.932837 | 46.647635 |
| R89continue_no_local_disc | 47.118085 | 46.938602 |

| 基座/接口 | 有效样本/pairs | 正间隔 % | ≥训练间隔 % | 平均间隔 |
|---|---:|---:|---:|---:|
| R76/query_disc | 95/191 | 92.670157 | 91.099476 | 2.839017 |
| R76/full_ce | 95/191 | 92.670157 | 91.623037 | 3.160596 |
| R76/full_disc | 95/191 | 92.670157 | 91.099476 | 3.894583 |
| R76/no_ref_disc | 95/191 | 92.146597 | 91.623037 | 4.009368 |
| R76/no_local_disc | 95/191 | 92.146597 | 90.575916 | 2.779028 |
| R76/native_baseline | 95/191 | 92.146597 | 90.575916 | 2.778836 |
| R89continue/query_disc | 95/191 | 92.670157 | 91.099476 | 3.096726 |
| R89continue/full_ce | 95/191 | 93.193717 | 92.146597 | 3.427963 |
| R89continue/full_disc | 95/191 | 93.193717 | 91.623037 | 3.750580 |
| R89continue/no_ref_disc | 95/191 | 93.193717 | 92.146597 | 3.738355 |
| R89continue/no_local_disc | 95/191 | 92.670157 | 91.623037 | 2.857196 |
| R89continue/native_baseline | 95/191 | 92.670157 | 91.623037 | 2.856068 |

全部40条训练曲线与128×58无标签内层更新结果在公开表保留。内层更新不看source标签，预测封存/原生审计后才接source IoU，不用于重选epoch或修改方法。

| Source128方法 | mIoU % | Acc25 % | Acc50 % |
|---|---:|---:|---:|
| R76_query_disc_frozen | 47.445664 | 85.937500 | 46.093750 |
| R76_query_disc_mean3 | 46.643133 | 84.375000 | 45.312500 |
| R76_query_disc_feature3 | 46.643133 | 84.375000 | 45.312500 |
| R76_query_disc_physical_raw | 47.281657 | 85.937500 | 46.093750 |
| R76_query_disc_physical_bt | 47.353288 | 85.937500 | 46.093750 |
| R76_full_ce_frozen | 46.740137 | 84.375000 | 45.312500 |
| R76_full_ce_mean3 | 47.120786 | 85.156250 | 46.093750 |
| R76_full_ce_feature3 | 47.120786 | 85.156250 | 46.093750 |
| R76_full_ce_physical_raw | 47.025491 | 85.156250 | 46.093750 |
| R76_full_ce_physical_bt | 46.694161 | 84.375000 | 46.093750 |
| R76_full_disc_frozen | 46.892290 | 84.375000 | 46.093750 |
| R76_full_disc_mean3 | 46.851196 | 84.375000 | 46.093750 |
| R76_full_disc_feature3 | 46.851196 | 84.375000 | 46.093750 |
| R76_full_disc_physical_raw | 46.765714 | 84.375000 | 46.093750 |
| R76_full_disc_physical_bt | 46.729142 | 84.375000 | 46.093750 |
| R76_no_ref_disc_frozen | 46.896928 | 84.375000 | 46.093750 |
| R76_no_ref_disc_mean3 | 46.855158 | 84.375000 | 46.093750 |
| R76_no_ref_disc_feature3 | 46.855158 | 84.375000 | 46.093750 |
| R76_no_ref_disc_physical_raw | 46.762669 | 84.375000 | 46.093750 |
| R76_no_ref_disc_physical_bt | 46.747689 | 84.375000 | 46.093750 |
| R76_no_local_disc_frozen | 47.385454 | 85.937500 | 45.312500 |
| R76_no_local_disc_mean3 | 47.315789 | 85.937500 | 45.312500 |
| R76_no_local_disc_feature3 | 47.315789 | 85.937500 | 45.312500 |
| R76_no_local_disc_physical_raw | 46.914871 | 85.156250 | 45.312500 |
| R76_no_local_disc_physical_bt | 47.224895 | 85.937500 | 45.312500 |
| R89continue_query_disc_frozen | 47.114669 | 85.937500 | 44.531250 |
| R89continue_query_disc_mean3 | 47.140733 | 85.937500 | 44.531250 |
| R89continue_query_disc_feature3 | 47.140733 | 85.937500 | 44.531250 |
| R89continue_query_disc_physical_raw | 46.843962 | 85.937500 | 44.531250 |
| R89continue_query_disc_physical_bt | 46.845976 | 85.937500 | 44.531250 |
| R89continue_full_ce_frozen | 46.811467 | 85.156250 | 45.312500 |
| R89continue_full_ce_mean3 | 46.829204 | 85.156250 | 45.312500 |
| R89continue_full_ce_feature3 | 46.829204 | 85.156250 | 45.312500 |
| R89continue_full_ce_physical_raw | 46.671930 | 85.156250 | 45.312500 |
| R89continue_full_ce_physical_bt | 46.715322 | 85.156250 | 45.312500 |
| R89continue_full_disc_frozen | 46.811063 | 85.156250 | 45.312500 |
| R89continue_full_disc_mean3 | 46.829204 | 85.156250 | 45.312500 |
| R89continue_full_disc_feature3 | 46.829204 | 85.156250 | 45.312500 |
| R89continue_full_disc_physical_raw | 46.451138 | 84.375000 | 45.312500 |
| R89continue_full_disc_physical_bt | 46.697623 | 85.156250 | 45.312500 |
| R89continue_no_ref_disc_frozen | 46.932837 | 85.156250 | 45.312500 |
| R89continue_no_ref_disc_mean3 | 46.879691 | 85.156250 | 45.312500 |
| R89continue_no_ref_disc_feature3 | 46.879691 | 85.156250 | 45.312500 |
| R89continue_no_ref_disc_physical_raw | 46.359820 | 84.375000 | 45.312500 |
| R89continue_no_ref_disc_physical_bt | 46.858362 | 85.156250 | 45.312500 |
| R89continue_no_local_disc_frozen | 47.118085 | 85.937500 | 45.312500 |
| R89continue_no_local_disc_mean3 | 47.193751 | 85.937500 | 45.312500 |
| R89continue_no_local_disc_feature3 | 47.193751 | 85.937500 | 45.312500 |
| R89continue_no_local_disc_physical_raw | 46.952981 | 85.937500 | 45.312500 |
| R89continue_no_local_disc_physical_bt | 46.894247 | 85.937500 | 45.312500 |
| R76_full_disc_conditional_raw | 46.615100 | 84.375000 | 46.093750 |
| R76_full_disc_conditional_bt | 46.842130 | 84.375000 | 46.093750 |
| R76_full_disc_wrong_physical_raw | 46.121229 | 82.812500 | 45.312500 |
| R76_full_disc_wrong_physical_bt | 46.875322 | 84.375000 | 46.093750 |
| R89continue_full_disc_conditional_raw | 46.459311 | 84.375000 | 45.312500 |
| R89continue_full_disc_conditional_bt | 46.705795 | 85.156250 | 45.312500 |
| R89continue_full_disc_wrong_physical_raw | 46.277841 | 83.593750 | 46.093750 |
| R89continue_full_disc_wrong_physical_bt | 46.714179 | 85.156250 | 46.093750 |

## 目标域更新严格沿用原R80/82范围

实际梯度主要来自原物理恢复：0.1点表征＋0.1实测RGB patch＋0.01图像单支logdepth。depth预测从反向融合前图像分支读取，不能复制反向LiDAR输入。Point目标是冻结表征代理，不冒充真实语义标签。两遮挡76001/76002；原102498个Adapter、双向交互投影与两个gate；SGD一步alpha0.5/clip1，raw版直接应用，BT版原Armijo c1e-4/tol1e-10、0.5/2^k(k0..8)。所有新候选接口、matching、RGB CNN、原3EED与恢复头均冻结，不扩大TTA层或调参。保留初始输出KL与0.1原生query保持；起点保持梯度为0，不把它当新监督信息。

物理首步方向在相同theta/恢复头/观测下相同，跨head共享方向与完全相同参数态的原生特征可节省计算。每个head的读出、保持曲率、实际接受步长、负结果各自保存；没有在参数改变后使用旧缓存。角色条件恢复单独加入0.1新条件项，仍只更新原102498参数，用相同alpha与视图；另保留原逻辑RGB–点绑定错配的raw/BT版本。错配BT可能接受不同幅度，raw同alpha比较用于隔离方向，不把所有BT差异归于对应可靠性。

两个主要零更新对照使用完全相同完整输入＋两遮挡观测：概率平均、原生query/text投影和实际判别输入的表示平均。旧FrozenABC/R59使用历史采样，是独立强参照，不冒充同三次遮挡。第三遮挡76003仅测留出恢复，不参与拟合、接受或超参选择，也不是新相机视角/独立场景。

历史R80/R82及R89/R90分数和UNKNOWN/回退状态按原封存复用。R80曾在TRAIN1024退化、R82增量脆弱；本轮不预设它们是稳定TTA。

## 全DEV336总体任务结果

| 方法 | Drone mIoU % | Quad mIoU % | 等权mIoU % | 等权Acc25 % | 等权Acc50 % |
|---|---:|---:|---:|---:|---:|
| source | 15.539892 | 28.120770 | 21.830331 | 40.773810 | 21.130952 |
| R76_frozen | 17.083330 | 28.139863 | 22.611596 | 41.071429 | 21.428571 |
| ABC | 15.736332 | 28.868588 | 22.302460 | 42.261905 | 21.428571 |
| R59 | 16.385568 | 28.836664 | 22.611116 | 43.154762 | 21.726190 |
| R80_raw | 17.128930 | 28.791222 | 22.960076 | 42.261905 | 21.130952 |
| R82_bt | 17.358929 | 28.163165 | 22.761047 | 41.666667 | 21.428571 |
| R89_continue_frozen | 17.308216 | 28.148111 | 22.728164 | 41.071429 | 21.130952 |
| R89_continue_oldTTA | 17.559059 | 28.097869 | 22.828464 | 41.369048 | 21.130952 |
| R90_joint_frozen | 16.805435 | 27.931775 | 22.368605 | 41.369048 | 20.833333 |
| R90_joint_tta | 16.879742 | 27.861492 | 22.370617 | 41.369048 | 21.130952 |
| R90_feature3 | 17.301687 | 28.489247 | 22.895467 | 41.964286 | 21.130952 |
| R76_query_disc_frozen | 16.918771 | 28.231960 | 22.575366 | 41.071429 | 21.130952 |
| R76_query_disc_mean3 | 16.780659 | 28.335628 | 22.558144 | 40.773810 | 21.130952 |
| R76_query_disc_feature3 | 16.780659 | 28.335628 | 22.558144 | 40.773810 | 21.130952 |
| R76_query_disc_physical_raw | 16.821158 | 28.858008 | 22.839583 | 41.666667 | 21.130952 |
| R76_query_disc_physical_bt | 16.982207 | 28.231960 | 22.607084 | 41.071429 | 21.130952 |
| R76_full_ce_frozen | 17.271896 | 28.259579 | 22.765738 | 41.369048 | 21.428571 |
| R76_full_ce_mean3 | 16.910296 | 28.415060 | 22.662678 | 41.369048 | 21.428571 |
| R76_full_ce_feature3 | 16.910296 | 28.415060 | 22.662678 | 41.369048 | 21.428571 |
| R76_full_ce_physical_raw | 17.243737 | 28.790413 | 23.017075 | 42.559524 | 21.130952 |
| R76_full_ce_physical_bt | 17.473736 | 28.162993 | 22.818365 | 41.964286 | 21.428571 |
| R76_full_disc_frozen | 17.346189 | 28.717901 | 23.032045 | 41.964286 | 21.726190 |
| R76_full_disc_mean3 | 17.145428 | 28.802929 | 22.974178 | 41.964286 | 21.726190 |
| R76_full_disc_feature3 | 17.145428 | 28.802929 | 22.974178 | 41.964286 | 21.726190 |
| R76_full_disc_physical_raw | 17.025536 | 28.934628 | 22.980082 | 42.559524 | 21.428571 |
| R76_full_disc_physical_bt | 17.073866 | 28.670921 | 22.872394 | 41.964286 | 21.726190 |
| R76_no_ref_disc_frozen | 17.363513 | 28.551395 | 22.957454 | 42.261905 | 21.726190 |
| R76_no_ref_disc_mean3 | 17.108744 | 28.717695 | 22.913219 | 41.964286 | 21.726190 |
| R76_no_ref_disc_feature3 | 17.108744 | 28.717695 | 22.913219 | 41.964286 | 21.726190 |
| R76_no_ref_disc_physical_raw | 16.922077 | 29.127372 | 23.024724 | 42.857143 | 21.130952 |
| R76_no_ref_disc_physical_bt | 17.037834 | 28.574697 | 22.806266 | 42.261905 | 21.726190 |
| R76_no_local_disc_frozen | 17.083330 | 28.139863 | 22.611596 | 41.071429 | 21.428571 |
| R76_no_local_disc_mean3 | 16.796001 | 28.415060 | 22.605530 | 41.071429 | 21.428571 |
| R76_no_local_disc_feature3 | 16.643802 | 28.415060 | 22.529431 | 40.773810 | 21.428571 |
| R76_no_local_disc_physical_raw | 17.128930 | 28.791222 | 22.960076 | 42.261905 | 21.130952 |
| R76_no_local_disc_physical_bt | 17.358929 | 28.163165 | 22.761047 | 41.666667 | 21.428571 |
| R89continue_query_disc_frozen | 17.176818 | 28.064121 | 22.620470 | 40.773810 | 21.130952 |
| R89continue_query_disc_mean3 | 17.071778 | 28.070734 | 22.571256 | 40.773810 | 21.130952 |
| R89continue_query_disc_feature3 | 17.071778 | 28.070734 | 22.571256 | 40.773810 | 21.130952 |
| R89continue_query_disc_physical_raw | 16.916614 | 28.360191 | 22.638402 | 41.071429 | 21.130952 |
| R89continue_query_disc_physical_bt | 17.299734 | 28.064121 | 22.681927 | 41.369048 | 21.130952 |
| R89continue_full_ce_frozen | 17.568891 | 28.466448 | 23.017669 | 41.666667 | 21.428571 |
| R89continue_full_ce_mean3 | 17.061247 | 28.391895 | 22.726571 | 41.666667 | 21.428571 |
| R89continue_full_ce_feature3 | 17.061247 | 28.391895 | 22.726571 | 41.666667 | 21.428571 |
| R89continue_full_ce_physical_raw | 16.950227 | 29.246297 | 23.098262 | 41.369048 | 21.130952 |
| R89continue_full_ce_physical_bt | 17.525638 | 28.417127 | 22.971382 | 41.666667 | 21.428571 |
| R89continue_full_disc_frozen | 17.463769 | 28.808022 | 23.135895 | 41.964286 | 21.726190 |
| R89continue_full_disc_mean3 | 17.281815 | 28.756902 | 23.019358 | 42.261905 | 21.726190 |
| R89continue_full_disc_feature3 | 17.281815 | 28.756902 | 23.019358 | 42.261905 | 21.726190 |
| R89continue_full_disc_physical_raw | 16.951841 | 29.146804 | 23.049323 | 41.071429 | 21.726190 |
| R89continue_full_disc_physical_bt | 17.489573 | 28.756822 | 23.123198 | 41.964286 | 21.726190 |
| R89continue_no_ref_disc_frozen | 17.337810 | 28.488019 | 22.912914 | 41.666667 | 21.428571 |
| R89continue_no_ref_disc_mean3 | 17.280201 | 28.758781 | 23.019491 | 42.261905 | 21.726190 |
| R89continue_no_ref_disc_feature3 | 17.280201 | 28.758781 | 23.019491 | 42.261905 | 21.726190 |
| R89continue_no_ref_disc_physical_raw | 16.798156 | 28.681808 | 22.739982 | 40.476190 | 21.130952 |
| R89continue_no_ref_disc_physical_bt | 17.333207 | 28.438698 | 22.885952 | 41.369048 | 21.428571 |
| R89continue_no_local_disc_frozen | 17.308216 | 28.148111 | 22.728164 | 41.071429 | 21.130952 |
| R89continue_no_local_disc_mean3 | 17.188336 | 28.071893 | 22.630114 | 41.369048 | 21.130952 |
| R89continue_no_local_disc_feature3 | 17.188336 | 28.071893 | 22.630114 | 41.369048 | 21.130952 |
| R89continue_no_local_disc_physical_raw | 16.878377 | 28.390885 | 22.634631 | 40.476190 | 21.130952 |
| R89continue_no_local_disc_physical_bt | 17.531334 | 28.065402 | 22.798368 | 41.071429 | 21.130952 |
| R76_full_disc_conditional_raw | 16.858545 | 28.921535 | 22.890040 | 42.261905 | 21.130952 |
| R76_full_disc_conditional_bt | 16.961761 | 28.670924 | 22.816342 | 41.666667 | 21.726190 |
| R76_full_disc_wrong_physical_raw | 16.729429 | 29.318644 | 23.024037 | 41.964286 | 21.726190 |
| R76_full_disc_wrong_physical_bt | 17.348236 | 28.721194 | 23.034715 | 41.964286 | 21.726190 |
| R89continue_full_disc_conditional_raw | 16.984651 | 28.714326 | 22.849489 | 41.369048 | 21.130952 |
| R89continue_full_disc_conditional_bt | 17.463769 | 28.756822 | 23.110296 | 41.964286 | 21.726190 |
| R89continue_full_disc_wrong_physical_raw | 16.535568 | 29.353463 | 22.944515 | 41.071429 | 21.130952 |
| R89continue_full_disc_wrong_physical_bt | 17.309718 | 28.845596 | 23.077657 | 41.369048 | 21.726190 |

## 组件、同信息增量与全部负例

| 项目 | 方法 | 对照 | 平台 | Δpp | 改善/恶化/不变 | 最大单例正/负贡献pp |
|---|---|---|---|---:|---:|---:|
| source_preparation | R76_full_disc_frozen | R76_frozen | balanced | +0.420449 | 9/7/320 | +0.170787/-0.042932 |
| source_preparation | R76_full_disc_frozen | R76_frozen | drone | +0.262860 | 3/4/161 | +0.278301/-0.085864 |
| source_preparation | R76_full_disc_frozen | R76_frozen | quad | +0.578039 | 6/3/159 | +0.341574/-0.051674 |
| spatial_vs_query | R76_full_disc_frozen | R76_query_disc_frozen | balanced | +0.456680 | 15/7/314 | +0.170787/-0.094792 |
| spatial_vs_query | R76_full_disc_frozen | R76_query_disc_frozen | drone | +0.427419 | 10/5/153 | +0.278301/-0.189584 |
| spatial_vs_query | R76_full_disc_frozen | R76_query_disc_frozen | quad | +0.485941 | 5/2/161 | +0.341574/-0.051674 |
| source_competition_loss | R76_full_disc_frozen | R76_full_ce_frozen | balanced | +0.266308 | 7/7/322 | +0.170787/-0.042932 |
| source_competition_loss | R76_full_disc_frozen | R76_full_ce_frozen | drone | +0.074293 | 2/4/162 | +0.204542/-0.085864 |
| source_competition_loss | R76_full_disc_frozen | R76_full_ce_frozen | quad | +0.458323 | 5/3/160 | +0.341574/-0.051674 |
| reference_reading | R76_full_disc_frozen | R76_no_ref_disc_frozen | balanced | +0.074592 | 2/3/331 | +0.048198/-0.008335 |
| reference_reading | R76_full_disc_frozen | R76_no_ref_disc_frozen | drone | -0.017323 | 0/2/166 | +0.000000/-0.016671 |
| reference_reading | R76_full_disc_frozen | R76_no_ref_disc_frozen | quad | +0.166506 | 2/1/165 | +0.096395/-0.000172 |
| local_visual_evidence | R76_full_disc_frozen | R76_no_local_disc_frozen | balanced | +0.420449 | 9/7/320 | +0.170787/-0.042932 |
| local_visual_evidence | R76_full_disc_frozen | R76_no_local_disc_frozen | drone | +0.262860 | 3/4/161 | +0.278301/-0.085864 |
| local_visual_evidence | R76_full_disc_frozen | R76_no_local_disc_frozen | quad | +0.578039 | 6/3/159 | +0.341574/-0.051674 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_full_disc_frozen | balanced | -0.159652 | 1/3/332 | +0.011651/-0.134811 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_full_disc_frozen | drone | -0.272323 | 0/2/166 | +0.000000/-0.269621 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_full_disc_frozen | quad | -0.046980 | 1/1/166 | +0.023302/-0.070283 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_full_disc_mean3 | balanced | -0.101785 | 1/4/331 | +0.008502/-0.051834 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_full_disc_mean3 | drone | -0.071562 | 1/2/165 | +0.017003/-0.085864 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_full_disc_mean3 | quad | -0.132008 | 0/2/166 | +0.000000/-0.103667 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_full_disc_feature3 | balanced | -0.101785 | 1/4/331 | +0.008502/-0.051834 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_full_disc_feature3 | drone | -0.071562 | 1/2/165 | +0.017003/-0.085864 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_full_disc_feature3 | quad | -0.132008 | 0/2/166 | +0.000000/-0.103667 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_full_disc_physical_raw | balanced | -0.107688 | 11/10/315 | +0.067712/-0.160001 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_full_disc_physical_raw | drone | +0.048330 | 8/5/155 | +0.135424/-0.189584 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_full_disc_physical_raw | quad | -0.263707 | 3/5/160 | +0.096395/-0.320003 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_full_disc_conditional_bt | balanced | +0.056051 | 1/2/333 | +0.057404/-0.001351 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_full_disc_conditional_bt | drone | +0.112106 | 1/1/166 | +0.114807/-0.002702 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_full_disc_conditional_bt | quad | -0.000003 | 0/1/167 | +0.000000/-0.000003 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_full_disc_wrong_physical_bt | balanced | -0.162321 | 2/3/331 | +0.010005/-0.134811 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_full_disc_wrong_physical_bt | drone | -0.274369 | 1/2/165 | +0.002721/-0.269621 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_full_disc_wrong_physical_bt | quad | -0.050273 | 1/1/166 | +0.020010/-0.070283 |
| adaptation_or_control | R76_full_disc_physical_bt | R80_raw | balanced | -0.087682 | 13/15/308 | +0.170787/-0.160001 |
| adaptation_or_control | R76_full_disc_physical_bt | R80_raw | drone | -0.055063 | 7/7/154 | +0.114807/-0.189584 |
| adaptation_or_control | R76_full_disc_physical_bt | R80_raw | quad | -0.120301 | 6/8/154 | +0.341574/-0.320003 |
| adaptation_or_control | R76_full_disc_physical_bt | R82_bt | balanced | +0.111347 | 7/8/321 | +0.170787/-0.134811 |
| adaptation_or_control | R76_full_disc_physical_bt | R82_bt | drone | -0.285063 | 2/5/161 | +0.114807/-0.269621 |
| adaptation_or_control | R76_full_disc_physical_bt | R82_bt | quad | +0.507756 | 5/3/160 | +0.341574/-0.051674 |
| adaptation_or_control | R76_full_disc_physical_bt | R59 | balanced | +0.261278 | 44/55/237 | +0.234305/-0.168124 |
| adaptation_or_control | R76_full_disc_physical_bt | R59 | drone | +0.688299 | 24/35/109 | +0.468611/-0.335152 |
| adaptation_or_control | R76_full_disc_physical_bt | R59 | quad | -0.165743 | 20/20/128 | +0.341574/-0.336248 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_query_disc_physical_bt | balanced | +0.265310 | 15/10/311 | +0.170787/-0.134811 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_query_disc_physical_bt | drone | +0.091660 | 10/8/150 | +0.278301/-0.269621 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_query_disc_physical_bt | quad | +0.438961 | 5/2/161 | +0.341574/-0.051674 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_full_ce_physical_bt | balanced | +0.054029 | 6/7/323 | +0.170787/-0.134811 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_full_ce_physical_bt | drone | -0.399870 | 1/5/162 | +0.033549/-0.269621 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_full_ce_physical_bt | quad | +0.507928 | 5/2/161 | +0.341574/-0.051674 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_no_ref_disc_physical_bt | balanced | +0.066128 | 2/3/331 | +0.048198/-0.008335 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_no_ref_disc_physical_bt | drone | +0.036032 | 1/2/165 | +0.053356/-0.016671 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_no_ref_disc_physical_bt | quad | +0.096224 | 1/1/166 | +0.096395/-0.000172 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_no_local_disc_physical_bt | balanced | +0.111347 | 7/8/321 | +0.170787/-0.134811 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_no_local_disc_physical_bt | drone | -0.285063 | 2/5/161 | +0.114807/-0.269621 |
| adaptation_or_control | R76_full_disc_physical_bt | R76_no_local_disc_physical_bt | quad | +0.507756 | 5/3/160 | +0.341574/-0.051674 |
| matched_raw_correspondence | R76_full_disc_physical_raw | R76_full_disc_wrong_physical_raw | balanced | -0.043955 | 8/10/318 | +0.094792/-0.130752 |
| matched_raw_correspondence | R76_full_disc_physical_raw | R76_full_disc_wrong_physical_raw | drone | +0.296107 | 4/3/161 | +0.189584/-0.085548 |
| matched_raw_correspondence | R76_full_disc_physical_raw | R76_full_disc_wrong_physical_raw | quad | -0.384017 | 4/7/157 | +0.055364/-0.261504 |
| matched_raw_conditional_target | R76_full_disc_conditional_raw | R76_full_disc_physical_raw | balanced | -0.090042 | 4/5/327 | +0.067712/-0.102271 |
| matched_raw_conditional_target | R76_full_disc_conditional_raw | R76_full_disc_physical_raw | drone | -0.166991 | 3/3/162 | +0.135424/-0.204542 |
| matched_raw_conditional_target | R76_full_disc_conditional_raw | R76_full_disc_physical_raw | quad | -0.013093 | 1/2/165 | +0.096395/-0.103992 |
| source_preparation | R89continue_full_disc_frozen | R89_continue_frozen | balanced | +0.407732 | 5/6/325 | +0.170787/-0.066365 |
| source_preparation | R89continue_full_disc_frozen | R89_continue_frozen | drone | +0.155553 | 3/4/161 | +0.278301/-0.132729 |
| source_preparation | R89continue_full_disc_frozen | R89_continue_frozen | quad | +0.659910 | 2/2/164 | +0.341574/-0.000921 |
| spatial_vs_query | R89continue_full_disc_frozen | R89continue_query_disc_frozen | balanced | +0.515426 | 13/17/306 | +0.170787/-0.026678 |
| spatial_vs_query | R89continue_full_disc_frozen | R89continue_query_disc_frozen | drone | +0.286951 | 7/12/149 | +0.278301/-0.053356 |
| spatial_vs_query | R89continue_full_disc_frozen | R89continue_query_disc_frozen | quad | +0.743900 | 6/5/157 | +0.341574/-0.025087 |
| source_competition_loss | R89continue_full_disc_frozen | R89continue_full_ce_frozen | balanced | +0.118226 | 2/1/333 | +0.170787/-0.053368 |
| source_competition_loss | R89continue_full_disc_frozen | R89continue_full_ce_frozen | drone | -0.105122 | 1/1/166 | +0.001614/-0.106735 |
| source_competition_loss | R89continue_full_disc_frozen | R89continue_full_ce_frozen | quad | +0.341574 | 1/0/167 | +0.341574/+0.000000 |
| reference_reading | R89continue_full_disc_frozen | R89continue_no_ref_disc_frozen | balanced | +0.222981 | 3/1/332 | +0.160001/-0.013863 |
| reference_reading | R89continue_full_disc_frozen | R89continue_no_ref_disc_frozen | drone | +0.125960 | 2/1/165 | +0.152071/-0.027725 |
| reference_reading | R89continue_full_disc_frozen | R89continue_no_ref_disc_frozen | quad | +0.320003 | 1/0/167 | +0.320003/+0.000000 |
| local_visual_evidence | R89continue_full_disc_frozen | R89continue_no_local_disc_frozen | balanced | +0.407732 | 5/6/325 | +0.170787/-0.066365 |
| local_visual_evidence | R89continue_full_disc_frozen | R89continue_no_local_disc_frozen | drone | +0.155553 | 3/4/161 | +0.278301/-0.132729 |
| local_visual_evidence | R89continue_full_disc_frozen | R89continue_no_local_disc_frozen | quad | +0.659910 | 2/2/164 | +0.341574/-0.000921 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_full_disc_frozen | balanced | -0.012698 | 5/1/330 | +0.012543/-0.041351 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_full_disc_frozen | drone | +0.025804 | 2/0/166 | +0.023123/+0.000000 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_full_disc_frozen | quad | -0.051199 | 3/1/164 | +0.025087/-0.082703 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_full_disc_mean3 | balanced | +0.103839 | 7/7/322 | +0.076036/-0.053369 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_full_disc_mean3 | drone | +0.207758 | 5/5/158 | +0.152071/-0.106738 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_full_disc_mean3 | quad | -0.000080 | 2/2/164 | +0.025087/-0.025311 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_full_disc_feature3 | balanced | +0.103839 | 7/7/322 | +0.076036/-0.053369 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_full_disc_feature3 | drone | +0.207758 | 5/5/158 | +0.152071/-0.106738 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_full_disc_feature3 | quad | -0.000080 | 2/2/164 | +0.025087/-0.025311 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_full_disc_physical_raw | balanced | +0.073875 | 13/11/312 | +0.134811/-0.138943 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_full_disc_physical_raw | drone | +0.537732 | 11/5/152 | +0.269621/-0.080816 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_full_disc_physical_raw | quad | -0.389982 | 2/6/160 | +0.015314/-0.277887 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_full_disc_conditional_bt | balanced | +0.012902 | 2/0/334 | +0.011561/+0.000000 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_full_disc_conditional_bt | drone | +0.025804 | 2/0/166 | +0.023123/+0.000000 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_full_disc_conditional_bt | quad | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_full_disc_wrong_physical_bt | balanced | +0.045541 | 6/8/322 | +0.042932/-0.041351 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_full_disc_wrong_physical_bt | drone | +0.179855 | 4/4/160 | +0.085864/-0.027725 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_full_disc_wrong_physical_bt | quad | -0.088774 | 2/4/162 | +0.020010/-0.082703 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R80_raw | balanced | +0.163122 | 17/19/300 | +0.170787/-0.130752 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R80_raw | drone | +0.360643 | 12/13/143 | +0.269621/-0.094759 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R80_raw | quad | -0.034400 | 5/6/157 | +0.341574/-0.261504 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R82_bt | balanced | +0.362151 | 16/16/304 | +0.170787/-0.047379 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R82_bt | drone | +0.130644 | 8/13/147 | +0.189584/-0.094759 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R82_bt | quad | +0.593657 | 8/3/157 | +0.341574/-0.082703 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R59 | balanced | +0.512082 | 50/58/228 | +0.234305/-0.168124 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R59 | drone | +1.104005 | 28/36/104 | +0.468611/-0.335152 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R59 | quad | -0.079842 | 22/22/124 | +0.341574/-0.336248 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_query_disc_physical_bt | balanced | +0.441270 | 13/12/311 | +0.170787/-0.026678 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_query_disc_physical_bt | drone | +0.189839 | 7/10/151 | +0.152071/-0.053356 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_query_disc_physical_bt | quad | +0.692701 | 6/2/160 | +0.341574/-0.003617 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_full_ce_physical_bt | balanced | +0.151815 | 4/2/330 | +0.170787/-0.053368 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_full_ce_physical_bt | drone | -0.036065 | 3/1/164 | +0.066375/-0.106735 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_full_ce_physical_bt | quad | +0.339696 | 1/1/166 | +0.341574/-0.001878 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_no_ref_disc_physical_bt | balanced | +0.237245 | 4/1/331 | +0.160001/-0.000939 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_no_ref_disc_physical_bt | drone | +0.156366 | 3/0/165 | +0.152071/+0.000000 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_no_ref_disc_physical_bt | quad | +0.318124 | 1/1/166 | +0.320003/-0.001878 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_no_local_disc_physical_bt | balanced | +0.324829 | 8/5/323 | +0.170787/-0.066365 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_no_local_disc_physical_bt | drone | -0.041761 | 5/3/160 | +0.152071/-0.132729 |
| adaptation_or_control | R89continue_full_disc_physical_bt | R89continue_no_local_disc_physical_bt | quad | +0.691420 | 3/2/163 | +0.341574/-0.001878 |
| matched_raw_correspondence | R89continue_full_disc_physical_raw | R89continue_full_disc_wrong_physical_raw | balanced | +0.104807 | 9/8/319 | +0.181738/-0.130752 |
| matched_raw_correspondence | R89continue_full_disc_physical_raw | R89continue_full_disc_wrong_physical_raw | drone | +0.416273 | 4/4/160 | +0.363477/-0.032626 |
| matched_raw_correspondence | R89continue_full_disc_physical_raw | R89continue_full_disc_wrong_physical_raw | quad | -0.206658 | 5/4/159 | +0.107040/-0.261504 |
| matched_raw_conditional_target | R89continue_full_disc_conditional_raw | R89continue_full_disc_physical_raw | balanced | -0.199834 | 4/5/327 | +0.021043/-0.170787 |
| matched_raw_conditional_target | R89continue_full_disc_conditional_raw | R89continue_full_disc_physical_raw | drone | +0.032810 | 2/1/165 | +0.042087/-0.023123 |
| matched_raw_conditional_target | R89continue_full_disc_conditional_raw | R89continue_full_disc_physical_raw | quad | -0.432478 | 2/4/162 | +0.015314/-0.341574 |

## 五个历史场景

不将同场景帧/描述当独立场景重复，不计算冒充全新总体的CI。两个平台等权为主指标，允许个别场景/样本退化，不按GT筛掉更新或负例。

| 平台/场景 | n | 方法 | mIoU % |
|---|---:|---|---:|
| drone/D_S1 | 106 | R76_frozen | 13.449026 |
| drone/D_S1 | 106 | R59 | 11.761612 |
| drone/D_S1 | 106 | R80_raw | 13.518440 |
| drone/D_S1 | 106 | R82_bt | 13.885825 |
| drone/D_S1 | 106 | R76_full_disc_frozen | 13.989152 |
| drone/D_S1 | 106 | R76_full_disc_feature3 | 13.697913 |
| drone/D_S1 | 106 | R76_full_disc_physical_raw | 13.595461 |
| drone/D_S1 | 106 | R76_full_disc_physical_bt | 13.557545 |
| drone/D_S1 | 106 | R89continue_full_disc_frozen | 14.347077 |
| drone/D_S1 | 106 | R89continue_full_disc_feature3 | 14.012027 |
| drone/D_S1 | 106 | R89continue_full_disc_physical_raw | 13.839565 |
| drone/D_S1 | 106 | R89continue_full_disc_physical_bt | 14.383724 |
| drone/D_S1 | 106 | R76_full_disc_conditional_bt | 13.379868 |
| drone/D_S1 | 106 | R76_full_disc_wrong_physical_bt | 13.984870 |
| drone/D_S1 | 106 | R89continue_full_disc_conditional_bt | 14.347077 |
| drone/D_S1 | 106 | R89continue_full_disc_wrong_physical_bt | 14.218210 |
| drone/D_S2 | 62 | R76_frozen | 23.296817 |
| drone/D_S2 | 62 | R59 | 24.291041 |
| drone/D_S2 | 62 | R80_raw | 23.301702 |
| drone/D_S2 | 62 | R82_bt | 23.296817 |
| drone/D_S2 | 62 | R76_full_disc_frozen | 23.085641 |
| drone/D_S2 | 62 | R76_full_disc_feature3 | 23.039568 |
| drone/D_S2 | 62 | R76_full_disc_physical_raw | 22.889857 |
| drone/D_S2 | 62 | R76_full_disc_physical_bt | 23.085641 |
| drone/D_S2 | 62 | R89continue_full_disc_frozen | 22.792308 |
| drone/D_S2 | 62 | R89continue_full_disc_feature3 | 22.872097 |
| drone/D_S2 | 62 | R89continue_full_disc_physical_raw | 22.272829 |
| drone/D_S2 | 62 | R89continue_full_disc_physical_bt | 22.799573 |
| drone/D_S2 | 62 | R76_full_disc_conditional_bt | 23.085641 |
| drone/D_S2 | 62 | R76_full_disc_wrong_physical_bt | 23.098506 |
| drone/D_S2 | 62 | R89continue_full_disc_conditional_bt | 22.792308 |
| drone/D_S2 | 62 | R89continue_full_disc_wrong_physical_bt | 22.595198 |
| quad/Q_S1 | 62 | R76_frozen | 10.447889 |
| quad/Q_S1 | 62 | R59 | 9.632116 |
| quad/Q_S1 | 62 | R80_raw | 10.667452 |
| quad/Q_S1 | 62 | R82_bt | 10.511031 |
| quad/Q_S1 | 62 | R76_full_disc_frozen | 10.708624 |
| quad/Q_S1 | 62 | R76_full_disc_feature3 | 11.129463 |
| quad/Q_S1 | 62 | R76_full_disc_physical_raw | 10.665260 |
| quad/Q_S1 | 62 | R76_full_disc_physical_bt | 10.771765 |
| quad/Q_S1 | 62 | R89continue_full_disc_frozen | 10.446969 |
| quad/Q_S1 | 62 | R89continue_full_disc_feature3 | 10.461468 |
| quad/Q_S1 | 62 | R89continue_full_disc_physical_raw | 10.551793 |
| quad/Q_S1 | 62 | R89continue_full_disc_physical_bt | 10.529837 |
| quad/Q_S1 | 62 | R76_full_disc_conditional_bt | 10.771773 |
| quad/Q_S1 | 62 | R76_full_disc_wrong_physical_bt | 10.717546 |
| quad/Q_S1 | 62 | R89continue_full_disc_conditional_bt | 10.529837 |
| quad/Q_S1 | 62 | R89continue_full_disc_wrong_physical_bt | 10.480199 |
| quad/Q_S2 | 49 | R76_frozen | 34.118085 |
| quad/Q_S2 | 49 | R59 | 36.067581 |
| quad/Q_S2 | 49 | R80_raw | 33.723220 |
| quad/Q_S2 | 49 | R82_bt | 34.118085 |
| quad/Q_S2 | 49 | R76_full_disc_frozen | 35.770023 |
| quad/Q_S2 | 49 | R76_full_disc_feature3 | 35.529054 |
| quad/Q_S2 | 49 | R76_full_disc_physical_raw | 35.114255 |
| quad/Q_S2 | 49 | R76_full_disc_physical_bt | 35.529054 |
| quad/Q_S2 | 49 | R89continue_full_disc_frozen | 35.312928 |
| quad/Q_S2 | 49 | R89continue_full_disc_feature3 | 35.119317 |
| quad/Q_S2 | 49 | R89continue_full_disc_physical_raw | 35.032535 |
| quad/Q_S2 | 49 | R89continue_full_disc_physical_bt | 35.032535 |
| quad/Q_S2 | 49 | R76_full_disc_conditional_bt | 35.529054 |
| quad/Q_S2 | 49 | R76_full_disc_wrong_physical_bt | 35.770023 |
| quad/Q_S2 | 49 | R89continue_full_disc_conditional_bt | 35.032535 |
| quad/Q_S2 | 49 | R89continue_full_disc_wrong_physical_bt | 35.399710 |
| quad/Q_S3 | 57 | R76_frozen | 42.244590 |
| quad/Q_S3 | 57 | R59 | 43.509770 |
| quad/Q_S3 | 57 | R80_raw | 44.265010 |
| quad/Q_S3 | 57 | R82_bt | 42.244590 |
| quad/Q_S3 | 57 | R76_full_disc_frozen | 42.244590 |
| quad/Q_S3 | 57 | R76_full_disc_feature3 | 42.244590 |
| quad/Q_S3 | 57 | R76_full_disc_physical_raw | 43.494260 |
| quad/Q_S3 | 57 | R76_full_disc_physical_bt | 42.244590 |
| quad/Q_S3 | 57 | R89continue_full_disc_frozen | 43.187756 |
| quad/Q_S3 | 57 | R89continue_full_disc_feature3 | 43.187756 |
| quad/Q_S3 | 57 | R89continue_full_disc_physical_raw | 44.313294 |
| quad/Q_S3 | 57 | R89continue_full_disc_physical_bt | 43.187756 |
| quad/Q_S3 | 57 | R76_full_disc_conditional_bt | 42.244590 |
| quad/Q_S3 | 57 | R76_full_disc_wrong_physical_bt | 42.244590 |
| quad/Q_S3 | 57 | R89continue_full_disc_conditional_bt | 43.187756 |
| quad/Q_S3 | 57 | R89continue_full_disc_wrong_physical_bt | 43.187756 |

## 正负案例与原图复核

按总体最大正/负贡献、旧TTA正例丢失、稳定高分和可恢复错误固定选页，并回访R90案例；病例只解释结果，不修改部署规则。原句、媒体、坐标、query索引和支持图只在本地私有图册；无法判断身份时保留UNKNOWN。换query与换物体、同物体重复框切换与实例混淆分别记录。

| 案例 | 平台 | 原R76 | 旧R80 | 新零步 | 同信息表示融合 | 物理raw | 物理BT | 条件BT |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CASE001 | quad | 0.000000 | 0.000000 | 0.573844 | 0.573844 | 0.573844 | 0.573844 | 0.573844 |
| CASE002 | drone | 0.000000 | 0.467545 | 0.467545 | 0.467545 | 0.467545 | 0.467545 | 0.467545 |
| CASE003 | drone | 0.144251 | 0.144251 | 0.000000 | 0.144251 | 0.000000 | 0.000000 | 0.000000 |
| CASE004 | drone | 0.194148 | 0.194148 | 0.087576 | 0.087576 | 0.194148 | 0.087576 | 0.087576 |
| CASE005 | quad | 0.233223 | 0.272370 | 0.233223 | 0.272370 | 0.272370 | 0.272370 | 0.272370 |
| CASE006 | drone | 0.452964 | 0.000000 | 0.452964 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| CASE007 | quad | 0.327987 | 0.327987 | 0.446062 | 0.327987 | 0.327987 | 0.327987 | 0.327987 |
| CASE008 | drone | 0.278824 | 0.278824 | 0.278824 | 0.250259 | 0.274253 | 0.278824 | 0.278824 |
| CASE009 | quad | 0.000000 | 0.000000 | 0.000000 | 0.174161 | 0.000000 | 0.000000 | 0.000000 |
| CASE010 | quad | 0.000000 | 0.537604 | 0.000000 | 0.000000 | 0.537604 | 0.000000 | 0.000000 |
| CASE011 | quad | 0.812701 | 0.812701 | 0.812701 | 0.812701 | 0.812701 | 0.812701 | 0.812701 |
| CASE012 | quad | 0.236613 | 0.236613 | 0.236613 | 0.236613 | 0.236613 | 0.236613 | 0.236613 |
| R90_CASE002 | quad | 0.327987 | 0.327987 | 0.446062 | 0.327987 | 0.327987 | 0.327987 | 0.327987 |
| R90_CASE003 | quad | 0.425162 | 0.425162 | 0.425162 | 0.425162 | 0.425162 | 0.425162 | 0.425162 |
| R90_CASE004 | drone | 0.255694 | 0.255694 | 0.255694 | 0.255694 | 0.255694 | 0.255694 | 0.255694 |
| R90_CASE005 | drone | 0.523128 | 0.413929 | 0.523128 | 0.523128 | 0.413929 | 0.523128 | 0.523128 |
| R90_CASE009 | quad | 0.000000 | 0.000000 | 0.573844 | 0.573844 | 0.573844 | 0.573844 | 0.573844 |
| R90_CASE012 | quad | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## 测量与实际成本

目标独立原生重放19488端点、23206项损失；概率最大误差0，特征0，参数态0，loss0，Top1不一致0。FP32容差事先定1e-6/参数1e-7，与方法接受阈值1e-10分开；没有因任务结果放宽。raw可能增加辅助目标，BT所有接受端点不增加。

十个source头训练合计197.671s，使用固定特征缓存，非整个3EED重训成本。DEV69臂集合预测390.680s，原生特征后缀3297次、条件梯度后缀2016次、条件试探后缀6546次；这些是整组实验成本，不能称单臂端到端时延。

源域第一次审计仅因两种等价平均的浮点归约顺序有7.1e-15pp差失败；修正审计口径匹配生产，逐例IoU、模型和容差未改，失败记录保留。工程阶段历史UNKNOWN元数据由占位改为复用原记录，工程封存未改、主实验开始前修正，没有数学预测变化。

边界：小规模source准备、部分额外物体标注、预测框点支持不纯、未有参照角色身份GT、固定A256上界和已暴露DEV均限制机制结论。额外source竞争监督不是无标签目标域TTA的身份真值。当前负结果不能穷尽所有竞争实例学习；当前正结果也不能由loss下降或一次开发均值升级为稳健新方法。FINAL未动，不追加source轮数/target超参/新损失或后续队列。

## 分阶段裁决与原图核验

本轮静态接口有开发增益，应保留为候选；物理TTA配方不采纳，条件恢复也未补上实例纠错。静态增益不能写成TTA增益。源域完整接口仍未超过query-only/no-local，目标开发集的正信号不能倒过来证明源域绑定前提或用来重选checkpoint。

R76静态接口相对原双向模型 **+0.420449pp**；R89匹配续训基座静态接口相对自身基座 **+0.407732pp**。两份基座均有静态净收益，但属于同一历史DEV，不能称独立重复确认。R89的物理BT虽超过同信息三观测读出，却低于自身完整输入零步；不能因它超过较弱直接融合就宣布参数更新成立。

| 端点 | 换query数 | 任务改善/恶化 | 留出恢复改善数 | 拟合目标下降数 | 平均实际alpha | 参数L2均值 |
|---|---:|---:|---:|---:|---:|---:|
| R76_full_disc_physical_raw | 53 | 11/14 | 219/336 | 40/336 | 0.500000 | 0.04046462 |
| R76_full_disc_physical_bt | 9 | 1/3 | 335/336 | 336/336 | 0.159953 | 0.01186035 |
| R76_full_disc_conditional_raw | 46 | 9/13 | 225/336 | 31/336 | 0.500000 | 0.04196919 |
| R76_full_disc_conditional_bt | 10 | 2/3 | 331/336 | 336/336 | 0.149007 | 0.01153274 |
| R76_full_disc_wrong_physical_raw | 51 | 12/13 | 134/336 | 63/336 | 0.500000 | 0.04451811 |
| R76_full_disc_wrong_physical_bt | 10 | 2/3 | 269/336 | 336/336 | 0.186797 | 0.01527873 |
| R89continue_full_disc_physical_raw | 48 | 16/14 | 219/336 | 48/336 | 0.500000 | 0.04046462 |
| R89continue_full_disc_physical_bt | 9 | 5/1 | 335/336 | 336/336 | 0.171677 | 0.01290987 |
| R89continue_full_disc_conditional_raw | 47 | 14/13 | 231/336 | 50/336 | 0.500000 | 0.04142505 |
| R89continue_full_disc_conditional_bt | 6 | 3/1 | 333/336 | 336/336 | 0.172514 | 0.01336526 |
| R89continue_full_disc_wrong_physical_raw | 60 | 19/19 | 134/336 | 74/336 | 0.500000 | 0.04451811 |
| R89continue_full_disc_wrong_physical_bt | 24 | 11/5 | 264/336 | 336/336 | 0.192697 | 0.01593132 |

主方法在未用于拟合与接受的第三遮挡恢复上335/336改善，最终grounding仍净退化。这比只检查拟合loss更强地说明：当前恢复目标确实能泛化到同观测的另一遮挡，但这种改善没有稳定指导候选身份/框的正确排序。它不证明新视角迁移，也不能据此断言所有恢复型TTA失败。

原图及独立原生重放共18页、16条唯一观测，包含两条重复回访。注意力/点权重是计算线索，未当作实例mask；恢复贡献覆盖车群、路面、植被和人物区域，缺少点级身份真值，不能定量断言全部改善都在背景。

| 案例 | 静态身份变化 | 零步→TTA身份变化 | 原图观察 |
|---|---|---|---|
| CASE001 | different_visible_objects | same_candidate_no_identity_change | 静态接口把远处小行人改为近景主体，原图支持不同实例纠错；TTA沿用新零步候选。局部权重集中主体附近但非实例真值。 |
| CASE002 | different_visible_objects | same_candidate_no_identity_change | 原3EED已选到近处车辆，R76选向后方另一车辆，新接口恢复原选择；TTA不改变候选。静态收益包含修复融合损害，并非全为纠正原3EED错误。 |
| CASE003 | UNKNOWN | same_candidate_no_identity_change | 新静态接口由右侧车辆附近转向树旁远处小框，图中缺少清晰主体支持；TTA保留错误候选。直接三观测融合恢复先前车辆框但IoU仍低。 |
| CASE004 | UNKNOWN | same_candidate_no_identity_change | 静态接口在画面右边缘相邻人形区域切换到较差框，遮挡和边缘裁切使实例归属不能可靠区分；TTA未再换框。静态误差身份归因保持UNKNOWN。 |
| CASE005 | same_candidate_no_identity_change | same_visible_object_duplicate_boxes | TTA与直接三观测读出都把同一可见SUV附近的框改为另一重复框，跨过Acc25阈值；没有图像依据把它计为换对物体。 |
| CASE006 | same_candidate_no_identity_change | different_visible_objects | 零步对准近处左侧车辆，TTA转向画面上沿远处区域，属于不同可见空间对象的误改；同信息直接融合也失败。恢复误差图广泛覆盖路面与车群，不能视为目标身份监督。 |
| CASE007 | same_visible_object_duplicate_boxes | same_visible_object_duplicate_boxes | TTA回到同一个左侧行人附近较差的重复框，未见换人物；新静态接口曾恢复更好框。错配BT保留更好框，不支持正确绑定必然带来此收益。 |
| CASE008 | same_candidate_no_identity_change | same_candidate_no_identity_change | TTA保留静态首选，而三观测融合换到同一车辆附近另一框；这是相对融合的保护，不是TTA修复静态错误。 |
| CASE009 | same_candidate_no_identity_change | same_candidate_no_identity_change | TTA与零步均选到树/车辆上方区域，三观测读出选择更接近左侧被遮挡车辆的框；主体在边缘且遮挡严重，对全部假设身份保持不确定。 |
| CASE010 | same_candidate_no_identity_change | same_candidate_no_identity_change | 主体与另一个背包行人分处不同位置。旧物理raw可选择左侧主体，新接口及其TTA保留另一人物，旧收益丢失；不能归因候选池无好框。 |
| CASE011 | same_candidate_no_identity_change | same_candidate_no_identity_change | 各状态均稳定选择同一可见行人，属于高IoU保留例；恢复图变化不应记为额外纠错。 |
| CASE012 | same_candidate_no_identity_change | same_candidate_no_identity_change | 各新状态选择同一受树干遮挡车辆的重复框，仍低于Acc25；原3EED有稍好框，候选恢复空间存在但新接口未利用。 |
| R90_CASE002 | same_visible_object_duplicate_boxes | same_visible_object_duplicate_boxes | TTA回到同一个左侧行人附近较差的重复框，未见换人物；新静态接口曾恢复更好框。错配BT保留更好框，不支持正确绑定必然带来此收益。 |
| R90_CASE003 | same_candidate_no_identity_change | same_candidate_no_identity_change | 回访近处滑板行人：原3EED选向另一人，R76已纠正；新接口与TTA保持同一主体，没有新增纠错。 |
| R90_CASE004 | same_candidate_no_identity_change | same_candidate_no_identity_change | 回访车辆：新静态与TTA均保持R76相同可见白色车辆，邻车在图上存在但本例无新增身份改变。 |
| R90_CASE005 | same_candidate_no_identity_change | same_candidate_no_identity_change | 回访遮挡车辆：原3EED误选前景邻车，R76已恢复远处目标；新静态/TTA不变，保住旧融合收益。 |
| R90_CASE009 | different_visible_objects | same_candidate_no_identity_change | 静态接口把远处小行人改为近景主体，原图支持不同实例纠错；TTA沿用新零步候选。局部权重集中主体附近但非实例真值。 |
| R90_CASE012 | same_candidate_no_identity_change | same_candidate_no_identity_change | 回访主体/邻车错绑：新接口仍在车辆位置，文字主体对应的另一物体未被选中；完整候选含更好框但排名靠后。恢复图未提供可信身份纠正。 |

几个区分性的例子：CASE001是静态接口从远处行人转回近处主体；CASE002包含恢复原3EED已正确、R76融合后丢失的车辆选择。CASE006中TTA从近处目标车辆转向远处区域；CASE007则是同一人物附近的重复框退化，不能统一称目标身份错。CASE005的唯一主TTA改善也是同一车辆重复框切换，同信息直接读出获得相同任务结果。CASE010保留了旧raw TTA可纠错、当前新接口及TTA没有纠错的负例。CASE003/004的静态误改身份判断保留UNKNOWN。

采取的结论是保留冻结融合和竞争实例静态判别作为开发候选，停止本轮“接回原物理恢复即可获得额外TTA”的配方，不增加epoch、步长、回溯门或损失。后续若重新授权研究，需要检验任务相关的竞争候选监督与自监督更新是否真正同向；本轮没有已经成立的无标签身份判定规则，不能从这些案例再补硬门。
