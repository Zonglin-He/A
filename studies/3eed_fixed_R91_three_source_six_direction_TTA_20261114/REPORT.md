# R97 — 固定R91方法的三个source、六方向TTA

本轮固定用户选定的R91 R76_full_disc双向结构及原物理恢复一步回溯physical_bt。未采用已停止的R95 LM，不新增损失、视觉专家、更新层或目标域搜索；原始source100参数与每输入的A256框固定。每输入恢复其source准备状态。FINAL帧在清单生成前排除。

## 六方向固定主结果

| source→target | n | 原始source | 准备后零步 | 同信息概率均值 | 同信息表示均值 | 固定TTA | TTA−零步 pp | TTA−概率均值 pp |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| waymo→drone | 2792 | 8.832240 | 9.285471 | 9.302691 | 9.292908 | 9.369391 | +0.083920 | +0.066700 |
| waymo→quad | 2736 | 17.588082 | 18.750207 | 18.862412 | 18.862412 | 18.791360 | +0.041153 | -0.071052 |
| drone→waymo | 3782 | 14.142348 | 14.609560 | 14.624784 | 14.640926 | 14.572323 | -0.037236 | -0.052461 |
| drone→quad | 2736 | 13.077292 | 12.901516 | 12.961345 | 12.966480 | 12.837646 | -0.063870 | -0.123699 |
| quad→waymo | 3782 | 10.829191 | 10.898764 | 10.823982 | 10.862020 | 10.873899 | -0.024865 | +0.049917 |
| quad→drone | 2792 | 6.055598 | 6.725807 | 6.555154 | 6.555154 | 6.705594 | -0.020213 | +0.150440 |

## 等权汇总与目标平台

六方向各占1/6；各目标平台由两个入向source等权平均。另保留所有逐方向、逐场景和逐例值，不用样本量较大的Waymo掩盖其他平台。

| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |
|---|---|---:|---:|---:|
| all_six_equal | original_source | 11.754125 | 22.475706 | 6.148478 |
| all_six_equal | prepared_frozen | 12.195221 | 22.844980 | 5.750723 |
| all_six_equal | direct_mean3 | 12.188395 | 22.859984 | 5.803582 |
| all_six_equal | direct_feature3 | 12.196650 | 22.888232 | 5.809673 |
| all_six_equal | physical_raw | 12.196428 | 22.926516 | 5.765605 |
| all_six_equal | physical_bt | 12.191702 | 22.859922 | 5.725441 |
| all_six_equal | wrong_physical_raw | 12.162627 | 22.704803 | 5.840389 |
| all_six_equal | wrong_physical_bt | 12.179638 | 22.774480 | 5.752408 |
| waymo | original_source | 12.485769 | 26.070862 | 6.795346 |
| waymo | prepared_frozen | 12.754162 | 26.692226 | 6.041777 |
| waymo | direct_mean3 | 12.724383 | 26.626124 | 6.107879 |
| waymo | direct_feature3 | 12.751473 | 26.692226 | 6.107879 |
| waymo | physical_raw | 12.648186 | 26.480698 | 5.975674 |
| waymo | physical_bt | 12.723111 | 26.665785 | 6.002115 |
| waymo | wrong_physical_raw | 12.619829 | 26.414595 | 5.962454 |
| waymo | wrong_physical_bt | 12.734829 | 26.718667 | 6.028556 |
| drone | original_source | 7.443919 | 14.272923 | 2.238539 |
| drone | prepared_frozen | 8.005639 | 14.613181 | 2.292264 |
| drone | direct_mean3 | 7.928922 | 14.541547 | 2.238539 |
| drone | direct_feature3 | 7.924031 | 14.523639 | 2.238539 |
| drone | physical_raw | 7.987108 | 14.648997 | 2.238539 |
| drone | physical_bt | 8.037493 | 14.702722 | 2.274355 |
| drone | wrong_physical_raw | 7.956255 | 14.452006 | 2.238539 |
| drone | wrong_physical_bt | 8.008033 | 14.631089 | 2.292264 |
| quad | original_source | 15.332687 | 27.083333 | 9.411550 |
| quad | prepared_frozen | 15.825862 | 27.229532 | 8.918129 |
| quad | direct_mean3 | 15.911878 | 27.412281 | 9.064327 |
| quad | direct_feature3 | 15.914446 | 27.448830 | 9.082602 |
| quad | physical_raw | 15.953991 | 27.649854 | 9.082602 |
| quad | physical_bt | 15.814503 | 27.211257 | 8.899854 |
| quad | wrong_physical_raw | 15.911796 | 27.247807 | 9.320175 |
| quad | wrong_physical_bt | 15.796052 | 26.973684 | 8.936404 |

## 改善、损伤与全部负比较

| 方向 | 对照 | ΔmIoU pp | 改善/恶化/不变 | Acc25救回/破坏 |
|---|---|---:|---:|---:|
| waymo→drone | original_source | +0.537151 | 417/399/1976 | 118/127 |
| waymo→drone | prepared_frozen | +0.083920 | 31/27/2734 | 11/4 |
| waymo→drone | direct_mean3 | +0.066700 | 59/54/2679 | 19/16 |
| waymo→drone | direct_feature3 | +0.076483 | 60/54/2678 | 20/16 |
| waymo→drone | wrong_physical_bt | +0.054885 | 32/25/2735 | 7/5 |
| waymo→quad | original_source | +1.203278 | 527/355/1854 | 204/165 |
| waymo→quad | prepared_frozen | +0.041153 | 16/10/2710 | 6/3 |
| waymo→quad | direct_mean3 | -0.071052 | 40/50/2646 | 8/21 |
| waymo→quad | direct_feature3 | -0.071052 | 40/50/2646 | 8/21 |
| waymo→quad | wrong_physical_bt | +0.034126 | 25/9/2702 | 10/2 |
| drone→waymo | original_source | +0.429976 | 594/546/2642 | 224/190 |
| drone→waymo | prepared_frozen | -0.037236 | 20/28/3734 | 5/6 |
| drone→waymo | direct_mean3 | -0.052461 | 87/101/3594 | 26/27 |
| drone→waymo | direct_feature3 | -0.068603 | 82/99/3601 | 24/27 |
| drone→waymo | wrong_physical_bt | -0.047950 | 28/40/3714 | 3/9 |
| drone→quad | original_source | -0.239645 | 510/514/1712 | 133/165 |
| drone→quad | prepared_frozen | -0.063870 | 30/28/2678 | 5/9 |
| drone→quad | direct_mean3 | -0.123699 | 78/67/2591 | 16/14 |
| drone→quad | direct_feature3 | -0.128834 | 76/64/2596 | 14/14 |
| drone→quad | wrong_physical_bt | +0.002776 | 51/36/2649 | 10/5 |
| quad→waymo | original_source | +0.044708 | 595/640/2547 | 274/263 |
| quad→waymo | prepared_frozen | -0.024865 | 7/13/3762 | 2/3 |
| quad→waymo | direct_mean3 | +0.049917 | 92/99/3591 | 50/46 |
| quad→waymo | direct_feature3 | +0.011879 | 87/98/3597 | 46/45 |
| quad→waymo | wrong_physical_bt | +0.024515 | 5/3/3774 | 3/1 |
| quad→drone | original_source | +0.649996 | 608/514/1670 | 169/136 |
| quad→drone | prepared_frozen | -0.020213 | 19/20/2753 | 1/3 |
| quad→drone | direct_mean3 | +0.150440 | 65/36/2691 | 14/8 |
| quad→drone | direct_feature3 | +0.150440 | 65/36/2691 | 14/8 |
| quad→drone | wrong_physical_bt | +0.004034 | 10/5/2777 | 3/1 |

## 固定方法与source准备

三个原始模型沿用R96的Waymo-only/Drone-only/Quad-only epoch100。Waymo配套模块复用已有R76/R91参数；Drone/Quad不能加载Waymo准备权重，各自只用source TRAIN512/VAL128按同一结构和损失准备：双向融合8epoch，AdamW3e-4；冻结融合后full_disc接口4epoch，AdamW1e-4。均clip1、weight_decay1e-4，sourceVAL最早最大值选择；无target选择。只迁移当前full_disc分支，未重跑原R91十个消融分支的联合epoch选择，故新增source的选epoch统计口径是单分支sourceVAL。本差异在目标评分前注册。

源附加物体7D框和主体9D框并存，依官方box_util的xyz/lwh/yaw[,pitch,roll]定义补齐缺失角度后做原native变换；没有改变目标评价几何。全部source额外物体标签仅服务source竞争训练，不进入TTA。源与目标在不同方向中的身份按各模型独立定义，不能把另一方向的source标签当当前方向测试监督。

- drone: fusion epoch1, head epoch1, 6144次训练尝试，5924次有效优化器更新，耗时534.313s；原始100epoch模型未变，无角色信号的尝试保留。
- quad: fusion epoch2, head epoch1, 6144次训练尝试，6064次有效优化器更新，耗时487.037s；原始100epoch模型未变，无角色信号的尝试保留。

TTA保持原102498参数范围；完整输入及两固定互补遮挡76001/76002共用点云、RGB、标定和文字。物理loss为0.1点表征＋0.1RGB＋0.01图像单支深度；SGD clip1、初始0.5，Armijo1e-4/tol1e-10，最多9个二分幅度。判别接口参与实际读出和保持验收，但首步物理梯度仍沿原R91接口关闭路径；本轮不修此已知限制。错配绑定raw/BT与无回溯raw完整保留，不在看到目标GT后升级为主方法。

概率均值和表示均值使用同一完整输入和相同两遮挡，不更新参数；与TTA具有相同观测，避免仅以裸source宣称适应价值。UNKNOWN、无可见支持、无有效更新、回退和所有零IoU样本均在总分分母内。

## 采样、审计与适用范围

使用表达式级固定点采样，所有实验臂共用；R96用官方batch采样，因而其原表仅作为同清单历史参照。本轮原始source在匹配输入上重算，不把采样变化混入TTA增量。RGB物理缓存只在seed/sample哈希完全一致时复用，当前source自己的query anchor indices重新绑定；抽查原始媒体重放不使用该缓存。

原R91已存Drone/Quad案例的7个准备后读出由新执行入口重放：概率最大误差0，Top1完全一致。正式全部六方向预测封存并完成原生抽查后才评分；主选框IoU另用通用3D凸体交核对，完整标量由独立汇总程序复核。

Drone仅2个验证场景，Quad仅3个，Waymo93个；18,620个有向任务输入不是18,620个独立场景。来源于已暴露验证范围，不宣称新独立测试或总体显著性。未查看FINAL输入/评分；未进行按平台选步幅、按负例删样本或后验切换方法。

## 成本

| 方向 | 完整执行秒 | GPU峰值bytes |
|---|---:|---:|
| waymo_to_drone | 1313.142 | 1004657664 |
| waymo_to_quad | 1497.623 | 1002399744 |
| drone_to_waymo | 1779.789 | 1004680704 |
| drone_to_quad | 1078.247 | 1002363904 |
| quad_to_waymo | 1978.995 | 1004681728 |
| quad_to_drone | 1411.635 | 1005067776 |

## 固定配方裁决

固定主方法相对自身零步-0.003518pp，相对同信息概率均值+0.003308pp，相对同信息表示均值-0.004948pp。等权六方向同时超过三项主要对照=False。该判断只涉及本固定配方和已有范围；不按控制组或单域最佳结果替换主配方，也不把辅助loss下降当grounding增益。

## 结果解释与记账说明

固定TTA只有Waymo→Drone同时超过自身零步与两项同信息直接读出；新增的Drone→Waymo/Quad与Quad→Waymo/Drone四方向均低于自身准备后零步。全六方向平均相对自身零步−0.003518pp，相对更强的同信息表示均值−0.004948pp，不能将对较弱概率均值的+0.003308pp解释成稳定适应收益。

源域配套准备相对同采样原始source的mIoU有+0.441096pp，但Acc50从6.148478%降为5.750723%。这属于源域新增模块整体的静态变化，不是测试时适应收益，也不能单独归因于某个融合组件。

代码中的legacy actual_parameter_L2字段记录提案向量范数，非逐元素FP32量化后的真实权重差。额外72个封存端点的原生写回复核确认所有抽查非零提案确实改变参数；精确范数与提案范数均在NATIVE_AUDIT中披露。更新数量汇总以非零提案计数。每方向耗时覆盖全部八实验臂及其共享计算，不是单方法时延。

不采用本固定物理恢复TTA作为六方向通用改进；保留全部模型、准备后冻结参照、直接读出、正负改变和错配控制。未根据本轮目标成绩改超参数、重新选择source checkpoint、切换为raw或删掉负方向。
