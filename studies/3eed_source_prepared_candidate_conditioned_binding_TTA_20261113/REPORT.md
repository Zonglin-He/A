# R89 — 源域准备的候选条件化角色空间绑定与TTA

## 最终综合裁决（同预算强对照与原图复核）

主要一步更新的同模型增量为+0.073904pp，但只来自1条改善、2条恶化、333条不变；唯一改善的全体正贡献为+0.132602pp，超过全部净增量。新冻结结构22.565283低于R7622.611596，也低于仅query残差22.623837。相同源域预算继续训练既有匹配头＋同样TTA达到22.828464，比完整新方法高0.189277pp。只做物理恢复为22.696590，比完整方法高0.057404pp；去掉对应项与其任务输出相同。去掉参照物后22.641382也未下降。

因此必须同时保留两项事实：**参数更新相对自身零步和同信息平均有微弱开发正增量；完整候选条件化绑定方案没有战胜更简单的同预算对照，不能据此采纳整套TTA或宣称可靠实体绑定已建立。** 数值增量检查通过不等于完整机制或稳健方法通过。

第三种未拟合遮挡的局部表示误差均值从0.00267514647变为0.00271294790，未显示平均恢复改善。加入对应项不应以“优化成功”作为正证据。

### 原图复核的代表性结论

已逐页查看11个不同样本及3个重复固定跟踪页。CASE005确实从其他区域回到目标车辆，但直接平均也成功，且纯3EED原先已有更高IoU；CASE002主要是同一车辆的重叠框切换，新增对应项撤回零步收益；CASE003/004显示新接口先把原有目标区域改为邻物或背景；CASE008直接平均选到前景目标，TTA仍选择背景人物；CASE011文字图已区分主体和参照类型，最终却仍选择参照车辆。CASE001静态收益来自无角色时的query后备路径，不能计作空间绑定成功。

这些案例解释总体差异，不用于修改规则。不能把所有坏例归为同一种语言错误，也不能把个别成功归给全部新组件。

### 整体按空间角色可用性分层

仅按无标签图中的角色是否存在分组，事后列性能；不筛预测、不重选参数。

| 有空间角色 | n | 方法 | 对照 | 子集Δpp | 对全336贡献pp | 改善/恶化 |
|---|---:|---|---|---:|---:|---:|
| False | 26 | candidate_frozen | frozen_bi | +1.005588 | +0.077813 | 2/1 |
| False | 26 | candidate_frozen | continue_frozen | +0.981156 | +0.075923 | 4/1 |
| False | 26 | candidate_frozen | query_frozen | +0.000000 | +0.000000 | 0/0 |
| False | 26 | candidate_tta1 | candidate_frozen | +0.000000 | +0.000000 | 0/0 |
| False | 26 | candidate_tta1 | continue_tta1 | +0.819058 | +0.063380 | 3/1 |
| False | 26 | candidate_tta1 | candidate_physical1 | +0.000000 | +0.000000 | 0/0 |
| True | 310 | candidate_frozen | frozen_bi | -0.134538 | -0.124127 | 5/12 |
| True | 310 | candidate_frozen | continue_frozen | -0.258833 | -0.238804 | 12/14 |
| True | 310 | candidate_frozen | query_frozen | -0.063465 | -0.058554 | 5/7 |
| True | 310 | candidate_tta1 | candidate_frozen | +0.080102 | +0.073904 | 1/2 |
| True | 310 | candidate_tta1 | continue_tta1 | -0.273848 | -0.252657 | 14/19 |
| True | 310 | candidate_tta1 | candidate_physical1 | -0.062218 | -0.057404 | 0/1 |

结论限定为冻结R76特征＋轻量接口、512条源训练及当前固定适应规则。更充分的源域准备、不同局部支持或任务对齐仍是竞争解释；本轮没有检验它们，也不继续按已暴露目标结果加步数/倍率/规则。当前不采纳完整方法，保留新架构、三个源域准备对照及全部正负例作为证据。

---

# R89 — 源域准备的候选条件化角色—空间证据与TTA

北京时间 2026-11-13 09:10。已完成三分支源域准备、DEV336全部29臂、原生状态/损失复核及保存IoU独立核算。全部为已暴露开发结果，未访问FINAL。

## 结果与裁决

固定主要方法 candidate_tta1 为 **22.639187%**，相对同模型零步 **+0.073904 pp**，相对同信息三读出平均 **+0.156981 pp**，相对R59 **+0.028071 pp**。新结构冻结分数为 **22.565283%**，相对R76 **-0.046314 pp**，相对同源域预算匹配头继续训练 **-0.162881 pp**，相对仅query残差对照 **-0.058554 pp**。

这轮实际训练了新读取/判别接口；并非仅复用未训练的Q/K/V。是否有任务增量由上述整体比较决定，不能由非零梯度、参数变化或恢复误差下降代替。

固定主要方法通过本轮全部预定同信息与基础增量比较，但仅保留为开发候选，不作稳健或独立确认。

## 结构与实际训练范围

保留原始Waymo-only epoch100、R76双向RGB–LiDAR结构、原始A256以及原生全局cross-encoder/decoder。最后文字→点云注意力作为观察接口，分别读取每个角色的三个局部空间假设；每个候选另有自己的主体点支持。假设先保留，直到候选条件化判别时才通过学习到的权重聚合。新残差head接收query、候选条件主体/参照物证据、完整上下文角色文字和相对几何，修正最终log匹配分数。零初始化末层使初始输出恢复R76。

源域轻量准备冻结R76特征，仅训练新增接口；继续训练对照仅更新既有匹配MLP。新增query-only对照不读取局部空间证据。三者相同512条Waymo训练、128条源域验证、四轮、相同完整输入＋两份遮挡grounding监督、AdamW lr1e-4/decay1e-4、batch1、clip1。训练/验证为94/82个场景，场景与帧交集均为0。三分支按源域验证等权均值选同一轮，最早最大值为第1轮，读取目标数据前已封存。

参数与计算并非完全匹配：继续训练184960个活跃参数；query-only分配72930、梯度活跃35585；完整新接口72930。每分支每轮512次更新，四轮2048次；完整和两份遮挡均参与源域监督。R76物理恢复头已继承，冻结特征上的常量辅助损失不被记作新接口训练。这里不是重训原始3EED、不是完整338k R76继续训练，也不是元辅助训练。

测试冻结源域准备的匹配头/新接口，仅更新R88沿用的102498个双向Adapter/交互参数。掩码88001/88002、物理恢复、对应、局部偏好、保持项与Armijo回溯固定；主要1步、预定次要3步，不根据目标GT验收或选步。完整+相同两份遮挡的概率平均是同信息零更新对照。

## Source VAL128：源域准备是否已学会更好的判别

源域初始化R76为 **47.385454%**；最终选定完整新接口为 **47.293237%**。训练本身不等于判别能力已成立，源域保留的负结果不能省略。

| 分支 | epoch | 完整mIoU % | 遮挡均值mIoU % | 完整训练loss | 遮挡训练loss |
|---|---:|---:|---:|---:|---:|
| continue | 1 | 47.118085 | 46.401568 | 3.880519 | 3.994785 |
| continue | 2 | 46.440242 | 45.653673 | 3.863087 | 3.970164 |
| continue | 3 | 46.149632 | 45.191191 | 3.852274 | 3.957540 |
| continue | 4 | 46.841085 | 45.987825 | 3.846707 | 3.945749 |
| query | 1 | 47.344616 | 46.417986 | 3.883490 | 4.019357 |
| query | 2 | 47.555019 | 46.656558 | 3.881774 | 4.009437 |
| query | 3 | 47.247504 | 46.606386 | 3.878755 | 4.002055 |
| query | 4 | 47.140930 | 46.054908 | 3.876048 | 3.997801 |
| candidate | 1 | 47.293237 | 46.280170 | 3.878908 | 4.012789 |
| candidate | 2 | 46.563463 | 45.459191 | 3.861428 | 3.974355 |
| candidate | 3 | 46.260659 | 44.914855 | 3.842378 | 3.938741 |
| candidate | 4 | 46.709266 | 45.258383 | 3.830187 | 3.917036 |

源域选定接口的事后撤回诊断（不用于重选轮次）：

| 读出 | 源域VAL128 mIoU % |
|---|---:|
| prepared_full | 47.293237 |
| frozen_R76 | 47.385454 |
| no_ref | 47.382678 |
| shared_mix | 47.272538 |
| query_only | 47.435623 |

## DEV336整体：所有方法及负例

两平台各168条，主指标两平台等权平均；五个历史场景分别报告。不以每场景必须上涨作门槛，不把336条当独立场景证据。全部UNKNOWN、无空间支持、回退和零更新都在分母。

| 方法 | Drone mIoU % | Quad mIoU % | 等权mIoU % | 等权Acc@25 % | 等权Acc@50 % |
|---|---:|---:|---:|---:|---:|
| source | 15.539892 | 28.120770 | 21.830331 | 40.773810 | 21.130952 |
| frozen_bi | 17.083330 | 28.139863 | 22.611596 | 41.071429 | 21.428571 |
| R59 | 16.385568 | 28.836664 | 22.611116 | 43.154762 | 21.726190 |
| R87_hard | 16.345995 | 29.374788 | 22.860391 | 42.261905 | 22.321429 |
| R88_direct | 17.198137 | 28.163165 | 22.680651 | 41.666667 | 21.428571 |
| R88_TTA1 | 16.811007 | 28.162993 | 22.487000 | 41.071429 | 21.428571 |
| R88_mean3 | 16.807529 | 28.623186 | 22.715357 | 41.666667 | 21.428571 |
| continue_frozen | 17.308216 | 28.148111 | 22.728164 | 41.071429 | 21.130952 |
| continue_mean3 | 17.147928 | 28.413035 | 22.780481 | 41.369048 | 21.130952 |
| continue_tta1 | 17.559059 | 28.097869 | 22.828464 | 41.369048 | 21.130952 |
| query_frozen | 16.992460 | 28.255214 | 22.623837 | 41.369048 | 21.130952 |
| query_mean3 | 16.664465 | 28.641204 | 22.652834 | 41.369048 | 20.833333 |
| query_tta1 | 16.980393 | 28.255214 | 22.617804 | 41.369048 | 21.130952 |
| candidate_frozen | 17.061770 | 28.068796 | 22.565283 | 41.071429 | 21.130952 |
| candidate_mean3 | 16.526323 | 28.438088 | 22.482206 | 40.773810 | 21.130952 |
| candidate_tta1 | 17.209577 | 28.068796 | 22.639187 | 41.071429 | 21.130952 |
| candidate_tta3 | 17.328775 | 28.068796 | 22.698785 | 41.369048 | 21.130952 |
| no_ref_frozen | 17.212167 | 28.068796 | 22.640481 | 41.071429 | 21.130952 |
| no_ref_tta1 | 17.213967 | 28.068796 | 22.641382 | 41.071429 | 21.130952 |
| shared_mix_frozen | 16.885471 | 28.255214 | 22.570343 | 41.369048 | 21.130952 |
| shared_mix_tta1 | 16.888645 | 28.255214 | 22.571929 | 41.369048 | 21.130952 |
| wrong_space_frozen | 16.870182 | 28.255214 | 22.562698 | 41.071429 | 21.130952 |
| wrong_space_tta1 | 16.884767 | 28.255214 | 22.569990 | 41.369048 | 21.130952 |
| role_swap_frozen | 16.909571 | 28.068796 | 22.489183 | 40.773810 | 21.130952 |
| role_swap_tta1 | 17.057379 | 28.068796 | 22.563087 | 40.773810 | 21.130952 |
| no_residual_frozen | 17.083330 | 28.139863 | 22.611596 | 41.071429 | 21.428571 |
| no_residual_tta1 | 17.080628 | 28.162993 | 22.621811 | 41.369048 | 21.428571 |
| candidate_physical1 | 17.324385 | 28.068796 | 22.696590 | 41.369048 | 21.130952 |
| candidate_no_correspondence1 | 17.324385 | 28.068796 | 22.696590 | 41.369048 | 21.130952 |

## 组件与配对变化

以下为完整方法减对照。no_ref同时撤回参照读取与相应上下文/几何；shared_mix同时撤回候选专属主体支持和候选/几何条件，是组合撤回，不当作单一纯因果消融。wrong_space置换局部空间点支持，保留原生双向RGB–LiDAR输入对应和观测目标；role_swap仅置换全局角色支持，候选自己的主体支持保留。它们不是真实身份GT。

| 组件/比较 | 方法 | 对照 | 平台 | ΔmIoU pp | 改善/恶化/不变 | 最大单例正/负贡献pp |
|---|---|---|---|---:|---:|---:|
| source_prepared_interface | candidate_frozen | frozen_bi | balanced | -0.046314 | 7/13/316 | +0.094792/-0.102271 |
| source_prepared_interface | candidate_frozen | frozen_bi | drone | -0.021560 | 5/9/154 | +0.189584/-0.204542 |
| source_prepared_interface | candidate_frozen | frozen_bi | quad | -0.071067 | 2/4/162 | +0.096395/-0.176049 |
| same_source_budget_matching_continuation | candidate_frozen | continue_frozen | balanced | -0.162881 | 16/15/305 | +0.048198/-0.102271 |
| same_source_budget_matching_continuation | candidate_frozen | continue_frozen | drone | -0.246447 | 12/9/147 | +0.094759/-0.204542 |
| same_source_budget_matching_continuation | candidate_frozen | continue_frozen | quad | -0.079316 | 4/6/158 | +0.096395/-0.176794 |
| query_only_source_prepared_control | candidate_frozen | query_frozen | balanced | -0.058554 | 5/7/324 | +0.076099/-0.102271 |
| query_only_source_prepared_control | candidate_frozen | query_frozen | drone | +0.069310 | 5/5/158 | +0.152199/-0.204542 |
| query_only_source_prepared_control | candidate_frozen | query_frozen | quad | -0.186419 | 0/2/166 | +0.000000/-0.176049 |
| parameter_update | candidate_tta1 | candidate_frozen | balanced | +0.073904 | 1/2/333 | +0.132602/-0.057404 |
| parameter_update | candidate_tta1 | candidate_frozen | drone | +0.147808 | 1/2/165 | +0.265204/-0.114807 |
| parameter_update | candidate_tta1 | candidate_frozen | quad | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| same_information_three_read_control | candidate_tta1 | candidate_mean3 | balanced | +0.156981 | 18/15/303 | +0.134811/-0.170787 |
| same_information_three_read_control | candidate_tta1 | candidate_mean3 | drone | +0.683254 | 10/6/152 | +0.269621/-0.048683 |
| same_information_three_read_control | candidate_tta1 | candidate_mean3 | quad | -0.369293 | 8/9/151 | +0.080098/-0.341574 |
| old_strong_no_update_control | candidate_tta1 | R59 | balanced | +0.028071 | 43/63/230 | +0.234305/-0.168124 |
| old_strong_no_update_control | candidate_tta1 | R59 | drone | +0.824010 | 25/39/104 | +0.468611/-0.335152 |
| old_strong_no_update_control | candidate_tta1 | R59 | quad | -0.767868 | 18/24/126 | +0.292609/-0.336248 |
| secondary_three_steps | candidate_tta3 | candidate_tta1 | balanced | +0.059599 | 2/0/334 | +0.057404/+0.000000 |
| secondary_three_steps | candidate_tta3 | candidate_tta1 | drone | +0.119197 | 2/0/166 | +0.114807/+0.000000 |
| secondary_three_steps | candidate_tta3 | candidate_tta1 | quad | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| correspondence_plus_rank | candidate_tta1 | candidate_physical1 | balanced | -0.057404 | 0/1/335 | +0.000000/-0.057404 |
| correspondence_plus_rank | candidate_tta1 | candidate_physical1 | drone | -0.114807 | 0/1/167 | +0.000000/-0.114807 |
| correspondence_plus_rank | candidate_tta1 | candidate_physical1 | quad | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| local_correspondence | candidate_tta1 | candidate_no_correspondence1 | balanced | -0.057404 | 0/1/335 | +0.000000/-0.057404 |
| local_correspondence | candidate_tta1 | candidate_no_correspondence1 | drone | -0.114807 | 0/1/167 | +0.000000/-0.114807 |
| local_correspondence | candidate_tta1 | candidate_no_correspondence1 | quad | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| no_ref | candidate_frozen | no_ref_frozen | balanced | -0.075198 | 1/1/334 | +0.057404/-0.132602 |
| no_ref | candidate_frozen | no_ref_frozen | drone | -0.150397 | 1/1/166 | +0.114807/-0.265204 |
| no_ref | candidate_frozen | no_ref_frozen | quad | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| no_ref | candidate_tta1 | no_ref_tta1 | balanced | -0.002195 | 0/1/335 | +0.000000/-0.002195 |
| no_ref | candidate_tta1 | no_ref_tta1 | drone | -0.004390 | 0/1/167 | +0.000000/-0.004390 |
| no_ref | candidate_tta1 | no_ref_tta1 | quad | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| shared_mix | candidate_frozen | shared_mix_frozen | balanced | -0.005060 | 6/5/325 | +0.076099/-0.102271 |
| shared_mix | candidate_frozen | shared_mix_frozen | drone | +0.176298 | 6/3/159 | +0.152199/-0.204542 |
| shared_mix | candidate_frozen | shared_mix_frozen | quad | -0.186419 | 0/2/166 | +0.000000/-0.176049 |
| shared_mix | candidate_tta1 | shared_mix_tta1 | balanced | +0.067257 | 6/6/324 | +0.132602/-0.102271 |
| shared_mix | candidate_tta1 | shared_mix_tta1 | drone | +0.320933 | 6/4/158 | +0.265204/-0.204542 |
| shared_mix | candidate_tta1 | shared_mix_tta1 | quad | -0.186419 | 0/2/166 | +0.000000/-0.176049 |
| wrong_space | candidate_frozen | wrong_space_frozen | balanced | +0.002585 | 6/6/324 | +0.076099/-0.102271 |
| wrong_space | candidate_frozen | wrong_space_frozen | drone | +0.191588 | 6/4/158 | +0.152199/-0.204542 |
| wrong_space | candidate_frozen | wrong_space_frozen | quad | -0.186419 | 0/2/166 | +0.000000/-0.176049 |
| wrong_space | candidate_tta1 | wrong_space_tta1 | balanced | +0.069196 | 6/6/324 | +0.132602/-0.102271 |
| wrong_space | candidate_tta1 | wrong_space_tta1 | drone | +0.324811 | 6/4/158 | +0.265204/-0.204542 |
| wrong_space | candidate_tta1 | wrong_space_tta1 | quad | -0.186419 | 0/2/166 | +0.000000/-0.176049 |
| role_swap | candidate_frozen | role_swap_frozen | balanced | +0.076099 | 1/0/335 | +0.076099/+0.000000 |
| role_swap | candidate_frozen | role_swap_frozen | drone | +0.152199 | 1/0/167 | +0.152199/+0.000000 |
| role_swap | candidate_frozen | role_swap_frozen | quad | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| role_swap | candidate_tta1 | role_swap_tta1 | balanced | +0.076099 | 1/0/335 | +0.076099/+0.000000 |
| role_swap | candidate_tta1 | role_swap_tta1 | drone | +0.152199 | 1/0/167 | +0.152199/+0.000000 |
| role_swap | candidate_tta1 | role_swap_tta1 | quad | +0.000000 | 0/0/168 | +0.000000/+0.000000 |
| no_residual | candidate_frozen | no_residual_frozen | balanced | -0.046314 | 7/13/316 | +0.094792/-0.102271 |
| no_residual | candidate_frozen | no_residual_frozen | drone | -0.021560 | 5/9/154 | +0.189584/-0.204542 |
| no_residual | candidate_frozen | no_residual_frozen | quad | -0.071067 | 2/4/162 | +0.096395/-0.176049 |
| no_residual | candidate_tta1 | no_residual_tta1 | balanced | +0.017376 | 8/14/314 | +0.132602/-0.102271 |
| no_residual | candidate_tta1 | no_residual_tta1 | drone | +0.128949 | 6/10/152 | +0.265204/-0.204542 |
| no_residual | candidate_tta1 | no_residual_tta1 | quad | -0.094198 | 2/4/162 | +0.096395/-0.176049 |

## 五个场景（已暴露，不作独立确认）

| 平台/场景 | n | 方法 | mIoU % |
|---|---:|---|---:|
| drone/D_S1 | 106 | frozen_bi | 13.449026 |
| drone/D_S1 | 106 | R59 | 11.761612 |
| drone/D_S1 | 106 | continue_frozen | 14.078596 |
| drone/D_S1 | 106 | query_frozen | 13.391830 |
| drone/D_S1 | 106 | candidate_frozen | 13.534854 |
| drone/D_S1 | 106 | candidate_mean3 | 12.723383 |
| drone/D_S1 | 106 | candidate_tta1 | 13.769115 |
| drone/D_S2 | 62 | frozen_bi | 23.296817 |
| drone/D_S2 | 62 | R59 | 24.291041 |
| drone/D_S2 | 62 | continue_frozen | 22.829826 |
| drone/D_S2 | 62 | query_frozen | 23.148376 |
| drone/D_S2 | 62 | candidate_frozen | 23.091659 |
| drone/D_S2 | 62 | candidate_mean3 | 23.028124 |
| drone/D_S2 | 62 | candidate_tta1 | 23.091659 |
| quad/Q_S1 | 62 | frozen_bi | 10.447889 |
| quad/Q_S1 | 62 | R59 | 9.632116 |
| quad/Q_S1 | 62 | continue_frozen | 10.448987 |
| quad/Q_S1 | 62 | query_frozen | 10.848428 |
| quad/Q_S1 | 62 | candidate_frozen | 10.343294 |
| quad/Q_S1 | 62 | candidate_mean3 | 10.549466 |
| quad/Q_S1 | 62 | candidate_tta1 | 10.343294 |
| quad/Q_S2 | 49 | frozen_bi | 34.118085 |
| quad/Q_S2 | 49 | R59 | 36.067581 |
| quad/Q_S2 | 49 | continue_frozen | 34.144976 |
| quad/Q_S2 | 49 | query_frozen | 34.006772 |
| quad/Q_S2 | 49 | candidate_frozen | 34.006772 |
| quad/Q_S2 | 49 | candidate_mean3 | 35.137328 |
| quad/Q_S2 | 49 | candidate_tta1 | 34.006772 |
| quad/Q_S3 | 57 | frozen_bi | 42.244590 |
| quad/Q_S3 | 57 | R59 | 43.509770 |
| quad/Q_S3 | 57 | continue_frozen | 42.244590 |
| quad/Q_S3 | 57 | query_frozen | 42.244590 |
| quad/Q_S3 | 57 | candidate_frozen | 42.244590 |
| quad/Q_S3 | 57 | candidate_mean3 | 42.136893 |
| quad/Q_S3 | 57 | candidate_tta1 | 42.244590 |

## Good / bad cases与读取机制

按全体固定规则选择最大收益/损伤、绝对高IoU和可恢复错误，不用案例重新改方法。另保留R88三个固定案例。公开仅匿名标量，原句/图像/点云/框索引/坐标/特征均留本地。具体图页核验见本地CASE_REVIEW及atlas；框IoU、框间重叠不能单独证明换了物体。

| 案例 | 平台 | 原R76 | 新冻结 | 同三读平均 | 一步 | 三步 | 冻结/一步选框重叠 |
|---|---|---:|---:|---:|---:|---:|---:|
| CASE001 | drone | 0.000000 | 0.318500 | 0.000000 | 0.318500 | 0.318500 | 1.000001 |
| CASE002 | drone | 0.217958 | 0.410834 | 0.217958 | 0.217958 | 0.410834 | 0.270812 |
| CASE003 | drone | 0.343630 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000000 |
| CASE004 | quad | 0.295762 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.999999 |
| CASE005 | drone | 0.000000 | 0.000000 | 0.445543 | 0.445543 | 0.445543 | 0.000000 |
| CASE006 | drone | 0.004350 | 0.004350 | 0.004350 | 0.000000 | 0.000000 | 0.347340 |
| CASE007 | drone | 0.452964 | 0.452964 | 0.000000 | 0.452964 | 0.452964 | 1.000001 |
| CASE008 | quad | 0.000000 | 0.000000 | 0.573844 | 0.000000 | 0.000000 | 1.000001 |
| CASE009 | quad | 0.443224 | 0.443224 | 0.617932 | 0.443224 | 0.443224 | 0.999998 |
| CASE010 | quad | 0.812701 | 0.812701 | 0.812701 | 0.812701 | 0.812701 | 1.000000 |
| CASE011 | quad | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.000001 |

候选条件化读取的计算诊断：222/336条具有至少一个参照角色；主体读取权重的候选间标准差均值为 0.11316214。权重确实可以随候选改变，但这一统计不是绑定正确率。完整逐条角色覆盖、权重变化、残差范围与所有更新轨迹均保留。

## 审计、成本与边界

源域缓存640条2560次原生suffix前向；源域三分支训练合计 46.391s（缓存轻量头训练，不能表述为整模型源训练耗时）。源域独立1152次原生重放，输入特征误差为0，原生与现场缓存头误差为0；保存FP32读出最大差 9.62043350539e-09，没有第一名变化。先前1e-10绝对概率阈值失败已保留；最终记录1e-6容差并额外强制特征相等、现场读出一致及所有Top1相同。

目标全部29臂合集含复用基线，原生main调用 49878，objective调用 14498，目标预测合集 754.748s、峰值已分配显存 1549454336bytes。该合集总时间含保存与所有对照，不是假定单方法延迟。新接口每次前向使用冻结3EED的既有语言/RGB–LiDAR表示，无外部视觉专家。

原生独立复核 7392 个读出、24192 个实际loss标量，最大概率误差 0、路由误差 0、loss误差 0；验收后总loss增加数为0。全部原A256坐标与读出、源域准备头在TTA中冻结、原始checkpoint未改均核验。预测先封存，评分仅连接历史保存A256 IoU，未新增访问目标标注。

新增空间假设仍来自有限原始候选和注意力；候选自身框内点也可能包含背景。源域grounding监督不提供参照物身份真值。当前轻量准备不等同于论文中的端到端语义/几何联合训练；负结果限定在本实现、源域样本量和固定TTA设置，不能排除缺少更充分准备或任务对齐条件。

不触碰FINAL，不重新搜索目标阈值，不新增更大模型/元学习/调参轮次。保留完整结果作为后续训练准备或监督设计的依据，当前主方法的采纳由DECISION的配对增量决定。

## 证据入口

- `DEV336/`：29臂全部匿名任务值、所有有序负比较、分场景结果。
- `results/`：源域12轮结果、组件、所有更新、覆盖与good/bad匿名标量。
- `own_code/`：本轮实现、工程检查、native/scalar/public审计。
- 本地私有atlas与原始状态/中间特征不公开。

## 保留的历史三采样基线

除本轮29臂外，原Frozen ABC直接复用R88在相同336条上的封存结果，不运行新模型、不改方法或源域选择。新增全体逐条与全部29臂对比均保留，包含所有负差。

| 平台 | n | Frozen ABC mIoU % | 主要方法−ABC pp | 改善/恶化 |
|---|---:|---:|---:|---:|
| balanced | 336 | 22.302460 | +0.336726 | 44/62 |
| drone | 168 | 15.736332 | +1.473245 | 27/36 |
| quad | 168 | 28.868588 | -0.799793 | 17/26 |

## 局部读取范围的额外边界

本轮沿用0.05全局底权重，并非严格实例mask。基于头平均原生注意力估计的候选自身支持内质量比例，先对每条候选取均值、再对样本取均值，为 0.191905。这是头平均输入的描述性代理，并不等于逐head归一化后实际向量的精确占比，更不是实例纯度。它保留了原方案的全局安全通路，也说明“候选专属局部向量”不能被解释为只含该物体。后续不能把这个实现的阴性结果推广到所有局部语义绑定结构。

本地案例图册与人工查看记录：`crossplatform/round89/atlas_PRIVATE/index.html`、`research/round89/CASE_REVIEW.md`。它们仅用于封存后的机制解释，不进入方法选择。公开包不含原句、原图或私有向量。

本轮主要一步相对自身冻结的3条IoU变化全部位于同一个Drone历史场景；Quad一步与自身零步的全部IoU均不变。该事实限制了跨采集条件稳定性的解释，不作为逐场景必须上涨的采用门槛。
