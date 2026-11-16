# R113 — 固定原生几何路径的 RGB 桥物理适应

本轮沿用三个 source100 和已有 R91 参数，不增加专家、source 准备、伪目标或辅助 loss。数据为历史暴露的 DEV768/384条独立表达、六方向；原 A256 框、六层位置、文字视角均固定。44个保护媒体别名仅复用既有冻结回退，保留总分分母。FINAL 未访问。

## 为什么改变更新范围

R112 撤回 point Adapter 与早期 RGB 注入后，冻结组合达到17.265658%，但这是开发选出的静态对照，不是 TTA。本轮固定这两个旁路，保护原生点云—文字表示，仅允许原有 point_to_image 的 Q/K/V/O 与 gate_I（45313参数）在目标输入上更新；扩大 image Adapter 的47441参数和原102498全范围作为比较。源 RGB CNN、late、匹配头、竞争接口和原始3EED均冻结。结构仍包含 RGB 分支、几何到RGB和late RGB到query；早期RGB到point路径被固定关闭，不宣称两条早期方向都仍活跃。

## 信号、接受和信息对照

监督仍为源域已准备的两个互补遮挡76001/76002上的0.1点特征＋0.1RGB＋0.01单图像深度恢复。工程反向检查证明深度项对主桥参数梯度精确为零，只是保留常数项。没有实例真值伪标签；该信号能否服务语言选框必须看任务结果。两种起点的完整102498参数梯度用于建立同范数父方向；主方法仅写45313参数，不夸称只付出小模块反向成本。一次clip1/0.5提议，固定9个二分回溯，物理目标＋原KL/表示保持、Armijo1e-4，目标GT不参与任何接受。

各输入先完成全部七组更新选择，再读取B/C对齐观测；C未参与拟合或回溯。当前A的RGB视角不变，C只是已有实测LiDAR观测，缺失时保留A回退；不称新RGB视角或独立确认集。原始full-all的当前A结果和步幅须复现R97。相同两遮挡的冻结概率均值/表示均值、noLate(r-only)、lateOnly和纯原生表示均值保留，防止只比弱裸source。

## 工程与审计边界

12条真实输入检查实际写参数、完整模型与缓存基础路径重放位等、冻结状态恢复、独立CPU匹配/接口、旧R97结果和C边界。首个工程失败是空角色UNKNOWN没有接口输入时的序列化问题，未改模型；另一次CPU/GPU FP32极近分数导致赢家歧义，保留原生GPU选择并记录数值未知，概率5e-6容差未放宽，所有修正在评分之前。正式审计保留每次实际状态、范围、正负尾部与该歧义计数。

## 全体与平台结果

| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |
|---|---|---:|---:|---:|
| all_six_equal | original_source | 16.811383 | 30.989583 | 14.973958 |
| all_six_equal | original_source_matchedcoverage | 16.775157 | 30.859375 | 14.843750 |
| all_six_equal | prepared_frozen | 16.228652 | 29.947917 | 12.630208 |
| all_six_equal | real_frozen_C | 16.128363 | 29.557292 | 12.239583 |
| all_six_equal | real_frozen_fusedABC | 16.703230 | 30.989583 | 13.151042 |
| all_six_equal | native_real_fusedABC | 16.974318 | 31.250000 | 14.843750 |
| all_six_equal | native_real_ensemble | 16.734016 | 30.729167 | 14.843750 |
| all_six_equal | real_TTA_C | 16.313584 | 29.817708 | 12.760417 |
| all_six_equal | real_TTA_fusedABC | 16.574958 | 30.598958 | 13.020833 |
| all_six_equal | direct_three_geometric | 16.991306 | 32.291667 | 13.671875 |
| all_six_equal | neither_correct_A | 16.874661 | 31.380208 | 14.192708 |
| all_six_equal | neither_correct_fusedABC | 17.265658 | 31.901042 | 14.322917 |
| all_six_equal | neither_wrong_fusedABC | 17.116358 | 31.770833 | 14.322917 |
| all_six_equal | noPA_correct_fusedABC | 17.005609 | 31.901042 | 14.062500 |
| all_six_equal | noEarlyRGB_correct_fusedABC | 16.352394 | 30.208333 | 13.020833 |
| all_six_equal | full_all_raw_A | 16.253866 | 29.817708 | 12.500000 |
| all_six_equal | full_all_raw_C | 16.145988 | 29.817708 | 12.109375 |
| all_six_equal | full_all_raw_fusedABC | 16.482887 | 30.729167 | 12.760417 |
| all_six_equal | full_all_raw_feature3_fusedABC | 16.812134 | 31.380208 | 13.151042 |
| all_six_equal | full_all_bt_A | 16.202689 | 29.557292 | 12.500000 |
| all_six_equal | full_all_bt_C | 16.276544 | 29.817708 | 12.239583 |
| all_six_equal | full_all_bt_fusedABC | 16.711584 | 30.859375 | 13.151042 |
| all_six_equal | full_all_bt_feature3_fusedABC | 16.944159 | 31.510417 | 13.411458 |
| all_six_equal | full_bridge_nm_raw_A | 16.578559 | 30.468750 | 13.020833 |
| all_six_equal | full_bridge_nm_raw_C | 16.377275 | 29.947917 | 12.500000 |
| all_six_equal | full_bridge_nm_raw_fusedABC | 16.802549 | 30.989583 | 13.281250 |
| all_six_equal | full_bridge_nm_raw_feature3_fusedABC | 16.840615 | 31.250000 | 13.541667 |
| all_six_equal | full_bridge_nm_bt_A | 16.249638 | 30.078125 | 12.630208 |
| all_six_equal | full_bridge_nm_bt_C | 16.165262 | 29.687500 | 12.239583 |
| all_six_equal | full_bridge_nm_bt_fusedABC | 16.748285 | 30.989583 | 13.151042 |
| all_six_equal | full_bridge_nm_bt_feature3_fusedABC | 16.899997 | 31.510417 | 13.411458 |
| all_six_equal | anchor_all_raw_A | 16.419549 | 30.729167 | 13.541667 |
| all_six_equal | anchor_all_raw_C | 16.203307 | 30.338542 | 12.630208 |
| all_six_equal | anchor_all_raw_fusedABC | 16.400023 | 30.859375 | 12.890625 |
| all_six_equal | anchor_all_raw_feature3_fusedABC | 16.339131 | 30.598958 | 12.760417 |
| all_six_equal | anchor_all_bt_A | 16.955556 | 31.640625 | 14.192708 |
| all_six_equal | anchor_all_bt_C | 16.276671 | 30.468750 | 13.411458 |
| all_six_equal | anchor_all_bt_fusedABC | 17.292972 | 31.901042 | 14.322917 |
| all_six_equal | anchor_all_bt_feature3_fusedABC | 17.080387 | 31.510417 | 14.192708 |
| all_six_equal | anchor_bridge_nm_raw_A | 16.779714 | 30.859375 | 14.192708 |
| all_six_equal | anchor_bridge_nm_raw_C | 16.149885 | 29.817708 | 13.541667 |
| all_six_equal | anchor_bridge_nm_raw_fusedABC | 17.189686 | 31.510417 | 14.322917 |
| all_six_equal | anchor_bridge_nm_raw_feature3_fusedABC | 16.993625 | 31.119792 | 14.192708 |
| all_six_equal | anchor_bridge_nm_bt_A | 16.826273 | 31.380208 | 14.062500 |
| all_six_equal | anchor_bridge_nm_bt_C | 16.112773 | 30.208333 | 13.281250 |
| all_six_equal | anchor_bridge_nm_bt_fusedABC | 17.297074 | 31.901042 | 14.322917 |
| all_six_equal | anchor_bridge_nm_bt_feature3_fusedABC | 17.011681 | 31.250000 | 14.192708 |
| all_six_equal | anchor_rgb_nm_raw_A | 16.702393 | 30.729167 | 14.062500 |
| all_six_equal | anchor_rgb_nm_raw_C | 16.130839 | 29.947917 | 13.411458 |
| all_six_equal | anchor_rgb_nm_raw_fusedABC | 17.185507 | 31.510417 | 14.192708 |
| all_six_equal | anchor_rgb_nm_raw_feature3_fusedABC | 17.010482 | 31.250000 | 14.192708 |
| all_six_equal | anchor_rgb_nm_bt_A | 16.862650 | 31.250000 | 14.192708 |
| all_six_equal | anchor_rgb_nm_bt_C | 16.195293 | 30.208333 | 13.411458 |
| all_six_equal | anchor_rgb_nm_bt_fusedABC | 17.313328 | 31.770833 | 14.453125 |
| all_six_equal | anchor_rgb_nm_bt_feature3_fusedABC | 17.023781 | 31.380208 | 14.192708 |
| all_six_equal | anchor_bridge_natural_raw_A | 16.859737 | 31.380208 | 14.192708 |
| all_six_equal | anchor_bridge_natural_raw_C | 16.133144 | 30.208333 | 13.411458 |
| all_six_equal | anchor_bridge_natural_raw_fusedABC | 17.383951 | 32.031250 | 14.453125 |
| all_six_equal | anchor_bridge_natural_raw_feature3_fusedABC | 17.073875 | 31.510417 | 14.192708 |
| all_six_equal | anchor_bridge_natural_bt_A | 16.826273 | 31.380208 | 14.062500 |
| all_six_equal | anchor_bridge_natural_bt_C | 16.114449 | 30.208333 | 13.281250 |
| all_six_equal | anchor_bridge_natural_bt_fusedABC | 17.297074 | 31.901042 | 14.322917 |
| all_six_equal | anchor_bridge_natural_bt_feature3_fusedABC | 17.052550 | 31.510417 | 14.192708 |
| all_six_equal | anchor_bridge_nm_wrong_raw_A | 16.627443 | 30.078125 | 14.192708 |
| all_six_equal | anchor_bridge_nm_wrong_raw_C | 15.962614 | 29.166667 | 13.541667 |
| all_six_equal | anchor_bridge_nm_wrong_raw_fusedABC | 17.200327 | 31.250000 | 14.453125 |
| all_six_equal | anchor_bridge_nm_wrong_raw_feature3_fusedABC | 16.795259 | 30.338542 | 14.322917 |
| all_six_equal | anchor_bridge_nm_wrong_bt_A | 16.785365 | 31.119792 | 14.062500 |
| all_six_equal | anchor_bridge_nm_wrong_bt_C | 16.041401 | 29.817708 | 13.281250 |
| all_six_equal | anchor_bridge_nm_wrong_bt_fusedABC | 17.225732 | 31.510417 | 14.322917 |
| all_six_equal | anchor_bridge_nm_wrong_bt_feature3_fusedABC | 16.889009 | 30.859375 | 14.192708 |
| all_six_equal | full_frozen_mean3_A | 16.469418 | 30.468750 | 13.020833 |
| all_six_equal | full_frozen_mean3_C | 16.294764 | 30.078125 | 12.500000 |
| all_six_equal | full_frozen_mean3_fusedABC | 16.670486 | 31.250000 | 13.151042 |
| all_six_equal | full_frozen_feature3_A | 16.512947 | 30.468750 | 13.020833 |
| all_six_equal | full_frozen_feature3_C | 16.343702 | 30.208333 | 12.500000 |
| all_six_equal | full_frozen_feature3_fusedABC | 16.720196 | 31.380208 | 13.151042 |
| all_six_equal | anchor_frozen_mean3_A | 16.908682 | 31.250000 | 14.192708 |
| all_six_equal | anchor_frozen_mean3_C | 16.166385 | 30.208333 | 13.281250 |
| all_six_equal | anchor_frozen_mean3_fusedABC | 16.992984 | 31.510417 | 14.192708 |
| all_six_equal | anchor_frozen_feature3_A | 16.908682 | 31.250000 | 14.192708 |
| all_six_equal | anchor_frozen_feature3_C | 16.166385 | 30.208333 | 13.281250 |
| all_six_equal | anchor_frozen_feature3_fusedABC | 17.024394 | 31.510417 | 14.192708 |
| all_six_equal | anchor_frozen_C | 16.120576 | 30.208333 | 13.281250 |
| all_six_equal | anchor_noLate_feature3_fusedABC | 16.921576 | 31.250000 | 14.583333 |
| all_six_equal | anchor_noLate_fusedABC | 17.120908 | 31.510417 | 14.843750 |
| all_six_equal | anchor_lateOnly_feature3_fusedABC | 16.901711 | 31.250000 | 14.453125 |
| all_six_equal | anchor_lateOnly_fusedABC | 17.180150 | 31.901042 | 14.453125 |
| all_six_equal | anchor_native_feature3_fusedABC | 16.968411 | 31.119792 | 14.973958 |
| waymo | original_source | 11.473631 | 23.437500 | 7.421875 |
| waymo | original_source_matchedcoverage | 11.473631 | 23.437500 | 7.421875 |
| waymo | prepared_frozen | 12.615516 | 26.171875 | 5.078125 |
| waymo | real_frozen_C | 12.837875 | 26.562500 | 5.078125 |
| waymo | real_frozen_fusedABC | 12.569007 | 26.171875 | 5.078125 |
| waymo | native_real_fusedABC | 11.535583 | 23.437500 | 7.421875 |
| waymo | native_real_ensemble | 11.215742 | 22.656250 | 7.421875 |
| waymo | real_TTA_C | 12.770512 | 26.562500 | 5.078125 |
| waymo | real_TTA_fusedABC | 12.663880 | 26.171875 | 5.078125 |
| waymo | direct_three_geometric | 12.991047 | 27.343750 | 6.640625 |
| waymo | neither_correct_A | 12.021834 | 25.000000 | 7.812500 |
| waymo | neither_correct_fusedABC | 12.304763 | 25.390625 | 7.812500 |
| waymo | neither_wrong_fusedABC | 11.938272 | 24.609375 | 7.421875 |
| waymo | noPA_correct_fusedABC | 12.381182 | 26.171875 | 7.421875 |
| waymo | noEarlyRGB_correct_fusedABC | 12.410531 | 25.781250 | 5.859375 |
| waymo | full_all_raw_A | 12.212356 | 25.000000 | 4.687500 |
| waymo | full_all_raw_C | 12.320054 | 25.390625 | 4.687500 |
| waymo | full_all_raw_fusedABC | 12.184391 | 25.390625 | 4.687500 |
| waymo | full_all_raw_feature3_fusedABC | 12.530251 | 26.562500 | 4.687500 |
| waymo | full_all_bt_A | 12.614173 | 26.171875 | 5.078125 |
| waymo | full_all_bt_C | 12.836532 | 26.562500 | 5.078125 |
| waymo | full_all_bt_fusedABC | 12.567664 | 26.171875 | 5.078125 |
| waymo | full_all_bt_feature3_fusedABC | 13.132113 | 27.734375 | 5.468750 |
| waymo | full_bridge_nm_raw_A | 12.699392 | 26.171875 | 5.468750 |
| waymo | full_bridge_nm_raw_C | 12.936459 | 26.562500 | 5.468750 |
| waymo | full_bridge_nm_raw_fusedABC | 12.667591 | 26.171875 | 5.468750 |
| waymo | full_bridge_nm_raw_feature3_fusedABC | 12.668878 | 26.562500 | 5.859375 |
| waymo | full_bridge_nm_bt_A | 12.596826 | 26.171875 | 5.078125 |
| waymo | full_bridge_nm_bt_C | 12.819185 | 26.562500 | 5.078125 |
| waymo | full_bridge_nm_bt_fusedABC | 12.550317 | 26.171875 | 5.078125 |
| waymo | full_bridge_nm_bt_feature3_fusedABC | 13.012354 | 27.734375 | 5.468750 |
| waymo | anchor_all_raw_A | 12.350720 | 26.562500 | 6.250000 |
| waymo | anchor_all_raw_C | 12.562121 | 26.953125 | 6.250000 |
| waymo | anchor_all_raw_fusedABC | 12.508652 | 26.953125 | 6.250000 |
| waymo | anchor_all_raw_feature3_fusedABC | 12.270895 | 25.781250 | 5.859375 |
| waymo | anchor_all_bt_A | 12.102694 | 25.000000 | 7.812500 |
| waymo | anchor_all_bt_C | 12.097419 | 24.218750 | 7.812500 |
| waymo | anchor_all_bt_fusedABC | 12.385622 | 25.390625 | 7.812500 |
| waymo | anchor_all_bt_feature3_fusedABC | 11.779585 | 23.828125 | 7.421875 |
| waymo | anchor_bridge_nm_raw_A | 11.625060 | 24.218750 | 7.421875 |
| waymo | anchor_bridge_nm_raw_C | 11.689770 | 23.828125 | 7.421875 |
| waymo | anchor_bridge_nm_raw_fusedABC | 11.907989 | 24.609375 | 7.421875 |
| waymo | anchor_bridge_nm_raw_feature3_fusedABC | 11.517136 | 23.437500 | 7.421875 |
| waymo | anchor_bridge_nm_bt_A | 12.102694 | 25.000000 | 7.812500 |
| waymo | anchor_bridge_nm_bt_C | 12.135169 | 24.609375 | 7.812500 |
| waymo | anchor_bridge_nm_bt_fusedABC | 12.385622 | 25.390625 | 7.812500 |
| waymo | anchor_bridge_nm_bt_feature3_fusedABC | 11.779585 | 23.828125 | 7.421875 |
| waymo | anchor_rgb_nm_raw_A | 11.881557 | 24.609375 | 7.421875 |
| waymo | anchor_rgb_nm_raw_C | 11.946266 | 24.218750 | 7.421875 |
| waymo | anchor_rgb_nm_raw_fusedABC | 12.164485 | 25.000000 | 7.421875 |
| waymo | anchor_rgb_nm_raw_feature3_fusedABC | 11.829301 | 24.218750 | 7.421875 |
| waymo | anchor_rgb_nm_bt_A | 12.102694 | 25.000000 | 7.812500 |
| waymo | anchor_rgb_nm_bt_C | 12.167403 | 24.609375 | 7.812500 |
| waymo | anchor_rgb_nm_bt_fusedABC | 12.385622 | 25.390625 | 7.812500 |
| waymo | anchor_rgb_nm_bt_feature3_fusedABC | 11.757207 | 24.218750 | 7.421875 |
| waymo | anchor_bridge_natural_raw_A | 12.102694 | 25.000000 | 7.812500 |
| waymo | anchor_bridge_natural_raw_C | 12.144983 | 24.609375 | 7.812500 |
| waymo | anchor_bridge_natural_raw_fusedABC | 12.385622 | 25.390625 | 7.812500 |
| waymo | anchor_bridge_natural_raw_feature3_fusedABC | 11.795658 | 24.218750 | 7.421875 |
| waymo | anchor_bridge_natural_bt_A | 12.102694 | 25.000000 | 7.812500 |
| waymo | anchor_bridge_natural_bt_C | 12.112749 | 24.609375 | 7.812500 |
| waymo | anchor_bridge_natural_bt_fusedABC | 12.385622 | 25.390625 | 7.812500 |
| waymo | anchor_bridge_natural_bt_feature3_fusedABC | 11.795658 | 24.218750 | 7.421875 |
| waymo | anchor_bridge_nm_wrong_raw_A | 11.670217 | 23.828125 | 7.421875 |
| waymo | anchor_bridge_nm_wrong_raw_C | 11.712463 | 23.437500 | 7.421875 |
| waymo | anchor_bridge_nm_wrong_raw_fusedABC | 11.919432 | 24.218750 | 7.421875 |
| waymo | anchor_bridge_nm_wrong_raw_feature3_fusedABC | 11.501806 | 23.046875 | 7.421875 |
| waymo | anchor_bridge_nm_wrong_bt_A | 12.087365 | 24.609375 | 7.812500 |
| waymo | anchor_bridge_nm_wrong_bt_C | 12.097377 | 24.218750 | 7.812500 |
| waymo | anchor_bridge_nm_wrong_bt_fusedABC | 12.370293 | 25.000000 | 7.812500 |
| waymo | anchor_bridge_nm_wrong_bt_feature3_fusedABC | 11.725804 | 23.437500 | 7.421875 |
| waymo | full_frozen_mean3_A | 12.727272 | 26.562500 | 5.468750 |
| waymo | full_frozen_mean3_C | 12.825962 | 26.562500 | 5.468750 |
| waymo | full_frozen_mean3_fusedABC | 12.901481 | 27.343750 | 5.468750 |
| waymo | full_frozen_feature3_A | 12.857860 | 26.562500 | 5.468750 |
| waymo | full_frozen_feature3_C | 12.975094 | 26.953125 | 5.468750 |
| waymo | full_frozen_feature3_fusedABC | 13.050613 | 27.734375 | 5.468750 |
| waymo | anchor_frozen_mean3_A | 11.756063 | 24.218750 | 7.421875 |
| waymo | anchor_frozen_mean3_C | 11.754512 | 23.437500 | 7.421875 |
| waymo | anchor_frozen_mean3_fusedABC | 11.794997 | 24.218750 | 7.421875 |
| waymo | anchor_frozen_feature3_A | 11.756063 | 24.218750 | 7.421875 |
| waymo | anchor_frozen_feature3_C | 11.754512 | 23.437500 | 7.421875 |
| waymo | anchor_frozen_feature3_fusedABC | 11.794997 | 24.218750 | 7.421875 |
| waymo | anchor_frozen_C | 12.016560 | 24.218750 | 7.812500 |
| waymo | anchor_noLate_feature3_fusedABC | 11.804911 | 23.828125 | 7.421875 |
| waymo | anchor_noLate_fusedABC | 11.484142 | 23.437500 | 7.421875 |
| waymo | anchor_lateOnly_feature3_fusedABC | 11.781109 | 23.828125 | 7.421875 |
| waymo | anchor_lateOnly_fusedABC | 12.181059 | 25.390625 | 7.421875 |
| waymo | anchor_native_feature3_fusedABC | 11.780221 | 23.437500 | 7.421875 |
| drone | original_source | 13.867693 | 24.609375 | 12.500000 |
| drone | original_source_matchedcoverage | 13.908446 | 24.218750 | 12.500000 |
| drone | prepared_frozen | 14.071678 | 24.609375 | 11.718750 |
| drone | real_frozen_C | 13.842086 | 23.828125 | 11.718750 |
| drone | real_frozen_fusedABC | 14.029708 | 25.000000 | 11.718750 |
| drone | native_real_fusedABC | 14.710633 | 26.171875 | 12.890625 |
| drone | native_real_ensemble | 14.188880 | 24.609375 | 12.500000 |
| drone | real_TTA_C | 13.743545 | 23.437500 | 11.718750 |
| drone | real_TTA_fusedABC | 14.080314 | 25.000000 | 11.718750 |
| drone | direct_three_geometric | 14.039081 | 26.171875 | 11.718750 |
| drone | neither_correct_A | 14.350360 | 25.781250 | 11.328125 |
| drone | neither_correct_fusedABC | 14.503438 | 26.171875 | 11.718750 |
| drone | neither_wrong_fusedABC | 14.630361 | 26.953125 | 11.718750 |
| drone | noPA_correct_fusedABC | 14.397922 | 26.171875 | 12.109375 |
| drone | noEarlyRGB_correct_fusedABC | 13.896284 | 25.000000 | 11.718750 |
| drone | full_all_raw_A | 14.092211 | 25.390625 | 11.328125 |
| drone | full_all_raw_C | 13.996790 | 25.000000 | 11.328125 |
| drone | full_all_raw_fusedABC | 13.968758 | 25.781250 | 11.328125 |
| drone | full_all_raw_feature3_fusedABC | 13.908045 | 25.390625 | 11.328125 |
| drone | full_all_bt_A | 14.123382 | 24.218750 | 11.718750 |
| drone | full_all_bt_C | 14.048358 | 24.218750 | 11.718750 |
| drone | full_all_bt_fusedABC | 14.014836 | 25.000000 | 11.718750 |
| drone | full_all_bt_feature3_fusedABC | 14.167088 | 25.390625 | 12.109375 |
| drone | full_bridge_nm_raw_A | 14.684812 | 25.781250 | 12.109375 |
| drone | full_bridge_nm_raw_C | 14.353510 | 24.609375 | 12.109375 |
| drone | full_bridge_nm_raw_fusedABC | 14.414351 | 25.390625 | 12.109375 |
| drone | full_bridge_nm_raw_feature3_fusedABC | 14.351474 | 25.390625 | 12.109375 |
| drone | full_bridge_nm_bt_A | 14.153325 | 25.000000 | 11.718750 |
| drone | full_bridge_nm_bt_C | 13.870945 | 23.828125 | 11.718750 |
| drone | full_bridge_nm_bt_fusedABC | 14.167202 | 25.000000 | 11.718750 |
| drone | full_bridge_nm_bt_feature3_fusedABC | 14.217506 | 25.000000 | 12.109375 |
| drone | anchor_all_raw_A | 13.667110 | 24.218750 | 11.328125 |
| drone | anchor_all_raw_C | 13.679673 | 24.218750 | 11.328125 |
| drone | anchor_all_raw_fusedABC | 13.908990 | 24.609375 | 11.718750 |
| drone | anchor_all_raw_feature3_fusedABC | 13.964072 | 25.000000 | 11.718750 |
| drone | anchor_all_bt_A | 14.501522 | 26.562500 | 11.328125 |
| drone | anchor_all_bt_C | 14.400879 | 26.953125 | 11.328125 |
| drone | anchor_all_bt_fusedABC | 14.504519 | 26.171875 | 11.718750 |
| drone | anchor_all_bt_feature3_fusedABC | 14.602058 | 26.171875 | 11.718750 |
| drone | anchor_bridge_nm_raw_A | 14.888921 | 25.781250 | 12.109375 |
| drone | anchor_bridge_nm_raw_C | 14.861768 | 26.171875 | 12.109375 |
| drone | anchor_bridge_nm_raw_fusedABC | 14.914243 | 26.171875 | 12.500000 |
| drone | anchor_bridge_nm_raw_feature3_fusedABC | 14.560401 | 25.390625 | 11.718750 |
| drone | anchor_bridge_nm_bt_A | 14.372304 | 26.171875 | 11.328125 |
| drone | anchor_bridge_nm_bt_C | 14.301248 | 26.562500 | 11.328125 |
| drone | anchor_bridge_nm_bt_fusedABC | 14.516825 | 26.171875 | 11.718750 |
| drone | anchor_bridge_nm_bt_feature3_fusedABC | 14.351651 | 25.390625 | 11.718750 |
| drone | anchor_rgb_nm_raw_A | 14.391260 | 25.000000 | 11.718750 |
| drone | anchor_rgb_nm_raw_C | 14.548134 | 26.171875 | 11.718750 |
| drone | anchor_rgb_nm_raw_fusedABC | 14.645209 | 25.781250 | 12.109375 |
| drone | anchor_rgb_nm_raw_feature3_fusedABC | 14.298809 | 25.000000 | 11.718750 |
| drone | anchor_rgb_nm_bt_A | 14.481438 | 25.781250 | 11.718750 |
| drone | anchor_rgb_nm_bt_C | 14.516572 | 26.562500 | 11.718750 |
| drone | anchor_rgb_nm_bt_fusedABC | 14.565587 | 25.781250 | 12.109375 |
| drone | anchor_rgb_nm_bt_feature3_fusedABC | 14.410329 | 25.390625 | 11.718750 |
| drone | anchor_bridge_natural_raw_A | 14.632936 | 26.562500 | 11.718750 |
| drone | anchor_bridge_natural_raw_C | 14.561879 | 26.953125 | 11.718750 |
| drone | anchor_bridge_natural_raw_fusedABC | 14.777456 | 26.562500 | 12.109375 |
| drone | anchor_bridge_natural_raw_feature3_fusedABC | 14.522161 | 25.781250 | 11.718750 |
| drone | anchor_bridge_natural_bt_A | 14.372304 | 26.171875 | 11.328125 |
| drone | anchor_bridge_natural_bt_C | 14.301248 | 26.562500 | 11.328125 |
| drone | anchor_bridge_natural_bt_fusedABC | 14.516825 | 26.171875 | 11.718750 |
| drone | anchor_bridge_natural_bt_feature3_fusedABC | 14.458185 | 25.781250 | 11.718750 |
| drone | anchor_bridge_nm_wrong_raw_A | 14.413686 | 24.609375 | 12.109375 |
| drone | anchor_bridge_nm_wrong_raw_C | 14.249578 | 24.609375 | 12.109375 |
| drone | anchor_bridge_nm_wrong_raw_fusedABC | 14.731458 | 25.781250 | 12.500000 |
| drone | anchor_bridge_nm_wrong_raw_feature3_fusedABC | 14.006900 | 24.218750 | 12.109375 |
| drone | anchor_bridge_nm_wrong_bt_A | 14.264912 | 25.781250 | 11.328125 |
| drone | anchor_bridge_nm_wrong_bt_C | 14.097474 | 25.781250 | 11.328125 |
| drone | anchor_bridge_nm_wrong_bt_fusedABC | 14.318129 | 25.390625 | 11.718750 |
| drone | anchor_bridge_nm_wrong_bt_feature3_fusedABC | 14.037416 | 24.609375 | 11.718750 |
| drone | full_frozen_mean3_A | 13.900299 | 24.609375 | 11.718750 |
| drone | full_frozen_mean3_C | 13.751240 | 23.828125 | 11.718750 |
| drone | full_frozen_mean3_fusedABC | 13.846353 | 25.000000 | 11.718750 |
| drone | full_frozen_feature3_A | 13.900299 | 24.609375 | 11.718750 |
| drone | full_frozen_feature3_C | 13.751240 | 23.828125 | 11.718750 |
| drone | full_frozen_feature3_fusedABC | 13.846353 | 25.000000 | 11.718750 |
| drone | anchor_frozen_mean3_A | 14.350303 | 25.781250 | 11.328125 |
| drone | anchor_frozen_mean3_C | 14.402733 | 26.953125 | 11.328125 |
| drone | anchor_frozen_mean3_fusedABC | 14.323969 | 25.781250 | 11.718750 |
| drone | anchor_frozen_feature3_A | 14.350303 | 25.781250 | 11.328125 |
| drone | anchor_frozen_feature3_C | 14.402733 | 26.953125 | 11.328125 |
| drone | anchor_frozen_feature3_fusedABC | 14.418200 | 25.781250 | 11.718750 |
| drone | anchor_frozen_C | 14.405591 | 26.953125 | 11.328125 |
| drone | anchor_noLate_feature3_fusedABC | 14.311257 | 25.781250 | 12.500000 |
| drone | anchor_noLate_fusedABC | 14.996071 | 26.953125 | 12.890625 |
| drone | anchor_lateOnly_feature3_fusedABC | 14.208546 | 25.390625 | 11.718750 |
| drone | anchor_lateOnly_fusedABC | 14.345223 | 25.781250 | 11.718750 |
| drone | anchor_native_feature3_fusedABC | 14.279733 | 25.390625 | 12.500000 |
| quad | original_source | 25.092825 | 44.921875 | 25.000000 |
| quad | original_source_matchedcoverage | 24.943393 | 44.921875 | 24.609375 |
| quad | prepared_frozen | 21.998762 | 39.062500 | 21.093750 |
| quad | real_frozen_C | 21.705129 | 38.281250 | 19.921875 |
| quad | real_frozen_fusedABC | 23.510976 | 41.796875 | 22.656250 |
| quad | native_real_fusedABC | 24.676737 | 44.140625 | 24.218750 |
| quad | native_real_ensemble | 24.797425 | 44.921875 | 24.609375 |
| quad | real_TTA_C | 22.426695 | 39.453125 | 21.484375 |
| quad | real_TTA_fusedABC | 22.980680 | 40.625000 | 22.265625 |
| quad | direct_three_geometric | 23.943791 | 43.359375 | 22.656250 |
| quad | neither_correct_A | 24.251789 | 43.359375 | 23.437500 |
| quad | neither_correct_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | neither_wrong_fusedABC | 24.780441 | 43.750000 | 23.828125 |
| quad | noPA_correct_fusedABC | 24.237723 | 43.359375 | 22.656250 |
| quad | noEarlyRGB_correct_fusedABC | 22.750368 | 39.843750 | 21.484375 |
| quad | full_all_raw_A | 22.457032 | 39.062500 | 21.484375 |
| quad | full_all_raw_C | 22.121121 | 39.062500 | 20.312500 |
| quad | full_all_raw_fusedABC | 23.295513 | 41.015625 | 22.265625 |
| quad | full_all_raw_feature3_fusedABC | 23.998107 | 42.187500 | 23.437500 |
| quad | full_all_bt_A | 21.870512 | 38.281250 | 20.703125 |
| quad | full_all_bt_C | 21.944741 | 38.671875 | 19.921875 |
| quad | full_all_bt_fusedABC | 23.552251 | 41.406250 | 22.656250 |
| quad | full_all_bt_feature3_fusedABC | 23.533275 | 41.406250 | 22.656250 |
| quad | full_bridge_nm_raw_A | 22.351472 | 39.453125 | 21.484375 |
| quad | full_bridge_nm_raw_C | 21.841855 | 38.671875 | 19.921875 |
| quad | full_bridge_nm_raw_fusedABC | 23.325705 | 41.406250 | 22.265625 |
| quad | full_bridge_nm_raw_feature3_fusedABC | 23.501495 | 41.796875 | 22.656250 |
| quad | full_bridge_nm_bt_A | 21.998762 | 39.062500 | 21.093750 |
| quad | full_bridge_nm_bt_C | 21.805656 | 38.671875 | 19.921875 |
| quad | full_bridge_nm_bt_fusedABC | 23.527335 | 41.796875 | 22.656250 |
| quad | full_bridge_nm_bt_feature3_fusedABC | 23.470130 | 41.796875 | 22.656250 |
| quad | anchor_all_raw_A | 23.240818 | 41.406250 | 23.046875 |
| quad | anchor_all_raw_C | 22.368127 | 39.843750 | 20.312500 |
| quad | anchor_all_raw_fusedABC | 22.782427 | 41.015625 | 20.703125 |
| quad | anchor_all_raw_feature3_fusedABC | 22.782427 | 41.015625 | 20.703125 |
| quad | anchor_all_bt_A | 24.262450 | 43.359375 | 23.437500 |
| quad | anchor_all_bt_C | 22.331714 | 40.234375 | 21.093750 |
| quad | anchor_all_bt_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | anchor_all_bt_feature3_fusedABC | 24.859517 | 44.531250 | 23.437500 |
| quad | anchor_bridge_nm_raw_A | 23.825162 | 42.578125 | 23.046875 |
| quad | anchor_bridge_nm_raw_C | 21.898116 | 39.453125 | 21.093750 |
| quad | anchor_bridge_nm_raw_fusedABC | 24.746827 | 43.750000 | 23.046875 |
| quad | anchor_bridge_nm_raw_feature3_fusedABC | 24.903337 | 44.531250 | 23.437500 |
| quad | anchor_bridge_nm_bt_A | 24.003819 | 42.968750 | 23.046875 |
| quad | anchor_bridge_nm_bt_C | 21.901903 | 39.453125 | 20.703125 |
| quad | anchor_bridge_nm_bt_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | anchor_bridge_nm_bt_feature3_fusedABC | 24.903806 | 44.531250 | 23.437500 |
| quad | anchor_rgb_nm_raw_A | 23.834364 | 42.578125 | 23.046875 |
| quad | anchor_rgb_nm_raw_C | 21.898116 | 39.453125 | 21.093750 |
| quad | anchor_rgb_nm_raw_fusedABC | 24.746827 | 43.750000 | 23.046875 |
| quad | anchor_rgb_nm_raw_feature3_fusedABC | 24.903337 | 44.531250 | 23.437500 |
| quad | anchor_rgb_nm_bt_A | 24.003819 | 42.968750 | 23.046875 |
| quad | anchor_rgb_nm_bt_C | 21.901903 | 39.453125 | 20.703125 |
| quad | anchor_rgb_nm_bt_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | anchor_rgb_nm_bt_feature3_fusedABC | 24.903806 | 44.531250 | 23.437500 |
| quad | anchor_bridge_natural_raw_A | 23.843580 | 42.578125 | 23.046875 |
| quad | anchor_bridge_natural_raw_C | 21.692570 | 39.062500 | 20.703125 |
| quad | anchor_bridge_natural_raw_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | anchor_bridge_natural_raw_feature3_fusedABC | 24.903806 | 44.531250 | 23.437500 |
| quad | anchor_bridge_natural_bt_A | 24.003819 | 42.968750 | 23.046875 |
| quad | anchor_bridge_natural_bt_C | 21.929352 | 39.453125 | 20.703125 |
| quad | anchor_bridge_natural_bt_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | anchor_bridge_natural_bt_feature3_fusedABC | 24.903806 | 44.531250 | 23.437500 |
| quad | anchor_bridge_nm_wrong_raw_A | 23.798427 | 41.796875 | 23.046875 |
| quad | anchor_bridge_nm_wrong_raw_C | 21.925800 | 39.453125 | 21.093750 |
| quad | anchor_bridge_nm_wrong_raw_fusedABC | 24.950090 | 43.750000 | 23.437500 |
| quad | anchor_bridge_nm_wrong_raw_feature3_fusedABC | 24.877071 | 43.750000 | 23.437500 |
| quad | anchor_bridge_nm_wrong_bt_A | 24.003819 | 42.968750 | 23.046875 |
| quad | anchor_bridge_nm_wrong_bt_C | 21.929352 | 39.453125 | 20.703125 |
| quad | anchor_bridge_nm_wrong_bt_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | anchor_bridge_nm_wrong_bt_feature3_fusedABC | 24.903806 | 44.531250 | 23.437500 |
| quad | full_frozen_mean3_A | 22.780682 | 40.234375 | 21.875000 |
| quad | full_frozen_mean3_C | 22.307090 | 39.843750 | 20.312500 |
| quad | full_frozen_mean3_fusedABC | 23.263623 | 41.406250 | 22.265625 |
| quad | full_frozen_feature3_A | 22.780682 | 40.234375 | 21.875000 |
| quad | full_frozen_feature3_C | 22.304773 | 39.843750 | 20.312500 |
| quad | full_frozen_feature3_fusedABC | 23.263623 | 41.406250 | 22.265625 |
| quad | anchor_frozen_mean3_A | 24.619681 | 43.750000 | 23.828125 |
| quad | anchor_frozen_mean3_C | 22.341908 | 40.234375 | 21.093750 |
| quad | anchor_frozen_mean3_fusedABC | 24.859986 | 44.531250 | 23.437500 |
| quad | anchor_frozen_feature3_A | 24.619681 | 43.750000 | 23.828125 |
| quad | anchor_frozen_feature3_C | 22.341908 | 40.234375 | 21.093750 |
| quad | anchor_frozen_feature3_fusedABC | 24.859986 | 44.531250 | 23.437500 |
| quad | anchor_frozen_C | 21.939577 | 39.453125 | 20.703125 |
| quad | anchor_noLate_feature3_fusedABC | 24.648561 | 44.140625 | 23.828125 |
| quad | anchor_noLate_fusedABC | 24.882511 | 44.140625 | 24.218750 |
| quad | anchor_lateOnly_feature3_fusedABC | 24.715478 | 44.531250 | 24.218750 |
| quad | anchor_lateOnly_fusedABC | 25.014169 | 44.531250 | 24.218750 |
| quad | anchor_native_feature3_fusedABC | 24.845278 | 44.531250 | 25.000000 |

## 固定主项与所有强对照

```json
{
  "primary": "anchor_bridge_nm_bt_feature3_fusedABC",
  "mIoU_percent": 17.011680703089954,
  "gains_pp": {
    "original_source": 0.20029770437270145,
    "prepared_frozen": 0.7830285439012741,
    "real_frozen_fusedABC": 0.3084504101311083,
    "native_real_fusedABC": 0.037363191370261006,
    "direct_three_geometric": 0.02037443859624588,
    "neither_correct_fusedABC": -0.2539776452033209,
    "anchor_frozen_mean3_fusedABC": 0.01869670955158398,
    "anchor_frozen_feature3_fusedABC": -0.012713566890123928,
    "anchor_noLate_fusedABC": -0.1092273917779174,
    "anchor_noLate_feature3_fusedABC": 0.09010429938793862,
    "anchor_native_feature3_fusedABC": 0.04326987976016028,
    "anchor_bridge_nm_wrong_bt_feature3_fusedABC": 0.12267192160919649,
    "full_all_bt_feature3_fusedABC": 0.0675221844577436,
    "anchor_all_bt_feature3_fusedABC": -0.0687061131089024
  },
  "passes_all_prelocked_controls": false,
  "robust_TTA_adoption": false,
  "decision": "do_not_adopt_this_RGB_bridge_physical_TTA_recipe",
  "primary_not_switched": true,
  "no_step_strength_grid_after_failure": true,
  "n": 768,
  "FINAL_access": false
}
```

## 更新作用

```json
[
  {
    "setting": "full_all",
    "n": 724,
    "scope_parameters": 102498,
    "accepted": 724,
    "mean_accepted_alpha": 0.20475440262430938,
    "mean_raw_L2": 0.03492534970568791,
    "mean_scope_gradient_L2": 0.06985069941137582,
    "mean_parent_gradient_L2": 0.06985069941137582,
    "aux_before_mean": 0.0371762719944387,
    "aux_accepted_mean": 0.03659900177774426,
    "main_readout_gain_vs_own_matched_frozen_pp": 0.2239622385804303
  },
  {
    "setting": "full_bridge_nm",
    "n": 724,
    "scope_parameters": 45313,
    "accepted": 724,
    "mean_accepted_alpha": 0.20797544026243095,
    "mean_raw_L2": 0.034925349700542486,
    "mean_scope_gradient_L2": 0.038114408364915676,
    "mean_parent_gradient_L2": 0.06985069941137582,
    "aux_before_mean": 0.0371762719944387,
    "aux_accepted_mean": 0.036817855887289946,
    "main_readout_gain_vs_own_matched_frozen_pp": 0.17980043646660349
  },
  {
    "setting": "anchor_all",
    "n": 724,
    "scope_parameters": 102498,
    "accepted": 662,
    "mean_accepted_alpha": 0.0515716721339779,
    "mean_raw_L2": 0.09739425920145535,
    "mean_scope_gradient_L2": 0.1947885184029107,
    "mean_parent_gradient_L2": 0.1947885184029107,
    "aux_before_mean": 0.04221477576927668,
    "aux_accepted_mean": 0.041140732872531394,
    "main_readout_gain_vs_own_matched_frozen_pp": 0.05599254621877847
  },
  {
    "setting": "anchor_bridge_nm",
    "n": 724,
    "scope_parameters": 45313,
    "accepted": 724,
    "mean_accepted_alpha": 0.19257974361187846,
    "mean_raw_L2": 0.09739425932237292,
    "mean_scope_gradient_L2": 0.04799515742857462,
    "mean_parent_gradient_L2": 0.1947885184029107,
    "aux_before_mean": 0.04221477576927668,
    "aux_accepted_mean": 0.0413181504996195,
    "main_readout_gain_vs_own_matched_frozen_pp": -0.012713566890123928
  },
  {
    "setting": "anchor_rgb_nm",
    "n": 724,
    "scope_parameters": 47441,
    "accepted": 724,
    "mean_accepted_alpha": 0.2879834254143646,
    "mean_raw_L2": 0.09739425920660078,
    "mean_scope_gradient_L2": 0.059861963092560436,
    "mean_parent_gradient_L2": 0.1947885184029107,
    "aux_before_mean": 0.04221477576927668,
    "aux_accepted_mean": 0.0403394249206131,
    "main_readout_gain_vs_own_matched_frozen_pp": -0.0006132727744372346
  },
  {
    "setting": "anchor_bridge_natural",
    "n": 724,
    "scope_parameters": 45313,
    "accepted": 724,
    "mean_accepted_alpha": 0.3590728591160221,
    "mean_raw_L2": 0.02399757871428731,
    "mean_scope_gradient_L2": 0.04799515742857462,
    "mean_parent_gradient_L2": 0.1947885184029107,
    "aux_before_mean": 0.04221477576927668,
    "aux_accepted_mean": 0.04147266681712553,
    "main_readout_gain_vs_own_matched_frozen_pp": 0.028155450387117753
  },
  {
    "setting": "anchor_bridge_nm_wrong",
    "n": 724,
    "scope_parameters": 45313,
    "accepted": 724,
    "mean_accepted_alpha": 0.19843911861187846,
    "mean_raw_L2": 0.0973942592657732,
    "mean_scope_gradient_L2": 0.040886437744532524,
    "mean_parent_gradient_L2": 0.17728661234331064,
    "aux_before_mean": 0.0497480247205751,
    "aux_accepted_mean": 0.048540679310854956,
    "main_readout_gain_vs_own_matched_frozen_pp": -0.13538548849932042
  }
]
```

任务分数、两个方向/平台和场景差异决定是否采用；辅助损失、参数非零和h/u/r恒等只解释机制。固定基础路径允许RGB改变候选排序，不能把不更新式保守回退当作纠错成功。任何静态组合或诊断臂的较高分都不改写本轮预锁定主项。若主项不能超过同信息零步与错配控制，停止此物理RGB更新配方，不继续调步数或加门槛挽救。

## 补充：未遮挡读出的正结果及缺失控制

原锁定主项不变，仍未通过。遮挡表示均值自身将零步17.265658%降到17.024394%，因此不能把该主项阴性推广成全部桥接TTA无效。开发次项、自然clip原始一步且最终不遮挡，达到17.383951%，较最强零步增加0.118293pp：5条IoU改善、2条恶化，最大单例正贡献0.086877pp，其他767条净贡献0.031416pp。原始A本身下降0.014924pp；C读出总体仅+0.012568pp，缺C回退包括在内，不宣称独立新场景迁移。

该次项暴露后另锁定补充控制；没有修改原89臂预测/主项/审计。新增6臂共4608任务值、4344非别名CPU独立读出；正确原次项概率位等复现。自然错配比正确低0.038456pp，但将错配提议L2匹配正确后，总体mIoU与正确次项完全相同。764/768条Top1相同，概率并非位等，不能声称其余4条对象身份相同。该结果未支持“正确RGB–点对应使参数更新有效”的解释，不据此采用该配方或进行步长网格。

后验计算归因显示：桥梯度平方量平均91.62%、中位97.63%集中于单个gate_I；仅208/724条更新减小门绝对值，不能笼统称为抑制坏桥。旁路后的辅助梯度与源域完整辅助梯度在96/724条上为负余弦，表明修改基础路径也改变了源域恢复任务的计算条件；这不是grounding梯度对齐证明。后续最小比较应包含固定gate_I=0及源域条件保留的RGB桥，区分静态校准和真实输入适应。

点支持检查：最终query锚点位于其框内的比例约90%，不支持“普遍锚点跑到框外”解释；这不认证实例纯度。私有图册检查保留可见车辆区域纠错、邻车不确定、稀疏小IoU变化及改向背景的负例；图册只作原主项的事后机制分析，不将其语义结论套到次项。

原89臂和补充6臂共95个不同任务读出、72960个任务值均公开匿名标量；所有别名、UNKNOWN、拒绝和失败attempt留存。主配方不采用；次项为开发正信号但未通过其绑定机制控制。
