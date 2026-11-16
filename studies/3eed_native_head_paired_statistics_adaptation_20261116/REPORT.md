# R110 — 原生分类头输入兼容性与配对统计状态适应

本轮真正更新的是原3EED最后分类头两个BN的运行均值/方差，1152个buffer值；不是学习权重、梯度或伪标签训练。三个source100/R91主干/卷积/affine/匹配MLP/原A256不变，无专家、新source准备或FINAL。

## 动机与边界

R91source准备不训练原CE头，却改变其query输入，存在待检验的接口分布偏移；维数一致不证明兼容。R108开发正增量在R109扩展失败，继续禁止拟合其输出。本轮用同一source模型当前A的原始表示O作参照，仅对prepared A表示P测量配对通道矩；不是假定O的预测正确。

在BN1/BN2预归一化激活上计算FP64总体矩，BN2的P激活必须在BN1已经更新后计算。a=sqrt(varP/varO)，meanNew=meanP+a(meanS-meanO)，varNew=a²(varS+eps)-eps；明确保留原BN epsilon的代数含义。低方差/非法方差通道保持source，不扫描阈值或幅度。P=O时状态与输出位等；真实写入/函数式实现/独立CPU重放/source恢复通过12实际工程输入。

C及fusedABC隐藏表示仅在正确/错配两个状态都构造完后读取，校准不用这些信息。实际C不足时存在A回退，故C总体和actual_C_available子集分开，不能将全部C项当独立新观测。DEV历史暴露，不是独立确认。

O是原本建立canonical A256时就计算的source状态，旧实验未保存h所以本轮只提取缺失状态，未重跑旧拟合或改变旧预测。提取窗口成本与缓存适应成本分开，不凭缓存时间宣称部署速度。所有原始框、source概率、native CE logits与既有实现核对，44保护别名不重读/模型，保留在768分母。

## 同信息对照

原生originalCE直接读出；originalCE+现有BBF融合的几何/算术池；originalCE、preparedCE、BBF三项等权直接池；原R108所有读出及物理/采样/TTA参照；同成本错配O矩anchor。不同描述之间只对矩做控制，不假定query身份。沿用相同几何池，不同时恢复/放大r。

## 全体结果

| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |
|---|---|---:|---:|---:|---:|
| all_six_equal | original_source | 16.811383 | 30.989583 | 14.973958 |
| all_six_equal | prepared_frozen | 16.228652 | 29.947917 | 12.630208 |
| all_six_equal | real_frozen_B | 16.352068 | 30.729167 | 12.369792 |
| all_six_equal | real_frozen_C | 16.128363 | 29.557292 | 12.239583 |
| all_six_equal | real_frozen_ensemble | 16.501679 | 30.729167 | 12.500000 |
| all_six_equal | real_frozen_fusedABC | 16.703230 | 30.989583 | 13.151042 |
| all_six_equal | real_TTA_C | 16.313584 | 29.817708 | 12.760417 |
| all_six_equal | real_TTA_fusedABC | 16.574958 | 30.598958 | 13.020833 |
| all_six_equal | randomA_frozen_fusedABC | 16.560112 | 30.598958 | 13.151042 |
| all_six_equal | randomA_TTA_fusedABC | 16.629515 | 30.729167 | 13.151042 |
| all_six_equal | wrong_frozen_fusedABC | 13.477241 | 24.869792 | 9.895833 |
| all_six_equal | wrong_TTA_fusedABC | 13.362302 | 24.609375 | 10.026042 |
| all_six_equal | teacher_shuffle_fusedABC | 16.280349 | 29.817708 | 12.500000 |
| all_six_equal | softtoken_A | 15.797063 | 28.906250 | 12.890625 |
| all_six_equal | softtoken_B | 15.870754 | 29.427083 | 12.630208 |
| all_six_equal | softtoken_C | 15.762694 | 28.515625 | 12.369792 |
| all_six_equal | softtoken_fusedABC | 16.159325 | 29.687500 | 13.020833 |
| all_six_equal | softtoken_geometric_A | 16.294275 | 30.338542 | 12.890625 |
| all_six_equal | softtoken_geometric_B | 16.561778 | 31.119792 | 12.890625 |
| all_six_equal | softtoken_geometric_C | 16.317403 | 30.078125 | 12.630208 |
| all_six_equal | softtoken_geometric_fusedABC | 16.884348 | 31.640625 | 13.281250 |
| all_six_equal | softtoken_arithmetic_A | 16.219504 | 29.557292 | 13.151042 |
| all_six_equal | softtoken_arithmetic_B | 16.454152 | 30.338542 | 13.151042 |
| all_six_equal | softtoken_arithmetic_C | 16.349115 | 29.687500 | 12.890625 |
| all_six_equal | softtoken_arithmetic_fusedABC | 16.907195 | 31.119792 | 13.671875 |
| all_six_equal | softtoken_shuffled_geometric_A | 7.528233 | 14.062500 | 1.953125 |
| all_six_equal | softtoken_shuffled_geometric_B | 7.286999 | 13.932292 | 2.213542 |
| all_six_equal | softtoken_shuffled_geometric_C | 7.585662 | 13.932292 | 1.822917 |
| all_six_equal | softtoken_shuffled_geometric_fusedABC | 7.424009 | 13.932292 | 1.953125 |
| all_six_equal | native_original_CE | 16.549116 | 29.947917 | 14.322917 |
| all_six_equal | original_source_matchedcoverage | 16.775157 | 30.859375 | 14.843750 |
| all_six_equal | original_native_pair_geometric | 16.895210 | 30.989583 | 14.453125 |
| all_six_equal | original_native_pair_arithmetic | 16.796663 | 30.729167 | 14.713542 |
| all_six_equal | original_CE_with_prepared_geometric_A | 16.477764 | 30.598958 | 13.411458 |
| all_six_equal | original_CE_with_prepared_arithmetic_A | 16.793812 | 30.989583 | 13.932292 |
| all_six_equal | original_CE_with_fused_geometric | 16.892267 | 31.770833 | 13.802083 |
| all_six_equal | original_CE_with_fused_arithmetic | 16.895199 | 31.119792 | 14.062500 |
| all_six_equal | original_CE_with_C_geometric | 16.508391 | 30.989583 | 13.281250 |
| all_six_equal | direct_three_geometric | 16.991306 | 32.291667 | 13.671875 |
| all_six_equal | direct_three_arithmetic | 16.930271 | 31.510417 | 14.062500 |
| all_six_equal | transport_CE_A | 15.135217 | 27.213542 | 12.239583 |
| all_six_equal | transport_geometric_A | 16.059077 | 29.036458 | 13.020833 |
| all_six_equal | transport_arithmetic_A | 16.230169 | 29.687500 | 13.151042 |
| all_six_equal | transport_CE_C | 15.131377 | 27.083333 | 11.979167 |
| all_six_equal | transport_geometric_C | 16.166792 | 29.036458 | 12.760417 |
| all_six_equal | transport_arithmetic_C | 16.227132 | 29.687500 | 12.760417 |
| all_six_equal | transport_CE_fusedABC | 15.482539 | 27.864583 | 12.630208 |
| all_six_equal | transport_geometric_fusedABC | 16.677866 | 30.989583 | 13.411458 |
| all_six_equal | transport_arithmetic_fusedABC | 16.642285 | 30.729167 | 13.411458 |
| all_six_equal | wrong_anchor_CE_A | 13.366400 | 25.000000 | 10.807292 |
| all_six_equal | wrong_anchor_geometric_A | 15.541169 | 28.645833 | 12.369792 |
| all_six_equal | wrong_anchor_arithmetic_A | 15.799829 | 29.296875 | 12.890625 |
| all_six_equal | wrong_anchor_CE_C | 13.243486 | 25.260417 | 10.156250 |
| all_six_equal | wrong_anchor_geometric_C | 15.341731 | 27.994792 | 11.718750 |
| all_six_equal | wrong_anchor_arithmetic_C | 15.812569 | 29.296875 | 12.369792 |
| all_six_equal | wrong_anchor_CE_fusedABC | 13.587936 | 26.302083 | 10.807292 |
| all_six_equal | wrong_anchor_geometric_fusedABC | 16.061798 | 29.947917 | 12.630208 |
| all_six_equal | wrong_anchor_arithmetic_fusedABC | 16.405184 | 30.859375 | 13.281250 |
| waymo | original_source | 11.473631 | 23.437500 | 7.421875 |
| waymo | prepared_frozen | 12.615516 | 26.171875 | 5.078125 |
| waymo | real_frozen_B | 12.944417 | 26.953125 | 5.468750 |
| waymo | real_frozen_C | 12.837875 | 26.562500 | 5.078125 |
| waymo | real_frozen_ensemble | 12.647714 | 26.562500 | 5.078125 |
| waymo | real_frozen_fusedABC | 12.569007 | 26.171875 | 5.078125 |
| waymo | real_TTA_C | 12.770512 | 26.562500 | 5.078125 |
| waymo | real_TTA_fusedABC | 12.663880 | 26.171875 | 5.078125 |
| waymo | randomA_frozen_fusedABC | 12.283274 | 25.000000 | 5.078125 |
| waymo | randomA_TTA_fusedABC | 12.338104 | 25.390625 | 5.078125 |
| waymo | wrong_frozen_fusedABC | 12.491509 | 26.171875 | 5.078125 |
| waymo | wrong_TTA_fusedABC | 12.325978 | 25.390625 | 5.078125 |
| waymo | teacher_shuffle_fusedABC | 12.723968 | 26.171875 | 5.078125 |
| waymo | softtoken_A | 12.200283 | 24.609375 | 5.468750 |
| waymo | softtoken_B | 12.564640 | 25.781250 | 5.468750 |
| waymo | softtoken_C | 12.515439 | 25.390625 | 5.468750 |
| waymo | softtoken_fusedABC | 12.644161 | 25.781250 | 5.468750 |
| waymo | softtoken_geometric_A | 13.155759 | 27.734375 | 5.468750 |
| waymo | softtoken_geometric_B | 13.479826 | 28.515625 | 5.468750 |
| waymo | softtoken_geometric_C | 13.390122 | 28.515625 | 5.468750 |
| waymo | softtoken_geometric_fusedABC | 13.415484 | 28.515625 | 5.468750 |
| waymo | softtoken_arithmetic_A | 13.096201 | 26.953125 | 5.859375 |
| waymo | softtoken_arithmetic_B | 13.178637 | 26.953125 | 5.859375 |
| waymo | softtoken_arithmetic_C | 13.346945 | 27.734375 | 5.859375 |
| waymo | softtoken_arithmetic_fusedABC | 13.307727 | 27.734375 | 5.859375 |
| waymo | softtoken_shuffled_geometric_A | 8.347614 | 16.796875 | 1.171875 |
| waymo | softtoken_shuffled_geometric_B | 8.050386 | 16.015625 | 1.562500 |
| waymo | softtoken_shuffled_geometric_C | 8.228805 | 16.406250 | 1.171875 |
| waymo | softtoken_shuffled_geometric_fusedABC | 7.964527 | 15.625000 | 1.171875 |
| waymo | native_original_CE | 11.117881 | 21.875000 | 7.031250 |
| waymo | original_source_matchedcoverage | 11.473631 | 23.437500 | 7.421875 |
| waymo | original_native_pair_geometric | 11.918529 | 24.218750 | 6.640625 |
| waymo | original_native_pair_arithmetic | 11.968590 | 25.000000 | 7.421875 |
| waymo | original_CE_with_prepared_geometric_A | 12.425512 | 25.781250 | 6.640625 |
| waymo | original_CE_with_prepared_arithmetic_A | 12.622200 | 26.562500 | 7.812500 |
| waymo | original_CE_with_fused_geometric | 12.584244 | 26.171875 | 7.031250 |
| waymo | original_CE_with_fused_arithmetic | 12.741501 | 26.562500 | 7.812500 |
| waymo | original_CE_with_C_geometric | 12.614893 | 26.171875 | 7.031250 |
| waymo | direct_three_geometric | 12.991047 | 27.343750 | 6.640625 |
| waymo | direct_three_arithmetic | 13.056678 | 27.343750 | 8.203125 |
| waymo | transport_CE_A | 11.569145 | 23.828125 | 4.687500 |
| waymo | transport_geometric_A | 12.475720 | 25.390625 | 5.859375 |
| waymo | transport_arithmetic_A | 12.970095 | 26.562500 | 6.250000 |
| waymo | transport_CE_C | 11.607331 | 23.828125 | 4.687500 |
| waymo | transport_geometric_C | 12.762537 | 26.171875 | 5.859375 |
| waymo | transport_arithmetic_C | 13.169974 | 27.343750 | 6.250000 |
| waymo | transport_CE_fusedABC | 11.591401 | 23.437500 | 4.687500 |
| waymo | transport_geometric_fusedABC | 12.530306 | 25.781250 | 5.859375 |
| waymo | transport_arithmetic_fusedABC | 12.903152 | 26.562500 | 6.250000 |
| waymo | wrong_anchor_CE_A | 10.986671 | 23.046875 | 4.687500 |
| waymo | wrong_anchor_geometric_A | 11.931138 | 25.390625 | 4.687500 |
| waymo | wrong_anchor_arithmetic_A | 12.381333 | 26.953125 | 5.468750 |
| waymo | wrong_anchor_CE_C | 11.456120 | 24.218750 | 5.078125 |
| waymo | wrong_anchor_geometric_C | 12.161394 | 25.781250 | 4.687500 |
| waymo | wrong_anchor_arithmetic_C | 12.948189 | 28.125000 | 5.468750 |
| waymo | wrong_anchor_CE_fusedABC | 11.419472 | 24.218750 | 4.687500 |
| waymo | wrong_anchor_geometric_fusedABC | 12.467739 | 26.562500 | 4.687500 |
| waymo | wrong_anchor_arithmetic_fusedABC | 12.761092 | 27.734375 | 5.468750 |
| drone | original_source | 13.867693 | 24.609375 | 12.500000 |
| drone | prepared_frozen | 14.071678 | 24.609375 | 11.718750 |
| drone | real_frozen_B | 13.984780 | 25.000000 | 11.718750 |
| drone | real_frozen_C | 13.842086 | 23.828125 | 11.718750 |
| drone | real_frozen_ensemble | 14.117679 | 25.000000 | 11.718750 |
| drone | real_frozen_fusedABC | 14.029708 | 25.000000 | 11.718750 |
| drone | real_TTA_C | 13.743545 | 23.437500 | 11.718750 |
| drone | real_TTA_fusedABC | 14.080314 | 25.000000 | 11.718750 |
| drone | randomA_frozen_fusedABC | 14.287690 | 25.781250 | 11.718750 |
| drone | randomA_TTA_fusedABC | 14.261923 | 25.781250 | 11.718750 |
| drone | wrong_frozen_fusedABC | 12.821754 | 22.656250 | 11.718750 |
| drone | wrong_TTA_fusedABC | 12.843616 | 22.656250 | 11.718750 |
| drone | teacher_shuffle_fusedABC | 13.903656 | 24.218750 | 11.718750 |
| drone | softtoken_A | 13.241966 | 23.046875 | 11.718750 |
| drone | softtoken_B | 13.245322 | 23.828125 | 11.718750 |
| drone | softtoken_C | 13.444898 | 23.046875 | 11.718750 |
| drone | softtoken_fusedABC | 13.274673 | 23.046875 | 11.718750 |
| drone | softtoken_geometric_A | 13.786263 | 23.828125 | 12.500000 |
| drone | softtoken_geometric_B | 13.969716 | 25.000000 | 12.500000 |
| drone | softtoken_geometric_C | 13.904868 | 23.828125 | 12.500000 |
| drone | softtoken_geometric_fusedABC | 13.957207 | 24.218750 | 12.500000 |
| drone | softtoken_arithmetic_A | 13.835012 | 23.437500 | 12.500000 |
| drone | softtoken_arithmetic_B | 14.176618 | 25.000000 | 12.500000 |
| drone | softtoken_arithmetic_C | 14.077124 | 23.828125 | 12.500000 |
| drone | softtoken_arithmetic_fusedABC | 14.196700 | 24.609375 | 12.500000 |
| drone | softtoken_shuffled_geometric_A | 7.454437 | 14.062500 | 1.171875 |
| drone | softtoken_shuffled_geometric_B | 7.339073 | 14.062500 | 1.562500 |
| drone | softtoken_shuffled_geometric_C | 8.204003 | 14.843750 | 1.562500 |
| drone | softtoken_shuffled_geometric_fusedABC | 7.796142 | 15.234375 | 1.171875 |
| drone | native_original_CE | 14.073336 | 23.828125 | 12.500000 |
| drone | original_source_matchedcoverage | 13.908446 | 24.218750 | 12.500000 |
| drone | original_native_pair_geometric | 14.196402 | 24.218750 | 12.500000 |
| drone | original_native_pair_arithmetic | 13.860068 | 23.046875 | 12.500000 |
| drone | original_CE_with_prepared_geometric_A | 13.932538 | 25.000000 | 11.328125 |
| drone | original_CE_with_prepared_arithmetic_A | 13.699276 | 23.437500 | 11.328125 |
| drone | original_CE_with_fused_geometric | 14.076427 | 25.781250 | 11.328125 |
| drone | original_CE_with_fused_arithmetic | 13.933085 | 24.218750 | 11.328125 |
| drone | original_CE_with_C_geometric | 13.810208 | 24.218750 | 11.328125 |
| drone | direct_three_geometric | 14.039081 | 26.171875 | 11.718750 |
| drone | direct_three_arithmetic | 13.961685 | 24.609375 | 11.328125 |
| drone | transport_CE_A | 12.588070 | 20.703125 | 11.328125 |
| drone | transport_geometric_A | 13.609855 | 22.265625 | 12.500000 |
| drone | transport_arithmetic_A | 13.962646 | 23.828125 | 12.500000 |
| drone | transport_CE_C | 12.715729 | 20.312500 | 11.328125 |
| drone | transport_geometric_C | 13.745269 | 21.875000 | 12.500000 |
| drone | transport_arithmetic_C | 13.994815 | 23.437500 | 12.500000 |
| drone | transport_CE_fusedABC | 12.518707 | 20.703125 | 11.328125 |
| drone | transport_geometric_fusedABC | 14.142526 | 23.828125 | 12.500000 |
| drone | transport_arithmetic_fusedABC | 14.035578 | 24.218750 | 12.500000 |
| drone | wrong_anchor_CE_A | 10.689414 | 18.359375 | 10.156250 |
| drone | wrong_anchor_geometric_A | 13.397311 | 22.265625 | 12.109375 |
| drone | wrong_anchor_arithmetic_A | 13.146092 | 21.875000 | 12.109375 |
| drone | wrong_anchor_CE_C | 10.502994 | 17.968750 | 10.156250 |
| drone | wrong_anchor_geometric_C | 13.314989 | 21.875000 | 12.109375 |
| drone | wrong_anchor_arithmetic_C | 13.072823 | 21.875000 | 12.109375 |
| drone | wrong_anchor_CE_fusedABC | 10.603580 | 19.140625 | 10.156250 |
| drone | wrong_anchor_geometric_fusedABC | 13.439952 | 22.656250 | 12.109375 |
| drone | wrong_anchor_arithmetic_fusedABC | 13.087041 | 22.265625 | 12.109375 |
| quad | original_source | 25.092825 | 44.921875 | 25.000000 |
| quad | prepared_frozen | 21.998762 | 39.062500 | 21.093750 |
| quad | real_frozen_B | 22.127007 | 40.234375 | 19.921875 |
| quad | real_frozen_C | 21.705129 | 38.281250 | 19.921875 |
| quad | real_frozen_ensemble | 22.739645 | 40.625000 | 20.703125 |
| quad | real_frozen_fusedABC | 23.510976 | 41.796875 | 22.656250 |
| quad | real_TTA_C | 22.426695 | 39.453125 | 21.484375 |
| quad | real_TTA_fusedABC | 22.980680 | 40.625000 | 22.265625 |
| quad | randomA_frozen_fusedABC | 23.109373 | 41.015625 | 22.656250 |
| quad | randomA_TTA_fusedABC | 23.288519 | 41.015625 | 22.656250 |
| quad | wrong_frozen_fusedABC | 15.118459 | 25.781250 | 12.890625 |
| quad | wrong_TTA_fusedABC | 14.917313 | 25.781250 | 13.281250 |
| quad | teacher_shuffle_fusedABC | 22.213422 | 39.062500 | 20.703125 |
| quad | softtoken_A | 21.948938 | 39.062500 | 21.484375 |
| quad | softtoken_B | 21.802300 | 38.671875 | 20.703125 |
| quad | softtoken_C | 21.327746 | 37.109375 | 19.921875 |
| quad | softtoken_fusedABC | 22.559141 | 40.234375 | 21.875000 |
| quad | softtoken_geometric_A | 21.940804 | 39.453125 | 20.703125 |
| quad | softtoken_geometric_B | 22.235792 | 39.843750 | 20.703125 |
| quad | softtoken_geometric_C | 21.657219 | 37.890625 | 19.921875 |
| quad | softtoken_geometric_fusedABC | 23.280353 | 42.187500 | 21.875000 |
| quad | softtoken_arithmetic_A | 21.727300 | 38.281250 | 21.093750 |
| quad | softtoken_arithmetic_B | 22.007202 | 39.062500 | 21.093750 |
| quad | softtoken_arithmetic_C | 21.623277 | 37.500000 | 20.312500 |
| quad | softtoken_arithmetic_fusedABC | 23.217157 | 41.015625 | 22.656250 |
| quad | softtoken_shuffled_geometric_A | 6.782647 | 11.328125 | 3.515625 |
| quad | softtoken_shuffled_geometric_B | 6.471538 | 11.718750 | 3.515625 |
| quad | softtoken_shuffled_geometric_C | 6.324177 | 10.546875 | 2.734375 |
| quad | softtoken_shuffled_geometric_fusedABC | 6.511357 | 10.937500 | 3.515625 |
| quad | native_original_CE | 24.456131 | 44.140625 | 23.437500 |
| quad | original_source_matchedcoverage | 24.943393 | 44.921875 | 24.609375 |
| quad | original_native_pair_geometric | 24.570698 | 44.531250 | 24.218750 |
| quad | original_native_pair_arithmetic | 24.561331 | 44.140625 | 24.218750 |
| quad | original_CE_with_prepared_geometric_A | 23.075243 | 41.015625 | 22.265625 |
| quad | original_CE_with_prepared_arithmetic_A | 24.059959 | 42.968750 | 22.656250 |
| quad | original_CE_with_fused_geometric | 24.016130 | 43.359375 | 23.046875 |
| quad | original_CE_with_fused_arithmetic | 24.011011 | 42.578125 | 23.046875 |
| quad | original_CE_with_C_geometric | 23.100071 | 42.578125 | 21.484375 |
| quad | direct_three_geometric | 23.943791 | 43.359375 | 22.656250 |
| quad | direct_three_arithmetic | 23.772449 | 42.578125 | 22.656250 |
| quad | transport_CE_A | 21.248435 | 37.109375 | 20.703125 |
| quad | transport_geometric_A | 22.091655 | 39.453125 | 20.703125 |
| quad | transport_arithmetic_A | 21.757767 | 38.671875 | 20.703125 |
| quad | transport_CE_C | 21.071071 | 37.109375 | 19.921875 |
| quad | transport_geometric_C | 21.992569 | 39.062500 | 19.921875 |
| quad | transport_arithmetic_C | 21.516606 | 38.281250 | 19.531250 |
| quad | transport_CE_fusedABC | 22.337508 | 39.453125 | 21.875000 |
| quad | transport_geometric_fusedABC | 23.360765 | 43.359375 | 21.875000 |
| quad | transport_arithmetic_fusedABC | 22.988126 | 41.406250 | 21.484375 |
| quad | wrong_anchor_CE_A | 18.423115 | 33.593750 | 17.578125 |
| quad | wrong_anchor_geometric_A | 21.295058 | 38.281250 | 20.312500 |
| quad | wrong_anchor_arithmetic_A | 21.872061 | 39.062500 | 21.093750 |
| quad | wrong_anchor_CE_C | 17.771344 | 33.593750 | 15.234375 |
| quad | wrong_anchor_geometric_C | 20.548809 | 36.328125 | 18.359375 |
| quad | wrong_anchor_arithmetic_C | 21.416695 | 37.890625 | 19.531250 |
| quad | wrong_anchor_CE_fusedABC | 18.740756 | 35.546875 | 17.578125 |
| quad | wrong_anchor_geometric_fusedABC | 22.277703 | 40.625000 | 21.093750 |
| quad | wrong_anchor_arithmetic_fusedABC | 23.367420 | 42.578125 | 22.265625 |

## 正负对照

| Source | Target | 方法 | 对照 | Δ pp | 改善/恶化/不变 |
|---|---|---|---|---:|---:|
| waymo | drone | transport_geometric_fusedABC | original_source | +0.753635 | 19/22/87 |
| waymo | drone | transport_geometric_fusedABC | original_source_matchedcoverage | +0.694356 | 18/22/88 |
| waymo | drone | transport_geometric_fusedABC | prepared_frozen | +0.414867 | 19/14/95 |
| waymo | drone | transport_geometric_fusedABC | real_frozen_fusedABC | +0.011477 | 16/15/97 |
| waymo | drone | transport_geometric_fusedABC | softtoken_geometric_fusedABC | +0.577450 | 10/6/112 |
| waymo | drone | transport_geometric_fusedABC | softtoken_arithmetic_fusedABC | +0.251116 | 11/6/111 |
| waymo | drone | transport_geometric_fusedABC | native_original_CE | +0.945311 | 18/20/90 |
| waymo | drone | transport_geometric_fusedABC | original_CE_with_fused_geometric | +0.710668 | 12/15/101 |
| waymo | drone | transport_geometric_fusedABC | original_CE_with_fused_arithmetic | +0.503507 | 15/14/99 |
| waymo | drone | transport_geometric_fusedABC | direct_three_geometric | +1.234548 | 11/13/104 |
| waymo | drone | transport_geometric_fusedABC | direct_three_arithmetic | +0.185666 | 12/14/102 |
| waymo | drone | transport_geometric_fusedABC | wrong_anchor_geometric_fusedABC | +0.797286 | 17/10/101 |
| waymo | drone | transport_CE_A | softtoken_A | -1.108373 | 5/16/107 |
| waymo | drone | transport_CE_A | native_original_CE | -1.660291 | 9/22/97 |
| waymo | drone | transport_geometric_C | original_CE_with_C_geometric | +0.110212 | 14/16/98 |
| waymo | drone | transport_geometric_C | real_frozen_C | -0.353713 | 15/13/100 |
| waymo | quad | transport_geometric_fusedABC | original_source | -1.520222 | 17/16/95 |
| waymo | quad | transport_geometric_fusedABC | original_source_matchedcoverage | -1.023295 | 14/14/100 |
| waymo | quad | transport_geometric_fusedABC | prepared_frozen | +2.212156 | 14/7/107 |
| waymo | quad | transport_geometric_fusedABC | real_frozen_fusedABC | -0.216554 | 8/6/114 |
| waymo | quad | transport_geometric_fusedABC | softtoken_geometric_fusedABC | -0.115200 | 2/1/125 |
| waymo | quad | transport_geometric_fusedABC | softtoken_arithmetic_fusedABC | +0.179302 | 4/2/122 |
| waymo | quad | transport_geometric_fusedABC | native_original_CE | -0.867766 | 16/15/97 |
| waymo | quad | transport_geometric_fusedABC | original_CE_with_fused_geometric | -0.820050 | 9/12/107 |
| waymo | quad | transport_geometric_fusedABC | original_CE_with_fused_arithmetic | -0.688897 | 12/10/106 |
| waymo | quad | transport_geometric_fusedABC | direct_three_geometric | -0.846578 | 9/8/111 |
| waymo | quad | transport_geometric_fusedABC | direct_three_arithmetic | -0.507185 | 10/9/109 |
| waymo | quad | transport_geometric_fusedABC | wrong_anchor_geometric_fusedABC | +1.785923 | 10/3/115 |
| waymo | quad | transport_CE_A | softtoken_A | -0.654081 | 1/7/120 |
| waymo | quad | transport_CE_A | native_original_CE | -3.340122 | 9/19/100 |
| waymo | quad | transport_geometric_C | original_CE_with_C_geometric | -1.233971 | 12/15/101 |
| waymo | quad | transport_geometric_C | real_frozen_C | +1.426551 | 7/3/118 |
| drone | waymo | transport_geometric_fusedABC | original_source | +0.084554 | 24/26/78 |
| drone | waymo | transport_geometric_fusedABC | original_source_matchedcoverage | +0.084554 | 24/26/78 |
| drone | waymo | transport_geometric_fusedABC | prepared_frozen | -0.605126 | 9/20/99 |
| drone | waymo | transport_geometric_fusedABC | real_frozen_fusedABC | -0.303140 | 10/19/99 |
| drone | waymo | transport_geometric_fusedABC | softtoken_geometric_fusedABC | -1.702636 | 4/14/110 |
| drone | waymo | transport_geometric_fusedABC | softtoken_arithmetic_fusedABC | -1.563252 | 3/15/110 |
| drone | waymo | transport_geometric_fusedABC | native_original_CE | +1.427395 | 27/17/84 |
| drone | waymo | transport_geometric_fusedABC | original_CE_with_fused_geometric | -0.517449 | 18/22/88 |
| drone | waymo | transport_geometric_fusedABC | original_CE_with_fused_arithmetic | -1.089722 | 18/21/89 |
| drone | waymo | transport_geometric_fusedABC | direct_three_geometric | -1.057119 | 13/16/99 |
| drone | waymo | transport_geometric_fusedABC | direct_three_arithmetic | -1.345728 | 14/19/95 |
| drone | waymo | transport_geometric_fusedABC | wrong_anchor_geometric_fusedABC | -0.750516 | 18/14/96 |
| drone | waymo | transport_CE_A | softtoken_A | -0.980465 | 8/13/107 |
| drone | waymo | transport_CE_A | native_original_CE | +0.158743 | 21/21/86 |
| drone | waymo | transport_geometric_C | original_CE_with_C_geometric | -0.324086 | 19/23/86 |
| drone | waymo | transport_geometric_C | real_frozen_C | -0.495303 | 9/20/99 |
| drone | quad | transport_geometric_fusedABC | original_source | -1.943897 | 16/23/89 |
| drone | quad | transport_geometric_fusedABC | original_source_matchedcoverage | -2.141960 | 14/23/91 |
| drone | quad | transport_geometric_fusedABC | prepared_frozen | +0.511850 | 16/11/101 |
| drone | quad | transport_geometric_fusedABC | real_frozen_fusedABC | -0.083867 | 7/7/114 |
| drone | quad | transport_geometric_fusedABC | softtoken_geometric_fusedABC | +0.276025 | 4/0/124 |
| drone | quad | transport_geometric_fusedABC | softtoken_arithmetic_fusedABC | +0.107915 | 6/3/119 |
| drone | quad | transport_geometric_fusedABC | native_original_CE | -1.322965 | 18/21/89 |
| drone | quad | transport_geometric_fusedABC | original_CE_with_fused_geometric | -0.490679 | 12/12/104 |
| drone | quad | transport_geometric_fusedABC | original_CE_with_fused_arithmetic | -0.611594 | 14/15/99 |
| drone | quad | transport_geometric_fusedABC | direct_three_geometric | -0.319473 | 9/7/112 |
| drone | quad | transport_geometric_fusedABC | direct_three_arithmetic | -0.316184 | 11/10/107 |
| drone | quad | transport_geometric_fusedABC | wrong_anchor_geometric_fusedABC | +0.380202 | 11/10/107 |
| drone | quad | transport_CE_A | softtoken_A | -0.746925 | 3/8/117 |
| drone | quad | transport_CE_A | native_original_CE | -3.075269 | 14/25/89 |
| drone | quad | transport_geometric_C | original_CE_with_C_geometric | -0.981034 | 17/13/98 |
| drone | quad | transport_geometric_C | real_frozen_C | -0.851671 | 7/9/112 |
| quad | waymo | transport_geometric_fusedABC | original_source | +2.028797 | 26/21/81 |
| quad | waymo | transport_geometric_fusedABC | original_source_matchedcoverage | +2.028797 | 26/21/81 |
| quad | waymo | transport_geometric_fusedABC | prepared_frozen | +0.434707 | 5/5/118 |
| quad | waymo | transport_geometric_fusedABC | real_frozen_fusedABC | +0.225739 | 5/5/118 |
| quad | waymo | transport_geometric_fusedABC | softtoken_geometric_fusedABC | -0.067720 | 3/4/121 |
| quad | waymo | transport_geometric_fusedABC | softtoken_arithmetic_fusedABC | +0.008411 | 4/4/120 |
| quad | waymo | transport_geometric_fusedABC | native_original_CE | +1.397456 | 20/18/90 |
| quad | waymo | transport_geometric_fusedABC | original_CE_with_fused_geometric | +0.409573 | 12/15/101 |
| quad | waymo | transport_geometric_fusedABC | original_CE_with_fused_arithmetic | +0.667332 | 16/19/93 |
| quad | waymo | transport_geometric_fusedABC | direct_three_geometric | +0.135638 | 8/7/113 |
| quad | waymo | transport_geometric_fusedABC | direct_three_arithmetic | +0.292984 | 9/10/109 |
| quad | waymo | transport_geometric_fusedABC | wrong_anchor_geometric_fusedABC | +0.875651 | 12/8/108 |
| quad | waymo | transport_CE_A | softtoken_A | -0.281810 | 11/7/110 |
| quad | waymo | transport_CE_A | native_original_CE | +0.743785 | 22/16/90 |
| quad | waymo | transport_geometric_C | original_CE_with_C_geometric | +0.619374 | 14/14/100 |
| quad | waymo | transport_geometric_C | real_frozen_C | +0.344627 | 6/5/117 |
| quad | drone | transport_geometric_fusedABC | original_source | -0.203968 | 28/22/78 |
| quad | drone | transport_geometric_fusedABC | original_source_matchedcoverage | -0.226196 | 23/20/85 |
| quad | drone | transport_geometric_fusedABC | prepared_frozen | -0.273170 | 4/10/114 |
| quad | drone | transport_geometric_fusedABC | real_frozen_fusedABC | +0.214159 | 4/5/119 |
| quad | drone | transport_geometric_fusedABC | softtoken_geometric_fusedABC | -0.206812 | 3/4/121 |
| quad | drone | transport_geometric_fusedABC | softtoken_arithmetic_fusedABC | -0.359463 | 3/4/121 |
| quad | drone | transport_geometric_fusedABC | native_original_CE | -0.806929 | 20/19/89 |
| quad | drone | transport_geometric_fusedABC | original_CE_with_fused_geometric | -0.578469 | 10/16/102 |
| quad | drone | transport_geometric_fusedABC | original_CE_with_fused_arithmetic | -0.084624 | 17/15/96 |
| quad | drone | transport_geometric_fusedABC | direct_three_geometric | -1.027658 | 5/12/111 |
| quad | drone | transport_geometric_fusedABC | direct_three_arithmetic | +0.176017 | 11/10/107 |
| quad | drone | transport_geometric_fusedABC | wrong_anchor_geometric_fusedABC | +0.607862 | 7/3/118 |
| quad | drone | transport_CE_A | softtoken_A | -0.199420 | 2/3/123 |
| quad | drone | transport_CE_A | native_original_CE | -1.310241 | 15/18/95 |
| quad | drone | transport_geometric_C | original_CE_with_C_geometric | -0.240090 | 13/15/100 |
| quad | drone | transport_geometric_C | real_frozen_C | +0.160077 | 2/4/122 |

## 接口诊断

{
  "native_original_CE": 16.54911593597195,
  "prepared_A_CE": 15.797062581438867,
  "prepared_fused_CE": 16.159324747746535,
  "transport_prepared_A_CE": 15.135216788404255,
  "transport_fused_CE": 15.482538809151999,
  "not_a_source_space_fault_proof": true
}

## 实际状态变化

[
  {
    "calibration": "transport",
    "n": 724,
    "mean_buffer_L2": 11.500897792505038,
    "mean_changed_values": 1152.0,
    "mean_eligible_channels_per_layer": [
      288.0,
      288.0
    ],
    "learned_parameters_changed": 0,
    "running_stat_values": 1152
  },
  {
    "calibration": "wrong_anchor",
    "n": 724,
    "mean_buffer_L2": 48.48167347609366,
    "mean_changed_values": 1152.0,
    "mean_eligible_channels_per_layer": [
      288.0,
      288.0
    ],
    "learned_parameters_changed": 0,
    "running_stat_values": 1152
  }
]

## 裁决

{
  "primary": "transport_geometric_fusedABC",
  "mIoU_percent": 16.67786603681623,
  "gains_pp": {
    "original_source": -0.13351696190102302,
    "original_source_matchedcoverage": -0.09729070588235444,
    "prepared_frozen": 0.4492138776275496,
    "real_frozen_fusedABC": -0.02536425614261617,
    "softtoken_geometric_fusedABC": -0.20648219157725123,
    "softtoken_arithmetic_fusedABC": -0.2293287312103196,
    "native_original_CE": 0.12875010084427885,
    "original_CE_with_fused_geometric": -0.2144008225654659,
    "original_CE_with_fused_arithmetic": -0.21733294031869832,
    "direct_three_geometric": -0.3134402276774786,
    "direct_three_arithmetic": -0.25240479009467265,
    "wrong_anchor_geometric_fusedABC": 0.6160680195363923
  },
  "passes_development_gates": false,
  "robust_adoption": false,
  "decision": "do_not_adopt_paired_BN_transport_recipe",
  "n": 768,
  "running_statistics_only_TTA": true,
  "learned_weights_updated": false,
  "gradient_or_pseudo_labels": false,
  "primary_not_switched": true,
  "FINAL_access": false
}

## 不可扩大结论

源original CE较差不能证明所有接口兼容，较好也不能证明全部错误来自BN。校准矩或归一化数值满足目标，不是grounding有效性。actual_C和全部输入分开；所有负例、常量通道、错配、别名回退均保留。该方法不是TTN复现，TTN原方法需source后训练学习插值权重，本轮没有做。未过任务门槛则停止这一配方，不增加校准强度/层数或按病例、平台选择统计。

## 逐例实际看图（封存后解释，不筛分母）

{
  "generated_pages": 11,
  "actually_viewed_pages": 4,
  "not_gold_identity_annotations": true,
  "unviewed_UNKNOWN": true,
  "observations": [
    "Viewed: CE/oldpool favor an elevated tree/background region, whereas transported pool returns to the visible near van already supported by frozen BBF. This partly restores a useful fusion choice and changes the overlapping box; not independent identity certification.",
    "Viewed: old and new choices overlap the same visible SUV vicinity. The decrease is consistent with different overlapping boxes rather than a certified different-instance switch; exact 3D identity UNKNOWN.",
    "Viewed: old CE/geometric favors a different region near the neighboring vehicle, while transport returns toward the partly out-of-image SUV supported by prepared/fused BBF. Out-of-image geometry and best-candidate overlap limit interpretation; identity not certified.",
    "Viewed: old CE/geometric favors the left-rear dark vehicle region, while transport returns to a spatially separated prepared/fused choice farther in BEV and loses overlap. This erases useful CE correction; not merely smaller movement within the same box."
  ],
  "no_case_specific_filter_or_GT_gate": true
}

正例同时包括从背景区域回到可见车辆、恢复冻结融合选择；负例既有同一车辆附近重复框的切换，也有抹掉原CE纠错后回到不同区域。不能将全部变化归为实例修复，也不能把矩配对有效等同于语义身份正确。
