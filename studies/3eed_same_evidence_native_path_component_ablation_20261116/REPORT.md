# R111 — 同一实测点云证据的原始路径撤回对照

R110统计适应失败后，先区分新增观测不足与source-prepared路径干扰。此轮没有TTA、梯度、更新或新监督；没有替换生产主方法。原始3EED和既有R91双向模型均保留。原A256/六层source A位置/三source100/物理配准/16384点预算/语句/UNKNOWN策略固定。全部768输入，44保护别名不重读而以prepared冻结回退计入。

## 最小改变与强对照

唯一主干差异是把与R106完全相同的已对齐B/C PointNet特征送入原始cross-encoder/decoder和原始匹配头，撤回R91新增分支。各层位置仍取source A，原框不回归；不是换一组候选。真实/同A随机重采样/错配三组都比较B、C、早期特征融合和最终概率均值。相同信息的native概率均值是必要强对照。其余58已有读出及评分严格复用，仅新增17读出。没有因为旧CE配方失败而再拟合其输出。

原始A重放的h/query/tokens与R110全部位等。首次正式尝试在第73条触发1e-7概率校验：批量矩乘读出与规范二维矩乘相差1.4685e-7，h/query/tokens不变。独立探针确认二维读出误差为0后，在GT读取之前修正；前72条复用隐藏表示重放读出，没有重跑PointNet/decoder。失败版本与封存哈希保留。12工程输入再次通过规范读出、身份融合、物理采样哈希、源指纹和独立CPU匹配投影验证。不是提高误差容忍来绕过断言。

## 原始与prepared匹配头的受控互换

O表示原始h与其文字表示，P表示prepared h与其文字表示。各输入系统内部只替换投影层；基础项均不含late RGB与竞争残差。完整prepared行保留作为应用系统参照。跨系统hidden即使维数相同也不假定语义基相同，互换是干预，不是把某一头当正确教师。

```json
{
  "source_native_projection": 16.775156742698584,
  "source_prepared_projection": 16.775156742698584,
  "preparedA_native_projection_base": 16.08464079448645,
  "preparedA_prepared_projection_base": 16.08464079448645,
  "preparedF_native_projection_base": 16.67646246040571,
  "preparedF_prepared_projection_base": 16.67646246040571,
  "preparedA_full": 16.22865215918868,
  "preparedF_full": 16.703230292958846,
  "head_comparison_h_and_text_fixed_within_input_system": true,
  "base_rows_exclude_late_RGB_and_competition_residual": true,
  "latent_system_comparison_includes_its_own_conditioned_text": true,
  "not_a_deployment_GT_selector": true
}
```

## 全体/平台结果

| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |
|---|---|---:|---:|---:|
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
| all_six_equal | native_real_B | 16.622544 | 30.338542 | 14.713542 |
| all_six_equal | native_real_C | 16.170858 | 29.817708 | 14.192708 |
| all_six_equal | native_real_fusedABC | 16.974318 | 31.250000 | 14.843750 |
| all_six_equal | native_real_ensemble | 16.734016 | 30.729167 | 14.843750 |
| all_six_equal | native_randomA_B | 16.897814 | 31.119792 | 14.713542 |
| all_six_equal | native_randomA_C | 16.801522 | 30.859375 | 14.583333 |
| all_six_equal | native_randomA_fusedABC | 16.781049 | 30.859375 | 14.843750 |
| all_six_equal | native_randomA_ensemble | 16.791696 | 30.729167 | 15.104167 |
| all_six_equal | native_wrong_B | 11.918293 | 21.093750 | 10.156250 |
| all_six_equal | native_wrong_C | 11.548736 | 20.703125 | 9.765625 |
| all_six_equal | native_wrong_fusedABC | 14.013089 | 25.520833 | 12.369792 |
| all_six_equal | native_wrong_ensemble | 15.302346 | 28.125000 | 13.802083 |
| all_six_equal | sourceA_prepared_head | 16.775157 | 30.859375 | 14.843750 |
| all_six_equal | preparedA_native_head_base | 16.084641 | 30.208333 | 12.760417 |
| all_six_equal | preparedA_prepared_head_base | 16.084641 | 30.208333 | 12.760417 |
| all_six_equal | preparedF_native_head_base | 16.676462 | 31.640625 | 13.020833 |
| all_six_equal | preparedF_prepared_head_base | 16.676462 | 31.640625 | 13.020833 |
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
| waymo | native_real_B | 11.231002 | 23.046875 | 7.421875 |
| waymo | native_real_C | 11.300386 | 22.656250 | 7.421875 |
| waymo | native_real_fusedABC | 11.535583 | 23.437500 | 7.421875 |
| waymo | native_real_ensemble | 11.215742 | 22.656250 | 7.421875 |
| waymo | native_randomA_B | 11.139854 | 23.046875 | 7.031250 |
| waymo | native_randomA_C | 11.390925 | 23.828125 | 7.421875 |
| waymo | native_randomA_fusedABC | 11.278876 | 23.046875 | 7.421875 |
| waymo | native_randomA_ensemble | 11.108901 | 23.046875 | 7.421875 |
| waymo | native_wrong_B | 10.319392 | 20.703125 | 6.640625 |
| waymo | native_wrong_C | 10.547527 | 21.484375 | 7.031250 |
| waymo | native_wrong_fusedABC | 10.867719 | 22.656250 | 6.640625 |
| waymo | native_wrong_ensemble | 10.531234 | 21.875000 | 6.640625 |
| waymo | sourceA_prepared_head | 11.473631 | 23.437500 | 7.421875 |
| waymo | preparedA_native_head_base | 12.512826 | 27.343750 | 5.078125 |
| waymo | preparedA_prepared_head_base | 12.512826 | 27.343750 | 5.078125 |
| waymo | preparedF_native_head_base | 12.680599 | 28.125000 | 5.078125 |
| waymo | preparedF_prepared_head_base | 12.680599 | 28.125000 | 5.078125 |
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
| drone | native_real_B | 14.266625 | 24.609375 | 12.500000 |
| drone | native_real_C | 14.016835 | 24.609375 | 12.500000 |
| drone | native_real_fusedABC | 14.710633 | 26.171875 | 12.890625 |
| drone | native_real_ensemble | 14.188880 | 24.609375 | 12.500000 |
| drone | native_randomA_B | 14.810559 | 26.171875 | 12.500000 |
| drone | native_randomA_C | 14.189341 | 25.000000 | 12.109375 |
| drone | native_randomA_fusedABC | 14.658134 | 25.781250 | 12.500000 |
| drone | native_randomA_ensemble | 14.058230 | 23.828125 | 12.500000 |
| drone | native_wrong_B | 13.738000 | 23.437500 | 12.500000 |
| drone | native_wrong_C | 13.443447 | 23.046875 | 12.109375 |
| drone | native_wrong_fusedABC | 13.843828 | 24.218750 | 12.109375 |
| drone | native_wrong_ensemble | 14.043059 | 24.609375 | 12.500000 |
| drone | sourceA_prepared_head | 13.908446 | 24.218750 | 12.500000 |
| drone | preparedA_native_head_base | 13.974539 | 25.000000 | 11.718750 |
| drone | preparedA_prepared_head_base | 13.974539 | 25.000000 | 11.718750 |
| drone | preparedF_native_head_base | 14.163699 | 25.390625 | 11.718750 |
| drone | preparedF_prepared_head_base | 14.163699 | 25.390625 | 11.718750 |
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
| quad | native_real_B | 24.370005 | 43.359375 | 24.218750 |
| quad | native_real_C | 23.195354 | 42.187500 | 22.656250 |
| quad | native_real_fusedABC | 24.676737 | 44.140625 | 24.218750 |
| quad | native_real_ensemble | 24.797425 | 44.921875 | 24.609375 |
| quad | native_randomA_B | 24.743028 | 44.140625 | 24.609375 |
| quad | native_randomA_C | 24.824301 | 43.750000 | 24.218750 |
| quad | native_randomA_fusedABC | 24.406137 | 43.750000 | 24.609375 |
| quad | native_randomA_ensemble | 25.207959 | 45.312500 | 25.390625 |
| quad | native_wrong_B | 11.697488 | 19.140625 | 11.328125 |
| quad | native_wrong_C | 10.655233 | 17.578125 | 10.156250 |
| quad | native_wrong_fusedABC | 17.327721 | 29.687500 | 18.359375 |
| quad | native_wrong_ensemble | 21.332745 | 37.890625 | 22.265625 |
| quad | sourceA_prepared_head | 24.943393 | 44.921875 | 24.609375 |
| quad | preparedA_native_head_base | 21.766557 | 38.281250 | 21.484375 |
| quad | preparedA_prepared_head_base | 21.766557 | 38.281250 | 21.484375 |
| quad | preparedF_native_head_base | 23.185090 | 41.406250 | 22.265625 |
| quad | preparedF_prepared_head_base | 23.185090 | 41.406250 | 22.265625 |

## 全部配对结果

| Source | Target | 方法 | 对照 | Δ pp | 改善/恶化/不变 |
|---|---|---|---|---:|---:|
| waymo | drone | native_real_fusedABC | original_source | +1.173821 | 7/3/118 |
| waymo | drone | native_real_fusedABC | original_source_matchedcoverage | +1.114543 | 6/3/119 |
| waymo | drone | native_real_fusedABC | prepared_frozen | +0.835053 | 31/14/83 |
| waymo | drone | native_real_fusedABC | native_real_ensemble | +0.356729 | 2/2/124 |
| waymo | drone | native_real_fusedABC | native_randomA_fusedABC | +0.402170 | 7/5/116 |
| waymo | drone | native_real_fusedABC | native_wrong_fusedABC | +0.939464 | 10/4/114 |
| waymo | drone | native_real_fusedABC | real_frozen_fusedABC | +0.431664 | 28/15/85 |
| waymo | drone | native_real_fusedABC | real_frozen_ensemble | +0.334049 | 27/14/87 |
| waymo | drone | native_real_fusedABC | real_TTA_fusedABC | +0.351359 | 28/15/85 |
| waymo | drone | native_real_fusedABC | direct_three_geometric | +1.654735 | 20/12/96 |
| waymo | drone | preparedA_native_head_base | preparedA_prepared_head_base | +0.000000 | 0/0/128 |
| waymo | drone | preparedF_native_head_base | preparedF_prepared_head_base | +0.000000 | 0/0/128 |
| waymo | drone | sourceA_prepared_head | original_source_matchedcoverage | +0.000000 | 0/0/128 |
| waymo | quad | native_real_fusedABC | original_source | -1.256103 | 6/10/112 |
| waymo | quad | native_real_fusedABC | original_source_matchedcoverage | -0.759175 | 3/8/117 |
| waymo | quad | native_real_fusedABC | prepared_frozen | +2.476275 | 15/14/99 |
| waymo | quad | native_real_fusedABC | native_real_ensemble | -1.617573 | 0/6/122 |
| waymo | quad | native_real_fusedABC | native_randomA_fusedABC | +0.046162 | 5/9/114 |
| waymo | quad | native_real_fusedABC | native_wrong_fusedABC | +5.709279 | 20/4/104 |
| waymo | quad | native_real_fusedABC | real_frozen_fusedABC | +0.047565 | 15/15/98 |
| waymo | quad | native_real_fusedABC | real_frozen_ensemble | +1.540615 | 14/13/101 |
| waymo | quad | native_real_fusedABC | real_TTA_fusedABC | +1.230738 | 15/13/100 |
| waymo | quad | native_real_fusedABC | direct_three_geometric | -0.582459 | 7/11/110 |
| waymo | quad | preparedA_native_head_base | preparedA_prepared_head_base | +0.000000 | 0/0/128 |
| waymo | quad | preparedF_native_head_base | preparedF_prepared_head_base | +0.000000 | 0/0/128 |
| waymo | quad | sourceA_prepared_head | original_source_matchedcoverage | +0.000000 | 0/0/128 |
| drone | waymo | native_real_fusedABC | original_source | +0.297950 | 4/1/123 |
| drone | waymo | native_real_fusedABC | original_source_matchedcoverage | +0.297950 | 4/1/123 |
| drone | waymo | native_real_fusedABC | prepared_frozen | -0.391730 | 20/22/86 |
| drone | waymo | native_real_fusedABC | native_real_ensemble | +0.553325 | 3/0/125 |
| drone | waymo | native_real_fusedABC | native_randomA_fusedABC | +0.427057 | 6/0/122 |
| drone | waymo | native_real_fusedABC | native_wrong_fusedABC | +1.250867 | 8/2/118 |
| drone | waymo | native_real_fusedABC | real_frozen_fusedABC | -0.089744 | 19/20/89 |
| drone | waymo | native_real_fusedABC | real_frozen_ensemble | -0.247158 | 19/21/88 |
| drone | waymo | native_real_fusedABC | real_TTA_fusedABC | -0.250073 | 19/21/88 |
| drone | waymo | native_real_fusedABC | direct_three_geometric | -0.843723 | 24/20/84 |
| drone | waymo | preparedA_native_head_base | preparedA_prepared_head_base | +0.000000 | 0/0/128 |
| drone | waymo | preparedF_native_head_base | preparedF_prepared_head_base | +0.000000 | 0/0/128 |
| drone | waymo | sourceA_prepared_head | original_source_matchedcoverage | +0.000000 | 0/0/128 |
| drone | quad | native_real_fusedABC | original_source | +0.423927 | 10/8/110 |
| drone | quad | native_real_fusedABC | original_source_matchedcoverage | +0.225864 | 8/8/112 |
| drone | quad | native_real_fusedABC | prepared_frozen | +2.879674 | 24/14/90 |
| drone | quad | native_real_fusedABC | native_real_ensemble | +1.376197 | 6/1/121 |
| drone | quad | native_real_fusedABC | native_randomA_fusedABC | +0.495037 | 9/13/106 |
| drone | quad | native_real_fusedABC | native_wrong_fusedABC | +8.988752 | 41/9/78 |
| drone | quad | native_real_fusedABC | real_frozen_fusedABC | +2.283957 | 17/11/100 |
| drone | quad | native_real_fusedABC | real_frozen_ensemble | +2.333569 | 21/10/97 |
| drone | quad | native_real_fusedABC | real_TTA_fusedABC | +2.161376 | 17/11/100 |
| drone | quad | native_real_fusedABC | direct_three_geometric | +2.048351 | 19/7/102 |
| drone | quad | preparedA_native_head_base | preparedA_prepared_head_base | +0.000000 | 0/0/128 |
| drone | quad | preparedF_native_head_base | preparedF_prepared_head_base | +0.000000 | 0/0/128 |
| drone | quad | sourceA_prepared_head | original_source_matchedcoverage | +0.000000 | 0/0/128 |
| quad | waymo | native_real_fusedABC | original_source | -0.174046 | 2/1/125 |
| quad | waymo | native_real_fusedABC | original_source_matchedcoverage | -0.174046 | 2/1/125 |
| quad | waymo | native_real_fusedABC | prepared_frozen | -1.768136 | 21/24/83 |
| quad | waymo | native_real_fusedABC | native_real_ensemble | +0.086356 | 1/0/127 |
| quad | waymo | native_real_fusedABC | native_randomA_fusedABC | +0.086356 | 1/0/127 |
| quad | waymo | native_real_fusedABC | native_wrong_fusedABC | +0.084861 | 2/0/126 |
| quad | waymo | native_real_fusedABC | real_frozen_fusedABC | -1.977104 | 21/25/82 |
| quad | waymo | native_real_fusedABC | real_frozen_ensemble | -1.977104 | 21/25/82 |
| quad | waymo | native_real_fusedABC | real_TTA_fusedABC | -2.006522 | 21/25/82 |
| quad | waymo | native_real_fusedABC | direct_three_geometric | -2.067205 | 17/20/91 |
| quad | waymo | preparedA_native_head_base | preparedA_prepared_head_base | +0.000000 | 0/0/128 |
| quad | waymo | preparedF_native_head_base | preparedF_prepared_head_base | +0.000000 | 0/0/128 |
| quad | waymo | sourceA_prepared_head | original_source_matchedcoverage | +0.000000 | 0/0/128 |
| quad | drone | native_real_fusedABC | original_source | +0.512058 | 13/3/112 |
| quad | drone | native_real_fusedABC | original_source_matchedcoverage | +0.489829 | 8/1/119 |
| quad | drone | native_real_fusedABC | prepared_frozen | +0.442856 | 21/26/81 |
| quad | drone | native_real_fusedABC | native_real_ensemble | +0.686777 | 6/3/119 |
| quad | drone | native_real_fusedABC | native_randomA_fusedABC | -0.297172 | 7/7/114 |
| quad | drone | native_real_fusedABC | native_wrong_fusedABC | +0.794146 | 8/4/116 |
| quad | drone | native_real_fusedABC | real_frozen_fusedABC | +0.930185 | 21/23/84 |
| quad | drone | native_real_fusedABC | real_frozen_ensemble | +0.851859 | 21/24/83 |
| quad | drone | native_real_fusedABC | real_TTA_fusedABC | +0.909278 | 21/23/84 |
| quad | drone | native_real_fusedABC | direct_three_geometric | -0.311632 | 18/25/85 |
| quad | drone | preparedA_native_head_base | preparedA_prepared_head_base | +0.000000 | 0/0/128 |
| quad | drone | preparedF_native_head_base | preparedF_prepared_head_base | +0.000000 | 0/0/128 |
| quad | drone | sourceA_prepared_head | original_source_matchedcoverage | +0.000000 | 0/0/128 |

## 裁决

```json
{
  "primary": "native_real_fusedABC",
  "mIoU_percent": 16.974317511719693,
  "gains_pp": {
    "original_source": 0.16293451300244044,
    "original_source_matchedcoverage": 0.19916076902110902,
    "prepared_frozen": 0.7456653525310131,
    "native_real_ensemble": 0.2403018170569169,
    "native_randomA_fusedABC": 0.19326832918524062,
    "native_randomA_ensemble": 0.1826211528078865,
    "native_wrong_fusedABC": 2.961228111472538,
    "real_frozen_fusedABC": 0.2710872187608473,
    "real_frozen_ensemble": 0.47263832739906775,
    "real_TTA_fusedABC": 0.3993593451770927,
    "direct_three_geometric": -0.016988752774015126
  },
  "passes_frozen_signal_gate": false,
  "TTA_run": false,
  "robust_TTA_adoption": false,
  "decision": "stop_original_native_fusion_recipe_no_TTA_from_this_signal",
  "n": 768,
  "primary_not_switched": true,
  "FINAL_access": false
}
```

此处冻结融合增量不是参数适应贡献。DEV历史暴露、同物理输入两incoming source相关、Drone/Quad场景很少，逐场景完整保留，不作独立输入置信区间。actual C与A回退单列，不能把所有C都宣称为新观测泛化。负例、错配、未知、失败尝试与44别名均不筛掉。未过门槛则停止这个冻结证据配方，不继续扫强度或训练新接口；通过也只能支持后续固定扩展验证，不直接支持新的TTA损失。

## 投影副本的实际参数核对（预测封存后）

```json
[
  {
    "source": "waymo",
    "all_native_and_prepared_projection_values_bitexact": true,
    "parameter_L2": 0.0,
    "max_abs_change": 0.0,
    "changed_values": 0
  },
  {
    "source": "drone",
    "all_native_and_prepared_projection_values_bitexact": true,
    "parameter_L2": 0.0,
    "max_abs_change": 0.0,
    "changed_values": 0
  },
  {
    "source": "quad",
    "all_native_and_prepared_projection_values_bitexact": true,
    "parameter_L2": 0.0,
    "max_abs_change": 0.0,
    "changed_values": 0
  }
]
```

若两个投影权重本来位等，互换分数一致是预期结果，不是判别性语义修复。该核对只排除投影副本漂移，不能据此把全部差异归给某一条RGB或点云模块。原native与prepared的差异发生在送入投影的表示/文字条件，以及额外late与竞争读出；后两者在当前基础项之上有正也有负作用。

## 实际图页核查与回放数量

```json
{
  "generated_pages": 12,
  "actually_viewed_pages": 4,
  "not_identity_gold_annotations": true,
  "unviewed_UNKNOWN": true,
  "observations": [
    "background-tree to visible truck rescue beyond originalsource",
    "neighboring/offset vehicle hypotheses with unresolved identity",
    "real and random fusion both restore originalperson choice",
    "overlapping rightedge person boxes and geometry change"
  ],
  "no_case_selection_gate": true,
  "attempt001_completed_inputs": 72,
  "attempt001_nonalias_hidden_states_reused": 62,
  "attempt001_alias_rows": 10
}
```

前72条完成输入中含10条保护别名，实际复用62条非别名隐藏表示/前缀，全部保留。图中既有从背景回到车辆、超出原始选择的纠错，也有只恢复原始人物选择；两类都有不同区域切换和重叠框变化，不能统一解释成实体身份学习。未看的8页保持UNKNOWN，不以图页筛掉评价样本。
