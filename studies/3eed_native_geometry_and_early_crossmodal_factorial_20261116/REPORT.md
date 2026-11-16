# R112 — 点云Adapter与早期跨模态交换的受控撤回

R111已确认原始与prepared匹配投影在三个source逐值位等，且native同证据路径比prepared好。因此本轮只改变新增特征路径，不再改匹配投影、BN或拟合新伪标签。全部768历史暴露开发输入，89读出：75旧读出/评分复用，14新增。44保护别名保留冻结回退。没有TTA、优化、新loss、source准备或专家，原A256/六层位置固定，FINAL未动。

## 实际干预与保留部分

2×2固定干预：point_adapter.up.weight+bias归零使残差成为恒等，gate_P归零阻止早期RGB→LiDAR；二者单独和组合对照。RGB编码器、image Adapter、反向point_to_image、lateRGB和竞争接口保留。正确与一次固定visible-only错配绑定均执行，A及同一实测点云fusedABC都比较。route成员和实体假设来自原fullR91当前A并在所有干预前固定。

12实际工程输入验证完整R91重放、实参改写与功能式旁路概率/隐藏表示/late/r位等、独立CPU读出、源状态恢复。二者都撤回时，匹配投影之前的h和u与R111native A/fused在全部非别名输入上位等。这只证明干预位置准确，不证明最终grounding正确。最终保留late/r，故不要求完整最终分数等于裸source。

## 因子作用（任务百分点）

```json
[
  {
    "binding": "correct",
    "view": "A",
    "full": 16.22865215918868,
    "noPA": 16.914970796537045,
    "noEarlyRGB": 16.26626124312036,
    "neither": 16.87466087565168,
    "remove_PA_with_RGB_active_pp": 0.6863186373483643,
    "remove_PA_with_RGB_off_pp": 0.6083996325313201,
    "remove_earlyRGB_with_PA_active_pp": 0.03760908393168094,
    "remove_earlyRGB_with_PA_identity_pp": -0.040309920885363226,
    "interaction_pp": -0.07791900481704417,
    "full_system_readout_late_and_competition_retained": true
  },
  {
    "binding": "correct",
    "view": "fusedABC",
    "full": 16.703230292958846,
    "noPA": 17.005608898471205,
    "noEarlyRGB": 16.352394301576417,
    "neither": 17.265658348293275,
    "remove_PA_with_RGB_active_pp": 0.3023786055123594,
    "remove_PA_with_RGB_off_pp": 0.9132640467168578,
    "remove_earlyRGB_with_PA_active_pp": -0.3508359913824286,
    "remove_earlyRGB_with_PA_identity_pp": 0.2600494498220698,
    "interaction_pp": 0.6108854412044984,
    "full_system_readout_late_and_competition_retained": true
  },
  {
    "binding": "wrong",
    "view": "A",
    "full": 16.25347031336018,
    "noPA": 16.97803535482183,
    "noEarlyRGB": 16.321487299316182,
    "neither": 16.544516586509378,
    "remove_PA_with_RGB_active_pp": 0.724565041461652,
    "remove_PA_with_RGB_off_pp": 0.2230292871931958,
    "remove_earlyRGB_with_PA_active_pp": 0.06801698595600314,
    "remove_earlyRGB_with_PA_identity_pp": -0.43351876831245306,
    "interaction_pp": -0.5015357542684562,
    "full_system_readout_late_and_competition_retained": true
  },
  {
    "binding": "wrong",
    "view": "fusedABC",
    "full": 16.63568420434579,
    "noPA": 16.88825920212319,
    "noEarlyRGB": 16.415155510770962,
    "neither": 17.116358156727113,
    "remove_PA_with_RGB_active_pp": 0.2525749977774012,
    "remove_PA_with_RGB_off_pp": 0.7012026459561511,
    "remove_earlyRGB_with_PA_active_pp": -0.22052869357482763,
    "remove_earlyRGB_with_PA_identity_pp": 0.22809895460392227,
    "interaction_pp": 0.4486276481787499,
    "full_system_readout_late_and_competition_retained": true
  }
]
```

该表是全系统固定读出的因果干预效应；其中包含后续文字交互、RGB反向路径和late/r的传播，不是某个通道的独立语义贡献。交互项非零时不可把单模块效果简单相加。

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
| all_six_equal | full_wrong_A | 16.253470 | 30.078125 | 12.890625 |
| all_six_equal | full_wrong_fusedABC | 16.635684 | 31.250000 | 13.020833 |
| all_six_equal | noPA_correct_A | 16.914971 | 31.510417 | 14.192708 |
| all_six_equal | noPA_correct_fusedABC | 17.005609 | 31.901042 | 14.062500 |
| all_six_equal | noPA_wrong_A | 16.978035 | 31.250000 | 14.322917 |
| all_six_equal | noPA_wrong_fusedABC | 16.888259 | 31.250000 | 14.322917 |
| all_six_equal | noEarlyRGB_correct_A | 16.266261 | 29.687500 | 12.890625 |
| all_six_equal | noEarlyRGB_correct_fusedABC | 16.352394 | 30.208333 | 13.020833 |
| all_six_equal | noEarlyRGB_wrong_A | 16.321487 | 30.338542 | 13.151042 |
| all_six_equal | noEarlyRGB_wrong_fusedABC | 16.415156 | 30.468750 | 13.411458 |
| all_six_equal | neither_correct_A | 16.874661 | 31.380208 | 14.192708 |
| all_six_equal | neither_correct_fusedABC | 17.265658 | 31.901042 | 14.322917 |
| all_six_equal | neither_wrong_A | 16.544517 | 30.468750 | 13.932292 |
| all_six_equal | neither_wrong_fusedABC | 17.116358 | 31.770833 | 14.322917 |
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
| waymo | full_wrong_A | 12.149245 | 25.000000 | 5.078125 |
| waymo | full_wrong_fusedABC | 12.553633 | 26.171875 | 5.468750 |
| waymo | noPA_correct_A | 12.010700 | 25.390625 | 7.421875 |
| waymo | noPA_correct_fusedABC | 12.381182 | 26.171875 | 7.421875 |
| waymo | noPA_wrong_A | 12.158049 | 24.218750 | 7.421875 |
| waymo | noPA_wrong_fusedABC | 12.270761 | 25.000000 | 7.421875 |
| waymo | noEarlyRGB_correct_A | 12.679673 | 26.562500 | 5.859375 |
| waymo | noEarlyRGB_correct_fusedABC | 12.410531 | 25.781250 | 5.859375 |
| waymo | noEarlyRGB_wrong_A | 12.739952 | 26.171875 | 6.250000 |
| waymo | noEarlyRGB_wrong_fusedABC | 12.593076 | 25.781250 | 6.250000 |
| waymo | neither_correct_A | 12.021834 | 25.000000 | 7.812500 |
| waymo | neither_correct_fusedABC | 12.304763 | 25.390625 | 7.812500 |
| waymo | neither_wrong_A | 11.830360 | 24.218750 | 7.421875 |
| waymo | neither_wrong_fusedABC | 11.938272 | 24.609375 | 7.421875 |
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
| drone | full_wrong_A | 13.991735 | 25.390625 | 11.718750 |
| drone | full_wrong_fusedABC | 14.097720 | 25.781250 | 11.718750 |
| drone | noPA_correct_A | 14.419496 | 26.171875 | 11.328125 |
| drone | noPA_correct_fusedABC | 14.397922 | 26.171875 | 12.109375 |
| drone | noPA_wrong_A | 14.519956 | 26.953125 | 11.328125 |
| drone | noPA_wrong_fusedABC | 14.032111 | 25.390625 | 11.718750 |
| drone | noEarlyRGB_correct_A | 13.660470 | 23.828125 | 11.718750 |
| drone | noEarlyRGB_correct_fusedABC | 13.896284 | 25.000000 | 11.718750 |
| drone | noEarlyRGB_wrong_A | 13.595839 | 25.390625 | 11.718750 |
| drone | noEarlyRGB_wrong_fusedABC | 13.600482 | 25.390625 | 11.718750 |
| drone | neither_correct_A | 14.350360 | 25.781250 | 11.328125 |
| drone | neither_correct_fusedABC | 14.503438 | 26.171875 | 11.718750 |
| drone | neither_wrong_A | 14.311268 | 25.390625 | 11.718750 |
| drone | neither_wrong_fusedABC | 14.630361 | 26.953125 | 11.718750 |
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
| quad | full_wrong_A | 22.619431 | 39.843750 | 21.875000 |
| quad | full_wrong_fusedABC | 23.255700 | 41.796875 | 21.875000 |
| quad | noPA_correct_A | 24.314716 | 42.968750 | 23.828125 |
| quad | noPA_correct_fusedABC | 24.237723 | 43.359375 | 22.656250 |
| quad | noPA_wrong_A | 24.256101 | 42.578125 | 24.218750 |
| quad | noPA_wrong_fusedABC | 24.361906 | 43.359375 | 23.828125 |
| quad | noEarlyRGB_correct_A | 22.458641 | 38.671875 | 21.093750 |
| quad | noEarlyRGB_correct_fusedABC | 22.750368 | 39.843750 | 21.484375 |
| quad | noEarlyRGB_wrong_A | 22.628671 | 39.453125 | 21.484375 |
| quad | noEarlyRGB_wrong_fusedABC | 23.051908 | 40.234375 | 22.265625 |
| quad | neither_correct_A | 24.251789 | 43.359375 | 23.437500 |
| quad | neither_correct_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | neither_wrong_A | 23.491922 | 41.796875 | 22.656250 |
| quad | neither_wrong_fusedABC | 24.780441 | 43.750000 | 23.828125 |

## 全部配对结果

| Source | Target | 方法 | 对照 | Δ pp | 改善/恶化/不变 |
|---|---|---|---|---:|---:|
| waymo | drone | noPA_correct_fusedABC | original_source | +0.938709 | 12/9/107 |
| waymo | drone | noPA_correct_fusedABC | original_source_matchedcoverage | +0.879431 | 11/9/108 |
| waymo | drone | noPA_correct_fusedABC | prepared_frozen | +0.599941 | 28/11/89 |
| waymo | drone | noPA_correct_fusedABC | real_frozen_fusedABC | +0.196552 | 25/13/90 |
| waymo | drone | noPA_correct_fusedABC | native_real_fusedABC | -0.235112 | 7/8/113 |
| waymo | drone | noPA_correct_fusedABC | direct_three_geometric | +1.419623 | 18/10/100 |
| waymo | drone | noPA_correct_fusedABC | noPA_wrong_fusedABC | +0.788991 | 11/6/111 |
| waymo | drone | noPA_correct_fusedABC | noEarlyRGB_correct_fusedABC | +0.250845 | 20/16/92 |
| waymo | drone | noPA_correct_fusedABC | neither_correct_fusedABC | -0.143055 | 7/7/114 |
| waymo | drone | noPA_correct_A | prepared_frozen | +0.198252 | 27/11/90 |
| waymo | drone | noEarlyRGB_correct_A | prepared_frozen | -0.284873 | 9/6/113 |
| waymo | drone | neither_correct_A | prepared_frozen | +0.667831 | 27/11/90 |
| waymo | quad | noPA_correct_fusedABC | original_source | -1.273849 | 9/8/111 |
| waymo | quad | noPA_correct_fusedABC | original_source_matchedcoverage | -0.776921 | 6/6/116 |
| waymo | quad | noPA_correct_fusedABC | prepared_frozen | +2.458529 | 15/12/101 |
| waymo | quad | noPA_correct_fusedABC | real_frozen_fusedABC | +0.029819 | 14/13/101 |
| waymo | quad | noPA_correct_fusedABC | native_real_fusedABC | -0.017746 | 8/5/115 |
| waymo | quad | noPA_correct_fusedABC | direct_three_geometric | -0.600204 | 9/10/109 |
| waymo | quad | noPA_correct_fusedABC | noPA_wrong_fusedABC | +0.181438 | 3/0/125 |
| waymo | quad | noPA_correct_fusedABC | noEarlyRGB_correct_fusedABC | +1.308891 | 15/10/103 |
| waymo | quad | noPA_correct_fusedABC | neither_correct_fusedABC | -0.310168 | 5/5/118 |
| waymo | quad | noPA_correct_A | prepared_frozen | +2.578160 | 14/12/102 |
| waymo | quad | noEarlyRGB_correct_A | prepared_frozen | +1.153218 | 9/6/113 |
| waymo | quad | neither_correct_A | prepared_frozen | +2.681652 | 17/11/100 |
| drone | waymo | noPA_correct_fusedABC | original_source | -0.027204 | 8/7/113 |
| drone | waymo | noPA_correct_fusedABC | original_source_matchedcoverage | -0.027204 | 8/7/113 |
| drone | waymo | noPA_correct_fusedABC | prepared_frozen | -0.716884 | 18/20/90 |
| drone | waymo | noPA_correct_fusedABC | real_frozen_fusedABC | -0.414897 | 17/19/92 |
| drone | waymo | noPA_correct_fusedABC | native_real_fusedABC | -0.325154 | 5/7/116 |
| drone | waymo | noPA_correct_fusedABC | direct_three_geometric | -1.168877 | 23/21/84 |
| drone | waymo | noPA_correct_fusedABC | noPA_wrong_fusedABC | +0.122529 | 7/3/118 |
| drone | waymo | noPA_correct_fusedABC | noEarlyRGB_correct_fusedABC | +0.125681 | 20/18/90 |
| drone | waymo | noPA_correct_fusedABC | neither_correct_fusedABC | -0.280145 | 4/4/120 |
| drone | waymo | noPA_correct_A | prepared_frozen | -1.004042 | 17/20/91 |
| drone | waymo | noEarlyRGB_correct_A | prepared_frozen | -0.249861 | 5/7/116 |
| drone | waymo | neither_correct_A | prepared_frozen | -0.729924 | 19/22/87 |
| drone | quad | noPA_correct_fusedABC | original_source | -0.436355 | 13/11/104 |
| drone | quad | noPA_correct_fusedABC | original_source_matchedcoverage | -0.634418 | 11/11/106 |
| drone | quad | noPA_correct_fusedABC | prepared_frozen | +2.019392 | 26/15/87 |
| drone | quad | noPA_correct_fusedABC | real_frozen_fusedABC | +1.423674 | 17/10/101 |
| drone | quad | noPA_correct_fusedABC | native_real_fusedABC | -0.860282 | 6/7/115 |
| drone | quad | noPA_correct_fusedABC | direct_three_geometric | +1.188069 | 19/9/100 |
| drone | quad | noPA_correct_fusedABC | noPA_wrong_fusedABC | -0.429803 | 7/6/115 |
| drone | quad | noPA_correct_fusedABC | noEarlyRGB_correct_fusedABC | +1.665818 | 21/10/97 |
| drone | quad | noPA_correct_fusedABC | neither_correct_fusedABC | -1.191934 | 8/11/109 |
| drone | quad | noPA_correct_A | prepared_frozen | +2.053748 | 25/11/92 |
| drone | quad | noEarlyRGB_correct_A | prepared_frozen | -0.233461 | 4/7/117 |
| drone | quad | neither_correct_A | prepared_frozen | +1.824401 | 25/15/88 |
| quad | waymo | noPA_correct_fusedABC | original_source | +1.842305 | 19/12/97 |
| quad | waymo | noPA_correct_fusedABC | original_source_matchedcoverage | +1.842305 | 19/12/97 |
| quad | waymo | noPA_correct_fusedABC | prepared_frozen | +0.248215 | 17/12/99 |
| quad | waymo | noPA_correct_fusedABC | real_frozen_fusedABC | +0.039247 | 17/13/98 |
| quad | waymo | noPA_correct_fusedABC | native_real_fusedABC | +2.016351 | 19/13/96 |
| quad | waymo | noPA_correct_fusedABC | direct_three_geometric | -0.050854 | 14/9/105 |
| quad | waymo | noPA_correct_fusedABC | noPA_wrong_fusedABC | +0.098312 | 3/2/123 |
| quad | waymo | noPA_correct_fusedABC | noEarlyRGB_correct_fusedABC | -0.184380 | 15/16/97 |
| quad | waymo | noPA_correct_fusedABC | neither_correct_fusedABC | +0.432983 | 14/13/101 |
| quad | waymo | noPA_correct_A | prepared_frozen | -0.205590 | 15/13/100 |
| quad | waymo | noEarlyRGB_correct_A | prepared_frozen | +0.378175 | 14/8/106 |
| quad | waymo | neither_correct_A | prepared_frozen | -0.457440 | 19/17/92 |
| quad | drone | noPA_correct_fusedABC | original_source | +0.121749 | 25/17/86 |
| quad | drone | noPA_correct_fusedABC | original_source_matchedcoverage | +0.099520 | 20/15/93 |
| quad | drone | noPA_correct_fusedABC | prepared_frozen | +0.052547 | 11/15/102 |
| quad | drone | noPA_correct_fusedABC | real_frozen_fusedABC | +0.539876 | 13/12/103 |
| quad | drone | noPA_correct_fusedABC | native_real_fusedABC | -0.390309 | 20/18/90 |
| quad | drone | noPA_correct_fusedABC | direct_three_geometric | -0.701941 | 8/15/105 |
| quad | drone | noPA_correct_fusedABC | noPA_wrong_fusedABC | -0.057368 | 3/3/122 |
| quad | drone | noPA_correct_fusedABC | noEarlyRGB_correct_fusedABC | +0.752432 | 16/18/94 |
| quad | drone | noPA_correct_fusedABC | neither_correct_fusedABC | -0.067977 | 13/15/100 |
| quad | drone | noPA_correct_A | prepared_frozen | +0.497384 | 11/11/106 |
| quad | drone | noEarlyRGB_correct_A | prepared_frozen | -0.537543 | 12/14/102 |
| quad | drone | neither_correct_A | prepared_frozen | -0.110468 | 16/21/91 |

## 固定主项裁决

```json
{
  "primary": "noPA_correct_fusedABC",
  "mIoU_percent": 17.005608898471205,
  "gains_pp": {
    "original_source": 0.19422589975395255,
    "original_source_matchedcoverage": 0.23045215577262113,
    "prepared_frozen": 0.7769567392825252,
    "real_frozen_fusedABC": 0.3023786055123594,
    "native_real_fusedABC": 0.03129138675151211,
    "direct_three_geometric": 0.014302633977496981,
    "noPA_wrong_fusedABC": 0.11734969634801473,
    "noEarlyRGB_correct_fusedABC": 0.653214596894788,
    "neither_correct_fusedABC": -0.2600494498220698
  },
  "passes_static_repair_gate": false,
  "TTA_run": false,
  "robust_TTA_adoption": false,
  "decision": "do_not_adopt_fixed_noPA_primary_as_method",
  "n": 768,
  "primary_not_switched": true,
  "FINAL_access": false
}
```

保持总体分母、两域/三域结果、所有场景聚类与正负尾部。固定撤回不是测试时学习；即便分数上涨，也不能包装为参数TTA贡献。错配有时不改变预测是合法结果，不要求每条反转。候选框覆盖/实体身份仍需独立分清；隐藏偏移变小不是有效性的替代。未过门槛不通过扩大倍率或额外门槛挽救，也不把事后最优对照改为本轮主项。

## 实际图页与剩余RGB路径

```json
{
  "generated_pages": 12,
  "actually_viewed_pages": 4,
  "unviewed_remain_UNKNOWN": true,
  "not_independent_instance_identity_gold": true,
  "observed_types": [
    "background to visible truck rescue",
    "neighboring and overlapping vehicle hypotheses unresolved",
    "same person-region with box extent changes",
    "noPA alone causes foregroundperson to distantregion switch; combined preserves it"
  ],
  "no_case_filter_or_sample_specific_patch": true
}
```

4/12页已实际查看，其余UNKNOWN。正例包含树/背景到车辆，负例既有相邻车/框范围变化，也有noPA单独引发原正确行人到远侧区域的真实切换；组合撤回保住该例。这说明交互必须测量，不能用单个模块名称概括因果。

```json
{
  "state": "PASS_BOTH_BYPASSED_CROSS_BINDING_H_U_RESIDUAL_BITEXACT_ALL_NONALIAS_A_F",
  "endpoints": 1448,
  "max_h_difference": 0.0,
  "max_text_difference": 0.0,
  "max_competition_residual_difference": 0.0,
  "mean_lateRGB_difference_L2": 4.30366773128145,
  "Top1_differences_from_binding_only_in_late_path": 178,
  "GT_used": false,
  "not_a_claim_late_RGB_beats_no_RGB_readout": true
}
```

同时撤回时，正确/错配绑定的h、文字和竞争残差在1448个端点逐值一致，绑定影响只经lateRGB回到候选评分。这提供后续“固定几何—语言基础，只适应剩余外观路径”的可核查计算边界；没有证明lateRGB本身优于完全去掉它，因此后续必须保留无late/无r读出和同信息直接均值，不把静态撤回当TTA。
