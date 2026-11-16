# R108 — 已训练soft-token头的冻结证据前提检查

这不是新增模型或TTA。三个epoch100原模型都已有独立参数的soft-token CE头；其训练目标与contrastive BBF不同，但共享同一原生decoder表示，不能称独立视觉专家。R4在早期权重/队列中测过BBS且较差，本轮仅检验当前三source/R91/真实注册点观测条件是否存在互补。

## 固定方案

直接用原prediction_heads[5].sem_cls_scores_head处理已封存R106的A/B/C/fusedABC query_hidden；真实全256token softmax分母、BN eval运行统计、dropout关闭。正跨度仅来自与当前BBF相同的纯文本mask，未用官方标注positive_map、类别或GT跨度；最后no-object255从不作为正token。区域与原A256完全不变。

CE的主体token概率质量在候选维归一化，然后分别固定等权几何池、等权算术池；打乱CE候选对应保留分数集合。锁定主方法为几何池fusedABC，不依据评分改成算术池。44条受保护媒体别名保留prepared冻结输出，不重解码或另跑头。

12条全原生工程测试验证原始sourceCE模块提取、真实R91decoder hook、缓存h与独立CPUConv/BN/softmax的一致性。整批预测封存后，独立CPU审计全部端点、source checkpoint权重与原框，再评分。

## 全部结果

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

## 六方向与全部负例

| Source | Target | 方法 | 对照 | Δ pp | 改善/恶化/不变 |
|---|---|---|---|---:|---:|
| waymo | drone | softtoken_geometric_fusedABC | original_source | +0.176185 | 15/21/92 |
| waymo | drone | softtoken_geometric_fusedABC | prepared_frozen | -0.162584 | 13/10/105 |
| waymo | drone | softtoken_geometric_fusedABC | real_frozen_fusedABC | -0.565973 | 10/11/107 |
| waymo | drone | softtoken_geometric_fusedABC | softtoken_fusedABC | +1.055938 | 12/9/107 |
| waymo | drone | softtoken_geometric_fusedABC | softtoken_arithmetic_fusedABC | -0.326335 | 2/3/123 |
| waymo | drone | softtoken_geometric_fusedABC | softtoken_shuffled_geometric_fusedABC | +8.230889 | 45/19/64 |
| waymo | drone | softtoken_geometric_fusedABC | real_TTA_fusedABC | -0.646278 | 10/11/107 |
| waymo | drone | softtoken_A | prepared_frozen | -1.082362 | 19/19/90 |
| waymo | drone | softtoken_geometric_A | prepared_frozen | -0.634675 | 10/10/108 |
| waymo | quad | softtoken_geometric_fusedABC | original_source | -1.405022 | 16/16/96 |
| waymo | quad | softtoken_geometric_fusedABC | prepared_frozen | +2.327356 | 12/7/109 |
| waymo | quad | softtoken_geometric_fusedABC | real_frozen_fusedABC | -0.101354 | 7/7/114 |
| waymo | quad | softtoken_geometric_fusedABC | softtoken_fusedABC | +1.524601 | 10/2/116 |
| waymo | quad | softtoken_geometric_fusedABC | softtoken_arithmetic_fusedABC | +0.294502 | 3/1/124 |
| waymo | quad | softtoken_geometric_fusedABC | softtoken_shuffled_geometric_fusedABC | +24.839628 | 66/7/55 |
| waymo | quad | softtoken_geometric_fusedABC | real_TTA_fusedABC | +1.081819 | 11/6/111 |
| waymo | quad | softtoken_A | prepared_frozen | +0.393881 | 12/6/110 |
| waymo | quad | softtoken_geometric_A | prepared_frozen | +0.464443 | 8/4/116 |
| drone | waymo | softtoken_geometric_fusedABC | original_source | +1.787190 | 29/25/74 |
| drone | waymo | softtoken_geometric_fusedABC | prepared_frozen | +1.097510 | 16/16/96 |
| drone | waymo | softtoken_geometric_fusedABC | real_frozen_fusedABC | +1.399496 | 17/15/96 |
| drone | waymo | softtoken_geometric_fusedABC | softtoken_fusedABC | +1.616655 | 19/9/100 |
| drone | waymo | softtoken_geometric_fusedABC | softtoken_arithmetic_fusedABC | +0.139384 | 5/8/115 |
| drone | waymo | softtoken_geometric_fusedABC | softtoken_shuffled_geometric_fusedABC | +5.691875 | 49/17/62 |
| drone | waymo | softtoken_geometric_fusedABC | real_TTA_fusedABC | +1.239167 | 17/16/95 |
| drone | waymo | softtoken_A | prepared_frozen | -0.893313 | 16/23/89 |
| drone | waymo | softtoken_geometric_A | prepared_frozen | +0.787026 | 15/15/98 |
| drone | quad | softtoken_geometric_fusedABC | original_source | -2.219922 | 14/24/90 |
| drone | quad | softtoken_geometric_fusedABC | prepared_frozen | +0.235825 | 15/12/101 |
| drone | quad | softtoken_geometric_fusedABC | real_frozen_fusedABC | -0.359892 | 6/8/114 |
| drone | quad | softtoken_geometric_fusedABC | softtoken_fusedABC | -0.082177 | 5/8/115 |
| drone | quad | softtoken_geometric_fusedABC | softtoken_arithmetic_fusedABC | -0.168110 | 3/3/122 |
| drone | quad | softtoken_geometric_fusedABC | softtoken_shuffled_geometric_fusedABC | +8.698363 | 50/15/63 |
| drone | quad | softtoken_geometric_fusedABC | real_TTA_fusedABC | -0.482473 | 8/12/108 |
| drone | quad | softtoken_A | prepared_frozen | -0.493529 | 11/11/106 |
| drone | quad | softtoken_geometric_A | prepared_frozen | -0.580359 | 4/8/116 |
| quad | waymo | softtoken_geometric_fusedABC | original_source | +2.096516 | 26/20/82 |
| quad | waymo | softtoken_geometric_fusedABC | prepared_frozen | +0.502426 | 4/2/122 |
| quad | waymo | softtoken_geometric_fusedABC | real_frozen_fusedABC | +0.293459 | 3/2/123 |
| quad | waymo | softtoken_geometric_fusedABC | softtoken_fusedABC | -0.074007 | 3/3/122 |
| quad | waymo | softtoken_geometric_fusedABC | softtoken_arithmetic_fusedABC | +0.076131 | 1/0/127 |
| quad | waymo | softtoken_geometric_fusedABC | softtoken_shuffled_geometric_fusedABC | +5.210039 | 44/25/59 |
| quad | waymo | softtoken_geometric_fusedABC | real_TTA_fusedABC | +0.264041 | 3/3/122 |
| quad | waymo | softtoken_A | prepared_frozen | +0.062846 | 5/5/118 |
| quad | waymo | softtoken_geometric_A | prepared_frozen | +0.293459 | 3/2/123 |
| quad | drone | softtoken_geometric_fusedABC | original_source | +0.002844 | 31/22/75 |
| quad | drone | softtoken_geometric_fusedABC | prepared_frozen | -0.066358 | 4/9/115 |
| quad | drone | softtoken_geometric_fusedABC | real_frozen_fusedABC | +0.420972 | 3/3/122 |
| quad | drone | softtoken_geometric_fusedABC | softtoken_fusedABC | +0.309131 | 4/3/121 |
| quad | drone | softtoken_geometric_fusedABC | softtoken_arithmetic_fusedABC | -0.152651 | 3/3/122 |
| quad | drone | softtoken_geometric_fusedABC | softtoken_shuffled_geometric_fusedABC | +4.091242 | 30/19/79 |
| quad | drone | softtoken_geometric_fusedABC | real_TTA_fusedABC | +0.400065 | 3/4/121 |
| quad | drone | softtoken_A | prepared_frozen | -0.577062 | 5/9/114 |
| quad | drone | softtoken_geometric_A | prepared_frozen | +0.063844 | 2/4/122 |

## 配对与集中度

[
  {
    "control": "original_source",
    "n": 768,
    "gain_pp": 0.07296522967622891,
    "better": 131,
    "worse": 128,
    "same": 509,
    "largest_positive_single_input_pp": 0.08108251948588525,
    "largest_negative_single_input_pp": -0.11153539380202347
  },
  {
    "control": "prepared_frozen",
    "n": 768,
    "gain_pp": 0.6556960692048003,
    "better": 64,
    "worse": 56,
    "same": 648,
    "largest_positive_single_input_pp": 0.09879236859757051,
    "largest_negative_single_input_pp": -0.0833832184462392
  },
  {
    "control": "real_frozen_fusedABC",
    "n": 768,
    "gain_pp": 0.18111793543463373,
    "better": 46,
    "worse": 46,
    "same": 676,
    "largest_positive_single_input_pp": 0.08186548601604034,
    "largest_negative_single_input_pp": -0.0762938956135653
  },
  {
    "control": "softtoken_fusedABC",
    "n": 768,
    "gain_pp": 0.7250234806469475,
    "better": 53,
    "worse": 34,
    "same": 681,
    "largest_positive_single_input_pp": 0.0903288215055313,
    "largest_negative_single_input_pp": -0.076377467268118
  },
  {
    "control": "softtoken_arithmetic_fusedABC",
    "n": 768,
    "gain_pp": -0.022846539633066106,
    "better": 17,
    "worse": 18,
    "same": 733,
    "largest_positive_single_input_pp": 0.05642673548899752,
    "largest_negative_single_input_pp": -0.044357020996713105
  },
  {
    "control": "softtoken_shuffled_geometric_fusedABC",
    "n": 768,
    "gain_pp": 9.460339310683441,
    "better": 284,
    "worse": 102,
    "same": 382,
    "largest_positive_single_input_pp": 0.11620391437363997,
    "largest_negative_single_input_pp": -0.07153117098545475
  },
  {
    "control": "real_TTA_fusedABC",
    "n": 768,
    "gain_pp": 0.3093900618508807,
    "better": 52,
    "worse": 52,
    "same": 664,
    "largest_positive_single_input_pp": 0.08186548601604034,
    "largest_negative_single_input_pp": -0.053708010201132854
  }
]

## 互补诊断

[
  {
    "scope": "all",
    "n": 768,
    "CE_better_than_BBF": 65,
    "CE_worse_than_BBF": 79,
    "geometric_better_than_BBF": 46,
    "geometric_worse_than_BBF": 46,
    "arithmetic_better_than_BBF": 48,
    "arithmetic_worse_than_BBF": 44,
    "GT_oracle_of_two_heads_mIoU_diagnostic": 17.84278770562926,
    "not_a_deployable_head_selector": true
  },
  {
    "scope": "waymo_to_drone",
    "n": 128,
    "CE_better_than_BBF": 16,
    "CE_worse_than_BBF": 22,
    "geometric_better_than_BBF": 10,
    "geometric_worse_than_BBF": 11,
    "arithmetic_better_than_BBF": 10,
    "arithmetic_worse_than_BBF": 11,
    "GT_oracle_of_two_heads_mIoU_diagnostic": 18.645567030450202,
    "not_a_deployable_head_selector": true
  },
  {
    "scope": "waymo_to_quad",
    "n": 128,
    "CE_better_than_BBF": 6,
    "CE_worse_than_BBF": 12,
    "geometric_better_than_BBF": 7,
    "geometric_worse_than_BBF": 7,
    "arithmetic_better_than_BBF": 5,
    "arithmetic_worse_than_BBF": 6,
    "GT_oracle_of_two_heads_mIoU_diagnostic": 32.96024247938007,
    "not_a_deployable_head_selector": true
  },
  {
    "scope": "drone_to_waymo",
    "n": 128,
    "CE_better_than_BBF": 19,
    "CE_worse_than_BBF": 23,
    "geometric_better_than_BBF": 17,
    "geometric_worse_than_BBF": 15,
    "arithmetic_better_than_BBF": 16,
    "arithmetic_worse_than_BBF": 10,
    "GT_oracle_of_two_heads_mIoU_diagnostic": 16.977168663535203,
    "not_a_deployable_head_selector": true
  },
  {
    "scope": "drone_to_quad",
    "n": 128,
    "CE_better_than_BBF": 12,
    "CE_worse_than_BBF": 12,
    "geometric_better_than_BBF": 6,
    "geometric_worse_than_BBF": 8,
    "arithmetic_better_than_BBF": 8,
    "arithmetic_worse_than_BBF": 10,
    "GT_oracle_of_two_heads_mIoU_diagnostic": 16.043464624982583,
    "not_a_deployable_head_selector": true
  },
  {
    "scope": "quad_to_waymo",
    "n": 128,
    "CE_better_than_BBF": 6,
    "CE_worse_than_BBF": 4,
    "geometric_better_than_BBF": 3,
    "geometric_worse_than_BBF": 2,
    "arithmetic_better_than_BBF": 3,
    "arithmetic_worse_than_BBF": 2,
    "GT_oracle_of_two_heads_mIoU_diagnostic": 11.155820817576902,
    "not_a_deployable_head_selector": true
  },
  {
    "scope": "quad_to_drone",
    "n": 128,
    "CE_better_than_BBF": 6,
    "CE_worse_than_BBF": 6,
    "geometric_better_than_BBF": 3,
    "geometric_worse_than_BBF": 3,
    "arithmetic_better_than_BBF": 6,
    "arithmetic_worse_than_BBF": 5,
    "GT_oracle_of_two_heads_mIoU_diagnostic": 11.274462617850586,
    "not_a_deployable_head_selector": true
  }
]

## 裁决

{
  "primary": "softtoken_geometric_fusedABC",
  "mIoU_percent": 16.88434822839348,
  "gains_pp": {
    "original_source": 0.0729652296762282,
    "prepared_frozen": 0.6556960692048008,
    "real_frozen_fusedABC": 0.18111793543463506,
    "softtoken_fusedABC": 0.7250234806469464,
    "softtoken_arithmetic_fusedABC": -0.022846539633068375,
    "softtoken_shuffled_geometric_fusedABC": 9.460339310683441,
    "real_TTA_fusedABC": 0.30939006185088047
  },
  "passes_locked_gate": false,
  "robust_adoption": false,
  "TTA_run": false,
  "decision": "do_not_fit_this_geometric_pool_target_under_failed_locked_gate",
  "frozen_complementarity_positive": true,
  "frozen_candidate_requires_locked_expansion": true,
  "n": 768,
  "FINAL_access": false,
  "source_retraining": false,
  "external_experts": false
}

存在冻结读出的开发正信号：CE单独低于BBF，但两种固定池化均提高fusedABC全体成绩，打乱对应明显下降。尚不能把混合分数解释为可靠身份标签，也不能宣布参数适应有效。锁定主方法没有胜过同信息算术池，因此按预设门槛不拟合这份几何池目标。算术池不事后替换主方法；若继续研究，只能锁定两种固定池并扩大评价，不扫权重或重新包装成已确认结果。

所有768条与正负例保留、29×768=22272任务读出。历史开发暴露、重复场景和几何注册的静态场景假设不变，不能将开发微小净增量称稳定泛化。未使用FINAL、外部专家、source重新训练，也未把seed objectness误当最终框质量。
