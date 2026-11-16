# R103 — 同一共识绑定证据的参数适应检验

固定R102共识G，使用原R99三视图L2–LM更新末端18,496参数。没有新增损失、扩大更新层或调参。以同信息共识直接读出为强对照；同范数旧G和打乱共识都以相同预算拟合。沿用768条开发输入、原A256及原样本，C在全部拟合后读取。不是独立确认；没有FINAL。

| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |
|---|---|---:|---:|---:|
| all_six_equal | original_source | 16.811383 | 30.989583 | 14.973958 |
| all_six_equal | prepared_frozen | 16.228652 | 29.947917 | 12.630208 |
| all_six_equal | binding_direct | 16.272845 | 30.078125 | 12.760417 |
| all_six_equal | direct_C | 16.163092 | 29.687500 | 12.500000 |
| all_six_equal | direct_FC | 16.241990 | 29.817708 | 12.760417 |
| all_six_equal | direct_four | 16.278525 | 29.947917 | 12.890625 |
| all_six_equal | lm_FC | 16.311431 | 29.947917 | 12.760417 |
| all_six_equal | joint3_FC | 16.517235 | 30.338542 | 13.020833 |
| all_six_equal | kl3_FC | 16.256596 | 29.947917 | 12.760417 |
| all_six_equal | kl3_shuffle_FC | 16.338772 | 30.078125 | 12.760417 |
| all_six_equal | single0_F | 16.273501 | 30.208333 | 12.630208 |
| all_six_equal | single0_C | 16.325118 | 30.338542 | 12.630208 |
| all_six_equal | single0_FC | 16.399077 | 30.468750 | 12.890625 |
| all_six_equal | single1_F | 16.541246 | 30.338542 | 13.020833 |
| all_six_equal | single1_C | 16.165992 | 29.947917 | 12.760417 |
| all_six_equal | single1_FC | 16.384475 | 29.947917 | 13.020833 |
| all_six_equal | signcommon_F | 16.374441 | 30.078125 | 12.760417 |
| all_six_equal | signcommon_C | 16.310744 | 29.817708 | 12.890625 |
| all_six_equal | signcommon_FC | 16.409823 | 30.208333 | 12.890625 |
| all_six_equal | scaledmean_F | 16.235272 | 29.947917 | 12.760417 |
| all_six_equal | scaledmean_C | 16.287346 | 29.687500 | 12.760417 |
| all_six_equal | scaledmean_FC | 16.259707 | 29.817708 | 12.630208 |
| all_six_equal | shufflecommon_F | 16.240945 | 29.947917 | 12.630208 |
| all_six_equal | shufflecommon_C | 16.178548 | 29.687500 | 12.630208 |
| all_six_equal | shufflecommon_FC | 16.301493 | 30.078125 | 12.760417 |
| all_six_equal | common3_F | 16.288628 | 30.078125 | 12.760417 |
| all_six_equal | common3_C | 16.369186 | 30.078125 | 13.020833 |
| all_six_equal | common3_FC | 16.376751 | 30.078125 | 12.890625 |
| all_six_equal | scaledmean3_F | 16.269396 | 30.078125 | 12.630208 |
| all_six_equal | scaledmean3_C | 16.276928 | 29.817708 | 12.760417 |
| all_six_equal | scaledmean3_FC | 16.290616 | 30.078125 | 12.630208 |
| all_six_equal | shufflecommon3_F | 16.347520 | 30.208333 | 12.630208 |
| all_six_equal | shufflecommon3_C | 16.215703 | 29.687500 | 12.630208 |
| all_six_equal | shufflecommon3_FC | 16.402570 | 30.078125 | 12.890625 |
| waymo | original_source | 11.473631 | 23.437500 | 7.421875 |
| waymo | prepared_frozen | 12.615516 | 26.171875 | 5.078125 |
| waymo | binding_direct | 12.652470 | 25.781250 | 5.468750 |
| waymo | direct_C | 12.647289 | 25.781250 | 5.078125 |
| waymo | direct_FC | 12.755842 | 25.390625 | 5.468750 |
| waymo | direct_four | 12.679658 | 25.781250 | 5.468750 |
| waymo | lm_FC | 12.914574 | 26.562500 | 5.468750 |
| waymo | joint3_FC | 13.067711 | 26.953125 | 5.859375 |
| waymo | kl3_FC | 12.815418 | 25.781250 | 5.468750 |
| waymo | kl3_shuffle_FC | 12.229305 | 25.781250 | 4.687500 |
| waymo | single0_F | 12.775049 | 25.781250 | 5.468750 |
| waymo | single0_C | 13.266971 | 26.953125 | 5.468750 |
| waymo | single0_FC | 13.188368 | 26.953125 | 5.468750 |
| waymo | single1_F | 12.496208 | 25.000000 | 5.468750 |
| waymo | single1_C | 12.395213 | 26.171875 | 5.078125 |
| waymo | single1_FC | 12.546399 | 25.390625 | 5.468750 |
| waymo | signcommon_F | 12.844233 | 26.171875 | 5.468750 |
| waymo | signcommon_C | 12.808026 | 25.781250 | 5.468750 |
| waymo | signcommon_FC | 12.776152 | 26.562500 | 5.078125 |
| waymo | scaledmean_F | 12.626779 | 26.171875 | 5.468750 |
| waymo | scaledmean_C | 12.987999 | 26.171875 | 5.468750 |
| waymo | scaledmean_FC | 12.711903 | 26.171875 | 5.078125 |
| waymo | shufflecommon_F | 12.373936 | 25.781250 | 4.687500 |
| waymo | shufflecommon_C | 12.436163 | 26.171875 | 4.687500 |
| waymo | shufflecommon_FC | 12.248883 | 25.781250 | 4.687500 |
| waymo | common3_F | 12.666801 | 26.171875 | 5.468750 |
| waymo | common3_C | 13.061071 | 26.562500 | 5.859375 |
| waymo | common3_FC | 12.659261 | 26.171875 | 5.078125 |
| waymo | scaledmean3_F | 12.574344 | 26.171875 | 5.078125 |
| waymo | scaledmean3_C | 12.823820 | 26.171875 | 5.468750 |
| waymo | scaledmean3_FC | 12.760112 | 26.562500 | 5.078125 |
| waymo | shufflecommon3_F | 12.652535 | 26.562500 | 4.687500 |
| waymo | shufflecommon3_C | 12.474746 | 25.781250 | 4.687500 |
| waymo | shufflecommon3_FC | 12.266676 | 25.390625 | 4.687500 |
| drone | original_source | 13.867693 | 24.609375 | 12.500000 |
| drone | prepared_frozen | 14.071678 | 24.609375 | 11.718750 |
| drone | binding_direct | 14.253223 | 25.781250 | 12.109375 |
| drone | direct_C | 13.794244 | 24.609375 | 11.328125 |
| drone | direct_FC | 13.844147 | 25.000000 | 11.718750 |
| drone | direct_four | 13.894029 | 25.000000 | 11.718750 |
| drone | lm_FC | 13.904753 | 24.218750 | 11.718750 |
| drone | joint3_FC | 14.238577 | 24.609375 | 12.109375 |
| drone | kl3_FC | 13.830349 | 25.000000 | 11.718750 |
| drone | kl3_shuffle_FC | 14.305533 | 25.000000 | 11.718750 |
| drone | single0_F | 14.070680 | 25.781250 | 11.718750 |
| drone | single0_C | 13.673235 | 24.609375 | 11.328125 |
| drone | single0_FC | 14.119800 | 25.781250 | 12.109375 |
| drone | single1_F | 14.499504 | 26.171875 | 12.109375 |
| drone | single1_C | 13.628243 | 24.218750 | 11.328125 |
| drone | single1_FC | 14.057129 | 25.000000 | 11.718750 |
| drone | signcommon_F | 14.261741 | 25.000000 | 12.109375 |
| drone | signcommon_C | 13.839946 | 24.609375 | 11.718750 |
| drone | signcommon_FC | 14.169148 | 25.000000 | 12.109375 |
| drone | scaledmean_F | 14.269543 | 25.390625 | 12.109375 |
| drone | scaledmean_C | 13.884551 | 24.609375 | 11.718750 |
| drone | scaledmean_FC | 13.984006 | 24.609375 | 11.718750 |
| drone | shufflecommon_F | 14.239825 | 25.000000 | 11.718750 |
| drone | shufflecommon_C | 14.042689 | 24.609375 | 12.109375 |
| drone | shufflecommon_FC | 14.160430 | 24.609375 | 11.718750 |
| drone | common3_F | 14.274606 | 25.390625 | 12.109375 |
| drone | common3_C | 13.754395 | 24.609375 | 11.718750 |
| drone | common3_FC | 14.178995 | 25.000000 | 12.109375 |
| drone | scaledmean3_F | 14.271562 | 25.000000 | 12.109375 |
| drone | scaledmean3_C | 13.898690 | 24.609375 | 11.718750 |
| drone | scaledmean3_FC | 13.909738 | 24.609375 | 11.718750 |
| drone | shufflecommon3_F | 14.241162 | 25.000000 | 11.718750 |
| drone | shufflecommon3_C | 14.066829 | 24.609375 | 12.109375 |
| drone | shufflecommon3_FC | 14.410966 | 25.000000 | 12.109375 |
| quad | original_source | 25.092825 | 44.921875 | 25.000000 |
| quad | prepared_frozen | 21.998762 | 39.062500 | 21.093750 |
| quad | binding_direct | 21.912842 | 38.671875 | 20.703125 |
| quad | direct_C | 22.047743 | 38.671875 | 21.093750 |
| quad | direct_FC | 22.125981 | 39.062500 | 21.093750 |
| quad | direct_four | 22.261887 | 39.062500 | 21.484375 |
| quad | lm_FC | 22.114967 | 39.062500 | 21.093750 |
| quad | joint3_FC | 22.245417 | 39.453125 | 21.093750 |
| quad | kl3_FC | 22.124021 | 39.062500 | 21.093750 |
| quad | kl3_shuffle_FC | 22.481477 | 39.453125 | 21.875000 |
| quad | single0_F | 21.974776 | 39.062500 | 20.703125 |
| quad | single0_C | 22.035149 | 39.453125 | 21.093750 |
| quad | single0_FC | 21.889065 | 38.671875 | 21.093750 |
| quad | single1_F | 22.628025 | 39.843750 | 21.484375 |
| quad | single1_C | 22.474518 | 39.453125 | 21.875000 |
| quad | single1_FC | 22.549898 | 39.453125 | 21.875000 |
| quad | signcommon_F | 22.017348 | 39.062500 | 20.703125 |
| quad | signcommon_C | 22.284261 | 39.062500 | 21.484375 |
| quad | signcommon_FC | 22.284169 | 39.062500 | 21.484375 |
| quad | scaledmean_F | 21.809494 | 38.281250 | 20.703125 |
| quad | scaledmean_C | 21.989489 | 38.281250 | 21.093750 |
| quad | scaledmean_FC | 22.083213 | 38.671875 | 21.093750 |
| quad | shufflecommon_F | 22.109075 | 39.062500 | 21.484375 |
| quad | shufflecommon_C | 22.056791 | 38.281250 | 21.093750 |
| quad | shufflecommon_FC | 22.495165 | 39.843750 | 21.875000 |
| quad | common3_F | 21.924477 | 38.671875 | 20.703125 |
| quad | common3_C | 22.292090 | 39.062500 | 21.484375 |
| quad | common3_FC | 22.291998 | 39.062500 | 21.484375 |
| quad | scaledmean3_F | 21.962282 | 39.062500 | 20.703125 |
| quad | scaledmean3_C | 22.108273 | 38.671875 | 21.093750 |
| quad | scaledmean3_FC | 22.201997 | 39.062500 | 21.093750 |
| quad | shufflecommon3_F | 22.148864 | 39.062500 | 21.484375 |
| quad | shufflecommon3_C | 22.105533 | 38.671875 | 21.093750 |
| quad | shufflecommon3_FC | 22.530068 | 39.843750 | 21.875000 |

## 六方向全部成对结果

| 方向 | 方法 | 对照 | Δ pp | 改善/恶化/不变 |
|---|---|---|---:|---:|
| waymo→drone | common3_FC | prepared_frozen | +0.342869 | 7/5/116 |
| waymo→drone | common3_FC | signcommon_FC | +0.034846 | 5/8/115 |
| waymo→drone | common3_FC | direct_FC | +0.011669 | 5/6/117 |
| waymo→drone | common3_FC | joint3_FC | +0.038585 | 2/1/125 |
| waymo→drone | common3_FC | scaledmean3_FC | +0.010274 | 2/1/125 |
| waymo→drone | common3_FC | shufflecommon3_FC | -0.016128 | 7/8/113 |
| waymo→drone | common3_C | signcommon_C | -0.006034 | 5/8/115 |
| waymo→drone | common3_F | signcommon_F | +0.040882 | 6/8/114 |
| waymo→drone | common3_C | scaledmean3_C | +0.114297 | 4/1/123 |
| waymo→drone | common3_F | scaledmean3_F | +0.018222 | 1/2/125 |
| waymo→quad | common3_FC | prepared_frozen | +0.066119 | 1/2/125 |
| waymo→quad | common3_FC | signcommon_FC | +0.015659 | 2/0/126 |
| waymo→quad | common3_FC | direct_FC | -0.000671 | 0/1/127 |
| waymo→quad | common3_FC | joint3_FC | +0.000000 | 0/0/128 |
| waymo→quad | common3_FC | scaledmean3_FC | +0.201400 | 1/0/127 |
| waymo→quad | common3_FC | shufflecommon3_FC | -0.493646 | 3/4/121 |
| waymo→quad | common3_C | signcommon_C | +0.015659 | 2/0/126 |
| waymo→quad | common3_F | signcommon_F | -0.185741 | 2/1/125 |
| waymo→quad | common3_C | scaledmean3_C | +0.201400 | 1/0/127 |
| waymo→quad | common3_F | scaledmean3_F | +0.000000 | 0/0/128 |
| drone→waymo | common3_FC | prepared_frozen | +0.055285 | 4/1/123 |
| drone→waymo | common3_FC | signcommon_FC | +0.008668 | 2/1/125 |
| drone→waymo | common3_FC | direct_FC | -0.090572 | 3/3/122 |
| drone→waymo | common3_FC | joint3_FC | -0.125660 | 0/1/127 |
| drone→waymo | common3_FC | scaledmean3_FC | +0.031273 | 1/0/127 |
| drone→waymo | common3_FC | shufflecommon3_FC | +0.606010 | 8/0/120 |
| drone→waymo | common3_C | signcommon_C | +0.754974 | 6/1/121 |
| drone→waymo | common3_F | signcommon_F | -0.241842 | 2/3/123 |
| drone→waymo | common3_C | scaledmean3_C | +0.474504 | 3/1/124 |
| drone→waymo | common3_F | scaledmean3_F | +0.131472 | 2/0/126 |
| drone→quad | common3_FC | prepared_frozen | +0.520353 | 3/5/120 |
| drone→quad | common3_FC | signcommon_FC | +0.000000 | 0/0/128 |
| drone→quad | common3_FC | direct_FC | +0.332706 | 4/3/121 |
| drone→quad | common3_FC | joint3_FC | +0.093162 | 2/3/123 |
| drone→quad | common3_FC | scaledmean3_FC | -0.021398 | 1/3/124 |
| drone→quad | common3_FC | shufflecommon3_FC | +0.017506 | 3/4/121 |
| drone→quad | common3_C | signcommon_C | +0.000000 | 0/0/128 |
| drone→quad | common3_F | signcommon_F | +0.000000 | 0/0/128 |
| drone→quad | common3_C | scaledmean3_C | +0.166235 | 1/1/126 |
| drone→quad | common3_F | scaledmean3_F | -0.075610 | 0/4/124 |
| quad→waymo | common3_FC | prepared_frozen | +0.032204 | 3/2/123 |
| quad→waymo | common3_FC | signcommon_FC | -0.242450 | 0/2/126 |
| quad→waymo | common3_FC | direct_FC | -0.102590 | 1/2/125 |
| quad→waymo | common3_FC | joint3_FC | -0.691241 | 0/2/126 |
| quad→waymo | common3_FC | scaledmean3_FC | -0.232976 | 0/1/127 |
| quad→waymo | common3_FC | shufflecommon3_FC | +0.179160 | 2/2/124 |
| quad→waymo | common3_C | signcommon_C | -0.248883 | 0/1/127 |
| quad→waymo | common3_F | signcommon_F | -0.113023 | 1/2/125 |
| quad→waymo | common3_C | scaledmean3_C | +0.000000 | 0/0/128 |
| quad→waymo | common3_F | scaledmean3_F | +0.053441 | 1/0/127 |
| quad→drone | common3_FC | prepared_frozen | -0.128235 | 0/3/125 |
| quad→drone | common3_FC | signcommon_FC | -0.015151 | 0/2/126 |
| quad→drone | common3_FC | direct_FC | +0.658026 | 4/2/122 |
| quad→drone | common3_FC | joint3_FC | -0.157749 | 0/2/126 |
| quad→drone | common3_FC | scaledmean3_FC | +0.528240 | 1/1/126 |
| quad→drone | common3_FC | shufflecommon3_FC | -0.447814 | 2/5/121 |
| quad→drone | common3_C | signcommon_C | -0.165067 | 1/4/123 |
| quad→drone | common3_F | signcommon_F | -0.015151 | 0/2/126 |
| quad→drone | common3_C | scaledmean3_C | -0.402886 | 0/3/125 |
| quad→drone | common3_F | scaledmean3_F | -0.012133 | 0/1/127 |

## 解释边界

G、所有弱证据和拟合目标在更新前冻结；不按GT筛反馈、选择步幅或停止更新。保留全部负例、UNKNOWN和零IoU。F/C/FC均为预设端点，主指标固定FC。原始source、prepared_frozen、旧G的三视图LM和同信息零更新都保留。旧G的R99开发优势在R100全量未保持，因此本轮即便开发通过，也不能宣称稳定有效；需固定后扩大检验。辅助误差下降、参数非零和偏好落实不替代任务指标。

## 裁决

{
  "primary": "common3_FC",
  "primary_mIoU_percent": 16.376751270063476,
  "gains_pp": {
    "prepared_frozen": 0.14809911087479577,
    "signcommon_FC": -0.03307146514755033,
    "direct_FC": 0.13476120096326838,
    "joint3_FC": -0.14048377958479463,
    "scaledmean3_FC": 0.08613556840769476,
    "shufflecommon3_FC": -0.02581869907749379
  },
  "passes_prelocked_primary_gates": false,
  "robust_adoption": false,
  "n": 768,
  "historical_development_exposure": true,
  "no_experts": true,
  "no_source_training": true,
  "FINAL_access": false,
  "primary_not_switched": true,
  "no_more_steps_or_parameter_grid": true
}
