# R102 — 两次错配绑定差分的一致性检验

R101更好拟合固定监督仍损害grounding，因此本轮只改变监督聚合，不修改模型、候选或更新器。每次错配绑定先对三视图真实–错配差分求平均，再对候选居中。两份差分同号时保留较小绝对值，异号记零，最后再居中。这是待验证的保守证据聚合，不把一致当身份真值。没有新的模型前向或参数更新。

先锁定开发768；观察到正信号后，原样固定扩展到历史R98同18620有向输入。完整集合共9310目标表达，各进入两个非自身source方向；不是18620独立场景。FINAL始终排除。主端点始终FC，禁止看到全量后改选F/C。所有负例、UNKNOWN、失败和零IoU保留。

同信息对照包括原G均值、两份单独差分、相同L2范数的原G整体缩放和打乱query对应。范数缩放排除“仅仅减小证据”的解释。开发均值差分余弦0.5463，同号比例72.20%，直接F第一名一致比例80.60%。这些只度量扰动稳定性，不是正确率。

## DEV768

| 方向 | n | 原始 | 冻结 | 旧G直接FC | 共识G直接FC | 同幅度旧G | 打乱共识G |
|---|---:|---:|---:|---:|---:|---:|---:|
| waymo→drone | 128 | 16.856414 | 17.195182 | 17.526382 | 17.503206 | 17.673296 | 17.357094 |
| waymo→quad | 128 | 33.347552 | 29.615174 | 29.681964 | 29.665634 | 29.480564 | 30.133118 |
| drone→waymo | 128 | 14.437008 | 15.126688 | 15.272546 | 15.173305 | 15.124098 | 14.935499 |
| drone→quad | 128 | 16.838098 | 14.382351 | 14.569998 | 14.902704 | 14.685862 | 14.857213 |
| quad→waymo | 128 | 8.510254 | 10.104344 | 10.239138 | 10.378998 | 10.299708 | 9.562268 |
| quad→drone | 128 | 10.878972 | 10.948174 | 10.161913 | 10.835090 | 10.294717 | 10.963766 |

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

| 方向 | 方法 | 对照 | Δ pp | 改善/恶化/不变 |
|---|---|---|---:|---:|
| waymo→drone | signcommon_FC | prepared_frozen | +0.308023 | 10/7/111 |
| waymo→drone | signcommon_FC | direct_FC | -0.023176 | 3/2/123 |
| waymo→drone | signcommon_FC | single0_FC | -0.012820 | 8/7/113 |
| waymo→drone | signcommon_FC | single1_FC | -0.252273 | 7/7/114 |
| waymo→drone | signcommon_FC | scaledmean_FC | -0.170090 | 3/3/122 |
| waymo→drone | signcommon_FC | shufflecommon_FC | +0.146112 | 14/10/104 |
| waymo→drone | signcommon_FC | joint3_FC | +0.003740 | 7/5/116 |
| waymo→drone | signcommon_FC | kl3_FC | +0.014741 | 4/1/123 |
| waymo→drone | signcommon_F | binding_direct | -0.037786 | 3/3/122 |
| waymo→drone | single0_FC | direct_FC | -0.010356 | 7/7/114 |
| waymo→drone | single1_FC | direct_FC | +0.229097 | 9/7/112 |
| waymo→quad | signcommon_FC | prepared_frozen | +0.050460 | 1/4/123 |
| waymo→quad | signcommon_FC | direct_FC | -0.016330 | 0/2/126 |
| waymo→quad | signcommon_FC | single0_FC | -0.013451 | 3/2/123 |
| waymo→quad | signcommon_FC | single1_FC | -0.570163 | 2/3/123 |
| waymo→quad | signcommon_FC | scaledmean_FC | +0.185070 | 1/2/125 |
| waymo→quad | signcommon_FC | shufflecommon_FC | -0.467483 | 3/6/119 |
| waymo→quad | signcommon_FC | joint3_FC | -0.015659 | 0/2/126 |
| waymo→quad | signcommon_FC | kl3_FC | -0.011105 | 0/1/127 |
| waymo→quad | signcommon_F | binding_direct | -0.016330 | 0/2/126 |
| waymo→quad | single0_FC | direct_FC | -0.002879 | 1/5/122 |
| waymo→quad | single1_FC | direct_FC | +0.553833 | 1/3/124 |
| drone→waymo | signcommon_FC | prepared_frozen | +0.046617 | 4/2/122 |
| drone→waymo | signcommon_FC | direct_FC | -0.099241 | 2/4/122 |
| drone→waymo | signcommon_FC | single0_FC | -0.939534 | 1/6/121 |
| drone→waymo | signcommon_FC | single1_FC | -0.096529 | 4/6/118 |
| drone→waymo | signcommon_FC | scaledmean_FC | +0.049207 | 2/1/125 |
| drone→waymo | signcommon_FC | shufflecommon_FC | +0.237806 | 7/3/118 |
| drone→waymo | signcommon_FC | joint3_FC | -0.134328 | 1/3/124 |
| drone→waymo | signcommon_FC | kl3_FC | -0.218392 | 1/4/123 |
| drone→waymo | signcommon_F | binding_direct | +0.388108 | 2/1/125 |
| drone→waymo | single0_FC | direct_FC | +0.840293 | 4/1/123 |
| drone→waymo | single1_FC | direct_FC | -0.002712 | 4/4/120 |
| drone→quad | signcommon_FC | prepared_frozen | +0.520353 | 3/5/120 |
| drone→quad | signcommon_FC | direct_FC | +0.332706 | 4/3/121 |
| drone→quad | signcommon_FC | single0_FC | +0.803658 | 4/5/119 |
| drone→quad | signcommon_FC | single1_FC | +0.038706 | 3/5/120 |
| drone→quad | signcommon_FC | scaledmean_FC | +0.216841 | 2/3/123 |
| drone→quad | signcommon_FC | shufflecommon_FC | +0.045491 | 4/4/120 |
| drone→quad | signcommon_FC | joint3_FC | +0.093162 | 2/3/123 |
| drone→quad | signcommon_FC | kl3_FC | +0.331400 | 3/3/122 |
| drone→quad | signcommon_F | binding_direct | +0.225342 | 3/4/121 |
| drone→quad | single0_FC | direct_FC | -0.470952 | 5/4/119 |
| drone→quad | single1_FC | direct_FC | +0.294000 | 5/2/121 |
| quad→waymo | signcommon_FC | prepared_frozen | +0.274654 | 5/2/121 |
| quad→waymo | signcommon_FC | direct_FC | +0.139860 | 1/0/127 |
| quad→waymo | signcommon_FC | single0_FC | +0.115102 | 1/1/126 |
| quad→waymo | signcommon_FC | single1_FC | +0.556033 | 3/1/124 |
| quad→waymo | signcommon_FC | scaledmean_FC | +0.079291 | 1/1/126 |
| quad→waymo | signcommon_FC | shufflecommon_FC | +0.816731 | 7/3/118 |
| quad→waymo | signcommon_FC | joint3_FC | -0.448790 | 1/1/126 |
| quad→waymo | signcommon_FC | kl3_FC | +0.139860 | 1/0/127 |
| quad→waymo | signcommon_F | binding_direct | -0.004583 | 2/1/125 |
| quad→waymo | single0_FC | direct_FC | +0.024758 | 2/1/125 |
| quad→waymo | single1_FC | direct_FC | -0.416173 | 1/3/124 |
| quad→drone | signcommon_FC | prepared_frozen | -0.113084 | 1/2/125 |
| quad→drone | signcommon_FC | direct_FC | +0.673177 | 4/0/124 |
| quad→drone | signcommon_FC | single0_FC | +0.111516 | 3/1/124 |
| quad→drone | signcommon_FC | single1_FC | +0.476310 | 5/2/121 |
| quad→drone | signcommon_FC | scaledmean_FC | +0.540373 | 1/0/127 |
| quad→drone | signcommon_FC | shufflecommon_FC | -0.128676 | 6/5/117 |
| quad→drone | signcommon_FC | joint3_FC | -0.142598 | 1/1/126 |
| quad→drone | signcommon_FC | kl3_FC | +0.662857 | 3/0/125 |
| quad→drone | signcommon_F | binding_direct | +0.054821 | 2/0/126 |
| quad→drone | single0_FC | direct_FC | +0.561661 | 3/1/124 |
| quad→drone | single1_FC | direct_FC | +0.196867 | 4/3/121 |

## FULL18620

| 方向 | n | 原始 | 冻结 | 旧G直接FC | 共识G直接FC | 同幅度旧G | 打乱共识G |
|---|---:|---:|---:|---:|---:|---:|---:|
| waymo→drone | 2792 | 8.832240 | 9.285471 | 9.253592 | 9.236819 | 9.301077 | 9.284675 |
| waymo→quad | 2736 | 17.588082 | 18.750207 | 18.808625 | 18.862290 | 18.789741 | 18.909557 |
| drone→waymo | 3782 | 14.142348 | 14.609560 | 14.600940 | 14.625480 | 14.602606 | 14.625298 |
| drone→quad | 2736 | 13.077292 | 12.901516 | 12.929359 | 12.934152 | 12.934061 | 12.887069 |
| quad→waymo | 3782 | 10.829191 | 10.898764 | 10.701449 | 10.803486 | 10.759689 | 10.848284 |
| quad→drone | 2792 | 6.055598 | 6.725807 | 6.683912 | 6.733094 | 6.661052 | 6.611051 |

| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |
|---|---|---:|---:|---:|
| all_six_equal | original_source | 11.754125 | 22.475706 | 6.148478 |
| all_six_equal | prepared_frozen | 12.195221 | 22.844980 | 5.750723 |
| all_six_equal | binding_direct | 12.222276 | 22.855125 | 5.770964 |
| all_six_equal | direct_C | 12.158858 | 22.632522 | 5.790887 |
| all_six_equal | direct_FC | 12.162979 | 22.670427 | 5.771452 |
| all_six_equal | direct_four | 12.164657 | 22.691301 | 5.806585 |
| all_six_equal | lm_FC | 12.179236 | 22.745868 | 5.772367 |
| all_six_equal | joint3_FC | 12.181929 | 22.735614 | 5.786113 |
| all_six_equal | single0_F | 12.180643 | 22.709233 | 5.805706 |
| all_six_equal | single0_C | 12.126219 | 22.520969 | 5.851826 |
| all_six_equal | single0_FC | 12.138056 | 22.527012 | 5.820734 |
| all_six_equal | single1_F | 12.204275 | 22.905969 | 5.790679 |
| all_six_equal | single1_C | 12.182693 | 22.780710 | 5.819135 |
| all_six_equal | single1_FC | 12.169919 | 22.767549 | 5.794805 |
| all_six_equal | signcommon_F | 12.198987 | 22.844309 | 5.743875 |
| all_six_equal | signcommon_C | 12.165243 | 22.717058 | 5.819257 |
| all_six_equal | signcommon_FC | 12.199220 | 22.788509 | 5.801666 |
| all_six_equal | scaledmean_F | 12.190976 | 22.816329 | 5.773563 |
| all_six_equal | scaledmean_C | 12.164707 | 22.693362 | 5.823175 |
| all_six_equal | scaledmean_FC | 12.174704 | 22.709966 | 5.787151 |
| all_six_equal | shufflecommon_F | 12.194806 | 22.742657 | 5.854866 |
| all_six_equal | shufflecommon_C | 12.148601 | 22.584949 | 5.865205 |
| all_six_equal | shufflecommon_FC | 12.194323 | 22.693095 | 5.864046 |
| waymo | original_source | 12.485769 | 26.070862 | 6.795346 |
| waymo | prepared_frozen | 12.754162 | 26.692226 | 6.041777 |
| waymo | binding_direct | 12.720859 | 26.560021 | 5.975674 |
| waymo | direct_C | 12.664009 | 26.467478 | 6.015336 |
| waymo | direct_FC | 12.651195 | 26.401375 | 5.975674 |
| waymo | direct_four | 12.665250 | 26.427816 | 6.081438 |
| waymo | lm_FC | 12.707832 | 26.573242 | 6.015336 |
| waymo | joint3_FC | 12.702271 | 26.560021 | 6.002115 |
| waymo | single0_F | 12.718375 | 26.520360 | 5.988895 |
| waymo | single0_C | 12.695654 | 26.493919 | 6.107879 |
| waymo | single0_FC | 12.675553 | 26.388154 | 6.015336 |
| waymo | single1_F | 12.686828 | 26.586462 | 5.962454 |
| waymo | single1_C | 12.674488 | 26.493919 | 6.081438 |
| waymo | single1_FC | 12.664932 | 26.493919 | 6.028556 |
| waymo | signcommon_F | 12.752254 | 26.652565 | 5.949233 |
| waymo | signcommon_C | 12.723929 | 26.612903 | 6.081438 |
| waymo | signcommon_FC | 12.714483 | 26.573242 | 5.975674 |
| waymo | scaledmean_F | 12.739640 | 26.679006 | 6.002115 |
| waymo | scaledmean_C | 12.715035 | 26.652565 | 6.094659 |
| waymo | scaledmean_FC | 12.681147 | 26.520360 | 6.041777 |
| waymo | shufflecommon_F | 12.740385 | 26.546801 | 6.028556 |
| waymo | shufflecommon_C | 12.708542 | 26.507139 | 6.094659 |
| waymo | shufflecommon_FC | 12.736791 | 26.560021 | 6.054997 |
| drone | original_source | 7.443919 | 14.272923 | 2.238539 |
| drone | prepared_frozen | 8.005639 | 14.613181 | 2.292264 |
| drone | binding_direct | 8.092262 | 14.702722 | 2.345989 |
| drone | direct_C | 7.920804 | 14.237106 | 2.256447 |
| drone | direct_FC | 7.968752 | 14.380372 | 2.274355 |
| drone | direct_four | 7.956074 | 14.398281 | 2.292264 |
| drone | lm_FC | 7.947229 | 14.398281 | 2.292264 |
| drone | joint3_FC | 7.947535 | 14.362464 | 2.310172 |
| drone | single0_F | 8.048543 | 14.505731 | 2.363897 |
| drone | single0_C | 7.901409 | 14.022206 | 2.310172 |
| drone | single0_FC | 7.969743 | 14.219198 | 2.345989 |
| drone | single1_F | 8.107708 | 14.792264 | 2.381805 |
| drone | single1_C | 7.918549 | 14.344556 | 2.238539 |
| drone | single1_FC | 7.982947 | 14.487822 | 2.328080 |
| drone | signcommon_F | 8.046492 | 14.559456 | 2.345989 |
| drone | signcommon_C | 7.922168 | 14.308739 | 2.220630 |
| drone | signcommon_FC | 7.984957 | 14.398281 | 2.310172 |
| drone | scaledmean_F | 8.041600 | 14.595272 | 2.363897 |
| drone | scaledmean_C | 7.941723 | 14.362464 | 2.292264 |
| drone | scaledmean_FC | 7.981065 | 14.398281 | 2.310172 |
| drone | shufflecommon_F | 7.994432 | 14.469914 | 2.453438 |
| drone | shufflecommon_C | 7.872167 | 14.219198 | 2.381805 |
| drone | shufflecommon_FC | 7.947863 | 14.344556 | 2.399713 |
| quad | original_source | 15.332687 | 27.083333 | 9.411550 |
| quad | prepared_frozen | 15.825862 | 27.229532 | 8.918129 |
| quad | binding_direct | 15.853709 | 27.302632 | 8.991228 |
| quad | direct_C | 15.891760 | 27.192982 | 9.100877 |
| quad | direct_FC | 15.868992 | 27.229532 | 9.064327 |
| quad | direct_four | 15.872648 | 27.247807 | 9.046053 |
| quad | lm_FC | 15.882647 | 27.266082 | 9.009503 |
| quad | joint3_FC | 15.895980 | 27.284357 | 9.046053 |
| quad | single0_F | 15.775010 | 27.101608 | 9.064327 |
| quad | single0_C | 15.781594 | 27.046784 | 9.137427 |
| quad | single0_FC | 15.768873 | 26.973684 | 9.100877 |
| quad | single1_F | 15.818290 | 27.339181 | 9.027778 |
| quad | single1_C | 15.955041 | 27.503655 | 9.137427 |
| quad | single1_FC | 15.861879 | 27.320906 | 9.027778 |
| quad | signcommon_F | 15.798215 | 27.320906 | 8.936404 |
| quad | signcommon_C | 15.849633 | 27.229532 | 9.155702 |
| quad | signcommon_FC | 15.898221 | 27.394006 | 9.119152 |
| quad | scaledmean_F | 15.791688 | 27.174708 | 8.954678 |
| quad | scaledmean_C | 15.837363 | 27.065058 | 9.082602 |
| quad | scaledmean_FC | 15.861901 | 27.211257 | 9.009503 |
| quad | shufflecommon_F | 15.849602 | 27.211257 | 9.082602 |
| quad | shufflecommon_C | 15.865095 | 27.028509 | 9.119152 |
| quad | shufflecommon_FC | 15.898313 | 27.174708 | 9.137427 |

| 方向 | 方法 | 对照 | Δ pp | 改善/恶化/不变 |
|---|---|---|---:|---:|
| waymo→drone | signcommon_FC | prepared_frozen | -0.048652 | 142/182/2468 |
| waymo→drone | signcommon_FC | direct_FC | -0.016773 | 86/78/2628 |
| waymo→drone | signcommon_FC | single0_FC | -0.040848 | 137/133/2522 |
| waymo→drone | signcommon_FC | single1_FC | -0.101802 | 137/144/2511 |
| waymo→drone | signcommon_FC | scaledmean_FC | -0.064258 | 68/84/2640 |
| waymo→drone | signcommon_FC | shufflecommon_FC | -0.047856 | 268/247/2277 |
| waymo→drone | signcommon_FC | joint3_FC | -0.031829 | 109/135/2548 |
| waymo→drone | signcommon_F | binding_direct | -0.033884 | 79/67/2646 |
| waymo→drone | single0_FC | direct_FC | +0.024075 | 115/119/2558 |
| waymo→drone | single1_FC | direct_FC | +0.085029 | 134/125/2533 |
| waymo→quad | signcommon_FC | prepared_frozen | +0.112083 | 156/125/2455 |
| waymo→quad | signcommon_FC | direct_FC | +0.053665 | 75/59/2602 |
| waymo→quad | signcommon_FC | single0_FC | +0.099028 | 127/99/2510 |
| waymo→quad | signcommon_FC | single1_FC | +0.076370 | 125/110/2501 |
| waymo→quad | signcommon_FC | scaledmean_FC | +0.072549 | 77/52/2607 |
| waymo→quad | signcommon_FC | shufflecommon_FC | -0.047267 | 210/188/2338 |
| waymo→quad | signcommon_FC | joint3_FC | +0.036589 | 98/76/2562 |
| waymo→quad | signcommon_F | binding_direct | -0.014647 | 68/77/2591 |
| waymo→quad | single0_FC | direct_FC | -0.045363 | 100/119/2517 |
| waymo→quad | single1_FC | direct_FC | -0.022705 | 103/102/2531 |
| drone→waymo | signcommon_FC | prepared_frozen | +0.015921 | 138/139/3505 |
| drone→waymo | signcommon_FC | direct_FC | +0.024541 | 59/52/3671 |
| drone→waymo | signcommon_FC | single0_FC | -0.089503 | 72/94/3616 |
| drone→waymo | signcommon_FC | single1_FC | -0.000551 | 101/82/3599 |
| drone→waymo | signcommon_FC | scaledmean_FC | +0.022875 | 55/47/3680 |
| drone→waymo | signcommon_FC | shufflecommon_FC | +0.000182 | 160/184/3438 |
| drone→waymo | signcommon_FC | joint3_FC | -0.028366 | 67/66/3649 |
| drone→waymo | signcommon_F | binding_direct | +0.028408 | 57/46/3679 |
| drone→waymo | single0_FC | direct_FC | +0.114044 | 92/64/3626 |
| drone→waymo | single1_FC | direct_FC | +0.025091 | 77/89/3616 |
| drone→quad | signcommon_FC | prepared_frozen | +0.032635 | 116/127/2493 |
| drone→quad | signcommon_FC | direct_FC | +0.004793 | 61/54/2621 |
| drone→quad | signcommon_FC | single0_FC | +0.159667 | 105/86/2545 |
| drone→quad | signcommon_FC | single1_FC | -0.003686 | 96/102/2538 |
| drone→quad | signcommon_FC | scaledmean_FC | +0.000090 | 45/43/2648 |
| drone→quad | signcommon_FC | shufflecommon_FC | +0.047082 | 151/144/2441 |
| drone→quad | signcommon_FC | joint3_FC | -0.032107 | 73/66/2597 |
| drone→quad | signcommon_F | binding_direct | -0.096341 | 55/50/2631 |
| drone→quad | single0_FC | direct_FC | -0.154874 | 92/99/2545 |
| drone→quad | single1_FC | direct_FC | +0.008479 | 80/73/2583 |
| quad→waymo | signcommon_FC | prepared_frozen | -0.095278 | 97/107/3578 |
| quad→waymo | signcommon_FC | direct_FC | +0.102037 | 44/32/3706 |
| quad→waymo | signcommon_FC | single0_FC | +0.167364 | 68/53/3661 |
| quad→waymo | signcommon_FC | single1_FC | +0.099654 | 64/48/3670 |
| quad→waymo | signcommon_FC | scaledmean_FC | +0.043797 | 28/25/3729 |
| quad→waymo | signcommon_FC | shufflecommon_FC | -0.044798 | 110/114/3558 |
| quad→waymo | signcommon_FC | joint3_FC | +0.052791 | 45/40/3697 |
| quad→waymo | signcommon_F | binding_direct | +0.034383 | 37/34/3711 |
| quad→waymo | single0_FC | direct_FC | -0.065327 | 55/59/3668 |
| quad→waymo | single1_FC | direct_FC | +0.002382 | 45/53/3684 |
| quad→drone | signcommon_FC | prepared_frozen | +0.007287 | 68/89/2635 |
| quad→drone | signcommon_FC | direct_FC | +0.049182 | 49/39/2704 |
| quad→drone | signcommon_FC | single0_FC | +0.071275 | 71/60/2661 |
| quad→drone | signcommon_FC | single1_FC | +0.105820 | 80/64/2648 |
| quad→drone | signcommon_FC | scaledmean_FC | +0.072042 | 35/28/2729 |
| quad→drone | signcommon_FC | shufflecommon_FC | +0.122043 | 174/157/2461 |
| quad→drone | signcommon_FC | joint3_FC | +0.106671 | 81/69/2642 |
| quad→drone | signcommon_F | binding_direct | -0.057657 | 52/47/2693 |
| quad→drone | single0_FC | direct_FC | -0.022092 | 58/63/2671 |
| quad→drone | single1_FC | direct_FC | -0.056638 | 64/70/2658 |

## 扩大部分单列

| 部分 | 对照 | 等权 Δ pp | 改善/恶化 |
|---|---|---:|---:|
| DEV768 | prepared_frozen | +0.181171 | 24/22 |
| DEV768 | direct_FC | +0.167833 | 14/11 |
| DEV768 | scaledmean_FC | +0.150115 | 10/10 |
| DEV768 | shufflecommon_FC | +0.108330 | 41/31 |
| DEV768 | joint3_FC | -0.107412 | 12/15 |
| EXPANSION_ONLY | prepared_frozen | -0.003717 | 693/747 |
| EXPANSION_ONLY | direct_FC | +0.029687 | 360/303 |
| EXPANSION_ONLY | scaledmean_FC | +0.018559 | 298/269 |
| EXPANSION_ONLY | shufflecommon_FC | +0.002385 | 1032/1003 |
| EXPANSION_ONLY | joint3_FC | +0.021952 | 461/437 |

## 文献边界

[Reg-TTA3D原文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/05971.pdf)用噪声稳定性筛伪框，但原任务是检测、监督定位回归。本轮只借鉴先检查稳定性的研究逻辑；两次反事实差分取同号最小值是这里的实验设计，不是其NPG复现。稳定和正确必须由本轮任务对照区分。

## 裁决

{
  "state": "R102_FIXED_COUNTERFACTUAL_CONSENSUS_DIRECT_READOUT_EVALUATED",
  "scopes": {
    "DEV768": {
      "primary": "signcommon_FC",
      "mIoU_percent": 16.409822735211026,
      "gains_pp": {
        "prepared_frozen": 0.1811705760223461,
        "direct_FC": 0.1678326661108187,
        "scaledmean_FC": 0.150115301476081,
        "shufflecommon_FC": 0.10832983354397285
      },
      "passes_primary_comparisons": true,
      "robust_adoption": false
    },
    "FULL18620": {
      "primary": "signcommon_FC",
      "mIoU_percent": 12.199220240747776,
      "gains_pp": {
        "prepared_frozen": 0.0039995488351891595,
        "direct_FC": 0.0362408296553447,
        "scaledmean_FC": 0.02451583051164441,
        "shufflecommon_FC": 0.004897685526874085
      },
      "passes_primary_comparisons": true,
      "robust_adoption": false
    }
  },
  "robustly_adopted": false,
  "TTA_run": false,
  "FINAL_access": false,
  "external_experts": false,
  "source_training": false,
  "full_historically_exposed": true,
  "next_action": "consider_same_signal_parameter_comparison_only_if_full_primary_passes"
}

好坏变化按固定规则收集，IoU变化不等于身份标签；不得据它们筛测试输入。若全量未保持开发优势，不以换阈值、增减场景或选择更好的单错配结果挽救。本轮没有运行TTA；直接证据不成立时，不让优化器承担监督纠错。


## 实际图页复核

本地生成12页，实际查看2页（不能写成12页均人工核验）。两个相反的Quad→Drone例子均显示：F和C各自第一名可保持不变，最终FC却因概率间隔变化而切换。一条由旧G直接FC的0恢复到0.6917，另一条由0.3980变为0；后者的F仍为0.3980。可视化的投影与场景支持它们涉及不同空间区域，但不把投影或IoU当正式身份真值。两条仅帮助解释融合干扰，不生成个案门槛，所有样本仍在总体分母内。原句、图像、坐标和query索引仅保留本地。

全量相对冻结仅+0.004000pp，扩大部分17852条为-0.003717pp；对旧G的+0.036241pp更适合解释为减轻旧证据路径的损伤，尚不是稳定的增量TTA结果。下一轮若接参数，必须保持相同证据，并明确区分原输入F读出与加入未拟合遮挡C的FC读出；不能事后换端点声称原主实验成功。
