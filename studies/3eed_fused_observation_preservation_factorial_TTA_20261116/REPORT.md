# R107 — 固定实测点观测的监督来源与融合状态保持

沿用R106真实注册B/C点云、同帧randomA和错配支持，原A RGB、坐标、文本视角、1024空间seed及256最终框不变。冻结三个source100与R91双向结构；只更新18496参数末端匹配投影，不增加专家或source训练。

## 锁定对照

R106原始均值目标两视图拟合原样复用。主实验保持相同目标q=(pA+pB)/2，在A/B拟合之外，以source原生fusedAB分布为第三视图的完整输出保持目标。两个诊断分别改为q=pAB两视图拟合，以及q=pAB+第三视图保持；等视图权重、LM10×6不变。R30已有输出保持思想，本轮不将保持项宣称为新贡献。

额外AB源状态的计算属于新增信息/成本，完整提供冻结AB、AB+C、四分布平均、直接q及覆盖匹配q+C。R106旧方向按新主实验的参数L2范数缩放，独立检查效果是否仅来自减弱更新；只使用权重，不用C或GT。随机A/错配/打乱目标均保留。

C/fusedABC缓存首次进入本轮程序在所有拟合和范数控制之后，事件顺序经独立CPU核验。但C和DEV已历史暴露，不能称独立确认。新增原生AB提取的B采样、物理点哈希、注册哈希与R106逐条一致；12条完整原生写参数前向验证，不依赖重放自洽而已。

## 全体结果

| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |
|---|---|---:|---:|---:|
| all_six_equal | original_source | 16.811383 | 30.989583 | 14.973958 |
| all_six_equal | prepared_frozen | 16.228652 | 29.947917 | 12.630208 |
| all_six_equal | frozen_four | 16.249883 | 29.947917 | 12.890625 |
| all_six_equal | direct_FC | 16.241990 | 29.817708 | 12.760417 |
| all_six_equal | joint3_FC | 16.517235 | 30.338542 | 13.020833 |
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
| all_six_equal | R106_teacher_direct_all | 16.428953 | 30.468750 | 12.760417 |
| all_six_equal | R106_teacher_direct_matched | 16.494894 | 30.598958 | 12.890625 |
| all_six_equal | R106_teacher_direct_matched_C | 16.363293 | 30.338542 | 12.500000 |
| all_six_equal | real_frozen_AB | 16.509657 | 30.468750 | 13.020833 |
| all_six_equal | real_frozen_AB_C | 16.504551 | 30.598958 | 12.630208 |
| all_six_equal | real_frozen_ensemble4 | 16.549388 | 30.729167 | 12.500000 |
| all_six_equal | randomA_frozen_AB | 16.569063 | 30.859375 | 12.890625 |
| all_six_equal | randomA_frozen_AB_C | 16.518506 | 30.468750 | 13.020833 |
| all_six_equal | randomA_frozen_ensemble4 | 16.363643 | 30.078125 | 13.020833 |
| all_six_equal | wrong_frozen_AB | 15.026499 | 27.604167 | 11.458333 |
| all_six_equal | wrong_frozen_AB_C | 13.436231 | 24.739583 | 10.156250 |
| all_six_equal | wrong_frozen_ensemble4 | 15.295641 | 27.604167 | 12.239583 |
| all_six_equal | real_mean_direct_all | 16.428953 | 30.468750 | 12.760417 |
| all_six_equal | real_mean_direct_matched | 16.494894 | 30.598958 | 12.890625 |
| all_six_equal | real_mean_direct_matched_C | 16.451364 | 30.598958 | 12.500000 |
| all_six_equal | real_native_direct_all | 16.509657 | 30.468750 | 13.020833 |
| all_six_equal | real_native_direct_matched | 16.538798 | 30.598958 | 13.020833 |
| all_six_equal | real_native_direct_matched_C | 16.504551 | 30.598958 | 12.630208 |
| all_six_equal | real_mean_keep_C | 16.355654 | 30.208333 | 12.760417 |
| all_six_equal | real_mean_keep_fusedABC | 16.583791 | 30.598958 | 13.151042 |
| all_six_equal | real_mean_keep_AB_C | 16.339738 | 30.208333 | 12.500000 |
| all_six_equal | real_native2_C | 16.179669 | 29.427083 | 12.760417 |
| all_six_equal | real_native2_fusedABC | 16.491111 | 30.468750 | 13.020833 |
| all_six_equal | real_native2_AB_C | 16.251081 | 29.947917 | 12.369792 |
| all_six_equal | real_native_keep_C | 16.378459 | 30.078125 | 12.630208 |
| all_six_equal | real_native_keep_fusedABC | 16.490384 | 30.468750 | 13.020833 |
| all_six_equal | real_native_keep_AB_C | 16.358593 | 30.208333 | 12.500000 |
| all_six_equal | randomA_mean_keep_C | 16.039112 | 29.687500 | 12.369792 |
| all_six_equal | randomA_mean_keep_fusedABC | 16.563499 | 30.468750 | 13.020833 |
| all_six_equal | randomA_mean_keep_AB_C | 16.648723 | 30.598958 | 13.020833 |
| all_six_equal | wrong_mean_keep_C | 11.862371 | 21.744792 | 8.333333 |
| all_six_equal | wrong_mean_keep_fusedABC | 13.328739 | 24.348958 | 10.156250 |
| all_six_equal | wrong_mean_keep_AB_C | 13.712889 | 25.520833 | 10.156250 |
| all_six_equal | teacher_shuffle_keep_C | 16.106692 | 29.296875 | 12.500000 |
| all_six_equal | teacher_shuffle_keep_fusedABC | 16.367031 | 30.078125 | 12.630208 |
| all_six_equal | teacher_shuffle_keep_AB_C | 16.094797 | 29.687500 | 11.979167 |
| all_six_equal | norm_old_direction_C | 16.344438 | 29.947917 | 12.760417 |
| all_six_equal | norm_old_direction_fusedABC | 16.544129 | 30.468750 | 13.020833 |
| all_six_equal | norm_old_direction_AB_C | 16.335632 | 30.208333 | 12.500000 |
| waymo | original_source | 11.473631 | 23.437500 | 7.421875 |
| waymo | prepared_frozen | 12.615516 | 26.171875 | 5.078125 |
| waymo | frozen_four | 12.546748 | 25.781250 | 5.468750 |
| waymo | direct_FC | 12.755842 | 25.390625 | 5.468750 |
| waymo | joint3_FC | 13.067711 | 26.953125 | 5.859375 |
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
| waymo | R106_teacher_direct_all | 12.643751 | 26.171875 | 5.078125 |
| waymo | R106_teacher_direct_matched | 12.643751 | 26.171875 | 5.078125 |
| waymo | R106_teacher_direct_matched_C | 12.505120 | 26.171875 | 5.078125 |
| waymo | real_frozen_AB | 12.629042 | 26.171875 | 5.078125 |
| waymo | real_frozen_AB_C | 12.643537 | 26.562500 | 5.078125 |
| waymo | real_frozen_ensemble4 | 12.647714 | 26.562500 | 5.078125 |
| waymo | randomA_frozen_AB | 12.549646 | 26.171875 | 4.687500 |
| waymo | randomA_frozen_AB_C | 12.301471 | 25.000000 | 5.078125 |
| waymo | randomA_frozen_ensemble4 | 12.392873 | 25.390625 | 5.078125 |
| waymo | wrong_frozen_AB | 12.719721 | 26.171875 | 5.078125 |
| waymo | wrong_frozen_AB_C | 11.909563 | 24.609375 | 5.078125 |
| waymo | wrong_frozen_ensemble4 | 12.481666 | 25.390625 | 5.078125 |
| waymo | real_mean_direct_all | 12.643751 | 26.171875 | 5.078125 |
| waymo | real_mean_direct_matched | 12.643751 | 26.171875 | 5.078125 |
| waymo | real_mean_direct_matched_C | 12.505120 | 26.171875 | 5.078125 |
| waymo | real_native_direct_all | 12.629042 | 26.171875 | 5.078125 |
| waymo | real_native_direct_matched | 12.629042 | 26.171875 | 5.078125 |
| waymo | real_native_direct_matched_C | 12.643537 | 26.562500 | 5.078125 |
| waymo | real_mean_keep_C | 12.690348 | 26.562500 | 5.078125 |
| waymo | real_mean_keep_fusedABC | 12.663880 | 26.171875 | 5.078125 |
| waymo | real_mean_keep_AB_C | 12.727878 | 26.562500 | 5.078125 |
| waymo | real_native2_C | 12.770512 | 26.562500 | 5.078125 |
| waymo | real_native2_fusedABC | 12.649171 | 26.171875 | 5.078125 |
| waymo | real_native2_AB_C | 12.727878 | 26.562500 | 5.078125 |
| waymo | real_native_keep_C | 12.913106 | 26.953125 | 5.078125 |
| waymo | real_native_keep_fusedABC | 12.649171 | 26.171875 | 5.078125 |
| waymo | real_native_keep_AB_C | 12.727878 | 26.562500 | 5.078125 |
| waymo | randomA_mean_keep_C | 12.265264 | 25.000000 | 5.078125 |
| waymo | randomA_mean_keep_fusedABC | 12.283274 | 25.000000 | 5.078125 |
| waymo | randomA_mean_keep_AB_C | 12.297704 | 25.000000 | 5.078125 |
| waymo | wrong_mean_keep_C | 11.689546 | 23.828125 | 5.078125 |
| waymo | wrong_mean_keep_fusedABC | 12.340383 | 25.390625 | 5.078125 |
| waymo | wrong_mean_keep_AB_C | 11.937798 | 24.609375 | 5.078125 |
| waymo | teacher_shuffle_keep_C | 12.863807 | 26.953125 | 5.078125 |
| waymo | teacher_shuffle_keep_fusedABC | 12.776311 | 26.562500 | 5.078125 |
| waymo | teacher_shuffle_keep_AB_C | 12.826246 | 26.953125 | 5.078125 |
| waymo | norm_old_direction_C | 12.770512 | 26.562500 | 5.078125 |
| waymo | norm_old_direction_fusedABC | 12.544687 | 25.781250 | 5.078125 |
| waymo | norm_old_direction_AB_C | 12.742587 | 26.562500 | 5.078125 |
| drone | original_source | 13.867693 | 24.609375 | 12.500000 |
| drone | prepared_frozen | 14.071678 | 24.609375 | 11.718750 |
| drone | frozen_four | 13.894458 | 24.609375 | 11.718750 |
| drone | direct_FC | 13.844147 | 25.000000 | 11.718750 |
| drone | joint3_FC | 14.238577 | 24.609375 | 12.109375 |
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
| drone | R106_teacher_direct_all | 13.999520 | 24.609375 | 11.718750 |
| drone | R106_teacher_direct_matched | 14.025223 | 24.609375 | 11.718750 |
| drone | R106_teacher_direct_matched_C | 13.985618 | 24.609375 | 11.718750 |
| drone | real_frozen_AB | 13.955078 | 24.218750 | 11.718750 |
| drone | real_frozen_AB_C | 14.093610 | 25.000000 | 11.718750 |
| drone | real_frozen_ensemble4 | 14.079439 | 24.609375 | 11.718750 |
| drone | randomA_frozen_AB | 14.221114 | 25.390625 | 11.718750 |
| drone | randomA_frozen_AB_C | 14.169714 | 25.000000 | 11.718750 |
| drone | randomA_frozen_ensemble4 | 14.117425 | 25.000000 | 11.718750 |
| drone | wrong_frozen_AB | 13.092185 | 23.046875 | 11.718750 |
| drone | wrong_frozen_AB_C | 13.380032 | 23.828125 | 11.718750 |
| drone | wrong_frozen_ensemble4 | 13.303630 | 23.437500 | 11.718750 |
| drone | real_mean_direct_all | 13.999520 | 24.609375 | 11.718750 |
| drone | real_mean_direct_matched | 14.025223 | 24.609375 | 11.718750 |
| drone | real_mean_direct_matched_C | 14.093610 | 25.000000 | 11.718750 |
| drone | real_native_direct_all | 13.955078 | 24.218750 | 11.718750 |
| drone | real_native_direct_matched | 14.025223 | 24.609375 | 11.718750 |
| drone | real_native_direct_matched_C | 14.093610 | 25.000000 | 11.718750 |
| drone | real_mean_keep_C | 13.972478 | 24.609375 | 11.718750 |
| drone | real_mean_keep_fusedABC | 14.071892 | 25.000000 | 11.718750 |
| drone | real_mean_keep_AB_C | 14.264892 | 25.390625 | 11.718750 |
| drone | real_native2_C | 13.710987 | 23.437500 | 11.718750 |
| drone | real_native2_fusedABC | 14.034676 | 25.000000 | 11.718750 |
| drone | real_native2_AB_C | 14.239955 | 25.390625 | 11.718750 |
| drone | real_native_keep_C | 13.914914 | 24.609375 | 11.718750 |
| drone | real_native_keep_fusedABC | 14.061439 | 25.000000 | 11.718750 |
| drone | real_native_keep_AB_C | 14.241710 | 25.390625 | 11.718750 |
| drone | randomA_mean_keep_C | 13.903680 | 24.609375 | 11.718750 |
| drone | randomA_mean_keep_fusedABC | 14.316458 | 25.781250 | 11.718750 |
| drone | randomA_mean_keep_AB_C | 14.294594 | 25.390625 | 11.718750 |
| drone | wrong_mean_keep_C | 13.189602 | 23.437500 | 11.328125 |
| drone | wrong_mean_keep_fusedABC | 12.716710 | 22.265625 | 11.718750 |
| drone | wrong_mean_keep_AB_C | 13.414342 | 23.828125 | 11.718750 |
| drone | teacher_shuffle_keep_C | 13.856018 | 23.828125 | 11.718750 |
| drone | teacher_shuffle_keep_fusedABC | 13.898231 | 24.218750 | 11.718750 |
| drone | teacher_shuffle_keep_AB_C | 14.109267 | 24.609375 | 11.718750 |
| drone | norm_old_direction_C | 13.805893 | 23.828125 | 11.718750 |
| drone | norm_old_direction_fusedABC | 13.932660 | 24.609375 | 11.718750 |
| drone | norm_old_direction_AB_C | 14.227676 | 25.390625 | 11.718750 |
| quad | original_source | 25.092825 | 44.921875 | 25.000000 |
| quad | prepared_frozen | 21.998762 | 39.062500 | 21.093750 |
| quad | frozen_four | 22.308444 | 39.453125 | 21.484375 |
| quad | direct_FC | 22.125981 | 39.062500 | 21.093750 |
| quad | joint3_FC | 22.245417 | 39.453125 | 21.093750 |
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
| quad | R106_teacher_direct_all | 22.643588 | 40.625000 | 21.484375 |
| quad | R106_teacher_direct_matched | 22.815707 | 41.015625 | 21.875000 |
| quad | R106_teacher_direct_matched_C | 22.599142 | 40.234375 | 20.703125 |
| quad | real_frozen_AB | 22.944851 | 41.015625 | 22.265625 |
| quad | real_frozen_AB_C | 22.776505 | 40.234375 | 21.093750 |
| quad | real_frozen_ensemble4 | 22.921010 | 41.015625 | 20.703125 |
| quad | randomA_frozen_AB | 22.936430 | 41.015625 | 22.265625 |
| quad | randomA_frozen_AB_C | 23.084334 | 41.406250 | 22.265625 |
| quad | randomA_frozen_ensemble4 | 22.580631 | 39.843750 | 22.265625 |
| quad | wrong_frozen_AB | 19.267589 | 33.593750 | 17.578125 |
| quad | wrong_frozen_AB_C | 15.019099 | 25.781250 | 13.671875 |
| quad | wrong_frozen_ensemble4 | 20.101629 | 33.984375 | 19.921875 |
| quad | real_mean_direct_all | 22.643588 | 40.625000 | 21.484375 |
| quad | real_mean_direct_matched | 22.815707 | 41.015625 | 21.875000 |
| quad | real_mean_direct_matched_C | 22.755362 | 40.625000 | 20.703125 |
| quad | real_native_direct_all | 22.944851 | 41.015625 | 22.265625 |
| quad | real_native_direct_matched | 22.962130 | 41.015625 | 22.265625 |
| quad | real_native_direct_matched_C | 22.776505 | 40.234375 | 21.093750 |
| quad | real_mean_keep_C | 22.404135 | 39.453125 | 21.484375 |
| quad | real_mean_keep_fusedABC | 23.015601 | 40.625000 | 22.656250 |
| quad | real_mean_keep_AB_C | 22.026444 | 38.671875 | 20.703125 |
| quad | real_native2_C | 22.057508 | 38.281250 | 21.484375 |
| quad | real_native2_fusedABC | 22.789486 | 40.234375 | 22.265625 |
| quad | real_native2_AB_C | 21.785411 | 37.890625 | 20.312500 |
| quad | real_native_keep_C | 22.307356 | 38.671875 | 21.093750 |
| quad | real_native_keep_fusedABC | 22.760541 | 40.234375 | 22.265625 |
| quad | real_native_keep_AB_C | 22.106190 | 38.671875 | 20.703125 |
| quad | randomA_mean_keep_C | 21.948393 | 39.453125 | 20.312500 |
| quad | randomA_mean_keep_fusedABC | 23.090765 | 40.625000 | 22.265625 |
| quad | randomA_mean_keep_AB_C | 23.353870 | 41.406250 | 22.265625 |
| quad | wrong_mean_keep_C | 10.707963 | 17.968750 | 8.593750 |
| quad | wrong_mean_keep_fusedABC | 14.929123 | 25.390625 | 13.671875 |
| quad | wrong_mean_keep_AB_C | 15.786526 | 28.125000 | 13.671875 |
| quad | teacher_shuffle_keep_C | 21.600251 | 37.109375 | 20.703125 |
| quad | teacher_shuffle_keep_fusedABC | 22.426550 | 39.453125 | 21.093750 |
| quad | teacher_shuffle_keep_AB_C | 21.348877 | 37.500000 | 19.140625 |
| quad | norm_old_direction_C | 22.456910 | 39.453125 | 21.484375 |
| quad | norm_old_direction_fusedABC | 23.155040 | 41.015625 | 22.265625 |
| quad | norm_old_direction_AB_C | 22.036632 | 38.671875 | 20.703125 |

## 锁定主方法与全部强对照

| 对照 | Δ pp | 改善/恶化/不变 |
|---|---:|---:|
| original_source | -0.227592 | 128/127/513 |
| prepared_frozen | +0.355139 | 32/26/710 |
| real_frozen_fusedABC | -0.119439 | 10/9/749 |
| real_frozen_AB | +0.074134 | 15/12/741 |
| real_frozen_AB_C | +0.079241 | 21/17/730 |
| real_frozen_ensemble4 | +0.034404 | 19/16/733 |
| real_mean_direct_matched_C | +0.132427 | 25/17/726 |
| real_native_direct_matched_C | +0.079241 | 21/17/730 |
| real_TTA_fusedABC | +0.008833 | 2/3/763 |
| norm_old_direction_fusedABC | +0.039662 | 4/4/760 |
| randomA_mean_keep_fusedABC | +0.020292 | 34/28/706 |
| wrong_mean_keep_fusedABC | +3.255053 | 107/27/634 |
| teacher_shuffle_keep_fusedABC | +0.216761 | 17/11/740 |

## 2×2机制结果

[
  {
    "endpoint": "fusedABC",
    "mean2": 16.5749581665426,
    "mean_keep": 16.583791331956178,
    "native2": 16.491111205959474,
    "native_keep": 16.490383827913487,
    "preservation_effect_at_mean_pp": 0.008833165413577149,
    "supervision_effect_without_keep_pp": -0.08384696058312713,
    "supervision_effect_with_keep_pp": -0.09340750404269116,
    "interaction_pp": -0.00956054345956403
  },
  {
    "endpoint": "C",
    "mean2": 16.31358406378749,
    "mean_keep": 16.355653752594765,
    "native2": 16.179669341824127,
    "native_keep": 16.378458694786456,
    "preservation_effect_at_mean_pp": 0.04206968880727402,
    "supervision_effect_without_keep_pp": -0.1339147219633645,
    "supervision_effect_with_keep_pp": 0.022804942191690714,
    "interaction_pp": 0.15671966415505523
  }
]

## 范数与方向

{
  "n": 258,
  "old_zero_direction": 0,
  "mean": {
    "old_parameter_L2": 1.6257636215062619,
    "protected_parameter_L2": 1.5412187970110105,
    "normmatched_parameter_L2": 1.5412187970125264,
    "scale_old_direction": 0.9584250438257123,
    "direction_cosine": 0.9755616002385382
  }
}

## 拟合

[
  {
    "fit": "real_mean_keep",
    "n": 258,
    "initial_error": 0.01281986053668921,
    "final_error": 0.006600520198024897,
    "relative_error": 0.520331107128766,
    "accepted_steps": 10.0,
    "attempted_steps": 15.213178294573643,
    "parameter_L2": 1.5412187972023867
  },
  {
    "fit": "real_native2",
    "n": 258,
    "initial_error": 0.019001102273591194,
    "final_error": 0.006358838988325303,
    "relative_error": 0.3631373064784499,
    "accepted_steps": 10.0,
    "attempted_steps": 15.069767441860465,
    "parameter_L2": 1.8122975824144436
  },
  {
    "fit": "real_native_keep",
    "n": 258,
    "initial_error": 0.01266740151572746,
    "final_error": 0.0049787580415026345,
    "relative_error": 0.41663607224444704,
    "accepted_steps": 10.0,
    "attempted_steps": 15.065891472868216,
    "parameter_L2": 1.732224124424264
  },
  {
    "fit": "randomA_mean_keep",
    "n": 258,
    "initial_error": 0.009012630713415563,
    "final_error": 0.004902449367311352,
    "relative_error": 0.5249243397421728,
    "accepted_steps": 10.0,
    "attempted_steps": 15.062015503875969,
    "parameter_L2": 1.5004459957221143
  },
  {
    "fit": "wrong_mean_keep",
    "n": 258,
    "initial_error": 0.08763297625053906,
    "final_error": 0.05252332464974172,
    "relative_error": 0.6142414871254861,
    "accepted_steps": 10.0,
    "attempted_steps": 16.073643410852714,
    "parameter_L2": 1.8418157622977838
  },
  {
    "fit": "teacher_shuffle_keep",
    "n": 258,
    "initial_error": 0.23616828217492336,
    "final_error": 0.1690479654659658,
    "relative_error": 0.7099188387839198,
    "accepted_steps": 10.0,
    "attempted_steps": 16.151162790697676,
    "parameter_L2": 2.320525688628147
  }
]

## 裁决

{
  "primary": "real_mean_keep_fusedABC",
  "mIoU_percent": 16.583791331956178,
  "gains_pp": {
    "original_source": -0.22759166676107512,
    "prepared_frozen": 0.3551391727674975,
    "real_frozen_fusedABC": -0.11943896100266826,
    "real_frozen_AB": 0.07413441523282316,
    "real_frozen_AB_C": 0.07924070867234789,
    "real_frozen_ensemble4": 0.03440375319368272,
    "real_mean_direct_matched_C": 0.1324274951297859,
    "real_native_direct_matched_C": 0.07924070867234789,
    "real_TTA_fusedABC": 0.008833165413577149,
    "norm_old_direction_fusedABC": 0.03966244716020384,
    "randomA_mean_keep_fusedABC": 0.020292123265235773,
    "wrong_mean_keep_fusedABC": 3.255052663677491,
    "teacher_shuffle_keep_fusedABC": 0.21676074044800941
  },
  "passes_development_gates": false,
  "robust_adoption": false,
  "decision": "do_not_adopt_fused_state_preservation_recipe",
  "n": 768,
  "primary_not_switched": true,
  "FINAL_access": false,
  "source_retraining": false,
  "external_experts": false
}

## 证据边界

55臂×768=42240读出，全部负例、UNKNOWN、保护媒体别名回退及C缺失回退均留在总体。258条执行拟合，178条实际使用更新输出；这些不能代表全体。目标分布和直接读出在评分前封存，GT只用于事后测量。相同场景/开发标签重复暴露，不能用最佳诊断臂代替预先主方法，也不能以某一平台个别病例决定门控。原source作为系统质量参照，FINAL从未读取。

## Good / failure 的总体来源

[
  {
    "scope": "all",
    "n": 768,
    "primary_better_than_fused": 10,
    "primary_worse_than_fused": 9,
    "primary_improvements_exceed_prepared": 5,
    "primary_improvements_only_repair_fusion": 5,
    "primary_harms_erase_fusion_improvement": 5,
    "old_TTA_harms_reverted_to_fusion_query": 1,
    "old_TTA_harms_still_present": 9,
    "old_TTA_benefits_preserved": 9
  },
  {
    "scope": "waymo_to_drone",
    "n": 128,
    "primary_better_than_fused": 2,
    "primary_worse_than_fused": 0,
    "primary_improvements_exceed_prepared": 1,
    "primary_improvements_only_repair_fusion": 1,
    "primary_harms_erase_fusion_improvement": 0,
    "old_TTA_harms_reverted_to_fusion_query": 0,
    "old_TTA_harms_still_present": 0,
    "old_TTA_benefits_preserved": 1
  },
  {
    "scope": "waymo_to_quad",
    "n": 128,
    "primary_better_than_fused": 1,
    "primary_worse_than_fused": 4,
    "primary_improvements_exceed_prepared": 0,
    "primary_improvements_only_repair_fusion": 1,
    "primary_harms_erase_fusion_improvement": 2,
    "old_TTA_harms_reverted_to_fusion_query": 1,
    "old_TTA_harms_still_present": 4,
    "old_TTA_benefits_preserved": 1
  },
  {
    "scope": "drone_to_waymo",
    "n": 128,
    "primary_better_than_fused": 1,
    "primary_worse_than_fused": 0,
    "primary_improvements_exceed_prepared": 0,
    "primary_improvements_only_repair_fusion": 1,
    "primary_harms_erase_fusion_improvement": 0,
    "old_TTA_harms_reverted_to_fusion_query": 0,
    "old_TTA_harms_still_present": 0,
    "old_TTA_benefits_preserved": 1
  },
  {
    "scope": "drone_to_quad",
    "n": 128,
    "primary_better_than_fused": 4,
    "primary_worse_than_fused": 5,
    "primary_improvements_exceed_prepared": 2,
    "primary_improvements_only_repair_fusion": 2,
    "primary_harms_erase_fusion_improvement": 3,
    "old_TTA_harms_reverted_to_fusion_query": 0,
    "old_TTA_harms_still_present": 5,
    "old_TTA_benefits_preserved": 4
  },
  {
    "scope": "quad_to_waymo",
    "n": 128,
    "primary_better_than_fused": 1,
    "primary_worse_than_fused": 0,
    "primary_improvements_exceed_prepared": 1,
    "primary_improvements_only_repair_fusion": 0,
    "primary_harms_erase_fusion_improvement": 0,
    "old_TTA_harms_reverted_to_fusion_query": 0,
    "old_TTA_harms_still_present": 0,
    "old_TTA_benefits_preserved": 1
  },
  {
    "scope": "quad_to_drone",
    "n": 128,
    "primary_better_than_fused": 1,
    "primary_worse_than_fused": 0,
    "primary_improvements_exceed_prepared": 1,
    "primary_improvements_only_repair_fusion": 0,
    "primary_harms_erase_fusion_improvement": 0,
    "old_TTA_harms_reverted_to_fusion_query": 0,
    "old_TTA_harms_still_present": 0,
    "old_TTA_benefits_preserved": 1
  }
]

案例索引从每方向正负尾部与中位数事后生成；没有新看图，不将IoU替代实例身份。R106已实际查看的4页仍只提供历史机制线索，本轮不声称新增视觉身份复核。
