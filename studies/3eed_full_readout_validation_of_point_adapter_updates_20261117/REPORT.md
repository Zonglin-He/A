# R126 — 完整多模态读出验收现成pointAdapter更新

本轮在读取任何R125任务分数之前锁定并生成。它不改R125梯度、模型、框或主项，不增加GPU训练，而是检验一个明确接口差异：R125的提案/回溯只看native点云—文字分布，实际输出还包含RGB late与竞争判别。能否用完整读出的一致性，从同一批现成更新中保留有益变化？

## 只有验收改变

对R125 source-zero A/B的h/U，用现有sourceA RGB late及竞争残差rA构造完整候选分布，冻结均值Tfull。对每个现成端点，用同一late/rA重算A/B完整匹配分布，目标为平均KL(Tfull||studentA/B)加原单位点特征保持。所有信息来自同一个已准备的3EED/R91；没有外部模型、CLIP缓存、规则颜色、GT或新source训练。

主项仅在{零步、原R125三步}之间选择；错配、随机和自由特征对照也各只有{零步、三步}两个选项。选择最小目标，且改善须超过max(1e−8,1e−6×|零步目标|)，否则回退零步。{0,1,3}正确提案早停只作为预设次项，不升级主项。没有新增步幅网格或根据本轮GT选强度。

完整教师仍是模型自己的预测，不是实例身份真值。RGB/竞争读出只参与本轮验收，**没有重新生成完整读出梯度**；现成native提案可能已经丢失有益方向或大步，本轮不能否定所有完整任务梯度。

所有A/B损失与状态选择确定后，才读取C相关最终读出；C不参与选择。选中输出逐值复制原R125实际写权重后的封存预测，不是额外改分或假写参数。部署时可使用所选原R125权重端点；零步不计为有效参数更新。C评价仍使用A256及A标注，不是新的独立场景/新帧GT。

## 必须保留的强直接读出

包含完整教师桥接logTfull+rF−rA；在相同最终late/rF下的完整AB概率平均；完整ABC概率平均。它们均无网络更新，后者直接使用相同三份观测。另保留原冻结均值、head均值、中位数、原生ABC、lateOnly等全部强基线，以及正确/错配/随机/自由变量相同验收。全部负比较在表内。

验收和新直接读出采用float64匹配MLP/归一化/logsumexp，以NumPy64独立复算Torch64生产；所选参数更新结果仍是原R125 GPU输出。另列相同float64零步F/C，避免将CPU读出精度变化误认作适应。

## 全体条件与结果

所有2304历史曝光输入、154保护别名、未知和无B/C在分母；原三source100/R91、A256、六层位置、完整文字和相机条件保持。别名不解码新特征，只沿用preparedA。R125原主项与所有27臂完整保留，R126新增15臂合计42臂96768任务读出值，非独立样本数。候选预测先封存、独立审计后再评分。

| 范围 | 方法 | mIoU% | Acc25% | Acc50% |
|---|---|---:|---:|---:|
| DEV | original_source | 16.811383 | 30.989583 | 14.973958 |
| DEV | prepared_A | 16.228652 | 29.947917 | 12.630208 |
| DEV | prepared_F | 16.703230 | 30.989583 | 13.151042 |
| DEV | native_F | 16.974318 | 31.250000 | 14.843750 |
| DEV | native_ABC | 16.734016 | 30.729167 | 14.843750 |
| DEV | lateOnly_F | 17.180150 | 31.901042 | 14.453125 |
| DEV | mean_context | 17.349002 | 31.901042 | 14.453125 |
| DEV | headmean_context | 17.176454 | 31.640625 | 14.583333 |
| DEV | robust_context | 17.383000 | 32.161458 | 14.583333 |
| DEV | weight_mean_context | 17.385491 | 32.031250 | 14.453125 |
| DEV | zero_C | 16.493471 | 30.729167 | 13.671875 |
| DEV | corr1_F | 17.263535 | 31.901042 | 14.192708 |
| DEV | corr1_C | 16.453460 | 30.468750 | 13.671875 |
| DEV | corr3_F | 17.183657 | 31.901042 | 14.192708 |
| DEV | corr3_C | 16.395135 | 30.338542 | 13.541667 |
| DEV | wrong1_F | 17.073532 | 31.640625 | 14.192708 |
| DEV | wrong1_C | 16.269577 | 30.468750 | 13.281250 |
| DEV | wrong3_F | 16.753181 | 30.859375 | 14.062500 |
| DEV | wrong3_C | 16.083455 | 30.208333 | 13.411458 |
| DEV | random1_F | 17.349002 | 31.901042 | 14.453125 |
| DEV | random1_C | 16.493471 | 30.729167 | 13.671875 |
| DEV | random3_F | 17.353267 | 32.031250 | 14.453125 |
| DEV | random3_C | 16.504216 | 30.729167 | 13.671875 |
| DEV | free3_F | 17.049592 | 31.119792 | 14.062500 |
| DEV | free3_C | 16.281418 | 29.817708 | 13.671875 |
| DEV | direct_F | 17.004480 | 31.640625 | 13.802083 |
| DEV | direct_C | 16.750924 | 31.250000 | 13.671875 |
| DEV | fullval_corr_F | 17.183657 | 31.901042 | 14.192708 |
| DEV | fullval_corr_C | 16.395135 | 30.338542 | 13.541667 |
| DEV | fullval_wrong_F | 17.209736 | 31.640625 | 14.322917 |
| DEV | fullval_wrong_C | 16.492428 | 30.729167 | 13.671875 |
| DEV | fullval_random_F | 17.353267 | 32.031250 | 14.453125 |
| DEV | fullval_random_C | 16.493471 | 30.729167 | 13.671875 |
| DEV | fullval_free_F | 17.050257 | 31.119792 | 14.062500 |
| DEV | fullval_free_C | 16.368549 | 29.947917 | 13.802083 |
| DEV | fullval_corr013_F | 17.183657 | 31.901042 | 14.192708 |
| DEV | fullval_corr013_C | 16.395135 | 30.338542 | 13.541667 |
| DEV | full_teacher_bridge | 17.219834 | 32.161458 | 14.322917 |
| DEV | full_direct_AB | 17.219834 | 32.161458 | 14.322917 |
| DEV | full_direct_ABC | 17.058556 | 31.250000 | 14.453125 |
| DEV | double_zero_F | 17.349002 | 31.901042 | 14.453125 |
| DEV | double_zero_C | 16.493471 | 30.729167 | 13.671875 |
| EXP | original_source | 16.144619 | 30.598958 | 12.434896 |
| EXP | prepared_A | 16.251067 | 30.598958 | 11.783854 |
| EXP | prepared_F | 16.238436 | 30.598958 | 12.044271 |
| EXP | native_F | 16.237263 | 30.859375 | 12.434896 |
| EXP | native_ABC | 16.321800 | 31.250000 | 12.434896 |
| EXP | lateOnly_F | 16.439770 | 31.184896 | 12.174479 |
| EXP | mean_context | 16.499566 | 31.119792 | 12.174479 |
| EXP | headmean_context | 16.533556 | 31.315104 | 12.369792 |
| EXP | robust_context | 16.435239 | 31.184896 | 12.239583 |
| EXP | weight_mean_context | 16.434424 | 30.989583 | 12.109375 |
| EXP | zero_C | 16.185584 | 30.794271 | 12.174479 |
| EXP | corr1_F | 16.506737 | 31.250000 | 12.109375 |
| EXP | corr1_C | 16.202944 | 30.794271 | 12.109375 |
| EXP | corr3_F | 16.435119 | 31.119792 | 12.044271 |
| EXP | corr3_C | 16.215379 | 30.794271 | 12.174479 |
| EXP | wrong1_F | 16.334969 | 30.859375 | 12.109375 |
| EXP | wrong1_C | 16.178852 | 30.924479 | 12.044271 |
| EXP | wrong3_F | 16.077595 | 30.533854 | 11.848958 |
| EXP | wrong3_C | 15.961698 | 30.403646 | 12.044271 |
| EXP | random1_F | 16.499566 | 31.119792 | 12.174479 |
| EXP | random1_C | 16.199495 | 30.794271 | 12.174479 |
| EXP | random3_F | 16.520437 | 31.184896 | 12.174479 |
| EXP | random3_C | 16.184336 | 30.729167 | 12.174479 |
| EXP | free3_F | 16.379069 | 31.054688 | 12.044271 |
| EXP | free3_C | 16.139837 | 30.729167 | 12.174479 |
| EXP | direct_F | 16.243921 | 30.729167 | 11.914062 |
| EXP | direct_C | 16.176727 | 30.664062 | 11.588542 |
| EXP | fullval_corr_F | 16.435119 | 31.119792 | 12.044271 |
| EXP | fullval_corr_C | 16.215379 | 30.794271 | 12.174479 |
| EXP | fullval_wrong_F | 16.488765 | 31.119792 | 12.174479 |
| EXP | fullval_wrong_C | 16.323117 | 31.054688 | 12.304688 |
| EXP | fullval_random_F | 16.499566 | 31.119792 | 12.174479 |
| EXP | fullval_random_C | 16.185584 | 30.794271 | 12.174479 |
| EXP | fullval_free_F | 16.395741 | 31.119792 | 12.044271 |
| EXP | fullval_free_C | 16.156514 | 30.794271 | 12.174479 |
| EXP | fullval_corr013_F | 16.435119 | 31.119792 | 12.044271 |
| EXP | fullval_corr013_C | 16.215379 | 30.794271 | 12.174479 |
| EXP | full_teacher_bridge | 16.327731 | 30.924479 | 11.914062 |
| EXP | full_direct_AB | 16.327731 | 30.924479 | 11.914062 |
| EXP | full_direct_ABC | 16.422767 | 31.184896 | 12.109375 |
| EXP | double_zero_F | 16.499566 | 31.119792 | 12.174479 |
| EXP | double_zero_C | 16.185584 | 30.794271 | 12.174479 |

## 目标平台

| 范围 | 目标 | 方法 | mIoU% | Acc25% | Acc50% |
|---|---|---|---:|---:|---:|
| DEV | waymo | original_source | 11.473631 | 23.437500 | 7.421875 |
| DEV | waymo | prepared_A | 12.615516 | 26.171875 | 5.078125 |
| DEV | waymo | mean_context | 12.247576 | 24.609375 | 7.812500 |
| DEV | waymo | corr3_F | 12.247576 | 24.609375 | 7.812500 |
| DEV | waymo | fullval_corr_F | 12.247576 | 24.609375 | 7.812500 |
| DEV | waymo | fullval_wrong_F | 12.247576 | 24.609375 | 7.812500 |
| DEV | waymo | fullval_random_F | 12.247576 | 24.609375 | 7.812500 |
| DEV | waymo | fullval_free_F | 12.242181 | 24.609375 | 7.812500 |
| DEV | waymo | fullval_corr013_F | 12.247576 | 24.609375 | 7.812500 |
| DEV | waymo | full_teacher_bridge | 12.000796 | 24.218750 | 7.812500 |
| DEV | waymo | full_direct_AB | 12.000796 | 24.218750 | 7.812500 |
| DEV | waymo | full_direct_ABC | 11.937716 | 23.828125 | 7.812500 |
| DEV | waymo | double_zero_F | 12.247576 | 24.609375 | 7.812500 |
| DEV | drone | original_source | 13.867693 | 24.609375 | 12.500000 |
| DEV | drone | prepared_A | 14.071678 | 24.609375 | 11.718750 |
| DEV | drone | mean_context | 14.805341 | 26.953125 | 11.718750 |
| DEV | drone | corr3_F | 14.932339 | 27.734375 | 11.718750 |
| DEV | drone | fullval_corr_F | 14.932339 | 27.734375 | 11.718750 |
| DEV | drone | fullval_wrong_F | 14.805341 | 26.953125 | 11.718750 |
| DEV | drone | fullval_random_F | 14.805341 | 26.953125 | 11.718750 |
| DEV | drone | fullval_free_F | 14.594248 | 26.562500 | 11.328125 |
| DEV | drone | fullval_corr013_F | 14.932339 | 27.734375 | 11.718750 |
| DEV | drone | full_teacher_bridge | 14.843298 | 27.343750 | 11.718750 |
| DEV | drone | full_direct_AB | 14.843298 | 27.343750 | 11.718750 |
| DEV | drone | full_direct_ABC | 14.389845 | 25.781250 | 11.328125 |
| DEV | drone | double_zero_F | 14.805341 | 26.953125 | 11.718750 |
| DEV | quad | original_source | 25.092825 | 44.921875 | 25.000000 |
| DEV | quad | prepared_A | 21.998762 | 39.062500 | 21.093750 |
| DEV | quad | mean_context | 24.994087 | 44.140625 | 23.828125 |
| DEV | quad | corr3_F | 24.371055 | 43.359375 | 23.046875 |
| DEV | quad | fullval_corr_F | 24.371055 | 43.359375 | 23.046875 |
| DEV | quad | fullval_wrong_F | 24.576291 | 43.359375 | 23.437500 |
| DEV | quad | fullval_random_F | 25.006882 | 44.531250 | 23.828125 |
| DEV | quad | fullval_free_F | 24.314343 | 42.187500 | 23.046875 |
| DEV | quad | fullval_corr013_F | 24.371055 | 43.359375 | 23.046875 |
| DEV | quad | full_teacher_bridge | 24.815408 | 44.921875 | 23.437500 |
| DEV | quad | full_direct_AB | 24.815408 | 44.921875 | 23.437500 |
| DEV | quad | full_direct_ABC | 24.848108 | 44.140625 | 24.218750 |
| DEV | quad | double_zero_F | 24.994087 | 44.140625 | 23.828125 |
| EXP | waymo | original_source | 13.450978 | 28.906250 | 6.835938 |
| EXP | waymo | prepared_A | 13.214120 | 28.125000 | 6.250000 |
| EXP | waymo | mean_context | 12.855988 | 27.343750 | 6.054688 |
| EXP | waymo | corr3_F | 12.744220 | 27.148438 | 6.054688 |
| EXP | waymo | fullval_corr_F | 12.744220 | 27.148438 | 6.054688 |
| EXP | waymo | fullval_wrong_F | 12.827214 | 27.343750 | 6.054688 |
| EXP | waymo | fullval_random_F | 12.855988 | 27.343750 | 6.054688 |
| EXP | waymo | fullval_free_F | 12.567778 | 27.148438 | 6.054688 |
| EXP | waymo | fullval_corr013_F | 12.744220 | 27.148438 | 6.054688 |
| EXP | waymo | full_teacher_bridge | 13.061895 | 27.929688 | 6.250000 |
| EXP | waymo | full_direct_AB | 13.061895 | 27.929688 | 6.250000 |
| EXP | waymo | full_direct_ABC | 13.024688 | 28.125000 | 6.054688 |
| EXP | waymo | double_zero_F | 12.855988 | 27.343750 | 6.054688 |
| EXP | drone | original_source | 11.525665 | 20.898438 | 9.375000 |
| EXP | drone | prepared_A | 12.009923 | 20.312500 | 9.179688 |
| EXP | drone | mean_context | 12.279426 | 22.265625 | 9.570312 |
| EXP | drone | corr3_F | 12.290231 | 22.265625 | 9.570312 |
| EXP | drone | fullval_corr_F | 12.290231 | 22.265625 | 9.570312 |
| EXP | drone | fullval_wrong_F | 12.246733 | 22.070312 | 9.570312 |
| EXP | drone | fullval_random_F | 12.279426 | 22.265625 | 9.570312 |
| EXP | drone | fullval_free_F | 12.339063 | 22.265625 | 9.570312 |
| EXP | drone | fullval_corr013_F | 12.290231 | 22.265625 | 9.570312 |
| EXP | drone | full_teacher_bridge | 12.073256 | 22.070312 | 9.375000 |
| EXP | drone | full_direct_AB | 12.073256 | 22.070312 | 9.375000 |
| EXP | drone | full_direct_ABC | 12.018980 | 21.679688 | 9.375000 |
| EXP | drone | double_zero_F | 12.279426 | 22.265625 | 9.570312 |
| EXP | quad | original_source | 23.457215 | 41.992188 | 21.093750 |
| EXP | quad | prepared_A | 23.529157 | 43.359375 | 19.921875 |
| EXP | quad | mean_context | 24.363284 | 43.750000 | 20.898438 |
| EXP | quad | corr3_F | 24.270907 | 43.945312 | 20.507812 |
| EXP | quad | fullval_corr_F | 24.270907 | 43.945312 | 20.507812 |
| EXP | quad | fullval_wrong_F | 24.392348 | 43.945312 | 20.898438 |
| EXP | quad | fullval_random_F | 24.363284 | 43.750000 | 20.898438 |
| EXP | quad | fullval_free_F | 24.280383 | 43.945312 | 20.507812 |
| EXP | quad | fullval_corr013_F | 24.270907 | 43.945312 | 20.507812 |
| EXP | quad | full_teacher_bridge | 23.848041 | 42.773438 | 20.117188 |
| EXP | quad | full_direct_AB | 23.848041 | 42.773438 | 20.117188 |
| EXP | quad | full_direct_ABC | 24.224633 | 43.750000 | 20.898438 |
| EXP | quad | double_zero_F | 24.363284 | 43.750000 | 20.898438 |

## 原场景频数后分层

| 范围 | 方法 | 最小支持覆盖 | 后分层mIoU% |
|---|---|---:|---:|
| DEV | original_source | 1.000000 | 12.186315 |
| DEV | prepared_A | 1.000000 | 11.764418 |
| DEV | prepared_F | 1.000000 | 11.980683 |
| DEV | native_F | 1.000000 | 12.237360 |
| DEV | native_ABC | 1.000000 | 11.834713 |
| DEV | lateOnly_F | 1.000000 | 12.479188 |
| DEV | mean_context | 1.000000 | 12.673432 |
| DEV | headmean_context | 1.000000 | 12.326706 |
| DEV | robust_context | 1.000000 | 12.646340 |
| DEV | weight_mean_context | 1.000000 | 12.698587 |
| DEV | zero_C | 1.000000 | 12.169366 |
| DEV | corr1_F | 1.000000 | 12.550540 |
| DEV | corr1_C | 1.000000 | 12.152160 |
| DEV | corr3_F | 1.000000 | 12.467380 |
| DEV | corr3_C | 1.000000 | 12.120706 |
| DEV | wrong1_F | 1.000000 | 12.444584 |
| DEV | wrong1_C | 1.000000 | 12.074738 |
| DEV | wrong3_F | 1.000000 | 12.097912 |
| DEV | wrong3_C | 1.000000 | 11.862898 |
| DEV | random1_F | 1.000000 | 12.673432 |
| DEV | random1_C | 1.000000 | 12.169366 |
| DEV | random3_F | 1.000000 | 12.682810 |
| DEV | random3_C | 1.000000 | 12.179548 |
| DEV | free3_F | 1.000000 | 12.334511 |
| DEV | free3_C | 1.000000 | 12.102099 |
| DEV | direct_F | 1.000000 | 12.454153 |
| DEV | direct_C | 1.000000 | 12.404671 |
| DEV | fullval_corr_F | 1.000000 | 12.467380 |
| DEV | fullval_corr_C | 1.000000 | 12.120706 |
| DEV | fullval_wrong_F | 1.000000 | 12.559903 |
| DEV | fullval_wrong_C | 1.000000 | 12.163598 |
| DEV | fullval_random_F | 1.000000 | 12.682810 |
| DEV | fullval_random_C | 1.000000 | 12.169366 |
| DEV | fullval_free_F | 1.000000 | 12.352216 |
| DEV | fullval_free_C | 1.000000 | 12.131676 |
| DEV | fullval_corr013_F | 1.000000 | 12.467380 |
| DEV | fullval_corr013_C | 1.000000 | 12.120706 |
| DEV | full_teacher_bridge | 1.000000 | 12.387076 |
| DEV | full_direct_AB | 1.000000 | 12.387076 |
| DEV | full_direct_ABC | 1.000000 | 12.350675 |
| DEV | double_zero_F | 1.000000 | 12.673432 |
| DEV | double_zero_C | 1.000000 | 12.169366 |
| EXP | original_source | 0.998149 | 12.339915 |
| EXP | prepared_A | 0.998149 | 13.127455 |
| EXP | prepared_F | 0.998149 | 12.899456 |
| EXP | native_F | 0.998149 | 12.670824 |
| EXP | native_ABC | 0.998149 | 12.817131 |
| EXP | lateOnly_F | 0.998149 | 12.880574 |
| EXP | mean_context | 0.998149 | 12.951703 |
| EXP | headmean_context | 0.998149 | 12.951158 |
| EXP | robust_context | 0.998149 | 12.772886 |
| EXP | weight_mean_context | 0.998149 | 12.817846 |
| EXP | zero_C | 0.998149 | 12.599336 |
| EXP | corr1_F | 0.998149 | 12.957582 |
| EXP | corr1_C | 0.998149 | 12.712740 |
| EXP | corr3_F | 0.998149 | 12.879481 |
| EXP | corr3_C | 0.998149 | 12.638783 |
| EXP | wrong1_F | 0.998149 | 12.757652 |
| EXP | wrong1_C | 0.998149 | 12.807323 |
| EXP | wrong3_F | 0.998149 | 12.599756 |
| EXP | wrong3_C | 0.998149 | 12.503689 |
| EXP | random1_F | 0.998149 | 12.951703 |
| EXP | random1_C | 0.998149 | 12.629566 |
| EXP | random3_F | 0.998149 | 12.958788 |
| EXP | random3_C | 0.998149 | 12.596626 |
| EXP | free3_F | 0.998149 | 12.780594 |
| EXP | free3_C | 0.998149 | 12.681485 |
| EXP | direct_F | 0.998149 | 12.659956 |
| EXP | direct_C | 0.998149 | 12.788927 |
| EXP | fullval_corr_F | 0.998149 | 12.879481 |
| EXP | fullval_corr_C | 0.998149 | 12.638783 |
| EXP | fullval_wrong_F | 0.998149 | 12.912602 |
| EXP | fullval_wrong_C | 0.998149 | 12.701684 |
| EXP | fullval_random_F | 0.998149 | 12.951703 |
| EXP | fullval_random_C | 0.998149 | 12.599336 |
| EXP | fullval_free_F | 0.998149 | 12.809309 |
| EXP | fullval_free_C | 0.998149 | 12.692040 |
| EXP | fullval_corr013_F | 0.998149 | 12.879481 |
| EXP | fullval_corr013_C | 0.998149 | 12.638783 |
| EXP | full_teacher_bridge | 0.998149 | 12.854354 |
| EXP | full_direct_AB | 0.998149 | 12.854354 |
| EXP | full_direct_ABC | 0.998149 | 12.936857 |
| EXP | double_zero_F | 0.998149 | 12.951703 |
| EXP | double_zero_C | 0.998149 | 12.599336 |

## DECISION

```json
{
  "primary": "fullval_corr_F",
  "cohorts": {
    "DEV": {
      "mIoU_percent": 17.183656701859316,
      "gains_pp": {
        "original_source": 0.37227370314206354,
        "prepared_A": 0.9550045426706362,
        "prepared_F": 0.4804264089004704,
        "native_F": 0.2093391901396231,
        "native_ABC": 0.44964100719654,
        "lateOnly_F": 0.003506404069909763,
        "mean_context": -0.16534495539436733,
        "headmean_context": 0.007202767299233415,
        "robust_context": -0.19934368735394514,
        "weight_mean_context": -0.20183429604251302,
        "corr3_F": 0.0,
        "direct_F": 0.17917623108110803,
        "full_teacher_bridge": -0.036177394687072706,
        "full_direct_AB": -0.036177394687072706,
        "full_direct_ABC": 0.12510052266765825,
        "double_zero_F": -0.16534495539436733,
        "fullval_wrong_F": -0.026079545945727034,
        "fullval_random_F": -0.16960989688652361,
        "fullval_free_F": 0.13339930382031184
      },
      "poststratified_mIoU_percent": 12.467380004185516,
      "poststratified_gains_pp": {
        "original_source": 0.28106533119092525,
        "prepared_A": 0.7029618199081504,
        "prepared_F": 0.4866973537290509,
        "native_F": 0.23001954458865015,
        "native_ABC": 0.6326667249443112,
        "lateOnly_F": -0.011808311748559674,
        "mean_context": -0.20605201100867987,
        "headmean_context": 0.14067404575681408,
        "robust_context": -0.17896015116996367,
        "weight_mean_context": -0.2312068336670201,
        "corr3_F": 0.0,
        "direct_F": 0.013227072181168253,
        "full_teacher_bridge": 0.08030393510443368,
        "full_direct_AB": 0.08030393510443368,
        "full_direct_ABC": 0.11670504515283575,
        "double_zero_F": -0.20605201100867987,
        "fullval_wrong_F": -0.0925229609141951,
        "fullval_random_F": -0.21542989405576485,
        "fullval_free_F": 0.1151641865710804
      },
      "passes_all_prelocked_controls": false,
      "unused_C_available_macro_gain_pp": -0.18184563140164337,
      "unused_C_gate": false
    },
    "EXP": {
      "mIoU_percent": 16.435119415607137,
      "gains_pp": {
        "original_source": 0.2905001554830733,
        "prepared_A": 0.18405283845471132,
        "prepared_F": 0.19668376206420035,
        "native_F": 0.19785601742481518,
        "native_ABC": 0.11331920323922162,
        "lateOnly_F": -0.004650508696045108,
        "mean_context": -0.06444645957989792,
        "headmean_context": -0.09843627636841035,
        "robust_context": -0.00011913867685819923,
        "weight_mean_context": 0.0006949176594019946,
        "corr3_F": 0.0,
        "direct_F": 0.191198503230531,
        "full_teacher_bridge": 0.10738889522491846,
        "full_direct_AB": 0.10738889522491846,
        "full_direct_ABC": 0.012352347767091487,
        "double_zero_F": -0.06444645957989792,
        "fullval_wrong_F": -0.0536457433444788,
        "fullval_random_F": -0.06444645957989792,
        "fullval_free_F": 0.03937829526550729
      },
      "poststratified_mIoU_percent": 12.879481104770235,
      "poststratified_gains_pp": {
        "original_source": 0.5395657094169639,
        "prepared_A": -0.24797435579725402,
        "prepared_F": -0.019974873297508466,
        "native_F": 0.20865726770408877,
        "native_ABC": 0.062349693894445224,
        "lateOnly_F": -0.0010925471349647609,
        "mean_context": -0.07222177064015334,
        "headmean_context": -0.07167714616479692,
        "robust_context": 0.10659529601856832,
        "weight_mean_context": 0.06163460712513391,
        "corr3_F": 0.0,
        "direct_F": 0.21952529728775438,
        "full_teacher_bridge": 0.025127208292106573,
        "full_direct_AB": 0.025127208292106573,
        "full_direct_ABC": -0.057375904402304556,
        "double_zero_F": -0.07222177064015334,
        "fullval_wrong_F": -0.03312113912384973,
        "fullval_random_F": -0.07222177064015334,
        "fullval_free_F": 0.07017222784819843
      },
      "passes_all_prelocked_controls": false,
      "unused_C_available_macro_gain_pp": -0.054295825275164224,
      "unused_C_gate": false
    }
  },
  "FINAL_access": false,
  "posthoc_primary_switch": false,
  "development_gate_pass": false,
  "TTA_robustly_adopted": false
}
```

## HELDOUT_C_TASK_ROWS

```json
[
  {
    "cohort": "DEV",
    "source": "drone",
    "target": "quad",
    "n": 62,
    "gain_pp": 0.16741116432578865,
    "better": 2,
    "worse": 3
  },
  {
    "cohort": "DEV",
    "source": "drone",
    "target": "waymo",
    "n": 17,
    "gain_pp": 0.08125417317613264,
    "better": 1,
    "worse": 0
  },
  {
    "cohort": "DEV",
    "source": "quad",
    "target": "drone",
    "n": 10,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0
  },
  {
    "cohort": "DEV",
    "source": "quad",
    "target": "waymo",
    "n": 17,
    "gain_pp": 0.48540869110279977,
    "better": 1,
    "worse": 0
  },
  {
    "cohort": "DEV",
    "source": "waymo",
    "target": "drone",
    "n": 10,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0
  },
  {
    "cohort": "DEV",
    "source": "waymo",
    "target": "quad",
    "n": 62,
    "gain_pp": -1.8251478170145812,
    "better": 1,
    "worse": 2
  },
  {
    "cohort": "EXP",
    "source": "drone",
    "target": "quad",
    "n": 109,
    "gain_pp": 0.6737924679498973,
    "better": 5,
    "worse": 3
  },
  {
    "cohort": "EXP",
    "source": "drone",
    "target": "waymo",
    "n": 40,
    "gain_pp": 0.015494983208688695,
    "better": 1,
    "worse": 0
  },
  {
    "cohort": "EXP",
    "source": "quad",
    "target": "drone",
    "n": 26,
    "gain_pp": 0.27622587352972183,
    "better": 1,
    "worse": 0
  },
  {
    "cohort": "EXP",
    "source": "quad",
    "target": "waymo",
    "n": 40,
    "gain_pp": -0.7705307931511775,
    "better": 0,
    "worse": 1
  },
  {
    "cohort": "EXP",
    "source": "waymo",
    "target": "drone",
    "n": 26,
    "gain_pp": -0.7604341903696776,
    "better": 1,
    "worse": 2
  },
  {
    "cohort": "EXP",
    "source": "waymo",
    "target": "quad",
    "n": 109,
    "gain_pp": 0.23967670718156184,
    "better": 1,
    "worse": 1
  }
]
```

## VALIDATOR_COMPONENT_SUMMARY

```json
[
  {
    "cohort": "DEV",
    "all_n": 768,
    "nonalias": 724,
    "categories": [
      {
        "postscore_GT_outcome_NOT_selector": "raw_update_helped",
        "n": 6,
        "accepted": 6,
        "rejected": 0,
        "retained_gain_or_harm_pp_full_denominator": 0.1177452981904203,
        "removed_gain_or_harm_pp_full_denominator": 0.0
      },
      {
        "postscore_GT_outcome_NOT_selector": "raw_update_harmed",
        "n": 10,
        "accepted": 10,
        "rejected": 0,
        "retained_gain_or_harm_pp_full_denominator": -0.2830902535847869,
        "removed_gain_or_harm_pp_full_denominator": 0.0
      },
      {
        "postscore_GT_outcome_NOT_selector": "raw_update_IoU_unchanged",
        "n": 708,
        "accepted": 237,
        "rejected": 471,
        "retained_gain_or_harm_pp_full_denominator": 0.0,
        "removed_gain_or_harm_pp_full_denominator": 0.0
      }
    ],
    "selection_corr": {
      "corr3": 253,
      "zero": 471
    },
    "selection_wrong": {
      "wrong3": 39,
      "zero": 685
    },
    "selection_random": {
      "random3": 101,
      "zero": 623
    },
    "selection_free": {
      "free3": 253,
      "zero": 471
    },
    "selection_corr013": {
      "corr1": 6,
      "corr3": 247,
      "zero": 471
    }
  },
  {
    "cohort": "EXP",
    "all_n": 1536,
    "nonalias": 1426,
    "categories": [
      {
        "postscore_GT_outcome_NOT_selector": "raw_update_helped",
        "n": 10,
        "accepted": 10,
        "rejected": 0,
        "retained_gain_or_harm_pp_full_denominator": 0.09708387173275017,
        "removed_gain_or_harm_pp_full_denominator": 0.0
      },
      {
        "postscore_GT_outcome_NOT_selector": "raw_update_harmed",
        "n": 12,
        "accepted": 12,
        "rejected": 0,
        "retained_gain_or_harm_pp_full_denominator": -0.1615303313126472,
        "removed_gain_or_harm_pp_full_denominator": 0.0
      },
      {
        "postscore_GT_outcome_NOT_selector": "raw_update_IoU_unchanged",
        "n": 1404,
        "accepted": 495,
        "rejected": 909,
        "retained_gain_or_harm_pp_full_denominator": 0.0,
        "removed_gain_or_harm_pp_full_denominator": 0.0
      }
    ],
    "selection_corr": {
      "corr3": 517,
      "zero": 909
    },
    "selection_wrong": {
      "wrong3": 80,
      "zero": 1346
    },
    "selection_random": {
      "random3": 187,
      "zero": 1239
    },
    "selection_free": {
      "free3": 518,
      "zero": 908
    },
    "selection_corr013": {
      "corr1": 12,
      "corr3": 509,
      "zero": 905
    }
  }
]
```

## CONCENTRATION

```json
[
  {
    "cohort": "DEV",
    "method": "fullval_corr_F",
    "control": "mean_context",
    "n": 768,
    "gain_pp": -0.16534495539436658,
    "better": 6,
    "worse": 10,
    "same": 752,
    "query_changes": 24,
    "largest_positive_single_input_pp": 0.04186852011328779,
    "largest_negative_single_input_pp": -0.07486691872986709
  },
  {
    "cohort": "DEV",
    "method": "fullval_corr_F",
    "control": "corr3_F",
    "n": 768,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0,
    "same": 768,
    "query_changes": 0,
    "largest_positive_single_input_pp": 0.0,
    "largest_negative_single_input_pp": 0.0
  },
  {
    "cohort": "DEV",
    "method": "fullval_corr_F",
    "control": "full_direct_AB",
    "n": 768,
    "gain_pp": -0.03617739468707651,
    "better": 16,
    "worse": 16,
    "same": 736,
    "query_changes": 61,
    "largest_positive_single_input_pp": 0.048454273699162144,
    "largest_negative_single_input_pp": -0.0893579070700335
  },
  {
    "cohort": "DEV",
    "method": "fullval_corr_F",
    "control": "full_direct_ABC",
    "n": 768,
    "gain_pp": 0.1251005226676577,
    "better": 21,
    "worse": 16,
    "same": 731,
    "query_changes": 56,
    "largest_positive_single_input_pp": 0.08265650351809108,
    "largest_negative_single_input_pp": -0.0893579070700335
  },
  {
    "cohort": "DEV",
    "method": "fullval_corr_F",
    "control": "full_teacher_bridge",
    "n": 768,
    "gain_pp": -0.03617739468707651,
    "better": 16,
    "worse": 16,
    "same": 736,
    "query_changes": 61,
    "largest_positive_single_input_pp": 0.048454273699162144,
    "largest_negative_single_input_pp": -0.0893579070700335
  },
  {
    "cohort": "DEV",
    "method": "fullval_corr_F",
    "control": "fullval_wrong_F",
    "n": 768,
    "gain_pp": -0.02607954594572675,
    "better": 6,
    "worse": 8,
    "same": 754,
    "query_changes": 21,
    "largest_positive_single_input_pp": 0.04186852011328779,
    "largest_negative_single_input_pp": -0.07265321711525753
  },
  {
    "cohort": "DEV",
    "method": "fullval_corr_F",
    "control": "fullval_random_F",
    "n": 768,
    "gain_pp": -0.1696098968865245,
    "better": 5,
    "worse": 10,
    "same": 753,
    "query_changes": 23,
    "largest_positive_single_input_pp": 0.04186852011328779,
    "largest_negative_single_input_pp": -0.07486691872986709
  },
  {
    "cohort": "DEV",
    "method": "fullval_corr_F",
    "control": "fullval_free_F",
    "n": 768,
    "gain_pp": 0.13339930382031134,
    "better": 14,
    "worse": 13,
    "same": 741,
    "query_changes": 43,
    "largest_positive_single_input_pp": 0.06864713716211572,
    "largest_negative_single_input_pp": -0.07486691872986709
  },
  {
    "cohort": "DEV",
    "method": "fullval_corr_F",
    "control": "double_zero_F",
    "n": 768,
    "gain_pp": -0.16534495539436658,
    "better": 6,
    "worse": 10,
    "same": 752,
    "query_changes": 24,
    "largest_positive_single_input_pp": 0.04186852011328779,
    "largest_negative_single_input_pp": -0.07486691872986709
  },
  {
    "cohort": "EXP",
    "method": "fullval_corr_F",
    "control": "mean_context",
    "n": 1536,
    "gain_pp": -0.06444645957989703,
    "better": 10,
    "worse": 12,
    "same": 1514,
    "query_changes": 39,
    "largest_positive_single_input_pp": 0.020871559304455223,
    "largest_negative_single_input_pp": -0.03803720941588019
  },
  {
    "cohort": "EXP",
    "method": "fullval_corr_F",
    "control": "corr3_F",
    "n": 1536,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0,
    "same": 1536,
    "query_changes": 0,
    "largest_positive_single_input_pp": 0.0,
    "largest_negative_single_input_pp": 0.0
  },
  {
    "cohort": "EXP",
    "method": "fullval_corr_F",
    "control": "full_direct_AB",
    "n": 1536,
    "gain_pp": 0.10738889522491796,
    "better": 37,
    "worse": 29,
    "same": 1470,
    "query_changes": 132,
    "largest_positive_single_input_pp": 0.046892743444216685,
    "largest_negative_single_input_pp": -0.041478933278917746
  },
  {
    "cohort": "EXP",
    "method": "fullval_corr_F",
    "control": "full_direct_ABC",
    "n": 1536,
    "gain_pp": 0.012352347767092924,
    "better": 37,
    "worse": 25,
    "same": 1474,
    "query_changes": 110,
    "largest_positive_single_input_pp": 0.04503449860016842,
    "largest_negative_single_input_pp": -0.041491082509298706
  },
  {
    "cohort": "EXP",
    "method": "fullval_corr_F",
    "control": "full_teacher_bridge",
    "n": 1536,
    "gain_pp": 0.10738889522491796,
    "better": 37,
    "worse": 29,
    "same": 1470,
    "query_changes": 131,
    "largest_positive_single_input_pp": 0.046892743444216685,
    "largest_negative_single_input_pp": -0.041478933278917746
  },
  {
    "cohort": "EXP",
    "method": "fullval_corr_F",
    "control": "fullval_wrong_F",
    "n": 1536,
    "gain_pp": -0.053645743344477456,
    "better": 13,
    "worse": 13,
    "same": 1510,
    "query_changes": 46,
    "largest_positive_single_input_pp": 0.020871559304455223,
    "largest_negative_single_input_pp": -0.03803720941588019
  },
  {
    "cohort": "EXP",
    "method": "fullval_corr_F",
    "control": "fullval_random_F",
    "n": 1536,
    "gain_pp": -0.06444645957989703,
    "better": 10,
    "worse": 12,
    "same": 1514,
    "query_changes": 39,
    "largest_positive_single_input_pp": 0.020871559304455223,
    "largest_negative_single_input_pp": -0.03803720941588019
  },
  {
    "cohort": "EXP",
    "method": "fullval_corr_F",
    "control": "fullval_free_F",
    "n": 1536,
    "gain_pp": 0.03937829526550585,
    "better": 22,
    "worse": 22,
    "same": 1492,
    "query_changes": 79,
    "largest_positive_single_input_pp": 0.051036878261435326,
    "largest_negative_single_input_pp": -0.041478933278917746
  },
  {
    "cohort": "EXP",
    "method": "fullval_corr_F",
    "control": "double_zero_F",
    "n": 1536,
    "gain_pp": -0.06444645957989703,
    "better": 10,
    "worse": 12,
    "same": 1514,
    "query_changes": 39,
    "largest_positive_single_input_pp": 0.020871559304455223,
    "largest_negative_single_input_pp": -0.03803720941588019
  }
]
```

## NATIVE_AUDIT

```json
{
  "state": "PASS_NUMPY64_FULL_TEACHER_CHOICES_READOUTS_AND_UNCHANGED_ACTUAL_UPDATES",
  "n": 2304,
  "aliases": 154,
  "choice_checks": 10750,
  "direct_readouts": 10750,
  "max_probability_error": 2.3869795029440866e-15,
  "max_objective_error": 3.774758283725532e-15,
  "double_vs_GPU_zero_Top1_changes": 0,
  "choice_counts": {
    "corr": {
      "corr3": 770,
      "zero": 1380
    },
    "wrong": {
      "zero": 2031,
      "wrong3": 119
    },
    "random": {
      "random3": 288,
      "zero": 1862
    },
    "free": {
      "free3": 771,
      "zero": 1379
    },
    "corr013": {
      "corr3": 756,
      "zero": 1376,
      "corr1": 18
    }
  },
  "no_new_gradients": true,
  "GT_used": false,
  "FINAL_access": false
}
```

## SCALAR_AUDIT

```json
{
  "state": "PASS_ALL_TASK_PLATFORM_SCENE_PAIRS_POSTSTRATIFICATION",
  "n": 2304,
  "arms": 42,
  "task_values": 96768,
  "new_task_values": 34560,
  "reused_task_values": 62208,
  "aliases": 154,
  "FINAL_access": false
}
```

## GEOMETRY_AUDIT

```json
{
  "n": 2304,
  "new_readouts": 34560,
  "reused_readouts": 62208,
  "geometry_max_error": 6.661338147750939e-16
}
```

## 裁决与边界

预设开发门未通过，停止这份完整读出验收配方，不事后升级0/1/3次项、平台特例或更多阈值。

组件表事后按真实IoU将原提案分成有益/有害/无变化，统计验收保留和拒绝；这些标签从不进入验收，也不自动证明物体身份。总体高于受损R125、却未超过冻结或同信息直接读出时，不能算TTA成立。

原场景频数仅来自既有全量清单，EXP缺少4个Waymo场景，只报告共同支持；同一物理输入两个incoming source不能当成独立场景。所有案例与负例保留，未经私有图像复核的身份保持UNKNOWN。没有FINAL、原始媒体、PointNet或外部专家访问。

本轮seconds是额外CPU多臂验证/读出开销，不包括生成R125更新及原RGB/context/PointNet缓存，不能包装成整个TTA端到端成本。原始媒体、原文、ID、坐标、特征、权重与梯度不公开。


## 后验：拟合、迁移与融合分开

这部分在主实验GT结果已曝光后提出，仅作已封存表征的机制诊断，不参与方法、阈值、状态选择或任何准入门。新增读出均先封存再计算其IoU；原主方法与失败裁决不变。F/C完整读出的逐条结果与原主评分完全一致，float64相对原GPU的F/C查询变化为0，独立审计通过。

不能将本轮失败一概归为信号完全没用：DEV全部样本的拟合A/B完整读出分别+0.188469/+0.320273pp，而F/C分别-.165345/-.098335pp。对于相同AB有效且真实C可用的178条，A/B完整读出+.441795/+1.307464pp，C却-.181846pp，存在实际的未拟合观测迁移缺口。

EXP相同可用C的350条上，F的native读出+.582120pp，加入冻结RGB late后-.002759pp，late+rA后-.123987pp，late+rF后-.342586pp。这是在完全相同零步/适应后h/U上仅更换读出的组件干预：RGB特征相加后归一化、竞争分数及其上下文提供者，均可能与适应后的native表示不相容，不能只怀疑动态残差数值漂移。DEV总体所有这些F读出仍下降，所以也不能把EXP局部读出提升升级为通用解法。

以下只支持区分已见A/B效果、C迁移和多模态接口，不能据此宣布因果身份真值或独立验证成功。若后续另开研究，应先检查适应与部署是否使用相容的输入聚合和上下文，而非直接继续增加原一致性步数。此次截止前不再由这些GT诊断启动新方法。

| 范围 | 子集 | 视图 | 读出 | 三步相对自身零步 pp |
|---|---|---|---|---:|
| DEV | all | A | native | +0.101434 |
| DEV | all | A | full | +0.188469 |
| DEV | all | B | native | +0.196336 |
| DEV | all | B | full | +0.320273 |
| DEV | all | F | native | -0.192742 |
| DEV | all | F | full | -0.165345 |
| DEV | all | C | native | -0.140234 |
| DEV | all | C | full | -0.098335 |
| DEV | same_AB_active_C_available | A | native | -0.860809 |
| DEV | same_AB_active_C_available | A | full | +0.441795 |
| DEV | same_AB_active_C_available | B | native | +0.701687 |
| DEV | same_AB_active_C_available | B | full | +1.307464 |
| DEV | same_AB_active_C_available | F | native | +0.020974 |
| DEV | same_AB_active_C_available | F | full | +0.685245 |
| DEV | same_AB_active_C_available | C | native | -0.482463 |
| DEV | same_AB_active_C_available | C | full | -0.181846 |
| EXP | all | A | native | +0.018450 |
| EXP | all | A | full | -0.158611 |
| EXP | all | B | native | -0.002848 |
| EXP | all | B | full | -0.157494 |
| EXP | all | F | native | +0.076281 |
| EXP | all | F | full | -0.064446 |
| EXP | all | C | native | -0.070183 |
| EXP | all | C | full | +0.029795 |
| EXP | same_AB_active_C_available | A | native | +0.049775 |
| EXP | same_AB_active_C_available | A | full | -0.130507 |
| EXP | same_AB_active_C_available | B | native | -0.203762 |
| EXP | same_AB_active_C_available | B | full | -0.615524 |
| EXP | same_AB_active_C_available | F | native | +0.582120 |
| EXP | same_AB_active_C_available | F | full | -0.342586 |
| EXP | same_AB_active_C_available | C | native | -0.375050 |
| EXP | same_AB_active_C_available | C | full | -0.054296 |

| 范围 | 子集 | 视图 | 组件读出 | 三步相对自身零步 pp |
|---|---|---|---|---:|
| DEV | all | F | native | -0.192742 |
| DEV | all | F | lateOnly | -0.207751 |
| DEV | all | F | fullA | -0.181686 |
| DEV | all | F | full | -0.165345 |
| DEV | all | C | native | -0.140234 |
| DEV | all | C | lateOnly | -0.080760 |
| DEV | all | C | fullA | -0.074361 |
| DEV | all | C | full | -0.098335 |
| DEV | same_AB_active_C_available | F | native | +0.020974 |
| DEV | same_AB_active_C_available | F | lateOnly | +0.206663 |
| DEV | same_AB_active_C_available | F | fullA | +0.562208 |
| DEV | same_AB_active_C_available | F | full | +0.685245 |
| DEV | same_AB_active_C_available | C | native | -0.482463 |
| DEV | same_AB_active_C_available | C | lateOnly | +0.410182 |
| DEV | same_AB_active_C_available | C | fullA | -0.167451 |
| DEV | same_AB_active_C_available | C | full | -0.181846 |
| EXP | all | F | native | +0.076281 |
| EXP | all | F | lateOnly | +0.130732 |
| EXP | all | F | fullA | +0.026656 |
| EXP | all | F | full | -0.064446 |
| EXP | all | C | native | -0.070183 |
| EXP | all | C | lateOnly | -0.020192 |
| EXP | all | C | fullA | +0.028534 |
| EXP | all | C | full | +0.029795 |
| EXP | same_AB_active_C_available | F | native | +0.582120 |
| EXP | same_AB_active_C_available | F | lateOnly | -0.002759 |
| EXP | same_AB_active_C_available | F | fullA | -0.123987 |
| EXP | same_AB_active_C_available | F | full | -0.342586 |
| EXP | same_AB_active_C_available | C | native | -0.375050 |
| EXP | same_AB_active_C_available | C | lateOnly | +0.041809 |
| EXP | same_AB_active_C_available | C | fullA | +0.011301 |
| EXP | same_AB_active_C_available | C | full | -0.054296 |
