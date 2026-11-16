# R120 — RGB物理更新通过实时语义上下文的受控TTA

本轮保持三个source100、已有源域R91双向模块、原A256/六层decoder位置/query身份不变。没有专家、新source训练或新增辅助损失。只更新既有RGB编码器三处GroupNorm的6个仿射张量，共224参数；原始点—文字grounding表示严格固定。

## 与R116和R118的区别

R116已经测试同一深度监督，但硬旁路earlyRGB后，其竞争残差r不随RGB更新变化，只有late能传递修改。本轮让更新后的RGB通过原source-compatible RGB→point→文字→竞争接口重算r，原生基础h/U仍冻结。对同一GN参数状态，比较实时r、冻结r、冻结late以及query打乱r。它检验一个实际计算路径，而非再搜索更新层或放大步长。

冻结基座使用R118的sourceA late/sourceF残差次项。它是在已经曝光的开发结果中被选出，不是R118已采用主项，也不是独立确认；R118同步A主项仍判失败，没有把其输出当伪标签拟合。现有GN更新直接来自实测稀疏深度。

主项一次原始梯度归一到source GN范数的1%，保留此前自然步、错配训练深度同范数、随机GN同范数控制；不回溯、不搜索步长。DEV复用R116已保存RAW方向，工程重算正确/错配梯度位等；EXP只做一次当前A梯度。75%有效深度格拟合、25%物理格留出，76001/2互补遮挡原样。所有方向固定后才读B/C和留出误差。深度头输入仅RGB，不能复制输入深度。

sourceA当前同步观测提供late，sourceF融合点上下文提供竞争残差；nativeA/B/C/F基础独立固定。最终F/C读出会使用融合上下文中的C，因此C没有参与梯度，但不能称不含C上下文的信息隔离验证，更不是新RGB视角。完整部署可能需要同一3EED的两个共享参数后缀，不声称缓存重放速度是端到端速度。

## 数据与技术核验

DEV768和EXP1536均已历史曝光，154保护别名在读取特征前prepared回退；全部负例、UNKNOWN、无变化和无C样本在总分分母。24真实工程输入验证旧sourceA/F隐藏态、文字、late、竞争残差位等，当前nativeA的h/U在实际GN写回后严格不变；作用域仅GN224。所有新预测先封存，再独立CPU读出/接口末端、真实GN写回范数和物理拟合/留出误差核验，最后评分。

## 全体与平台

| 数据 | 目标 | 方法 | mIoU% | Acc25% | Acc50% |
|---|---|---|---:|---:|---:|
| DEV | all_six_equal | original_source | 16.811383 | 30.989583 | 14.973958 |
| DEV | all_six_equal | prepared_A | 16.228652 | 29.947917 | 12.630208 |
| DEV | all_six_equal | prepared_F | 16.703230 | 30.989583 | 13.151042 |
| DEV | all_six_equal | native_F | 16.974318 | 31.250000 | 14.843750 |
| DEV | all_six_equal | repaired_F | 17.265658 | 31.901042 | 14.322917 |
| DEV | all_six_equal | lateOnly_F | 17.180150 | 31.901042 | 14.453125 |
| DEV | all_six_equal | context_zero_F | 17.349002 | 31.901042 | 14.453125 |
| DEV | all_six_equal | native_ABC | 16.734016 | 30.729167 | 14.843750 |
| DEV | all_six_equal | context_zero_noR_F | 17.228298 | 32.031250 | 14.453125 |
| DEV | all_six_equal | context_zero_C | 16.493471 | 30.729167 | 13.671875 |
| DEV | all_six_equal | context_zero_ABC | 17.058556 | 31.250000 | 14.453125 |
| DEV | all_six_equal | context_zero_rshuffle_F | 17.244811 | 31.901042 | 14.322917 |
| DEV | all_six_equal | depth_trust_live_F | 17.313490 | 31.770833 | 14.453125 |
| DEV | all_six_equal | depth_trust_frozenR_F | 17.313490 | 31.770833 | 14.453125 |
| DEV | all_six_equal | depth_trust_frozenLate_F | 17.313490 | 31.770833 | 14.453125 |
| DEV | all_six_equal | depth_trust_shuffleR_F | 17.244811 | 31.901042 | 14.322917 |
| DEV | all_six_equal | depth_trust_live_C | 16.493471 | 30.729167 | 13.671875 |
| DEV | all_six_equal | depth_trust_live_ABC | 17.058556 | 31.250000 | 14.453125 |
| DEV | all_six_equal | depth_natural_live_F | 17.313490 | 31.770833 | 14.453125 |
| DEV | all_six_equal | depth_natural_frozenR_F | 17.313490 | 31.770833 | 14.453125 |
| DEV | all_six_equal | depth_natural_frozenLate_F | 17.349002 | 31.901042 | 14.453125 |
| DEV | all_six_equal | depth_natural_shuffleR_F | 17.244811 | 31.901042 | 14.322917 |
| DEV | all_six_equal | depth_natural_live_C | 16.493471 | 30.729167 | 13.671875 |
| DEV | all_six_equal | depth_natural_live_ABC | 17.058556 | 31.250000 | 14.453125 |
| DEV | all_six_equal | wrong_depth_trust_live_F | 17.313490 | 31.770833 | 14.453125 |
| DEV | all_six_equal | wrong_depth_trust_frozenR_F | 17.313490 | 31.770833 | 14.453125 |
| DEV | all_six_equal | wrong_depth_trust_frozenLate_F | 17.313490 | 31.770833 | 14.453125 |
| DEV | all_six_equal | wrong_depth_trust_shuffleR_F | 17.244811 | 31.901042 | 14.322917 |
| DEV | all_six_equal | wrong_depth_trust_live_C | 16.493471 | 30.729167 | 13.671875 |
| DEV | all_six_equal | wrong_depth_trust_live_ABC | 17.058556 | 31.250000 | 14.453125 |
| DEV | all_six_equal | random_GN_trust_live_F | 17.313490 | 31.770833 | 14.453125 |
| DEV | all_six_equal | random_GN_trust_frozenR_F | 17.313490 | 31.770833 | 14.453125 |
| DEV | all_six_equal | random_GN_trust_frozenLate_F | 17.349002 | 31.901042 | 14.453125 |
| DEV | all_six_equal | random_GN_trust_shuffleR_F | 17.244811 | 31.901042 | 14.322917 |
| DEV | all_six_equal | random_GN_trust_live_C | 16.493471 | 30.729167 | 13.671875 |
| DEV | all_six_equal | random_GN_trust_live_ABC | 17.058556 | 31.250000 | 14.453125 |
| DEV | waymo | original_source | 11.473631 | 23.437500 | 7.421875 |
| DEV | waymo | prepared_A | 12.615516 | 26.171875 | 5.078125 |
| DEV | waymo | prepared_F | 12.569007 | 26.171875 | 5.078125 |
| DEV | waymo | native_F | 11.535583 | 23.437500 | 7.421875 |
| DEV | waymo | repaired_F | 12.304763 | 25.390625 | 7.812500 |
| DEV | waymo | lateOnly_F | 12.181059 | 25.390625 | 7.421875 |
| DEV | waymo | context_zero_F | 12.247576 | 24.609375 | 7.812500 |
| DEV | waymo | native_ABC | 11.215742 | 22.656250 | 7.421875 |
| DEV | waymo | context_zero_ABC | 11.937716 | 23.828125 | 7.812500 |
| DEV | waymo | depth_trust_live_F | 12.247576 | 24.609375 | 7.812500 |
| DEV | waymo | depth_trust_frozenR_F | 12.247576 | 24.609375 | 7.812500 |
| DEV | waymo | depth_trust_frozenLate_F | 12.247576 | 24.609375 | 7.812500 |
| DEV | waymo | depth_trust_shuffleR_F | 12.239533 | 25.000000 | 7.812500 |
| DEV | waymo | depth_natural_live_F | 12.247576 | 24.609375 | 7.812500 |
| DEV | waymo | wrong_depth_trust_live_F | 12.247576 | 24.609375 | 7.812500 |
| DEV | waymo | random_GN_trust_live_F | 12.247576 | 24.609375 | 7.812500 |
| DEV | drone | original_source | 13.867693 | 24.609375 | 12.500000 |
| DEV | drone | prepared_A | 14.071678 | 24.609375 | 11.718750 |
| DEV | drone | prepared_F | 14.029708 | 25.000000 | 11.718750 |
| DEV | drone | native_F | 14.710633 | 26.171875 | 12.890625 |
| DEV | drone | repaired_F | 14.503438 | 26.171875 | 11.718750 |
| DEV | drone | lateOnly_F | 14.345223 | 25.781250 | 11.718750 |
| DEV | drone | context_zero_F | 14.805341 | 26.953125 | 11.718750 |
| DEV | drone | native_ABC | 14.188880 | 24.609375 | 12.500000 |
| DEV | drone | context_zero_ABC | 14.389845 | 25.781250 | 11.328125 |
| DEV | drone | depth_trust_live_F | 14.698808 | 26.562500 | 11.718750 |
| DEV | drone | depth_trust_frozenR_F | 14.698808 | 26.562500 | 11.718750 |
| DEV | drone | depth_trust_frozenLate_F | 14.698808 | 26.562500 | 11.718750 |
| DEV | drone | depth_trust_shuffleR_F | 14.462357 | 26.562500 | 11.328125 |
| DEV | drone | depth_natural_live_F | 14.698808 | 26.562500 | 11.718750 |
| DEV | drone | wrong_depth_trust_live_F | 14.698808 | 26.562500 | 11.718750 |
| DEV | drone | random_GN_trust_live_F | 14.698808 | 26.562500 | 11.718750 |
| DEV | quad | original_source | 25.092825 | 44.921875 | 25.000000 |
| DEV | quad | prepared_A | 21.998762 | 39.062500 | 21.093750 |
| DEV | quad | prepared_F | 23.510976 | 41.796875 | 22.656250 |
| DEV | quad | native_F | 24.676737 | 44.140625 | 24.218750 |
| DEV | quad | repaired_F | 24.988774 | 44.140625 | 23.437500 |
| DEV | quad | lateOnly_F | 25.014169 | 44.531250 | 24.218750 |
| DEV | quad | context_zero_F | 24.994087 | 44.140625 | 23.828125 |
| DEV | quad | native_ABC | 24.797425 | 44.921875 | 24.609375 |
| DEV | quad | context_zero_ABC | 24.848108 | 44.140625 | 24.218750 |
| DEV | quad | depth_trust_live_F | 24.994087 | 44.140625 | 23.828125 |
| DEV | quad | depth_trust_frozenR_F | 24.994087 | 44.140625 | 23.828125 |
| DEV | quad | depth_trust_frozenLate_F | 24.994087 | 44.140625 | 23.828125 |
| DEV | quad | depth_trust_shuffleR_F | 25.032543 | 44.140625 | 23.828125 |
| DEV | quad | depth_natural_live_F | 24.994087 | 44.140625 | 23.828125 |
| DEV | quad | wrong_depth_trust_live_F | 24.994087 | 44.140625 | 23.828125 |
| DEV | quad | random_GN_trust_live_F | 24.994087 | 44.140625 | 23.828125 |
| EXP | all_six_equal | original_source | 16.144619 | 30.598958 | 12.434896 |
| EXP | all_six_equal | prepared_A | 16.251067 | 30.598958 | 11.783854 |
| EXP | all_six_equal | prepared_F | 16.238436 | 30.598958 | 12.044271 |
| EXP | all_six_equal | native_F | 16.237263 | 30.859375 | 12.434896 |
| EXP | all_six_equal | repaired_F | 16.314985 | 30.924479 | 12.109375 |
| EXP | all_six_equal | lateOnly_F | 16.439770 | 31.184896 | 12.174479 |
| EXP | all_six_equal | context_zero_F | 16.499566 | 31.119792 | 12.174479 |
| EXP | all_six_equal | native_ABC | 16.321800 | 31.250000 | 12.434896 |
| EXP | all_six_equal | context_zero_noR_F | 16.423666 | 30.989583 | 12.239583 |
| EXP | all_six_equal | context_zero_C | 16.185584 | 30.794271 | 12.174479 |
| EXP | all_six_equal | context_zero_ABC | 16.422767 | 31.184896 | 12.109375 |
| EXP | all_six_equal | context_zero_rshuffle_F | 16.433578 | 31.184896 | 12.109375 |
| EXP | all_six_equal | depth_trust_live_F | 16.499566 | 31.119792 | 12.174479 |
| EXP | all_six_equal | depth_trust_frozenR_F | 16.499566 | 31.119792 | 12.174479 |
| EXP | all_six_equal | depth_trust_frozenLate_F | 16.499566 | 31.119792 | 12.174479 |
| EXP | all_six_equal | depth_trust_shuffleR_F | 16.433578 | 31.184896 | 12.109375 |
| EXP | all_six_equal | depth_trust_live_C | 16.185584 | 30.794271 | 12.174479 |
| EXP | all_six_equal | depth_trust_live_ABC | 16.422767 | 31.184896 | 12.109375 |
| EXP | all_six_equal | depth_natural_live_F | 16.493879 | 31.119792 | 12.174479 |
| EXP | all_six_equal | depth_natural_frozenR_F | 16.493879 | 31.119792 | 12.174479 |
| EXP | all_six_equal | depth_natural_frozenLate_F | 16.504605 | 31.119792 | 12.174479 |
| EXP | all_six_equal | depth_natural_shuffleR_F | 16.471917 | 31.250000 | 12.174479 |
| EXP | all_six_equal | depth_natural_live_C | 16.184843 | 30.729167 | 12.174479 |
| EXP | all_six_equal | depth_natural_live_ABC | 16.406026 | 31.119792 | 12.109375 |
| EXP | all_six_equal | wrong_depth_trust_live_F | 16.499566 | 31.119792 | 12.174479 |
| EXP | all_six_equal | wrong_depth_trust_frozenR_F | 16.499566 | 31.119792 | 12.174479 |
| EXP | all_six_equal | wrong_depth_trust_frozenLate_F | 16.499566 | 31.119792 | 12.174479 |
| EXP | all_six_equal | wrong_depth_trust_shuffleR_F | 16.471917 | 31.250000 | 12.174479 |
| EXP | all_six_equal | wrong_depth_trust_live_C | 16.185584 | 30.794271 | 12.174479 |
| EXP | all_six_equal | wrong_depth_trust_live_ABC | 16.422767 | 31.184896 | 12.109375 |
| EXP | all_six_equal | random_GN_trust_live_F | 16.499566 | 31.119792 | 12.174479 |
| EXP | all_six_equal | random_GN_trust_frozenR_F | 16.499566 | 31.119792 | 12.174479 |
| EXP | all_six_equal | random_GN_trust_frozenLate_F | 16.499566 | 31.119792 | 12.174479 |
| EXP | all_six_equal | random_GN_trust_shuffleR_F | 16.433578 | 31.184896 | 12.109375 |
| EXP | all_six_equal | random_GN_trust_live_C | 16.185584 | 30.794271 | 12.174479 |
| EXP | all_six_equal | random_GN_trust_live_ABC | 16.422767 | 31.184896 | 12.109375 |
| EXP | waymo | original_source | 13.450978 | 28.906250 | 6.835938 |
| EXP | waymo | prepared_A | 13.214120 | 28.125000 | 6.250000 |
| EXP | waymo | prepared_F | 13.107853 | 27.734375 | 5.859375 |
| EXP | waymo | native_F | 13.085533 | 27.929688 | 6.445312 |
| EXP | waymo | repaired_F | 12.616854 | 27.148438 | 6.054688 |
| EXP | waymo | lateOnly_F | 12.839589 | 27.539062 | 6.054688 |
| EXP | waymo | context_zero_F | 12.855988 | 27.343750 | 6.054688 |
| EXP | waymo | native_ABC | 13.399290 | 28.710938 | 6.640625 |
| EXP | waymo | context_zero_ABC | 13.024688 | 28.125000 | 6.054688 |
| EXP | waymo | depth_trust_live_F | 12.855988 | 27.343750 | 6.054688 |
| EXP | waymo | depth_trust_frozenR_F | 12.855988 | 27.343750 | 6.054688 |
| EXP | waymo | depth_trust_frozenLate_F | 12.855988 | 27.343750 | 6.054688 |
| EXP | waymo | depth_trust_shuffleR_F | 13.136130 | 28.125000 | 6.250000 |
| EXP | waymo | depth_natural_live_F | 12.855988 | 27.343750 | 6.054688 |
| EXP | waymo | wrong_depth_trust_live_F | 12.855988 | 27.343750 | 6.054688 |
| EXP | waymo | random_GN_trust_live_F | 12.855988 | 27.343750 | 6.054688 |
| EXP | drone | original_source | 11.525665 | 20.898438 | 9.375000 |
| EXP | drone | prepared_A | 12.009923 | 20.312500 | 9.179688 |
| EXP | drone | prepared_F | 11.874628 | 20.312500 | 9.375000 |
| EXP | drone | native_F | 12.067931 | 22.265625 | 9.765625 |
| EXP | drone | repaired_F | 12.240490 | 22.460938 | 9.570312 |
| EXP | drone | lateOnly_F | 12.383083 | 22.656250 | 9.570312 |
| EXP | drone | context_zero_F | 12.279426 | 22.265625 | 9.570312 |
| EXP | drone | native_ABC | 11.933850 | 22.265625 | 9.570312 |
| EXP | drone | context_zero_ABC | 12.018980 | 21.679688 | 9.375000 |
| EXP | drone | depth_trust_live_F | 12.279426 | 22.265625 | 9.570312 |
| EXP | drone | depth_trust_frozenR_F | 12.279426 | 22.265625 | 9.570312 |
| EXP | drone | depth_trust_frozenLate_F | 12.279426 | 22.265625 | 9.570312 |
| EXP | drone | depth_trust_shuffleR_F | 12.442217 | 22.851562 | 9.570312 |
| EXP | drone | depth_natural_live_F | 12.279426 | 22.265625 | 9.570312 |
| EXP | drone | wrong_depth_trust_live_F | 12.279426 | 22.265625 | 9.570312 |
| EXP | drone | random_GN_trust_live_F | 12.279426 | 22.265625 | 9.570312 |
| EXP | quad | original_source | 23.457215 | 41.992188 | 21.093750 |
| EXP | quad | prepared_A | 23.529157 | 43.359375 | 19.921875 |
| EXP | quad | prepared_F | 23.732826 | 43.750000 | 20.898438 |
| EXP | quad | native_F | 23.558326 | 42.382812 | 21.093750 |
| EXP | quad | repaired_F | 24.087611 | 43.164062 | 20.703125 |
| EXP | quad | lateOnly_F | 24.096638 | 43.359375 | 20.898438 |
| EXP | quad | context_zero_F | 24.363284 | 43.750000 | 20.898438 |
| EXP | quad | native_ABC | 23.632260 | 42.773438 | 21.093750 |
| EXP | quad | context_zero_ABC | 24.224633 | 43.750000 | 20.898438 |
| EXP | quad | depth_trust_live_F | 24.363284 | 43.750000 | 20.898438 |
| EXP | quad | depth_trust_frozenR_F | 24.363284 | 43.750000 | 20.898438 |
| EXP | quad | depth_trust_frozenLate_F | 24.363284 | 43.750000 | 20.898438 |
| EXP | quad | depth_trust_shuffleR_F | 23.722387 | 42.578125 | 20.507812 |
| EXP | quad | depth_natural_live_F | 24.346224 | 43.750000 | 20.898438 |
| EXP | quad | wrong_depth_trust_live_F | 24.363284 | 43.750000 | 20.898438 |
| EXP | quad | random_GN_trust_live_F | 24.363284 | 43.750000 | 20.898438 |

## 场景比例敏感性

同R119，权重只取完整18620行的场景频数，各方法相同。EXP缺4个Waymo场景，按共同支持报告，不能冒充实际全量新推理。所有这些数据都已开发曝光；后分层不产生独立确认。

| 数据 | 方法 | 最小完整场景质量覆盖 | 后分层mIoU% |
|---|---|---:|---:|
| DEV | original_source | 1.000000 | 12.186315 |
| DEV | prepared_A | 1.000000 | 11.764418 |
| DEV | prepared_F | 1.000000 | 11.980683 |
| DEV | native_F | 1.000000 | 12.237360 |
| DEV | repaired_F | 1.000000 | 12.590408 |
| DEV | lateOnly_F | 1.000000 | 12.479188 |
| DEV | context_zero_F | 1.000000 | 12.673432 |
| DEV | native_ABC | 1.000000 | 11.834713 |
| DEV | context_zero_ABC | 1.000000 | 12.350675 |
| DEV | depth_trust_live_F | 1.000000 | 12.607014 |
| DEV | depth_trust_frozenR_F | 1.000000 | 12.607014 |
| DEV | depth_trust_frozenLate_F | 1.000000 | 12.607014 |
| DEV | depth_trust_shuffleR_F | 1.000000 | 12.718520 |
| DEV | depth_natural_live_F | 1.000000 | 12.607014 |
| DEV | wrong_depth_trust_live_F | 1.000000 | 12.607014 |
| DEV | random_GN_trust_live_F | 1.000000 | 12.607014 |
| EXP | original_source | 0.998149 | 12.339915 |
| EXP | prepared_A | 0.998149 | 13.127455 |
| EXP | prepared_F | 0.998149 | 12.899456 |
| EXP | native_F | 0.998149 | 12.670824 |
| EXP | repaired_F | 0.998149 | 12.706617 |
| EXP | lateOnly_F | 0.998149 | 12.880574 |
| EXP | context_zero_F | 0.998149 | 12.951703 |
| EXP | native_ABC | 0.998149 | 12.817131 |
| EXP | context_zero_ABC | 0.998149 | 12.936857 |
| EXP | depth_trust_live_F | 0.998149 | 12.951703 |
| EXP | depth_trust_frozenR_F | 0.998149 | 12.951703 |
| EXP | depth_trust_frozenLate_F | 0.998149 | 12.951703 |
| EXP | depth_trust_shuffleR_F | 0.998149 | 12.984208 |
| EXP | depth_natural_live_F | 0.998149 | 12.939346 |
| EXP | wrong_depth_trust_live_F | 0.998149 | 12.951703 |
| EXP | random_GN_trust_live_F | 0.998149 | 12.951703 |

## 预设裁决

```json
{
  "primary": "depth_trust_live_F",
  "cohorts": {
    "DEV": {
      "mIoU_percent": 17.31349044477975,
      "gains_pp": {
        "context_zero_F": -0.03551121247393496,
        "context_zero_ABC": 0.2549342655880906,
        "lateOnly_F": 0.13334014699034213,
        "native_ABC": 0.5794747501169724,
        "wrong_depth_trust_live_F": 0.0,
        "random_GN_trust_live_F": 0.0,
        "depth_natural_live_F": 0.0,
        "depth_trust_frozenR_F": 0.0,
        "depth_trust_shuffleR_F": 0.06867947057198265
      },
      "poststratified_mIoU_percent": 12.607013837708921,
      "poststratified_gains_pp": {
        "context_zero_F": -0.06641817748527323,
        "context_zero_ABC": 0.2563388786762406,
        "lateOnly_F": 0.12782552177484519,
        "native_ABC": 0.7723005584677161,
        "wrong_depth_trust_live_F": 0.0,
        "random_GN_trust_live_F": 0.0,
        "depth_natural_live_F": 0.0,
        "depth_trust_frozenR_F": 0.0,
        "depth_trust_shuffleR_F": -0.1115058814512917
      },
      "passes_all_prelocked_controls": false
    },
    "EXP": {
      "mIoU_percent": 16.499565875187034,
      "gains_pp": {
        "context_zero_F": 0.0,
        "context_zero_ABC": 0.0767988073469894,
        "lateOnly_F": 0.05979595088385281,
        "native_ABC": 0.17776566281911954,
        "wrong_depth_trust_live_F": 0.0,
        "random_GN_trust_live_F": 0.0,
        "depth_natural_live_F": 0.005686599300858575,
        "depth_trust_frozenR_F": 0.0,
        "depth_trust_shuffleR_F": 0.06598770124164233
      },
      "poststratified_mIoU_percent": 12.951702875410389,
      "poststratified_gains_pp": {
        "context_zero_F": 0.0,
        "context_zero_ABC": 0.014845866237848782,
        "lateOnly_F": 0.07112922350518858,
        "native_ABC": 0.13457146453459856,
        "wrong_depth_trust_live_F": 0.0,
        "random_GN_trust_live_F": 0.0,
        "depth_natural_live_F": 0.012356763145394467,
        "depth_trust_frozenR_F": 0.0,
        "depth_trust_shuffleR_F": -0.03250512904874725
      },
      "passes_all_prelocked_controls": false
    }
  },
  "FINAL_access": false,
  "posthoc_primary_switch": false,
  "development_gate_pass": false,
  "TTA_robustly_adopted": false
}
```

这里的gate仅是开发原型准入，即使通过也不自动替换生产方法或声称全量稳定TTA。没有将较有利的自然步/ABC读出事后改为主项，没有按平台或案例拼参数。若主项未优于同信息零步及错配/随机/冻结r控制，停止当前实时GN语义路径配方，不追加步长、层或损失网格。

## 物理恢复与语义路径分别报告

```json
[
  {
    "cohort": "DEV",
    "setting": "depth_trust",
    "n": 724,
    "zero_fit_mean": 0.23403052181952236,
    "updated_fit_mean": 0.2030060475033538,
    "zero_holdout_mean": 0.23363303444065278,
    "updated_holdout_mean": 0.2027495051090971,
    "holdout_improved": 705,
    "holdout_worsened": 19
  },
  {
    "cohort": "DEV",
    "setting": "depth_natural",
    "n": 724,
    "zero_fit_mean": 0.23403052181952236,
    "updated_fit_mean": 0.1752837127311289,
    "zero_holdout_mean": 0.23363303444065278,
    "updated_holdout_mean": 0.17516609914703757,
    "holdout_improved": 717,
    "holdout_worsened": 7
  },
  {
    "cohort": "DEV",
    "setting": "wrong_depth_trust",
    "n": 724,
    "zero_fit_mean": 0.23403052181952236,
    "updated_fit_mean": 0.20356215473820163,
    "zero_holdout_mean": 0.23363303444065278,
    "updated_holdout_mean": 0.203237635376786,
    "holdout_improved": 697,
    "holdout_worsened": 27
  },
  {
    "cohort": "DEV",
    "setting": "random_GN_trust",
    "n": 724,
    "zero_fit_mean": 0.23403052181952236,
    "updated_fit_mean": 0.23392265058992154,
    "zero_holdout_mean": 0.23363303444065278,
    "updated_holdout_mean": 0.23353584646890507,
    "holdout_improved": 368,
    "holdout_worsened": 356
  },
  {
    "cohort": "EXP",
    "setting": "depth_trust",
    "n": 1426,
    "zero_fit_mean": 0.22723296033884666,
    "updated_fit_mean": 0.19658424348673598,
    "zero_holdout_mean": 0.22743062631110825,
    "updated_holdout_mean": 0.1969298268413644,
    "holdout_improved": 1404,
    "holdout_worsened": 22
  },
  {
    "cohort": "EXP",
    "setting": "depth_natural",
    "n": 1426,
    "zero_fit_mean": 0.22723296033884666,
    "updated_fit_mean": 0.1697306267702939,
    "zero_holdout_mean": 0.22743062631110825,
    "updated_holdout_mean": 0.1700348771303934,
    "holdout_improved": 1420,
    "holdout_worsened": 6
  },
  {
    "cohort": "EXP",
    "setting": "wrong_depth_trust",
    "n": 1426,
    "zero_fit_mean": 0.22723296033884666,
    "updated_fit_mean": 0.1971039359090748,
    "zero_holdout_mean": 0.22743062631110825,
    "updated_holdout_mean": 0.19740275510418298,
    "holdout_improved": 1392,
    "holdout_worsened": 34
  },
  {
    "cohort": "EXP",
    "setting": "random_GN_trust",
    "n": 1426,
    "zero_fit_mean": 0.22723296033884666,
    "updated_fit_mean": 0.2270524443308777,
    "zero_holdout_mean": 0.22743062631110825,
    "updated_holdout_mean": 0.22727037486613716,
    "holdout_improved": 749,
    "holdout_worsened": 677
  }
]
[
  {
    "cohort": "DEV",
    "setting": "depth_trust",
    "n": 724,
    "residual_change_L2": 0.022580321624530764,
    "centered_residual_change_L2": 0.01069102404844398,
    "late_change_L2": 0.5833695893406713,
    "gradient_correct_wrong_cosine": 0.958412962288925
  },
  {
    "cohort": "DEV",
    "setting": "depth_natural",
    "n": 724,
    "residual_change_L2": 0.032080207043698274,
    "centered_residual_change_L2": 0.015898950850050265,
    "late_change_L2": 0.8411523136297011,
    "gradient_correct_wrong_cosine": 0.958412962288925
  },
  {
    "cohort": "DEV",
    "setting": "wrong_depth_trust",
    "n": 724,
    "residual_change_L2": 0.02287899498952296,
    "centered_residual_change_L2": 0.010820814005364249,
    "late_change_L2": 0.5837571916231177,
    "gradient_correct_wrong_cosine": 0.958412962288925
  },
  {
    "cohort": "DEV",
    "setting": "random_GN_trust",
    "n": 724,
    "residual_change_L2": 0.002212323509318521,
    "centered_residual_change_L2": 0.0011067000867757452,
    "late_change_L2": 0.06079858460750467,
    "gradient_correct_wrong_cosine": 0.958412962288925
  },
  {
    "cohort": "EXP",
    "setting": "depth_trust",
    "n": 1426,
    "residual_change_L2": 0.023173806087780996,
    "centered_residual_change_L2": 0.011041604711995624,
    "late_change_L2": 0.5808761850794805,
    "gradient_correct_wrong_cosine": 0.9610297730370027
  },
  {
    "cohort": "EXP",
    "setting": "depth_natural",
    "n": 1426,
    "residual_change_L2": 0.03198344937536006,
    "centered_residual_change_L2": 0.015513249439555944,
    "late_change_L2": 0.831286586697423,
    "gradient_correct_wrong_cosine": 0.9610297730370027
  },
  {
    "cohort": "EXP",
    "setting": "wrong_depth_trust",
    "n": 1426,
    "residual_change_L2": 0.023672237829754614,
    "centered_residual_change_L2": 0.011280199734929124,
    "late_change_L2": 0.5822681487120441,
    "gradient_correct_wrong_cosine": 0.9610297730370027
  },
  {
    "cohort": "EXP",
    "setting": "random_GN_trust",
    "n": 1426,
    "residual_change_L2": 0.002233871774300063,
    "centered_residual_change_L2": 0.0011375581220982969,
    "late_change_L2": 0.057971845054289144,
    "gradient_correct_wrong_cosine": 0.9610297730370027
  }
]
```

同一参数状态的实时r与冻结r比较控制了物理更新和观测；辅助误差一样不能代替最终选框收益。正确与错配深度保留相同目标值集合，故相似结果只否定空间对应的独立贡献，不能断言其所有全局深度统计都无效。没有把非零r变化或损失下降写成任务成功。

## 全部正负尾部

```json
[
  {
    "cohort": "DEV",
    "control": "context_zero_F",
    "n": 768,
    "gain_pp": -0.035511212473934566,
    "better": 0,
    "worse": 1,
    "same": 767,
    "largest_positive_single_input_pp": 0.0,
    "largest_negative_single_input_pp": -0.035511212473934566
  },
  {
    "cohort": "DEV",
    "control": "context_zero_ABC",
    "n": 768,
    "gain_pp": 0.2549342655880897,
    "better": 21,
    "worse": 12,
    "same": 735,
    "largest_positive_single_input_pp": 0.08265650351809108,
    "largest_negative_single_input_pp": -0.0893579070700335
  },
  {
    "cohort": "DEV",
    "control": "lateOnly_F",
    "n": 768,
    "gain_pp": 0.13334014699033728,
    "better": 10,
    "worse": 13,
    "same": 745,
    "largest_positive_single_input_pp": 0.08699899059864145,
    "largest_negative_single_input_pp": -0.03779742334418918
  },
  {
    "cohort": "DEV",
    "control": "depth_trust_frozenR_F",
    "n": 768,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0,
    "same": 768,
    "largest_positive_single_input_pp": 0.0,
    "largest_negative_single_input_pp": 0.0
  },
  {
    "cohort": "DEV",
    "control": "depth_trust_shuffleR_F",
    "n": 768,
    "gain_pp": 0.06867947057197822,
    "better": 35,
    "worse": 27,
    "same": 706,
    "largest_positive_single_input_pp": 0.044214014148823484,
    "largest_negative_single_input_pp": -0.05557292617527865
  },
  {
    "cohort": "DEV",
    "control": "wrong_depth_trust_live_F",
    "n": 768,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0,
    "same": 768,
    "largest_positive_single_input_pp": 0.0,
    "largest_negative_single_input_pp": 0.0
  },
  {
    "cohort": "DEV",
    "control": "random_GN_trust_live_F",
    "n": 768,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0,
    "same": 768,
    "largest_positive_single_input_pp": 0.0,
    "largest_negative_single_input_pp": 0.0
  },
  {
    "cohort": "EXP",
    "control": "context_zero_F",
    "n": 1536,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0,
    "same": 1536,
    "largest_positive_single_input_pp": 0.0,
    "largest_negative_single_input_pp": 0.0
  },
  {
    "cohort": "EXP",
    "control": "context_zero_ABC",
    "n": 1536,
    "gain_pp": 0.07679880734698998,
    "better": 35,
    "worse": 22,
    "same": 1479,
    "largest_positive_single_input_pp": 0.04503449860016842,
    "largest_negative_single_input_pp": -0.041491082509298706
  },
  {
    "cohort": "EXP",
    "control": "lateOnly_F",
    "n": 1536,
    "gain_pp": 0.05979595088385152,
    "better": 24,
    "worse": 27,
    "same": 1485,
    "largest_positive_single_input_pp": 0.04272420444360952,
    "largest_negative_single_input_pp": -0.04764303521813209
  },
  {
    "cohort": "EXP",
    "control": "depth_trust_frozenR_F",
    "n": 1536,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0,
    "same": 1536,
    "largest_positive_single_input_pp": 0.0,
    "largest_negative_single_input_pp": 0.0
  },
  {
    "cohort": "EXP",
    "control": "depth_trust_shuffleR_F",
    "n": 1536,
    "gain_pp": 0.0659877012416429,
    "better": 60,
    "worse": 60,
    "same": 1416,
    "largest_positive_single_input_pp": 0.04729911980131544,
    "largest_negative_single_input_pp": -0.04764303521813209
  },
  {
    "cohort": "EXP",
    "control": "wrong_depth_trust_live_F",
    "n": 1536,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0,
    "same": 1536,
    "largest_positive_single_input_pp": 0.0,
    "largest_negative_single_input_pp": 0.0
  },
  {
    "cohort": "EXP",
    "control": "random_GN_trust_live_F",
    "n": 1536,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0,
    "same": 1536,
    "largest_positive_single_input_pp": 0.0,
    "largest_negative_single_input_pp": 0.0
  }
]
```

所列good/failure按IoU差异定义，未独立看图的物体身份保持UNKNOWN；原source已正确的恢复和新身份纠错不同。所有匿名逐例、场景、参数写回与物理误差行公开，原图/文字/坐标/ID/特征/梯度/权重仅留本地。固定框在Acc50的上限限制沿用R119诊断，不因一轮更新改变。

## 文献与成本边界

文献适用条件与官方实现核对见LITERATURE.md。借鉴了有限参数适应和保留任务路径的条件，不是ProxyTTA复现。已有source-prepared模块不等于源域辅助更新已与grounding对齐；这项联系由本轮任务对照检验。每条seconds覆盖全部实验臂的缓存后缀与共享计算，不是单方法端到端延迟。

## 零步数值后端与决策核验

主项DEV17.313490%低于自身零步17.349002%，EXP16.499566%与零步完全相同；正确、错配与随机方向均无总体新收益。硬固定native基础后，GN的实时r路径已接通，但没有表现为额外任务收益。

```json
{
  "rows": [
    {
      "cohort": "DEV",
      "method": "depth_trust_live_F",
      "control": "context_zero_F",
      "n": 768,
      "query_changes": 1,
      "IoU_changes": 1,
      "reused_values_only": true
    },
    {
      "cohort": "DEV",
      "method": "depth_trust_live_F",
      "control": "depth_trust_frozenR_F",
      "n": 768,
      "query_changes": 0,
      "IoU_changes": 0,
      "reused_values_only": true
    },
    {
      "cohort": "DEV",
      "method": "depth_trust_live_F",
      "control": "depth_trust_frozenLate_F",
      "n": 768,
      "query_changes": 0,
      "IoU_changes": 0,
      "reused_values_only": true
    },
    {
      "cohort": "DEV",
      "method": "depth_trust_live_F",
      "control": "wrong_depth_trust_live_F",
      "n": 768,
      "query_changes": 0,
      "IoU_changes": 0,
      "reused_values_only": true
    },
    {
      "cohort": "DEV",
      "method": "depth_trust_live_F",
      "control": "random_GN_trust_live_F",
      "n": 768,
      "query_changes": 1,
      "IoU_changes": 0,
      "reused_values_only": true
    },
    {
      "cohort": "DEV",
      "method": "depth_trust_live_C",
      "control": "context_zero_C",
      "n": 768,
      "query_changes": 1,
      "IoU_changes": 0,
      "reused_values_only": true
    },
    {
      "cohort": "EXP",
      "method": "depth_trust_live_F",
      "control": "context_zero_F",
      "n": 1536,
      "query_changes": 2,
      "IoU_changes": 0,
      "reused_values_only": true
    },
    {
      "cohort": "EXP",
      "method": "depth_trust_live_F",
      "control": "depth_trust_frozenR_F",
      "n": 1536,
      "query_changes": 0,
      "IoU_changes": 0,
      "reused_values_only": true
    },
    {
      "cohort": "EXP",
      "method": "depth_trust_live_F",
      "control": "depth_trust_frozenLate_F",
      "n": 1536,
      "query_changes": 1,
      "IoU_changes": 0,
      "reused_values_only": true
    },
    {
      "cohort": "EXP",
      "method": "depth_trust_live_F",
      "control": "wrong_depth_trust_live_F",
      "n": 1536,
      "query_changes": 0,
      "IoU_changes": 0,
      "reused_values_only": true
    },
    {
      "cohort": "EXP",
      "method": "depth_trust_live_F",
      "control": "random_GN_trust_live_F",
      "n": 1536,
      "query_changes": 2,
      "IoU_changes": 0,
      "reused_values_only": true
    },
    {
      "cohort": "EXP",
      "method": "depth_trust_live_C",
      "control": "context_zero_C",
      "n": 1536,
      "query_changes": 3,
      "IoU_changes": 0,
      "reused_values_only": true
    }
  ],
  "new_GT_scoring": 0,
  "new_models": 0,
  "FINAL_access": false
}
{
  "postscore_diagnostic": true,
  "does_not_change_primary_or_predictions": true,
  "n": 2304,
  "aliases": 154,
  "cohorts": [
    {
      "cohort": "DEV",
      "n_nonalias": 724,
      "zero_backend_Top1_disagreements": 0,
      "updates_change_CPU_baseline": 1,
      "updates_change_actual_GPU_zero": 1,
      "max_zero_CPU_GPU_probability_error": 1.7097260975318385e-06,
      "max_zero_CPU_replay_error": 0.0,
      "all_disagreements_within_numeric_probability_bound": true,
      "new_task_GT_access": false
    },
    {
      "cohort": "EXP",
      "n_nonalias": 1426,
      "zero_backend_Top1_disagreements": 0,
      "updates_change_CPU_baseline": 2,
      "updates_change_actual_GPU_zero": 2,
      "max_zero_CPU_GPU_probability_error": 1.6791414015893125e-06,
      "max_zero_CPU_replay_error": 0.0,
      "all_disagreements_within_numeric_probability_bound": true,
      "new_task_GT_access": false
    }
  ],
  "FINAL_access": false
}
```

零步GPU头与原CPU基线在所有2150非别名输入的Top1一致，CPU逐值重放误差为0；因此这轮DEV单例退化不能归因于零步后端差异。主更新相对零步DEV换1个query且恶化1条，EXP换2个query但IoU不变。这个后核验不修改主结果、预测或更新规则。实时r与冻结r的总体分数相同不自动表示概率相同，精确query差异见上表。

物理留出仍显著改善：DEV705/724、EXP1404/1426，但不决定哪一个候选才是文字目标。保留自然步等全部次项，不继续加步、放大或切换归一化层来挽救。当前深度监督/实时GN语义路径配方不采纳；不据此否定一切无专家TTA。
