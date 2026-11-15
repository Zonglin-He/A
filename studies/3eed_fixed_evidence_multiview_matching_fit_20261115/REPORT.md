# R99 — 固定G，只改变拟合视图数

本轮在R98断点分析后，只把同一末端匹配投影的监督由完整F扩展为F及原有两个互补遮挡；G、原始A256、source准备权重、参数范围18,496、十次LM与每次六次阻尼试探均保留。没有引入外部专家、源域重训练或新目标身份标签。主方法预先固定joint3_FC，未拟合C在全部拟合完成后才读取。

## 已有全量证据的断点

以下不是新预测成绩，是对已审核R98全部18,620条的单因素重组。

| 因素 | 六方向等权Δ mIoU pp | 改善/恶化 |
|---|---:|---:|
| source_preparation | +0.441096 | 3235/2961 |
| mask_only | -0.041826 | 530/581 |
| add_C_readout_only | -0.023914 | 271/311 |
| G_only_F | +0.027056 | 796/834 |
| fit_execution_F | -0.013867 | 425/393 |
| weight_transfer_C | -0.014559 | 623/699 |
| weights_same_FC | +0.007930 | 650/666 |
| weights_vs_same_evidence | +0.016256 | 457/394 |
| complete_recipe | -0.015985 | 718/763 |

C单独降低冻结模型成绩；把C加入最终平均本身也有损失。不能将原完整配方的净损失全部称为参数更新失败。另一方面，F上G直接读出略好于参数拟合，说明单视图监督的利用仍有限。以上对比不证明mask中的RGB或LiDAR哪一项是根因。

## 新实验：六方向固定开发清单

每个目标平台按场景轮转和ID哈希固定128条，各入向source共用同一128条，合计768有向任务。Waymo覆盖93场景，Drone2、Quad3；这是已暴露开发数据，不能称独立确认。所有负例、UNKNOWN、无变化均保留。

| 方向 | 冻结 | 旧单视图LM FC | 同信息直接FC | 直接四视图 | 新三视图LM FC | 打乱监督FC |
|---|---:|---:|---:|---:|---:|---:|
| waymo→drone | 17.195182 | 17.372191 | 17.526382 | 17.504459 | 17.499466 | 17.173271 |
| waymo→quad | 29.615174 | 29.681293 | 29.681964 | 30.274718 | 29.681293 | 30.174939 |
| drone→waymo | 15.126688 | 15.659794 | 15.272546 | 15.402636 | 15.307633 | 14.684224 |
| drone→quad | 14.382351 | 14.548641 | 14.569998 | 14.249055 | 14.809542 | 14.674523 |
| quad→waymo | 10.104344 | 10.169355 | 10.239138 | 9.956681 | 10.827789 | 9.499123 |
| quad→drone | 10.948174 | 10.437315 | 10.161913 | 10.283599 | 10.977688 | 11.310180 |

## 平台、端点与全部负比较

| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |
|---|---|---:|---:|---:|
| all_six_equal | original_source | 16.811383 | 30.989583 | 14.973958 |
| all_six_equal | prepared_frozen | 16.228652 | 29.947917 | 12.630208 |
| all_six_equal | binding_direct | 16.272845 | 30.078125 | 12.760417 |
| all_six_equal | R94_Adam | 16.233257 | 29.947917 | 12.630208 |
| all_six_equal | frozen_C | 16.205204 | 30.078125 | 12.760417 |
| all_six_equal | frozen_FC | 16.301252 | 29.947917 | 12.760417 |
| all_six_equal | frozen_four | 16.249883 | 29.947917 | 12.890625 |
| all_six_equal | direct_C | 16.163092 | 29.687500 | 12.500000 |
| all_six_equal | direct_FC | 16.241990 | 29.817708 | 12.760417 |
| all_six_equal | direct_four | 16.278525 | 29.947917 | 12.890625 |
| all_six_equal | lm_F | 16.340603 | 30.078125 | 12.760417 |
| all_six_equal | lm_C | 16.407057 | 29.947917 | 13.020833 |
| all_six_equal | lm_FC | 16.311431 | 29.947917 | 12.760417 |
| all_six_equal | lm_shuffle_FC | 16.393525 | 30.338542 | 12.630208 |
| all_six_equal | joint3_F | 16.349062 | 30.208333 | 12.760417 |
| all_six_equal | joint3_C | 16.345003 | 30.078125 | 12.760417 |
| all_six_equal | joint3_FC | 16.517235 | 30.338542 | 13.020833 |
| all_six_equal | joint3_shuffle_F | 16.239719 | 30.078125 | 12.630208 |
| all_six_equal | joint3_shuffle_C | 16.432364 | 30.338542 | 12.760417 |
| all_six_equal | joint3_shuffle_FC | 16.252710 | 29.947917 | 12.500000 |
| waymo | original_source | 11.473631 | 23.437500 | 7.421875 |
| waymo | prepared_frozen | 12.615516 | 26.171875 | 5.078125 |
| waymo | binding_direct | 12.652470 | 25.781250 | 5.468750 |
| waymo | R94_Adam | 12.615516 | 26.171875 | 5.078125 |
| waymo | frozen_C | 12.703457 | 27.343750 | 5.468750 |
| waymo | frozen_FC | 12.440081 | 25.390625 | 5.078125 |
| waymo | frozen_four | 12.546748 | 25.781250 | 5.468750 |
| waymo | direct_C | 12.647289 | 25.781250 | 5.078125 |
| waymo | direct_FC | 12.755842 | 25.390625 | 5.468750 |
| waymo | direct_four | 12.679658 | 25.781250 | 5.468750 |
| waymo | lm_F | 12.843638 | 26.562500 | 5.468750 |
| waymo | lm_C | 13.292566 | 26.953125 | 6.250000 |
| waymo | lm_FC | 12.914574 | 26.562500 | 5.468750 |
| waymo | lm_shuffle_FC | 12.511596 | 26.562500 | 4.687500 |
| waymo | joint3_F | 12.735473 | 26.562500 | 5.468750 |
| waymo | joint3_C | 13.087062 | 26.953125 | 5.468750 |
| waymo | joint3_FC | 13.067711 | 26.953125 | 5.859375 |
| waymo | joint3_shuffle_F | 12.390714 | 26.171875 | 4.687500 |
| waymo | joint3_shuffle_C | 12.482943 | 26.171875 | 4.687500 |
| waymo | joint3_shuffle_FC | 12.091673 | 25.390625 | 4.296875 |
| drone | original_source | 13.867693 | 24.609375 | 12.500000 |
| drone | prepared_frozen | 14.071678 | 24.609375 | 11.718750 |
| drone | binding_direct | 14.253223 | 25.781250 | 12.109375 |
| drone | R94_Adam | 14.085494 | 24.609375 | 11.718750 |
| drone | frozen_C | 13.813703 | 24.609375 | 11.718750 |
| drone | frozen_FC | 14.272587 | 25.390625 | 12.109375 |
| drone | frozen_four | 13.894458 | 24.609375 | 11.718750 |
| drone | direct_C | 13.794244 | 24.609375 | 11.328125 |
| drone | direct_FC | 13.844147 | 25.000000 | 11.718750 |
| drone | direct_four | 13.894029 | 25.000000 | 11.718750 |
| drone | lm_F | 14.265667 | 25.000000 | 12.109375 |
| drone | lm_C | 13.874663 | 24.218750 | 11.718750 |
| drone | lm_FC | 13.904753 | 24.218750 | 11.718750 |
| drone | lm_shuffle_FC | 14.244249 | 25.000000 | 11.718750 |
| drone | joint3_F | 14.271562 | 25.000000 | 12.109375 |
| drone | joint3_C | 13.781419 | 24.218750 | 11.718750 |
| drone | joint3_FC | 14.238577 | 24.609375 | 12.109375 |
| drone | joint3_shuffle_F | 14.168917 | 25.000000 | 11.718750 |
| drone | joint3_shuffle_C | 14.403413 | 25.781250 | 12.109375 |
| drone | joint3_shuffle_FC | 14.241726 | 25.000000 | 11.718750 |
| quad | original_source | 25.092825 | 44.921875 | 25.000000 |
| quad | prepared_frozen | 21.998762 | 39.062500 | 21.093750 |
| quad | binding_direct | 21.912842 | 38.671875 | 20.703125 |
| quad | R94_Adam | 21.998762 | 39.062500 | 21.093750 |
| quad | frozen_C | 22.098451 | 38.281250 | 21.093750 |
| quad | frozen_FC | 22.191088 | 39.062500 | 21.093750 |
| quad | frozen_four | 22.308444 | 39.453125 | 21.484375 |
| quad | direct_C | 22.047743 | 38.671875 | 21.093750 |
| quad | direct_FC | 22.125981 | 39.062500 | 21.093750 |
| quad | direct_four | 22.261887 | 39.062500 | 21.484375 |
| quad | lm_F | 21.912506 | 38.671875 | 20.703125 |
| quad | lm_C | 22.053944 | 38.671875 | 21.093750 |
| quad | lm_FC | 22.114967 | 39.062500 | 21.093750 |
| quad | lm_shuffle_FC | 22.424731 | 39.453125 | 21.484375 |
| quad | joint3_F | 22.040152 | 39.062500 | 20.703125 |
| quad | joint3_C | 22.166527 | 39.062500 | 21.093750 |
| quad | joint3_FC | 22.245417 | 39.453125 | 21.093750 |
| quad | joint3_shuffle_F | 22.159525 | 39.062500 | 21.484375 |
| quad | joint3_shuffle_C | 22.410737 | 39.062500 | 21.484375 |
| quad | joint3_shuffle_FC | 22.424731 | 39.453125 | 21.484375 |

| 方向 | 方法 | 对照 | Δ pp | 改善/恶化/不变 |
|---|---|---|---:|---:|
| waymo→drone | joint3_FC | prepared_frozen | +0.304284 | 6/7/115 |
| waymo→drone | joint3_FC | lm_FC | +0.127275 | 2/0/126 |
| waymo→drone | joint3_FC | direct_FC | -0.026916 | 4/5/119 |
| waymo→drone | joint3_FC | direct_four | -0.004993 | 5/5/118 |
| waymo→drone | joint3_FC | joint3_shuffle_FC | +0.326194 | 10/12/106 |
| waymo→drone | joint3_C | lm_C | +0.058650 | 2/0/126 |
| waymo→drone | joint3_C | frozen_C | +0.282061 | 8/5/115 |
| waymo→drone | joint3_F | lm_F | +0.011790 | 1/0/127 |
| waymo→drone | joint3_F | binding_direct | -0.015126 | 4/4/120 |
| waymo→quad | joint3_FC | prepared_frozen | +0.066119 | 1/2/125 |
| waymo→quad | joint3_FC | lm_FC | +0.000000 | 0/0/128 |
| waymo→quad | joint3_FC | direct_FC | -0.000671 | 0/1/127 |
| waymo→quad | joint3_FC | direct_four | -0.593426 | 0/2/126 |
| waymo→quad | joint3_FC | joint3_shuffle_FC | -0.493646 | 3/4/121 |
| waymo→quad | joint3_C | lm_C | +0.000000 | 0/0/128 |
| waymo→quad | joint3_C | frozen_C | +0.221077 | 2/1/125 |
| waymo→quad | joint3_F | lm_F | +0.000000 | 0/0/128 |
| waymo→quad | joint3_F | binding_direct | -0.000671 | 0/1/127 |
| drone→waymo | joint3_FC | prepared_frozen | +0.180945 | 5/1/122 |
| drone→waymo | joint3_FC | lm_FC | -0.352161 | 0/2/126 |
| drone→waymo | joint3_FC | direct_FC | +0.035087 | 3/2/123 |
| drone→waymo | joint3_FC | direct_four | -0.095003 | 4/2/122 |
| drone→waymo | joint3_FC | joint3_shuffle_FC | +0.623409 | 8/1/119 |
| drone→waymo | joint3_C | lm_C | -0.334151 | 0/3/125 |
| drone→waymo | joint3_C | frozen_C | +0.550733 | 7/4/117 |
| drone→waymo | joint3_F | lm_F | -0.035925 | 0/2/126 |
| drone→waymo | joint3_F | binding_direct | +0.112564 | 3/5/120 |
| drone→quad | joint3_FC | prepared_frozen | +0.427191 | 3/4/121 |
| drone→quad | joint3_FC | lm_FC | +0.260901 | 2/0/126 |
| drone→quad | joint3_FC | direct_FC | +0.239544 | 2/0/126 |
| drone→quad | joint3_FC | direct_four | +0.560487 | 2/0/126 |
| drone→quad | joint3_FC | joint3_shuffle_FC | +0.135019 | 4/3/121 |
| drone→quad | joint3_C | lm_C | +0.225166 | 2/1/125 |
| drone→quad | joint3_C | frozen_C | -0.084927 | 2/3/123 |
| drone→quad | joint3_F | lm_F | +0.255292 | 2/1/125 |
| drone→quad | joint3_F | binding_direct | +0.255292 | 2/1/125 |
| quad→waymo | joint3_FC | prepared_frozen | +0.723444 | 5/2/121 |
| quad→waymo | joint3_FC | lm_FC | +0.658434 | 2/1/125 |
| quad→waymo | joint3_FC | direct_FC | +0.588650 | 2/1/125 |
| quad→waymo | joint3_FC | direct_four | +0.871107 | 3/2/123 |
| quad→waymo | joint3_FC | joint3_shuffle_FC | +1.328665 | 4/1/123 |
| quad→waymo | joint3_C | lm_C | -0.076856 | 0/1/127 |
| quad→waymo | joint3_C | frozen_C | +0.216476 | 2/2/124 |
| quad→waymo | joint3_F | lm_F | -0.180405 | 1/2/125 |
| quad→waymo | joint3_F | binding_direct | +0.053441 | 1/0/127 |
| quad→drone | joint3_FC | prepared_frozen | +0.029514 | 2/3/123 |
| quad→drone | joint3_FC | lm_FC | +0.540373 | 1/0/127 |
| quad→drone | joint3_FC | direct_FC | +0.815775 | 5/1/122 |
| quad→drone | joint3_FC | direct_four | +0.694089 | 5/2/121 |
| quad→drone | joint3_FC | joint3_shuffle_FC | -0.332492 | 4/5/119 |
| quad→drone | joint3_C | lm_C | -0.245137 | 0/1/127 |
| quad→drone | joint3_C | frozen_C | -0.346629 | 2/3/123 |
| quad→drone | joint3_F | lm_F | +0.000000 | 0/0/128 |
| quad→drone | joint3_F | binding_direct | +0.051803 | 2/1/125 |

## 技术核验与good/failure案例

三source六个真实工程样本退回单视图时，与原LM最终权重最大误差均为零。正式768条全部封存后，独立重放全部拟合端点、目标误差和接受条件，并在六个真实样本重算原生三视图特征及拟合权重。显式Jacobian与autograd比较，显式J·Jᵀ与分块Gram比较，另用通用3D交核验主结果IoU；旧对照与R98原逐例值保持一致。

从全量R98每方向每因素的正负变化中固定两条极值和一条中位变化样本，形成可追溯案例列表。另生成12页原图、投影、BEV和候选排名私有图册；实际人工式本地视觉复核仅两页，不宣称全部完成身份审核。复核发现一个纠错中F已纠正而C仍错，平均仍正确；另一个误改中F/C各自Top1均未改变，但概率间隔改变后使FC从正确行人候选转向另一候选。因此Top1是否改变本身不足以描述平均读出的损伤。身份不确定保持未知，案例不参与适应或逐例规则。

图册原句、媒体、框坐标、query索引和特征均不公开。可视化首尝试将非Waymo的空suffix转整数导致异常，尚无页面或方法结果受影响；修复仅为可视化坐标参数默认0，保留失败尝试，不改科学预测。

## 本轮裁决

固定主端点=16.517235%，相对旧单视图LM FC +0.205804pp，相对冻结+0.288583pp，相对同信息直接FC +0.275245pp，相对打乱监督+0.264525pp。未拟合C相对旧单视图C -0.062055pp，相对冻结C +0.139799pp。F相对直接证据+0.076217pp。不会将更高的次要端点事后升级为原主方法。

本轮只更新拟合视图数这一配方的证据。优化误差下降、更多非零步、局部好图均不能代替总体任务收益。夜间目标仍进行，后续实验必须根据这些差异锁定新假设，保留无外部专家、固定候选和同信息控制；截止2026-11-16北京时间09:30后不启动新模型、训练或评分。

开发信号说明：主端点对旧LM FC仅7条IoU改善、3条恶化，其余758条不变；相对冻结22条改善、19条恶化。尽管六个方向相对冻结均为正，样本改变量仍少，必须不改参数扩展全量才能判断。该场景均衡小清单上原始3EED为16.811383%，仍高于当前准备后系统；不能把适应相对自身冻结的增量称为整个系统已经超过原始3EED。下一步R100固定本算法和全部对照，原样扩展原18620清单，不按已看GT改方法。
