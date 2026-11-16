# R104 — 几何绑定的RGB末端投影适应

固定原R100均值G、三拟合视图、10次LM与每次6次阻尼验收、原A256及所有source状态，只把更新对象从18,496参数query投影换为已在source训练的4,160参数RGB末端投影。RGB编码器、双向交互、门控、query匹配、主干与竞争残差全部冻结。不是新增视觉网络，也不是引入专家。参数数目不同属于本更新结构差异，不能归为严格等参数量因果实验。

先用12条真实样本验证原始和实际写参数后的完整模型重放、显式Jacobian、未映射候选每视图原始分数不变。新RGB输入通过原始点–像素对应重算，不反求源权重或猜像素。随后全部768条封存预测，独立审计后评分。C只在两种拟合完成后生成；是同物理观测的遮挡版本，不是新相机或独立场景。

| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |
|---|---|---:|---:|---:|---:|
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
| all_six_equal | late_F | 16.214022 | 29.947917 | 12.890625 |
| all_six_equal | late_C | 16.131533 | 29.557292 | 12.630208 |
| all_six_equal | late_FC | 16.128391 | 29.557292 | 12.630208 |
| all_six_equal | late_shuffle_F | 16.243978 | 30.208333 | 12.760417 |
| all_six_equal | late_shuffle_C | 16.168329 | 29.817708 | 12.760417 |
| all_six_equal | late_shuffle_FC | 16.336928 | 30.338542 | 13.020833 |
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
| waymo | late_F | 12.572382 | 25.781250 | 5.859375 |
| waymo | late_C | 12.905173 | 25.781250 | 5.859375 |
| waymo | late_FC | 12.543123 | 25.390625 | 5.468750 |
| waymo | late_shuffle_F | 12.734907 | 26.171875 | 5.468750 |
| waymo | late_shuffle_C | 12.686454 | 26.562500 | 5.468750 |
| waymo | late_shuffle_FC | 12.548512 | 25.781250 | 5.468750 |
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
| drone | late_F | 14.292082 | 25.781250 | 12.109375 |
| drone | late_C | 13.931046 | 25.000000 | 11.718750 |
| drone | late_FC | 14.185283 | 25.390625 | 12.109375 |
| drone | late_shuffle_F | 14.191299 | 25.781250 | 11.718750 |
| drone | late_shuffle_C | 13.891654 | 25.000000 | 11.718750 |
| drone | late_shuffle_FC | 14.315694 | 26.171875 | 12.109375 |
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
| quad | late_F | 21.777603 | 38.281250 | 20.703125 |
| quad | late_C | 21.558379 | 37.890625 | 20.312500 |
| quad | late_FC | 21.656767 | 37.890625 | 20.312500 |
| quad | late_shuffle_F | 21.805728 | 38.671875 | 21.093750 |
| quad | late_shuffle_C | 21.926880 | 37.890625 | 21.093750 |
| quad | late_shuffle_FC | 22.146577 | 39.062500 | 21.484375 |

## 六方向固定比较

| 方向 | 方法 | 对照 | Δ pp | 改善/恶化/不变 |
|---|---|---|---:|---:|
| waymo→drone | late_FC | prepared_frozen | +0.220025 | 7/6/115 |
| waymo→drone | late_FC | frozen_FC | -0.073117 | 6/6/116 |
| waymo→drone | late_FC | direct_FC | -0.111175 | 4/6/118 |
| waymo→drone | late_FC | direct_four | -0.089251 | 5/6/117 |
| waymo→drone | late_FC | joint3_FC | -0.084259 | 3/3/122 |
| waymo→drone | late_FC | late_shuffle_FC | -0.152919 | 4/8/116 |
| waymo→drone | late_F | binding_direct | -0.079299 | 3/5/120 |
| waymo→drone | late_F | joint3_F | -0.064173 | 2/3/123 |
| waymo→drone | late_C | direct_C | -0.163535 | 4/6/118 |
| waymo→drone | late_C | joint3_C | -0.067570 | 2/3/123 |
| waymo→quad | late_FC | prepared_frozen | -0.146386 | 0/3/125 |
| waymo→quad | late_FC | frozen_FC | -0.019104 | 0/2/126 |
| waymo→quad | late_FC | direct_FC | -0.213177 | 0/3/125 |
| waymo→quad | late_FC | direct_four | -0.805931 | 0/4/124 |
| waymo→quad | late_FC | joint3_FC | -0.212505 | 0/2/126 |
| waymo→quad | late_FC | late_shuffle_FC | +0.033030 | 2/2/124 |
| waymo→quad | late_F | binding_direct | -0.221175 | 0/4/124 |
| waymo→quad | late_F | joint3_F | -0.220504 | 0/3/125 |
| waymo→quad | late_C | direct_C | -0.213177 | 0/3/125 |
| waymo→quad | late_C | joint3_C | -0.212505 | 0/2/126 |
| drone→waymo | late_FC | prepared_frozen | -0.007733 | 8/6/114 |
| drone→waymo | late_FC | frozen_FC | +0.196411 | 6/5/117 |
| drone→waymo | late_FC | direct_FC | -0.153590 | 4/5/119 |
| drone→waymo | late_FC | direct_four | -0.283680 | 5/5/118 |
| drone→waymo | late_FC | joint3_FC | -0.188677 | 4/6/118 |
| drone→waymo | late_FC | late_shuffle_FC | +0.040820 | 7/6/115 |
| drone→waymo | late_F | binding_direct | +0.201709 | 3/5/120 |
| drone→waymo | late_F | joint3_F | +0.089145 | 5/5/118 |
| drone→waymo | late_C | direct_C | -0.023315 | 5/4/119 |
| drone→waymo | late_C | joint3_C | -0.282055 | 3/5/120 |
| drone→quad | late_FC | prepared_frozen | -0.537604 | 4/6/118 |
| drone→quad | late_FC | frozen_FC | -1.049538 | 3/4/121 |
| drone→quad | late_FC | direct_FC | -0.725251 | 3/2/123 |
| drone→quad | late_FC | direct_four | -0.404309 | 3/2/123 |
| drone→quad | late_FC | joint3_FC | -0.964796 | 2/3/123 |
| drone→quad | late_FC | late_shuffle_FC | -1.012650 | 6/6/116 |
| drone→quad | late_F | binding_direct | -0.049302 | 3/1/124 |
| drone→quad | late_F | joint3_F | -0.304594 | 3/2/123 |
| drone→quad | late_C | direct_C | -0.765551 | 3/4/121 |
| drone→quad | late_C | joint3_C | -1.003790 | 3/5/120 |
| quad→waymo | late_FC | prepared_frozen | -0.137053 | 4/4/120 |
| quad→waymo | late_FC | frozen_FC | +0.009675 | 4/4/120 |
| quad→waymo | late_FC | direct_FC | -0.271847 | 1/3/124 |
| quad→waymo | late_FC | direct_four | +0.010610 | 3/4/121 |
| quad→waymo | late_FC | joint3_FC | -0.860497 | 1/4/123 |
| quad→waymo | late_FC | late_shuffle_FC | -0.051596 | 2/3/123 |
| quad→waymo | late_F | binding_direct | -0.361886 | 2/2/124 |
| quad→waymo | late_F | joint3_F | -0.415327 | 1/2/125 |
| quad→waymo | late_C | direct_C | +0.539084 | 3/0/125 |
| quad→waymo | late_C | joint3_C | -0.081723 | 3/1/124 |
| quad→drone | late_FC | prepared_frozen | +0.007185 | 1/2/125 |
| quad→drone | late_FC | frozen_FC | -0.101492 | 2/2/124 |
| quad→drone | late_FC | direct_FC | +0.793446 | 5/1/122 |
| quad→drone | late_FC | direct_four | +0.671760 | 5/2/121 |
| quad→drone | late_FC | joint3_FC | -0.022329 | 1/1/126 |
| quad→drone | late_FC | late_shuffle_FC | -0.107904 | 1/2/125 |
| quad→drone | late_F | binding_direct | +0.157017 | 3/2/123 |
| quad→drone | late_F | joint3_F | +0.105214 | 1/1/126 |
| quad→drone | late_C | direct_C | +0.437138 | 7/2/119 |
| quad→drone | late_C | joint3_C | +0.366823 | 2/0/126 |

## 机制与边界

{
  "mean_mapped_query_fraction": 0.8833363850911459,
  "top1_switches_vs_prepared_F": 100,
  "categories": [
    {
      "before_known": false,
      "after_known": false,
      "n": 1,
      "better": 0,
      "worse": 0
    },
    {
      "before_known": false,
      "after_known": true,
      "n": 3,
      "better": 0,
      "worse": 1
    },
    {
      "before_known": true,
      "after_known": false,
      "n": 8,
      "better": 2,
      "worse": 3
    },
    {
      "before_known": true,
      "after_known": true,
      "n": 88,
      "better": 22,
      "worse": 23
    }
  ],
  "invariant_is_raw_per_view_score_not_FC_ranking": true
}

未映射候选的每视图原始分数保持，不等于最终FC排名保持：概率归一化及跨视图平均仍可引入间接变化。可见性不等于正确实例身份；IoU高低也不作为身份真值。所有未知、无RGB、零更新、负例仍计入总体。G、更新接受与停止均不使用GT。拟合下降、参数非零或保护不变量不是任务成功。

## 裁决

{
  "primary": "late_FC",
  "primary_mIoU_percent": 16.128391222080957,
  "gains_pp": {
    "prepared_frozen": -0.10026093710772344,
    "frozen_FC": -0.17286081780189733,
    "direct_FC": -0.11359884701925083,
    "direct_four": -0.15013348758617795,
    "joint3_FC": -0.38884382756731384,
    "late_shuffle_FC": -0.2085365313421228
  },
  "passes_prelocked_primary_gates": false,
  "n": 768,
  "parameters": 4160,
  "query_projection_parameters_in_control": 18496,
  "scope_only_change_vs_R100": true,
  "robust_adoption": false,
  "development_exposed": true,
  "FINAL_access": false,
  "no_experts": true,
  "no_source_training": true,
  "primary_not_switched_to_F": true
}

若不能超过同信息直接读出和打乱监督，不采纳本更新位置配方，也不按个例增加幅度或改变主端点。开发通过也仍需固定扩展验证，不能宣称稳定跨域TTA。文献边界见LITERATURE.md，本轮不是READ或xMUDA复现。


## 不使用GT的可拟合性下界诊断

因为未映射候选原始分数固定，即使允许所有已映射候选分数自由变化，居中分数L2仍有下界：未映射候选G差分相对其自身均值的平方和除以2K。对768条，映射候选平均占88.33%；该下界均值仅0.00001831，约占初始误差的0.57%。实际RGB投影最终误差0.001605，query投影0.001313。因此，不能主要用“缺少RGB覆盖导致不可拟合”解释本轮失败；RGB表示基底/共享参数表达与监督质量仍需区分。这个下界不是grounding上界，不使用GT，也没有生成新的方法预测。

RGB投影平均参数位移L2为10.568，源投影范数约5，显示两个更新位置的参数尺度不同。但权重位移本身不能证明任务过冲，不据此追加步长网格或逐例选择更新幅度。所有接受步骤实际降低原无标签目标，任务仍低于直接使用相同信息与打乱对照。
