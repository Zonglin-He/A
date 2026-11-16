# R101 — 固定绑定监督的分布度量检验

R100固定全量扩展未保持R99的开发优势。本轮固定原三源、A256、RGB–点云绑定差分G、三拟合视图、18,496个末端线性参数和10×6优化预算，仅将均匀分数L2及其Gauss–Newton度量替换为精确候选KL及匹配Fisher度量。真实目标、打乱目标均运行。源模型逐输入重置，C在两种拟合完成后才读取，不参与验收。768条来自R99已暴露开发集；不宣称独立泛化。

| 方向 | n | 冻结 | 直接FC | L2三视图FC | KL三视图FC | KL打乱FC |
|---|---:|---:|---:|---:|---:|---:|
| waymo→drone | 128 | 17.195182 | 17.526382 | 17.499466 | 17.488464 | 17.306078 |
| waymo→quad | 128 | 29.615174 | 29.681964 | 29.681293 | 29.676740 | 30.101697 |
| drone→waymo | 128 | 15.126688 | 15.272546 | 15.307633 | 15.391697 | 14.653344 |
| drone→quad | 128 | 14.382351 | 14.569998 | 14.809542 | 14.571303 | 14.861257 |
| quad→waymo | 128 | 10.104344 | 10.239138 | 10.827789 | 10.239138 | 9.805267 |
| quad→drone | 128 | 10.948174 | 10.161913 | 10.977688 | 10.172233 | 11.304988 |

## 等权六方向及各目标平台

| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |
|---|---|---:|---:|---:|
| all_six_equal | original_source | 16.811383 | 30.989583 | 14.973958 |
| all_six_equal | prepared_frozen | 16.228652 | 29.947917 | 12.630208 |
| all_six_equal | binding_direct | 16.272845 | 30.078125 | 12.760417 |
| all_six_equal | direct_FC | 16.241990 | 29.817708 | 12.760417 |
| all_six_equal | direct_four | 16.278525 | 29.947917 | 12.890625 |
| all_six_equal | lm_FC | 16.311431 | 29.947917 | 12.760417 |
| all_six_equal | joint3_C | 16.345003 | 30.078125 | 12.760417 |
| all_six_equal | joint3_FC | 16.517235 | 30.338542 | 13.020833 |
| all_six_equal | kl3_F | 16.234662 | 29.947917 | 12.760417 |
| all_six_equal | kl3_C | 16.184528 | 29.947917 | 12.500000 |
| all_six_equal | kl3_FC | 16.256596 | 29.947917 | 12.760417 |
| all_six_equal | kl3_shuffle_FC | 16.338772 | 30.078125 | 12.760417 |
| waymo | original_source | 11.473631 | 23.437500 | 7.421875 |
| waymo | prepared_frozen | 12.615516 | 26.171875 | 5.078125 |
| waymo | binding_direct | 12.652470 | 25.781250 | 5.468750 |
| waymo | direct_FC | 12.755842 | 25.390625 | 5.468750 |
| waymo | direct_four | 12.679658 | 25.781250 | 5.468750 |
| waymo | lm_FC | 12.914574 | 26.562500 | 5.468750 |
| waymo | joint3_C | 13.087062 | 26.953125 | 5.468750 |
| waymo | joint3_FC | 13.067711 | 26.953125 | 5.859375 |
| waymo | kl3_F | 12.550952 | 25.390625 | 5.468750 |
| waymo | kl3_C | 12.707359 | 26.171875 | 5.078125 |
| waymo | kl3_FC | 12.815418 | 25.781250 | 5.468750 |
| waymo | kl3_shuffle_FC | 12.229305 | 25.781250 | 4.687500 |
| drone | original_source | 13.867693 | 24.609375 | 12.500000 |
| drone | prepared_frozen | 14.071678 | 24.609375 | 11.718750 |
| drone | binding_direct | 14.253223 | 25.781250 | 12.109375 |
| drone | direct_FC | 13.844147 | 25.000000 | 11.718750 |
| drone | direct_four | 13.894029 | 25.000000 | 11.718750 |
| drone | lm_FC | 13.904753 | 24.218750 | 11.718750 |
| drone | joint3_C | 13.781419 | 24.218750 | 11.718750 |
| drone | joint3_FC | 14.238577 | 24.609375 | 12.109375 |
| drone | kl3_F | 14.242804 | 25.781250 | 12.109375 |
| drone | kl3_C | 13.893257 | 25.000000 | 11.328125 |
| drone | kl3_FC | 13.830349 | 25.000000 | 11.718750 |
| drone | kl3_shuffle_FC | 14.305533 | 25.000000 | 11.718750 |
| quad | original_source | 25.092825 | 44.921875 | 25.000000 |
| quad | prepared_frozen | 21.998762 | 39.062500 | 21.093750 |
| quad | binding_direct | 21.912842 | 38.671875 | 20.703125 |
| quad | direct_FC | 22.125981 | 39.062500 | 21.093750 |
| quad | direct_four | 22.261887 | 39.062500 | 21.484375 |
| quad | lm_FC | 22.114967 | 39.062500 | 21.093750 |
| quad | joint3_C | 22.166527 | 39.062500 | 21.093750 |
| quad | joint3_FC | 22.245417 | 39.453125 | 21.093750 |
| quad | kl3_F | 21.910229 | 38.671875 | 20.703125 |
| quad | kl3_C | 21.952968 | 38.671875 | 21.093750 |
| quad | kl3_FC | 22.124021 | 39.062500 | 21.093750 |
| quad | kl3_shuffle_FC | 22.481477 | 39.453125 | 21.875000 |

## 全部预设比较

| 方向 | 方法 | 对照 | Δ pp | 改善/恶化/不变 |
|---|---|---|---:|---:|
| waymo→drone | kl3_FC | prepared_frozen | +0.293282 | 8/8/112 |
| waymo→drone | kl3_FC | joint3_FC | -0.011001 | 4/5/119 |
| waymo→drone | kl3_FC | lm_FC | +0.116273 | 6/5/117 |
| waymo→drone | kl3_FC | direct_FC | -0.037918 | 0/2/126 |
| waymo→drone | kl3_FC | direct_four | -0.015994 | 1/2/125 |
| waymo→drone | kl3_FC | kl3_shuffle_FC | +0.182386 | 15/8/105 |
| waymo→drone | kl3_C | joint3_C | -0.034068 | 7/6/115 |
| waymo→drone | kl3_C | frozen_C | +0.247993 | 12/8/108 |
| waymo→drone | kl3_F | joint3_F | -0.022791 | 3/5/120 |
| waymo→drone | kl3_F | binding_direct | -0.037918 | 0/2/126 |
| waymo→quad | kl3_FC | prepared_frozen | +0.061566 | 1/3/124 |
| waymo→quad | kl3_FC | joint3_FC | -0.004553 | 0/1/127 |
| waymo→quad | kl3_FC | lm_FC | -0.004553 | 0/1/127 |
| waymo→quad | kl3_FC | direct_FC | -0.005225 | 0/1/127 |
| waymo→quad | kl3_FC | direct_four | -0.597979 | 0/2/126 |
| waymo→quad | kl3_FC | kl3_shuffle_FC | -0.424957 | 5/5/118 |
| waymo→quad | kl3_C | joint3_C | -0.004553 | 0/1/127 |
| waymo→quad | kl3_C | frozen_C | +0.216524 | 2/2/124 |
| waymo→quad | kl3_F | joint3_F | -0.004553 | 0/1/127 |
| waymo→quad | kl3_F | binding_direct | -0.005225 | 0/1/127 |
| drone→waymo | kl3_FC | prepared_frozen | +0.265009 | 7/3/118 |
| drone→waymo | kl3_FC | joint3_FC | +0.084064 | 2/2/124 |
| drone→waymo | kl3_FC | lm_FC | -0.268097 | 1/3/124 |
| drone→waymo | kl3_FC | direct_FC | +0.119151 | 1/0/127 |
| drone→waymo | kl3_FC | direct_four | -0.010939 | 3/1/124 |
| drone→waymo | kl3_FC | kl3_shuffle_FC | +0.738353 | 11/4/113 |
| drone→waymo | kl3_C | joint3_C | -0.125470 | 2/2/124 |
| drone→waymo | kl3_C | frozen_C | +0.425263 | 7/4/117 |
| drone→waymo | kl3_F | joint3_F | -0.197995 | 1/3/124 |
| drone→waymo | kl3_F | binding_direct | -0.085431 | 0/4/124 |
| drone→quad | kl3_FC | prepared_frozen | +0.188952 | 3/5/120 |
| drone→quad | kl3_FC | joint3_FC | -0.238239 | 0/1/127 |
| drone→quad | kl3_FC | lm_FC | +0.022662 | 1/0/127 |
| drone→quad | kl3_FC | direct_FC | +0.001306 | 1/0/127 |
| drone→quad | kl3_FC | direct_four | +0.322248 | 1/0/127 |
| drone→quad | kl3_FC | kl3_shuffle_FC | -0.289954 | 4/4/120 |
| drone→quad | kl3_C | joint3_C | -0.422564 | 1/2/125 |
| drone→quad | kl3_C | frozen_C | -0.507491 | 3/5/120 |
| drone→quad | kl3_F | joint3_F | -0.255292 | 1/2/125 |
| drone→quad | kl3_F | binding_direct | +0.000000 | 0/0/128 |
| quad→waymo | kl3_FC | prepared_frozen | +0.134794 | 5/3/120 |
| quad→waymo | kl3_FC | joint3_FC | -0.588650 | 1/2/125 |
| quad→waymo | kl3_FC | lm_FC | +0.069784 | 2/1/125 |
| quad→waymo | kl3_FC | direct_FC | +0.000000 | 0/0/128 |
| quad→waymo | kl3_FC | direct_four | +0.282457 | 2/1/125 |
| quad→waymo | kl3_FC | kl3_shuffle_FC | +0.433871 | 7/5/116 |
| quad→waymo | kl3_C | joint3_C | -0.633936 | 0/2/126 |
| quad→waymo | kl3_C | frozen_C | -0.417460 | 1/3/124 |
| quad→waymo | kl3_F | joint3_F | -0.171047 | 0/1/127 |
| quad→waymo | kl3_F | binding_direct | -0.117606 | 1/1/126 |
| quad→drone | kl3_FC | prepared_frozen | -0.775941 | 1/5/122 |
| quad→drone | kl3_FC | joint3_FC | -0.805455 | 1/4/123 |
| quad→drone | kl3_FC | lm_FC | -0.265082 | 1/3/124 |
| quad→drone | kl3_FC | direct_FC | +0.010320 | 1/0/127 |
| quad→drone | kl3_FC | direct_four | -0.111366 | 1/1/126 |
| quad→drone | kl3_FC | kl3_shuffle_FC | -1.132756 | 6/9/113 |
| quad→drone | kl3_C | joint3_C | +0.257744 | 3/4/121 |
| quad→drone | kl3_C | frozen_C | -0.088885 | 3/5/120 |
| quad→drone | kl3_F | joint3_F | -0.034723 | 1/1/126 |
| quad→drone | kl3_F | binding_direct | +0.017079 | 1/0/127 |

## 机制与边界

Fisher双对偶矩阵已与显式参数Jacobian核验，KL梯度与自动求导比较；全部预测和每个接受/拒绝trial均独立重放。C为同一物理观测的新遮挡而非新相机或新场景。新方法复用的特征缓存有R100原生前向审计及逐文件哈希；没有新增图像专家。

好坏尾部保留于私有案例索引，公开表保留所有匿名结果。IoU变化只作为事后效果，不当作实例身份真值或更新筛选条件。对梯度与损失度量的假设不能替代grounding指标。

## 裁决

{
  "primary": "kl3_FC",
  "primary_mIoU_percent": 16.256595907080325,
  "gains_pp": {
    "prepared_frozen": 0.02794374789164422,
    "joint3_FC": -0.2606391425679462,
    "direct_FC": 0.014605837980116831,
    "direct_four": -0.021928802586810292,
    "kl3_shuffle_FC": -0.08217593181931448
  },
  "passes_primary_comparisons": false,
  "n": 768,
  "historically_exposed_development": true,
  "independent_confirmation": false,
  "robust_adoption": false,
  "one_changed_factor": "KL objective and matched Fisher metric",
  "no_external_experts": true,
  "no_source_retraining": true,
  "FINAL_access": false,
  "C_new_scene": false
}

若KL未超过同信息直接读出与L2对照，停止以同一G单纯更换度量作为性能主线，不扩大步数。若开发全部比较通过，也只作为候选，需固定后扩大验证；不能据开发集宣称稳健TTA。


## 已验证的拟合与任务差距

对全部768输入按原生GPU头重放，KL配方在690条上获得比L2配方更低的KL，均值由L2端点0.000986降至0.000530。但相对L2只有8条IoU改善、15条恶化；全部15条恶化都同时获得更低的KL。因此不能把这轮下降归因为KL目标没有真正优化，也不能用继续降低KL作为grounding验收。52条FC query切换中，14条在F和C各自Top1均未改变时发生，表明概率融合的置信度变化仍参与最终决策。这里的计数不等于人工确认的实例身份。

本轮度量假设的文献依据仅是目标并不等价：Hinton等的[原文第2.1节](https://arxiv.org/html/1503.02531v1)说明，居中logit平方拟合是高温蒸馏极限，低温目标较少关注很低的logit；哪种信息有用仍需实验。它不保证本任务KL更好，也没有向本系统加入外部teacher。实验已经否定本固定配方的采用依据；不扩大KL步数或挑选表现较好的方向。
