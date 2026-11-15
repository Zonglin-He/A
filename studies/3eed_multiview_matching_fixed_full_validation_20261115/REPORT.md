# R100 — R99固定配方的六方向全量扩展

R99开发768条出现正信号后，保持G、三视图拟合、18,496参数、原始A256、十次LM与每次六次阻尼、原source状态和C后读取规则不变。没有继续挑参数。原768条输出经哈希核验复用，新增输入使用完全相同程序；全部20臂、负例、UNKNOWN和零IoU保留。全量18620来自此前R98历史暴露范围，属于固定配置扩展，不能称全新独立测试。

| 方向 | n | 原始 | 冻结 | 旧单视图FC | 直接FC | 直接四视图 | 新三视图FC | 打乱三视图FC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| waymo→drone | 2792 | 8.832240 | 9.285471 | 9.310655 | 9.253592 | 9.297855 | 9.268648 | 9.183057 |
| waymo→quad | 2736 | 17.588082 | 18.750207 | 18.805533 | 18.808625 | 18.827405 | 18.825702 | 18.855284 |
| drone→waymo | 3782 | 14.142348 | 14.609560 | 14.646784 | 14.600940 | 14.678816 | 14.653846 | 14.597819 |
| drone→quad | 2736 | 13.077292 | 12.901516 | 12.959761 | 12.929359 | 12.917891 | 12.966259 | 12.824789 |
| quad→waymo | 3782 | 10.829191 | 10.898764 | 10.768880 | 10.701449 | 10.651684 | 10.750696 | 10.857332 |
| quad→drone | 2792 | 6.055598 | 6.725807 | 6.583802 | 6.683912 | 6.614292 | 6.626423 | 6.635238 |

## 开发部分与扩大部分分开看

原768条用于选择这一候选；其余17852条是本算法未用于开发输出的扩大部分，但数据本身存在R98历史暴露，仍不能称新独立测试。以下按方向等权列主要差值，不让复用的开发正例掩盖扩大部分。

| 部分 | 对照 | 等权Δ pp | 改善/恶化 |
|---|---|---:|---:|
| R99_DEV128 | prepared_frozen | +0.288583 | 22/19 |
| R99_DEV128 | lm_FC | +0.205804 | 7/3 |
| R99_DEV128 | direct_FC | +0.275245 | 16/10 |
| R99_DEV128 | direct_four | +0.238710 | 19/13 |
| R99_DEV128 | joint3_shuffle_FC | +0.264525 | 33/26 |
| EXPANSION_ONLY | prepared_frozen | -0.025670 | 674/753 |
| EXPANSION_ONLY | lm_FC | -0.006418 | 101/134 |
| EXPANSION_ONLY | direct_FC | +0.007735 | 396/376 |
| EXPANSION_ONLY | direct_four | +0.008169 | 478/455 |
| EXPANSION_ONLY | joint3_shuffle_FC | +0.015831 | 963/949 |

## 全部平台和读出

| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |
|---|---|---:|---:|---:|
| all_six_equal | original_source | 11.754125 | 22.475706 | 6.148478 |
| all_six_equal | prepared_frozen | 12.195221 | 22.844980 | 5.750723 |
| all_six_equal | binding_direct | 12.222276 | 22.855125 | 5.770964 |
| all_six_equal | R94_Adam | 12.219633 | 22.903881 | 5.746316 |
| all_six_equal | frozen_C | 12.153395 | 22.764288 | 5.763395 |
| all_six_equal | frozen_FC | 12.171306 | 22.758759 | 5.776371 |
| all_six_equal | frozen_four | 12.163235 | 22.800363 | 5.767802 |
| all_six_equal | direct_C | 12.158858 | 22.632522 | 5.790887 |
| all_six_equal | direct_FC | 12.162979 | 22.670427 | 5.771452 |
| all_six_equal | direct_four | 12.164657 | 22.691301 | 5.806585 |
| all_six_equal | lm_F | 12.208409 | 22.814888 | 5.736868 |
| all_six_equal | lm_C | 12.138836 | 22.657804 | 5.794524 |
| all_six_equal | lm_FC | 12.179236 | 22.745868 | 5.772367 |
| all_six_equal | lm_shuffle_FC | 12.164903 | 22.687674 | 5.799456 |
| all_six_equal | joint3_F | 12.210093 | 22.839377 | 5.766557 |
| all_six_equal | joint3_C | 12.149033 | 22.667020 | 5.798541 |
| all_six_equal | joint3_FC | 12.181929 | 22.735614 | 5.786113 |
| all_six_equal | joint3_shuffle_F | 12.190070 | 22.802840 | 5.771086 |
| all_six_equal | joint3_shuffle_C | 12.159000 | 22.731108 | 5.839887 |
| all_six_equal | joint3_shuffle_FC | 12.158920 | 22.729681 | 5.796856 |
| waymo | original_source | 12.485769 | 26.070862 | 6.795346 |
| waymo | prepared_frozen | 12.754162 | 26.692226 | 6.041777 |
| waymo | binding_direct | 12.720859 | 26.560021 | 5.975674 |
| waymo | R94_Adam | 12.746734 | 26.652565 | 6.028556 |
| waymo | frozen_C | 12.717750 | 26.665785 | 6.041777 |
| waymo | frozen_FC | 12.734092 | 26.560021 | 6.081438 |
| waymo | frozen_four | 12.694533 | 26.520360 | 6.054997 |
| waymo | direct_C | 12.664009 | 26.467478 | 6.015336 |
| waymo | direct_FC | 12.651195 | 26.401375 | 5.975674 |
| waymo | direct_four | 12.665250 | 26.427816 | 6.081438 |
| waymo | lm_F | 12.750847 | 26.692226 | 5.909572 |
| waymo | lm_C | 12.673350 | 26.507139 | 6.081438 |
| waymo | lm_FC | 12.707832 | 26.573242 | 6.015336 |
| waymo | lm_shuffle_FC | 12.749592 | 26.599683 | 6.041777 |
| waymo | joint3_F | 12.744379 | 26.692226 | 5.962454 |
| waymo | joint3_C | 12.655386 | 26.480698 | 6.002115 |
| waymo | joint3_FC | 12.702271 | 26.560021 | 6.002115 |
| waymo | joint3_shuffle_F | 12.737790 | 26.546801 | 5.975674 |
| waymo | joint3_shuffle_C | 12.732287 | 26.639344 | 6.107879 |
| waymo | joint3_shuffle_FC | 12.727576 | 26.507139 | 6.015336 |
| drone | original_source | 7.443919 | 14.272923 | 2.238539 |
| drone | prepared_frozen | 8.005639 | 14.613181 | 2.292264 |
| drone | binding_direct | 8.092262 | 14.702722 | 2.345989 |
| drone | R94_Adam | 8.068038 | 14.756447 | 2.292264 |
| drone | frozen_C | 7.892948 | 14.434097 | 2.220630 |
| drone | frozen_FC | 7.958780 | 14.541547 | 2.256447 |
| drone | frozen_four | 7.927265 | 14.541547 | 2.220630 |
| drone | direct_C | 7.920804 | 14.237106 | 2.256447 |
| drone | direct_FC | 7.968752 | 14.380372 | 2.274355 |
| drone | direct_four | 7.956074 | 14.398281 | 2.292264 |
| drone | lm_F | 8.017052 | 14.559456 | 2.328080 |
| drone | lm_C | 7.909283 | 14.255014 | 2.274355 |
| drone | lm_FC | 7.947229 | 14.398281 | 2.292264 |
| drone | lm_shuffle_FC | 7.929531 | 14.398281 | 2.292264 |
| drone | joint3_F | 8.017364 | 14.541547 | 2.345989 |
| drone | joint3_C | 7.909460 | 14.290831 | 2.274355 |
| drone | joint3_FC | 7.947535 | 14.362464 | 2.310172 |
| drone | joint3_shuffle_F | 8.018058 | 14.577364 | 2.328080 |
| drone | joint3_shuffle_C | 7.907319 | 14.434097 | 2.274355 |
| drone | joint3_shuffle_FC | 7.909148 | 14.434097 | 2.274355 |
| quad | original_source | 15.332687 | 27.083333 | 9.411550 |
| quad | prepared_frozen | 15.825862 | 27.229532 | 8.918129 |
| quad | binding_direct | 15.853709 | 27.302632 | 8.991228 |
| quad | R94_Adam | 15.844128 | 27.302632 | 8.918129 |
| quad | frozen_C | 15.849486 | 27.192982 | 9.027778 |
| quad | frozen_FC | 15.821046 | 27.174708 | 8.991228 |
| quad | frozen_four | 15.867908 | 27.339181 | 9.027778 |
| quad | direct_C | 15.891760 | 27.192982 | 9.100877 |
| quad | direct_FC | 15.868992 | 27.229532 | 9.064327 |
| quad | direct_four | 15.872648 | 27.247807 | 9.046053 |
| quad | lm_F | 15.857329 | 27.192982 | 8.972953 |
| quad | lm_C | 15.833875 | 27.211257 | 9.027778 |
| quad | lm_FC | 15.882647 | 27.266082 | 9.009503 |
| quad | lm_shuffle_FC | 15.815586 | 27.065058 | 9.064327 |
| quad | joint3_F | 15.868536 | 27.284357 | 8.991228 |
| quad | joint3_C | 15.882251 | 27.229532 | 9.119152 |
| quad | joint3_FC | 15.895980 | 27.284357 | 9.046053 |
| quad | joint3_shuffle_F | 15.814360 | 27.284357 | 9.009503 |
| quad | joint3_shuffle_C | 15.837395 | 27.119883 | 9.137427 |
| quad | joint3_shuffle_FC | 15.840036 | 27.247807 | 9.100877 |

## 全部预设成对比较

| 方向 | 方法 | 对照 | Δ pp | 改善/恶化/不变 |
|---|---|---|---:|---:|
| waymo→drone | joint3_FC | prepared_frozen | -0.016823 | 133/145/2514 |
| waymo→drone | joint3_FC | lm_FC | -0.042007 | 12/26/2754 |
| waymo→drone | joint3_FC | direct_FC | +0.015056 | 124/91/2577 |
| waymo→drone | joint3_FC | direct_four | -0.029207 | 129/96/2567 |
| waymo→drone | joint3_FC | joint3_shuffle_FC | +0.085591 | 228/198/2366 |
| waymo→drone | joint3_C | lm_C | +0.013988 | 23/16/2753 |
| waymo→drone | joint3_C | frozen_C | +0.066029 | 134/134/2524 |
| waymo→drone | joint3_F | lm_F | +0.011245 | 19/20/2753 |
| waymo→drone | joint3_F | binding_direct | +0.005987 | 124/91/2577 |
| waymo→quad | joint3_FC | prepared_frozen | +0.075495 | 126/123/2487 |
| waymo→quad | joint3_FC | lm_FC | +0.020169 | 11/15/2710 |
| waymo→quad | joint3_FC | direct_FC | +0.017076 | 71/85/2580 |
| waymo→quad | joint3_FC | direct_four | -0.001704 | 78/91/2567 |
| waymo→quad | joint3_FC | joint3_shuffle_FC | -0.029582 | 183/168/2385 |
| waymo→quad | joint3_C | lm_C | -0.022261 | 14/9/2713 |
| waymo→quad | joint3_C | frozen_C | -0.014376 | 106/116/2514 |
| waymo→quad | joint3_F | lm_F | -0.019188 | 12/16/2708 |
| waymo→quad | joint3_F | binding_direct | -0.019967 | 70/95/2571 |
| drone→waymo | joint3_FC | prepared_frozen | +0.044286 | 153/151/3478 |
| drone→waymo | joint3_FC | lm_FC | +0.007062 | 28/27/3727 |
| drone→waymo | joint3_FC | direct_FC | +0.052906 | 41/35/3706 |
| drone→waymo | joint3_FC | direct_four | -0.024970 | 67/70/3645 |
| drone→waymo | joint3_FC | joint3_shuffle_FC | +0.056027 | 183/187/3412 |
| drone→waymo | joint3_C | lm_C | +0.050610 | 39/39/3704 |
| drone→waymo | joint3_C | frozen_C | +0.107967 | 133/142/3507 |
| drone→waymo | joint3_F | lm_F | +0.020798 | 31/30/3721 |
| drone→waymo | joint3_F | binding_direct | +0.014183 | 48/48/3686 |
| drone→quad | joint3_FC | prepared_frozen | +0.064743 | 124/143/2469 |
| drone→quad | joint3_FC | lm_FC | +0.006498 | 26/33/2677 |
| drone→quad | joint3_FC | direct_FC | +0.036900 | 48/49/2639 |
| drone→quad | joint3_FC | direct_four | +0.048368 | 62/60/2614 |
| drone→quad | joint3_FC | joint3_shuffle_FC | +0.141470 | 176/155/2405 |
| drone→quad | joint3_C | lm_C | +0.119014 | 40/25/2671 |
| drone→quad | joint3_C | frozen_C | +0.079906 | 119/117/2500 |
| drone→quad | joint3_F | lm_F | +0.041603 | 22/30/2684 |
| drone→quad | joint3_F | binding_direct | +0.049621 | 55/46/2635 |
| quad→waymo | joint3_FC | prepared_frozen | -0.148068 | 97/113/3572 |
| quad→waymo | joint3_FC | lm_FC | -0.018184 | 19/22/3741 |
| quad→waymo | joint3_FC | direct_FC | +0.049246 | 43/36/3703 |
| quad→waymo | joint3_FC | direct_four | +0.099012 | 69/61/3652 |
| quad→waymo | joint3_FC | joint3_shuffle_FC | -0.106637 | 100/125/3557 |
| quad→waymo | joint3_C | lm_C | -0.086537 | 18/32/3732 |
| quad→waymo | joint3_C | frozen_C | -0.232695 | 56/91/3635 |
| quad→waymo | joint3_F | lm_F | -0.033733 | 14/23/3745 |
| quad→waymo | joint3_F | binding_direct | +0.032858 | 48/47/3687 |
| quad→drone | joint3_FC | prepared_frozen | -0.099384 | 63/97/2632 |
| quad→drone | joint3_FC | lm_FC | +0.042621 | 12/14/2766 |
| quad→drone | joint3_FC | direct_FC | -0.057488 | 85/90/2617 |
| quad→drone | joint3_FC | direct_four | +0.012131 | 92/90/2610 |
| quad→drone | joint3_FC | joint3_shuffle_FC | -0.008815 | 126/142/2524 |
| quad→drone | joint3_C | lm_C | -0.013634 | 11/16/2765 |
| quad→drone | joint3_C | frozen_C | -0.033006 | 65/83/2644 |
| quad→drone | joint3_F | lm_F | -0.010620 | 16/16/2760 |
| quad→drone | joint3_F | binding_direct | -0.155782 | 81/93/2618 |

## 辅助诊断与范围

对R99全部768输入的CPU头重放显示：三视图拟合减小A/B/C分数目标误差，但7条改善和3条恶化的C拟合误差都下降；22次FC选框变化中18次没有改变F或C各自Top1。CPU/GPU概率最大差约1.21e-6且无Top1差，数值差与初始过严格阈值失败均保留，不冒称逐位一致。全部正式成绩仍来自原CUDA封存预测。

另对R98全部18620条初始误差分解：源Top10承载38.68%概率，仅占7.50%等权分数误差；这不是参数梯度份额的结论。按相同头Jacobian实际投影，分数L2与分布KL初始参数梯度在768开发输入平均余弦.5578。这里只建立度量差异，KL未在本轮执行或选优，不能声称其必然更好。相关诊断与已实际读取的Hinton原文sec2.1见diagnostics；没有引入外部teacher。

本轮按hash复用R98固定G缓存（真实/错配三视图分数）和原768预测，再重算其余输入的三视图头拟合。所有新视图正确绑定基础分数与原缓存核对，source权重及原始框均固定。执行耗时包含缓存复用与多个控制，不是完整端到端部署延迟；原始G缓存生成也有成本。

## 判断

{
  "primary": "joint3_FC",
  "primary_mIoU_percent": 12.181928845294244,
  "gains_pp": {
    "original_source": 0.4278036946910202,
    "prepared_frozen": -0.013291846618342262,
    "lm_FC": 0.0026931102955032316,
    "direct_FC": 0.018949434201813276,
    "direct_four": 0.017271600683011812,
    "joint3_shuffle_FC": 0.02300890327546945
  },
  "heldout_C_gain_vs_oldC_pp": 0.010196617016015708,
  "F_gain_vs_direct_pp": -0.012183238345919278,
  "passes_main_comparisons": false,
  "n": 18620,
  "unchanged_R99_method": true,
  "FINAL_access": false,
  "expert_added": false,
  "source_retraining": false,
  "historical_exposure": true,
  "robust_adoption": false
}

候选方法必须超过同信息直接读出及错配控制，不能只超过裸source。F和C是预设诊断端点，不在看到全量标签后替换主端点。优化下降与非零参数不是任务成功。若开发优势未保持，降低该配方优先级；不要无依据增加步数或按少数案例补门。下一步按夜间状态和剩余时间选择最小区分实验，保持无外部专家、无源训练、FINAL受保护；2026-11-16北京时间09:30停止新增模型、训练和评分。

固定扩展裁决：开发768条相对旧LM FC为+0.205804pp，扩大17852条为-0.006418pp，完整矩阵相对自身冻结-0.013292pp。开发优势未保持，因此当前三视图L2配方不采纳为稳定TTA。保留全部有益与有害变化；不通过改主端点、删Quad方向或增加步数挽救。下一项若执行，应单独锁定并比较候选分布度量，与本固定矩阵区分。
