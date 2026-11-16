# R105 — 实际额外观测下的原生双向融合与匹配适应

本轮改变证据获取：同一A坐标、原A256、完整文字与原source准备参数不变，按固定传感器顺序引入真实B/C图像。没有新的预训练网络；source CNN对每张图分别编码，经过已训练的稀疏RGB↔LiDAR交互读取。额外点支持由传感器PnP/反向循环、实测深度及邻近回波约束，失败保持UNKNOWN。

这是新的离线多观测setting。计数器不被当作秒，未证明连续视频或因果在线；不是新场景测试，也不能将收益回写为旧单观测R95有效。当前开发数据多轮暴露，FINAL没有访问。

## 预检与实施验证

{
  "n": 692,
  "groups": [
    {
      "platform": "waymo",
      "attempts": 224,
      "geometry_cycle_supported": 40,
      "statuses": {
        "mask_free_sensor_geometry_and_cycle_supported": 40,
        "forward_sensor_geometry_unsupported": 175,
        "reverse_sensor_geometry_unsupported": 9
      }
    },
    {
      "platform": "drone",
      "attempts": 236,
      "geometry_cycle_supported": 50,
      "statuses": {
        "mask_free_sensor_geometry_and_cycle_supported": 50,
        "forward_sensor_geometry_unsupported": 169,
        "reverse_sensor_geometry_unsupported": 16,
        "sensor_cycle_unsupported": 1
      }
    },
    {
      "platform": "quad",
      "attempts": 232,
      "geometry_cycle_supported": 153,
      "statuses": {
        "forward_sensor_geometry_unsupported": 55,
        "mask_free_sensor_geometry_and_cycle_supported": 153,
        "reverse_sensor_geometry_unsupported": 24
      }
    }
  ],
  "reused_exact_probe": 47,
  "seconds": 500.550127488008,
  "no_model_or_GT": true
}

原24条错场景预检全部被拒绝，但不能据此认证正确实例或校准错误率。保护帧以及保护JPEG内容别名在解码前排除。对内容别名anchor仅复用既有冻结预测；所有768条仍留在分母。

{
  "groups": [
    {
      "target": "waymo",
      "n": 256,
      "attempted_fit": 42,
      "real_C_available": 42,
      "TTA_final_enabled": 34,
      "protected_alias_fallback": 0,
      "mean_seconds": 0.18924048734822918
    },
    {
      "target": "drone",
      "n": 256,
      "attempted_fit": 56,
      "real_C_available": 44,
      "TTA_final_enabled": 20,
      "protected_alias_fallback": 20,
      "mean_seconds": 0.19294773269962207
    },
    {
      "target": "quad",
      "n": 256,
      "attempted_fit": 154,
      "real_C_available": 146,
      "TTA_final_enabled": 118,
      "protected_alias_fallback": 24,
      "mean_seconds": 0.3360988985851918
    }
  ],
  "all_768_retained": true,
  "attempted_fit_cost_kept_even_without_C": true
}

工程验证包括原始A稀疏图逐项一致、额外但未使用RGB bank不影响输出、重复A bank数值等价、真实B图像的原生前向、实际写参数后的全模型重放与显式Jacobian。首个工程attempt因在no_grad中调用导数检查失败，完整保存；检查器显式enable_grad后重试，不涉及方法或标签选择。

## 固定方法

q=(p_A+p_AB)/2来自同一冻结模型，停止梯度。相同q监督A和A+B，只有末端18496参数query投影更新；LM仍为10次、每次最多6次阻尼验收。每个视图使用自己的原生竞争残差，该残差不依赖更新的投影，冻结它不偷换任务分数。C的网络预测在全部拟合结束后才计算，不用于目标、验收或停止。B不可用不拟合；C不可用则记录已发生拟合与成本，但最终回到同信息冻结读出。

主端点为更新后jointABC，必须与冻结jointABC及冻结概率集成比较；只超过原A或历史单观测结果不够。额外绑定打乱只改变新增图像的可见点归属，保留A图和可见性；另有q中B候选身份打乱控制。历史R100方法仅作上下文，输入信息不同。

## 全部输入结果

| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |
|---|---|---:|---:|---:|
| all_six_equal | original_source | 16.811383 | 30.989583 | 14.973958 |
| all_six_equal | prepared_frozen | 16.228652 | 29.947917 | 12.630208 |
| all_six_equal | frozen_four | 16.249883 | 29.947917 | 12.890625 |
| all_six_equal | direct_FC | 16.241990 | 29.817708 | 12.760417 |
| all_six_equal | joint3_FC | 16.517235 | 30.338542 | 13.020833 |
| all_six_equal | frozen_A_AB | 16.217763 | 29.947917 | 12.630208 |
| all_six_equal | frozen_AC | 16.228435 | 29.947917 | 12.630208 |
| all_six_equal | frozen_ensemble3 | 16.217763 | 29.947917 | 12.630208 |
| all_six_equal | frozen_jointABC | 16.217545 | 29.947917 | 12.630208 |
| all_six_equal | TTA_A | 16.217785 | 29.947917 | 12.630208 |
| all_six_equal | TTA_AB | 16.217763 | 29.947917 | 12.630208 |
| all_six_equal | TTA_AC | 16.217567 | 29.947917 | 12.630208 |
| all_six_equal | TTA_jointABC | 16.217243 | 29.947917 | 12.630208 |
| all_six_equal | TTA_ensemble3 | 16.217763 | 29.947917 | 12.630208 |
| all_six_equal | teacher_shuffle_AC | 16.196922 | 29.817708 | 12.630208 |
| all_six_equal | teacher_shuffle_jointABC | 16.191243 | 29.817708 | 12.630208 |
| all_six_equal | wrong_frozen_AC | 16.275559 | 30.208333 | 12.630208 |
| all_six_equal | wrong_frozen_jointABC | 16.298726 | 30.338542 | 12.630208 |
| all_six_equal | wrong_frozen_ensemble3 | 16.232104 | 30.078125 | 12.630208 |
| all_six_equal | wrong_TTA_AC | 16.214801 | 30.078125 | 12.630208 |
| all_six_equal | wrong_TTA_jointABC | 16.246236 | 30.208333 | 12.630208 |
| all_six_equal | wrong_TTA_ensemble3 | 16.232104 | 30.078125 | 12.630208 |
| waymo | original_source | 11.473631 | 23.437500 | 7.421875 |
| waymo | prepared_frozen | 12.615516 | 26.171875 | 5.078125 |
| waymo | frozen_four | 12.546748 | 25.781250 | 5.468750 |
| waymo | direct_FC | 12.755842 | 25.390625 | 5.468750 |
| waymo | joint3_FC | 13.067711 | 26.953125 | 5.859375 |
| waymo | frozen_A_AB | 12.615516 | 26.171875 | 5.078125 |
| waymo | frozen_AC | 12.615516 | 26.171875 | 5.078125 |
| waymo | frozen_ensemble3 | 12.615516 | 26.171875 | 5.078125 |
| waymo | frozen_jointABC | 12.615516 | 26.171875 | 5.078125 |
| waymo | TTA_A | 12.615516 | 26.171875 | 5.078125 |
| waymo | TTA_AB | 12.615516 | 26.171875 | 5.078125 |
| waymo | TTA_AC | 12.615516 | 26.171875 | 5.078125 |
| waymo | TTA_jointABC | 12.615516 | 26.171875 | 5.078125 |
| waymo | TTA_ensemble3 | 12.615516 | 26.171875 | 5.078125 |
| waymo | teacher_shuffle_AC | 12.416371 | 25.390625 | 5.078125 |
| waymo | teacher_shuffle_jointABC | 12.381250 | 25.390625 | 5.078125 |
| waymo | wrong_frozen_AC | 12.615516 | 26.171875 | 5.078125 |
| waymo | wrong_frozen_jointABC | 12.615516 | 26.171875 | 5.078125 |
| waymo | wrong_frozen_ensemble3 | 12.615516 | 26.171875 | 5.078125 |
| waymo | wrong_TTA_AC | 12.615516 | 26.171875 | 5.078125 |
| waymo | wrong_TTA_jointABC | 12.615516 | 26.171875 | 5.078125 |
| waymo | wrong_TTA_ensemble3 | 12.615516 | 26.171875 | 5.078125 |
| drone | original_source | 13.867693 | 24.609375 | 12.500000 |
| drone | prepared_frozen | 14.071678 | 24.609375 | 11.718750 |
| drone | frozen_four | 13.894458 | 24.609375 | 11.718750 |
| drone | direct_FC | 13.844147 | 25.000000 | 11.718750 |
| drone | joint3_FC | 14.238577 | 24.609375 | 12.109375 |
| drone | frozen_A_AB | 14.039010 | 24.609375 | 11.718750 |
| drone | frozen_AC | 14.071678 | 24.609375 | 11.718750 |
| drone | frozen_ensemble3 | 14.039010 | 24.609375 | 11.718750 |
| drone | frozen_jointABC | 14.039010 | 24.609375 | 11.718750 |
| drone | TTA_A | 14.039076 | 24.609375 | 11.718750 |
| drone | TTA_AB | 14.039010 | 24.609375 | 11.718750 |
| drone | TTA_AC | 14.039076 | 24.609375 | 11.718750 |
| drone | TTA_jointABC | 14.039010 | 24.609375 | 11.718750 |
| drone | TTA_ensemble3 | 14.039010 | 24.609375 | 11.718750 |
| drone | teacher_shuffle_AC | 13.988365 | 24.609375 | 11.718750 |
| drone | teacher_shuffle_jointABC | 14.018657 | 24.609375 | 11.718750 |
| drone | wrong_frozen_AC | 14.213959 | 25.390625 | 11.718750 |
| drone | wrong_frozen_jointABC | 14.181357 | 25.390625 | 11.718750 |
| drone | wrong_frozen_ensemble3 | 14.097585 | 25.000000 | 11.718750 |
| drone | wrong_TTA_AC | 14.192807 | 25.390625 | 11.718750 |
| drone | wrong_TTA_jointABC | 14.185011 | 25.390625 | 11.718750 |
| drone | wrong_TTA_ensemble3 | 14.097585 | 25.000000 | 11.718750 |
| quad | original_source | 25.092825 | 44.921875 | 25.000000 |
| quad | prepared_frozen | 21.998762 | 39.062500 | 21.093750 |
| quad | frozen_four | 22.308444 | 39.453125 | 21.484375 |
| quad | direct_FC | 22.125981 | 39.062500 | 21.093750 |
| quad | joint3_FC | 22.245417 | 39.453125 | 21.093750 |
| quad | frozen_A_AB | 21.998762 | 39.062500 | 21.093750 |
| quad | frozen_AC | 21.998110 | 39.062500 | 21.093750 |
| quad | frozen_ensemble3 | 21.998762 | 39.062500 | 21.093750 |
| quad | frozen_jointABC | 21.998110 | 39.062500 | 21.093750 |
| quad | TTA_A | 21.998762 | 39.062500 | 21.093750 |
| quad | TTA_AB | 21.998762 | 39.062500 | 21.093750 |
| quad | TTA_AC | 21.998110 | 39.062500 | 21.093750 |
| quad | TTA_jointABC | 21.997203 | 39.062500 | 21.093750 |
| quad | TTA_ensemble3 | 21.998762 | 39.062500 | 21.093750 |
| quad | teacher_shuffle_AC | 22.186030 | 39.453125 | 21.093750 |
| quad | teacher_shuffle_jointABC | 22.173823 | 39.453125 | 21.093750 |
| quad | wrong_frozen_AC | 21.997203 | 39.062500 | 21.093750 |
| quad | wrong_frozen_jointABC | 22.099305 | 39.453125 | 21.093750 |
| quad | wrong_frozen_ensemble3 | 21.983210 | 39.062500 | 21.093750 |
| quad | wrong_TTA_AC | 21.836079 | 38.671875 | 21.093750 |
| quad | wrong_TTA_jointABC | 21.938181 | 39.062500 | 21.093750 |
| quad | wrong_TTA_ensemble3 | 21.983210 | 39.062500 | 21.093750 |

## 所有正负变化与集中度

[
  {
    "control": "prepared_frozen",
    "better": 0,
    "worse": 4,
    "same": 764,
    "net_pp": -0.01140928506958555,
    "largest_positive_single_input_pp": 0.0,
    "largest_negative_single_input_pp": -0.010867489225137741
  },
  {
    "control": "frozen_jointABC",
    "better": 0,
    "worse": 1,
    "same": 767,
    "net_pp": -0.0003022424819331933,
    "largest_positive_single_input_pp": 0.0,
    "largest_negative_single_input_pp": -0.0003022424819331933
  },
  {
    "control": "frozen_ensemble3",
    "better": 0,
    "worse": 2,
    "same": 766,
    "net_pp": -0.0005198377225874584,
    "largest_positive_single_input_pp": 0.0,
    "largest_negative_single_input_pp": -0.0003022424819331933
  },
  {
    "control": "teacher_shuffle_jointABC",
    "better": 11,
    "worse": 12,
    "same": 745,
    "net_pp": 0.025999634044771073,
    "largest_positive_single_input_pp": 0.06367550154503418,
    "largest_negative_single_input_pp": -0.06277016420032257
  },
  {
    "control": "wrong_TTA_jointABC",
    "better": 4,
    "worse": 5,
    "same": 759,
    "net_pp": -0.028993256863800287,
    "largest_positive_single_input_pp": 0.053708010201132854,
    "largest_negative_single_input_pp": -0.03475282577478824
  },
  {
    "control": "joint3_FC",
    "better": 19,
    "worse": 25,
    "same": 724,
    "net_pp": -0.2999921755291776,
    "largest_positive_single_input_pp": 0.05280641669786784,
    "largest_negative_single_input_pp": -0.08842370123172709
  }
]

## 裁决

{
  "primary": "TTA_jointABC",
  "mIoU_percent": 16.217242874119098,
  "gains_pp": {
    "prepared_frozen": -0.011409285069582609,
    "frozen_jointABC": -0.0003022424819327796,
    "frozen_ensemble3": -0.0005198377225852369,
    "teacher_shuffle_jointABC": 0.02599963404477279,
    "wrong_TTA_jointABC": -0.028993256863799388,
    "joint3_FC": -0.299992175529173
  },
  "passes_development_gates": false,
  "robust_adoption": false,
  "decision": "do_not_adopt_this_multiview_TTA_recipe",
  "new_setting": "actual multiple RGB observations on canonical A LiDAR; offline/exposed scenes",
  "all_inputs": 768,
  "FINAL_access": false,
  "no_experts": true,
  "source_training": false,
  "primary_not_switched": true
}

辅助拟合下降不能证明实体身份正确；几何对应通过也不是语言判断真值。good/failure按每方向、每对照、符号分别取两个极端和中位用于事后机制诊断，不改变评价分母。场景聚类结果完整保留，不把共享场景/描述视为独立样本，不为单场景给总体CI。若未过同信息与错配控制，不加步幅/层位置网格挽救。文献借鉴与边界见LITERATURE.md。


## 事后观测与候选支持诊断

{
  "sensor": [
    {
      "platform": "waymo",
      "supported_pairs": 40,
      "same_pointcloud_bytes": 0,
      "estimated_translation_m": {
        "n": 40,
        "mean": 0.14008931035597222,
        "median": 0.0186114508877839,
        "p90": 0.3221902130632488,
        "max": 1.3078972450873632
      },
      "estimated_rotation_degrees": {
        "n": 40,
        "mean": 0.11316547723065418,
        "median": 0.05784566156218454,
        "p90": 0.2640653540824034,
        "max": 0.5726896263595292
      },
      "resized_RGB_mean_absolute_difference": {
        "n": 40,
        "mean": 0.03510853187710631,
        "median": 0.02892378929653458,
        "p90": 0.07136939753114789,
        "max": 0.13870737911049835
      },
      "not_an_identity_or_camera_motion_certificate": true
    },
    {
      "platform": "drone",
      "supported_pairs": 50,
      "same_pointcloud_bytes": 0,
      "estimated_translation_m": {
        "n": 50,
        "mean": 0.12695955536063644,
        "median": 0.11449827888079755,
        "p90": 0.19922233232326897,
        "max": 0.30962046876413274
      },
      "estimated_rotation_degrees": {
        "n": 50,
        "mean": 0.7565444683218066,
        "median": 0.5585145331889589,
        "p90": 1.5391089852790025,
        "max": 2.0330790638847804
      },
      "resized_RGB_mean_absolute_difference": {
        "n": 50,
        "mean": 0.0782907398365162,
        "median": 0.0849384848069002,
        "p90": 0.10488038306142772,
        "max": 0.13902964207601443
      },
      "not_an_identity_or_camera_motion_certificate": true
    },
    {
      "platform": "quad",
      "supported_pairs": 153,
      "same_pointcloud_bytes": 0,
      "estimated_translation_m": {
        "n": 153,
        "mean": 0.060372749547808156,
        "median": 0.028858593951191117,
        "p90": 0.14431354986853398,
        "max": 0.40378807574952374
      },
      "estimated_rotation_degrees": {
        "n": 153,
        "mean": 0.5977853723683646,
        "median": 0.4712525659413382,
        "p90": 1.2987521989883737,
        "max": 3.0018630485155136
      },
      "resized_RGB_mean_absolute_difference": {
        "n": 153,
        "mean": 0.05006174895826102,
        "median": 0.05214226749727669,
        "p90": 0.0911943556176811,
        "max": 0.13828240687551063
      },
      "not_an_identity_or_camera_motion_certificate": true
    }
  ],
  "directions": [
    {
      "direction": "waymo_to_drone",
      "n": 128,
      "TTA_output_enabled": 10,
      "query_switches_vs_prepared": 2,
      "positive_IoU_changes": 0,
      "negative_IoU_changes": 2,
      "B_mapped_queries_on_attempted": {
        "n": 28,
        "mean": 52.535714285714285,
        "median": 47.5,
        "p90": 95.3,
        "max": 136.0
      },
      "C_mapped_queries_on_enabled": {
        "n": 10,
        "mean": 43.8,
        "median": 34.0,
        "p90": 81.1,
        "max": 109.0
      }
    },
    {
      "direction": "waymo_to_quad",
      "n": 128,
      "TTA_output_enabled": 59,
      "query_switches_vs_prepared": 1,
      "positive_IoU_changes": 0,
      "negative_IoU_changes": 1,
      "B_mapped_queries_on_attempted": {
        "n": 77,
        "mean": 73.88311688311688,
        "median": 46.0,
        "p90": 173.0,
        "max": 195.0
      },
      "C_mapped_queries_on_enabled": {
        "n": 59,
        "mean": 83.40677966101696,
        "median": 60.0,
        "p90": 172.0,
        "max": 188.0
      }
    },
    {
      "direction": "drone_to_waymo",
      "n": 128,
      "TTA_output_enabled": 17,
      "query_switches_vs_prepared": 0,
      "positive_IoU_changes": 0,
      "negative_IoU_changes": 0,
      "B_mapped_queries_on_attempted": {
        "n": 21,
        "mean": 101.71428571428571,
        "median": 103.0,
        "p90": 156.0,
        "max": 167.0
      },
      "C_mapped_queries_on_enabled": {
        "n": 17,
        "mean": 107.17647058823529,
        "median": 125.0,
        "p90": 162.4,
        "max": 170.0
      }
    },
    {
      "direction": "drone_to_quad",
      "n": 128,
      "TTA_output_enabled": 59,
      "query_switches_vs_prepared": 2,
      "positive_IoU_changes": 0,
      "negative_IoU_changes": 1,
      "B_mapped_queries_on_attempted": {
        "n": 77,
        "mean": 68.85714285714286,
        "median": 42.0,
        "p90": 156.8,
        "max": 205.0
      },
      "C_mapped_queries_on_enabled": {
        "n": 59,
        "mean": 80.84745762711864,
        "median": 46.0,
        "p90": 160.4,
        "max": 190.0
      }
    },
    {
      "direction": "quad_to_waymo",
      "n": 128,
      "TTA_output_enabled": 17,
      "query_switches_vs_prepared": 0,
      "positive_IoU_changes": 0,
      "negative_IoU_changes": 0,
      "B_mapped_queries_on_attempted": {
        "n": 21,
        "mean": 99.76190476190476,
        "median": 105.0,
        "p90": 145.0,
        "max": 160.0
      },
      "C_mapped_queries_on_enabled": {
        "n": 17,
        "mean": 102.58823529411765,
        "median": 122.0,
        "p90": 150.2,
        "max": 169.0
      }
    },
    {
      "direction": "quad_to_drone",
      "n": 128,
      "TTA_output_enabled": 10,
      "query_switches_vs_prepared": 0,
      "positive_IoU_changes": 0,
      "negative_IoU_changes": 0,
      "B_mapped_queries_on_attempted": {
        "n": 28,
        "mean": 49.285714285714285,
        "median": 42.5,
        "p90": 89.80000000000001,
        "max": 122.0
      },
      "C_mapped_queries_on_enabled": {
        "n": 10,
        "mean": 43.9,
        "median": 35.0,
        "p90": 82.3,
        "max": 103.0
      }
    }
  ]
}

以上仅解释已封存结果。真实图像内容不同不等于有独立主体信息；变换由传感器估计，不认证相机运动或物体身份。启用子集单列用于机制理解，仍以全部768条作为主分母，未据此重新选择方法或步长。

## 新观测与实际看图复核

{
  "atlas_pages_generated": 5,
  "pages_actually_viewed": 2,
  "same_information_TTA_vs_frozen_jointABC_good_IoU_cases": 0,
  "all_Top1_changes_vs_prepared": 5,
  "before_after_AABB_pair_IoU_min": 0.5202667713165283,
  "before_after_AABB_pair_IoU_max": 0.7113606929779053,
  "same_instance_not_certified_by_pair_IoU": true,
  "review": "Two viewed pages are consistent with overlapping candidate changes on the same visible object/area. Others remain visually UNKNOWN; no claim of complete identity audit.",
  "actual_viewpoint_note": "Sensor estimates have sub-degree median rotation and small median translation; distinct image bytes do not ensure new discriminating target evidence. Not all variation is JPEG compression: resized RGB differences are nonzero, but target-specific information is unproven."
}

{
  "sensor": [
    {
      "platform": "waymo",
      "supported_pairs": 40,
      "same_pointcloud_bytes": 0,
      "estimated_translation_m": {
        "n": 40,
        "mean": 0.14008931035597222,
        "median": 0.0186114508877839,
        "p90": 0.3221902130632488,
        "max": 1.3078972450873632
      },
      "estimated_rotation_degrees": {
        "n": 40,
        "mean": 0.11316547723065418,
        "median": 0.05784566156218454,
        "p90": 0.2640653540824034,
        "max": 0.5726896263595292
      },
      "resized_RGB_mean_absolute_difference": {
        "n": 40,
        "mean": 0.03510853187710631,
        "median": 0.02892378929653458,
        "p90": 0.07136939753114789,
        "max": 0.13870737911049835
      },
      "not_an_identity_or_camera_motion_certificate": true
    },
    {
      "platform": "drone",
      "supported_pairs": 50,
      "same_pointcloud_bytes": 0,
      "estimated_translation_m": {
        "n": 50,
        "mean": 0.12695955536063644,
        "median": 0.11449827888079755,
        "p90": 0.19922233232326897,
        "max": 0.30962046876413274
      },
      "estimated_rotation_degrees": {
        "n": 50,
        "mean": 0.7565444683218066,
        "median": 0.5585145331889589,
        "p90": 1.5391089852790025,
        "max": 2.0330790638847804
      },
      "resized_RGB_mean_absolute_difference": {
        "n": 50,
        "mean": 0.0782907398365162,
        "median": 0.0849384848069002,
        "p90": 0.10488038306142772,
        "max": 0.13902964207601443
      },
      "not_an_identity_or_camera_motion_certificate": true
    },
    {
      "platform": "quad",
      "supported_pairs": 153,
      "same_pointcloud_bytes": 0,
      "estimated_translation_m": {
        "n": 153,
        "mean": 0.060372749547808156,
        "median": 0.028858593951191117,
        "p90": 0.14431354986853398,
        "max": 0.40378807574952374
      },
      "estimated_rotation_degrees": {
        "n": 153,
        "mean": 0.5977853723683646,
        "median": 0.4712525659413382,
        "p90": 1.2987521989883737,
        "max": 3.0018630485155136
      },
      "resized_RGB_mean_absolute_difference": {
        "n": 153,
        "mean": 0.05006174895826102,
        "median": 0.05214226749727669,
        "p90": 0.0911943556176811,
        "max": 0.13828240687551063
      },
      "not_an_identity_or_camera_motion_certificate": true
    }
  ],
  "directions": [
    {
      "direction": "waymo_to_drone",
      "n": 128,
      "TTA_output_enabled": 10,
      "query_switches_vs_prepared": 2,
      "positive_IoU_changes": 0,
      "negative_IoU_changes": 2,
      "B_mapped_queries_on_attempted": {
        "n": 28,
        "mean": 52.535714285714285,
        "median": 47.5,
        "p90": 95.3,
        "max": 136.0
      },
      "C_mapped_queries_on_enabled": {
        "n": 10,
        "mean": 43.8,
        "median": 34.0,
        "p90": 81.1,
        "max": 109.0
      }
    },
    {
      "direction": "waymo_to_quad",
      "n": 128,
      "TTA_output_enabled": 59,
      "query_switches_vs_prepared": 1,
      "positive_IoU_changes": 0,
      "negative_IoU_changes": 1,
      "B_mapped_queries_on_attempted": {
        "n": 77,
        "mean": 73.88311688311688,
        "median": 46.0,
        "p90": 173.0,
        "max": 195.0
      },
      "C_mapped_queries_on_enabled": {
        "n": 59,
        "mean": 83.40677966101696,
        "median": 60.0,
        "p90": 172.0,
        "max": 188.0
      }
    },
    {
      "direction": "drone_to_waymo",
      "n": 128,
      "TTA_output_enabled": 17,
      "query_switches_vs_prepared": 0,
      "positive_IoU_changes": 0,
      "negative_IoU_changes": 0,
      "B_mapped_queries_on_attempted": {
        "n": 21,
        "mean": 101.71428571428571,
        "median": 103.0,
        "p90": 156.0,
        "max": 167.0
      },
      "C_mapped_queries_on_enabled": {
        "n": 17,
        "mean": 107.17647058823529,
        "median": 125.0,
        "p90": 162.4,
        "max": 170.0
      }
    },
    {
      "direction": "drone_to_quad",
      "n": 128,
      "TTA_output_enabled": 59,
      "query_switches_vs_prepared": 2,
      "positive_IoU_changes": 0,
      "negative_IoU_changes": 1,
      "B_mapped_queries_on_attempted": {
        "n": 77,
        "mean": 68.85714285714286,
        "median": 42.0,
        "p90": 156.8,
        "max": 205.0
      },
      "C_mapped_queries_on_enabled": {
        "n": 59,
        "mean": 80.84745762711864,
        "median": 46.0,
        "p90": 160.4,
        "max": 190.0
      }
    },
    {
      "direction": "quad_to_waymo",
      "n": 128,
      "TTA_output_enabled": 17,
      "query_switches_vs_prepared": 0,
      "positive_IoU_changes": 0,
      "negative_IoU_changes": 0,
      "B_mapped_queries_on_attempted": {
        "n": 21,
        "mean": 99.76190476190476,
        "median": 105.0,
        "p90": 145.0,
        "max": 160.0
      },
      "C_mapped_queries_on_enabled": {
        "n": 17,
        "mean": 102.58823529411765,
        "median": 122.0,
        "p90": 150.2,
        "max": 169.0
      }
    },
    {
      "direction": "quad_to_drone",
      "n": 128,
      "TTA_output_enabled": 10,
      "query_switches_vs_prepared": 0,
      "positive_IoU_changes": 0,
      "negative_IoU_changes": 0,
      "B_mapped_queries_on_attempted": {
        "n": 28,
        "mean": 49.285714285714285,
        "median": 42.5,
        "p90": 89.80000000000001,
        "max": 122.0
      },
      "C_mapped_queries_on_enabled": {
        "n": 10,
        "mean": 43.9,
        "median": 35.0,
        "p90": 82.3,
        "max": 103.0
      }
    }
  ],
  "descriptive_only_no_method_change": true
}

这一轮没有相对自身冻结模型的good IoU案例，不能将与较弱历史或打乱对照的差异包装成有效纠错。新增观测仅使5条Top1变化，其中4条IoU下降、1条仍为零；5对所选框的AABB重叠均高于0.52，说明不能直接把query变化解释成换了物体。只实际查看2页，其余保持未知。真实额外RGB接入工程可用，但这份原生读取没有产生总体有益的排序信息，参数适应也没有额外收益；不再对本配方增加更新强度。
