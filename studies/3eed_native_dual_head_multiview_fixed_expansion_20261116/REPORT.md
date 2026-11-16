# R109 — 原生双头与真实点观测的固定配置扩展验证

先验证监督来源，不新增适应损失、参数更新、BN校准、模型或source训练。固定R106实测点特征对齐和R108独立soft-token头的等权读出；保持同一A256、坐标、文字视角、RGB与R91双向结构。主读出仍是几何池，不按结果换成算术池。

## 清单与观测边界

{
  "groups": [
    {
      "target": "waymo",
      "expressions": 256,
      "distinct_anchor_metapaths": 256,
      "scene_clusters": 89,
      "protected_alias_fallback": 0,
      "current_DEV_anchor_metapath_overlap": 0
    },
    {
      "target": "drone",
      "expressions": 256,
      "distinct_anchor_metapaths": 256,
      "scene_clusters": 2,
      "protected_alias_fallback": 26,
      "current_DEV_anchor_metapath_overlap": 0
    },
    {
      "target": "quad",
      "expressions": 256,
      "distinct_anchor_metapaths": 256,
      "scene_clusters": 3,
      "protected_alias_fallback": 29,
      "current_DEV_anchor_metapath_overlap": 0
    }
  ],
  "unique_target_expressions": 768,
  "directed_inputs": 1536,
  "no_model_GT_or_media_decode": true,
  "historically_exposed_full_VAL_not_independent_test": true
}

768新增表达/物理路径，两个incoming sources形成1536定向输入，排除当前DEV锚点路径。所有拒绝/未知/保护别名均留在分母。全VAL有历史模型和标签暴露，这不是全新独立测试；Waymo/Drone/Quad场景分别89/2/3，不按帧IID给总体CI。相邻观测由固定计数器顺序检索，时间单位未知，不宣称因果在线视频。

## 传感器对应

{
  "state": "R109_SENSOR_ONLY_REGISTRATION_PREREQUISITE_COMPLETE",
  "n": 1396,
  "groups": [
    {
      "platform": "waymo",
      "kind": "same_scene_distinct_sensor_candidate",
      "attempts": 482,
      "forward_numeric_supported": 123,
      "bidirectional_cycle_supported": 104,
      "statuses": {
        "forward_sensor_geometry_unsupported": 359,
        "mask_free_sensor_geometry_and_cycle_supported": 104,
        "reverse_sensor_geometry_unsupported": 17,
        "sensor_cycle_unsupported": 2
      }
    },
    {
      "platform": "waymo",
      "kind": "wrong_scene_control",
      "attempts": 0,
      "forward_numeric_supported": 0,
      "bidirectional_cycle_supported": 0,
      "statuses": {}
    },
    {
      "platform": "drone",
      "kind": "same_scene_distinct_sensor_candidate",
      "attempts": 460,
      "forward_numeric_supported": 144,
      "bidirectional_cycle_supported": 109,
      "statuses": {
        "mask_free_sensor_geometry_and_cycle_supported": 109,
        "forward_sensor_geometry_unsupported": 316,
        "reverse_sensor_geometry_unsupported": 35
      }
    },
    {
      "platform": "drone",
      "kind": "wrong_scene_control",
      "attempts": 0,
      "forward_numeric_supported": 0,
      "bidirectional_cycle_supported": 0,
      "statuses": {}
    },
    {
      "platform": "quad",
      "kind": "same_scene_distinct_sensor_candidate",
      "attempts": 454,
      "forward_numeric_supported": 311,
      "bidirectional_cycle_supported": 276,
      "statuses": {
        "mask_free_sensor_geometry_and_cycle_supported": 276,
        "forward_sensor_geometry_unsupported": 143,
        "reverse_sensor_geometry_unsupported": 34,
        "sensor_cycle_unsupported": 1
      }
    },
    {
      "platform": "quad",
      "kind": "wrong_scene_control",
      "attempts": 0,
      "forward_numeric_supported": 0,
      "bidirectional_cycle_supported": 0,
      "statuses": {}
    }
  ],
  "seconds": 1124.6625166680024,
  "model_or_GT_used": false,
  "foreground_masks_used": false,
  "old_selected_pose_cache_used": false,
  "protected_media_decoded": false,
  "sensor_support_not_identity_truth": true
}

复用原SIFT+独立实测深度PnP数值验证、反向注册与留出循环阈值；无实例mask/外部缓存/提供位姿。先锁定B/C，再检查几何，不在失败后搜索更有利邻帧。已注册FINAL路径不进入frame白名单，受保护JPEG内容别名不解码。初次注册启动缺少父目录，在读取任何传感器前失败；保留attempt001脚本/日志，仅修parents=True后重启，未重跑任何有效注册。

## 全体结果

| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |
|---|---|---:|---:|---:|
| all_six_equal | original_source | 16.144619 | 30.598958 | 12.434896 |
| all_six_equal | prepared_frozen | 16.251067 | 30.598958 | 11.783854 |
| all_six_equal | frozen_four | 16.047076 | 30.273438 | 11.718750 |
| all_six_equal | direct_FC | 16.232646 | 30.403646 | 11.914062 |
| all_six_equal | joint3_FC | 16.197053 | 30.338542 | 11.914062 |
| all_six_equal | real_BBF_B | 16.063307 | 29.947917 | 11.783854 |
| all_six_equal | real_BBF_C | 15.948717 | 30.078125 | 11.848958 |
| all_six_equal | real_BBF_ensemble | 16.120167 | 30.533854 | 11.783854 |
| all_six_equal | real_BBF_fusedABC | 16.238436 | 30.598958 | 12.044271 |
| all_six_equal | real_softtoken_fusedABC | 15.703252 | 29.492188 | 12.174479 |
| all_six_equal | real_softtoken_geometric_fusedABC | 16.032048 | 30.338542 | 11.979167 |
| all_six_equal | real_softtoken_arithmetic_fusedABC | 16.013494 | 30.338542 | 12.044271 |
| all_six_equal | real_softtoken_shuffled_geometric_fusedABC | 7.175905 | 13.085938 | 2.278646 |
| all_six_equal | randomA_BBF_B | 15.906004 | 30.143229 | 11.653646 |
| all_six_equal | randomA_BBF_C | 15.808070 | 29.492188 | 11.458333 |
| all_six_equal | randomA_BBF_ensemble | 16.172387 | 30.338542 | 11.914062 |
| all_six_equal | randomA_BBF_fusedABC | 16.134486 | 30.208333 | 11.848958 |
| all_six_equal | randomA_softtoken_fusedABC | 15.727445 | 29.361979 | 12.044271 |
| all_six_equal | randomA_softtoken_geometric_fusedABC | 16.068931 | 30.273438 | 11.914062 |
| all_six_equal | randomA_softtoken_arithmetic_fusedABC | 16.048101 | 30.208333 | 11.914062 |
| all_six_equal | randomA_softtoken_shuffled_geometric_fusedABC | 6.996777 | 12.890625 | 2.213542 |
| all_six_equal | wrong_BBF_B | 11.231992 | 20.507812 | 7.291667 |
| all_six_equal | wrong_BBF_C | 12.131712 | 22.265625 | 8.138021 |
| all_six_equal | wrong_BBF_ensemble | 14.600495 | 27.148438 | 10.611979 |
| all_six_equal | wrong_BBF_fusedABC | 13.627054 | 25.195312 | 9.505208 |
| all_six_equal | wrong_softtoken_fusedABC | 13.065188 | 24.348958 | 9.375000 |
| all_six_equal | wrong_softtoken_geometric_fusedABC | 13.375032 | 24.804688 | 9.440104 |
| all_six_equal | wrong_softtoken_arithmetic_fusedABC | 13.285601 | 24.609375 | 9.375000 |
| all_six_equal | wrong_softtoken_shuffled_geometric_fusedABC | 6.211258 | 11.328125 | 2.018229 |
| all_six_equal | softtoken_A | 15.600549 | 29.296875 | 11.718750 |
| all_six_equal | softtoken_geometric_A | 16.123040 | 30.403646 | 11.718750 |
| all_six_equal | softtoken_arithmetic_A | 16.073164 | 30.273438 | 11.848958 |
| all_six_equal | softtoken_shuffled_geometric_A | 7.038492 | 12.890625 | 2.213542 |
| waymo | original_source | 13.450978 | 28.906250 | 6.835938 |
| waymo | prepared_frozen | 13.214120 | 28.125000 | 6.250000 |
| waymo | frozen_four | 12.817701 | 27.734375 | 5.859375 |
| waymo | direct_FC | 12.982180 | 27.539062 | 5.664062 |
| waymo | joint3_FC | 13.021302 | 27.539062 | 5.664062 |
| waymo | real_BBF_B | 12.670460 | 26.562500 | 5.664062 |
| waymo | real_BBF_C | 12.913095 | 27.343750 | 6.445312 |
| waymo | real_BBF_ensemble | 13.155356 | 27.929688 | 5.859375 |
| waymo | real_BBF_fusedABC | 13.107853 | 27.734375 | 5.859375 |
| waymo | real_softtoken_fusedABC | 13.080012 | 27.734375 | 6.250000 |
| waymo | real_softtoken_geometric_fusedABC | 13.201998 | 27.734375 | 5.859375 |
| waymo | real_softtoken_arithmetic_fusedABC | 13.249430 | 27.929688 | 6.250000 |
| waymo | real_softtoken_shuffled_geometric_fusedABC | 6.578084 | 14.062500 | 1.367188 |
| waymo | randomA_BBF_B | 12.895446 | 26.953125 | 6.054688 |
| waymo | randomA_BBF_C | 12.781869 | 26.953125 | 6.250000 |
| waymo | randomA_BBF_ensemble | 13.293512 | 27.929688 | 6.445312 |
| waymo | randomA_BBF_fusedABC | 13.283757 | 27.929688 | 6.445312 |
| waymo | randomA_softtoken_fusedABC | 13.397425 | 28.125000 | 6.640625 |
| waymo | randomA_softtoken_geometric_fusedABC | 13.529579 | 28.125000 | 6.640625 |
| waymo | randomA_softtoken_arithmetic_fusedABC | 13.488205 | 28.320312 | 6.835938 |
| waymo | randomA_softtoken_shuffled_geometric_fusedABC | 6.611454 | 14.257812 | 1.367188 |
| waymo | wrong_BBF_B | 11.261328 | 22.656250 | 5.273438 |
| waymo | wrong_BBF_C | 11.531860 | 24.218750 | 5.664062 |
| waymo | wrong_BBF_ensemble | 12.481031 | 26.171875 | 6.054688 |
| waymo | wrong_BBF_fusedABC | 12.337976 | 25.585938 | 5.859375 |
| waymo | wrong_softtoken_fusedABC | 12.353298 | 25.781250 | 6.054688 |
| waymo | wrong_softtoken_geometric_fusedABC | 12.437713 | 25.585938 | 6.054688 |
| waymo | wrong_softtoken_arithmetic_fusedABC | 12.371061 | 25.585938 | 6.250000 |
| waymo | wrong_softtoken_shuffled_geometric_fusedABC | 6.454900 | 13.281250 | 1.367188 |
| waymo | softtoken_A | 13.287277 | 28.320312 | 6.250000 |
| waymo | softtoken_geometric_A | 13.459483 | 28.125000 | 6.250000 |
| waymo | softtoken_arithmetic_A | 13.397022 | 28.320312 | 6.640625 |
| waymo | softtoken_shuffled_geometric_A | 6.771857 | 14.453125 | 1.367188 |
| drone | original_source | 11.525665 | 20.898438 | 9.375000 |
| drone | prepared_frozen | 12.009923 | 20.312500 | 9.179688 |
| drone | frozen_four | 11.675251 | 19.726562 | 8.984375 |
| drone | direct_FC | 12.040575 | 19.531250 | 9.179688 |
| drone | joint3_FC | 11.874050 | 19.921875 | 9.179688 |
| drone | real_BBF_B | 11.753108 | 19.726562 | 8.984375 |
| drone | real_BBF_C | 11.851897 | 20.117188 | 9.179688 |
| drone | real_BBF_ensemble | 11.913191 | 20.507812 | 9.375000 |
| drone | real_BBF_fusedABC | 11.874628 | 20.312500 | 9.375000 |
| drone | real_softtoken_fusedABC | 11.063831 | 18.750000 | 9.570312 |
| drone | real_softtoken_geometric_fusedABC | 11.313362 | 19.531250 | 9.375000 |
| drone | real_softtoken_arithmetic_fusedABC | 11.522835 | 20.117188 | 9.375000 |
| drone | real_softtoken_shuffled_geometric_fusedABC | 7.236741 | 12.890625 | 2.343750 |
| drone | randomA_BBF_B | 12.028368 | 20.898438 | 9.179688 |
| drone | randomA_BBF_C | 12.025900 | 20.117188 | 9.179688 |
| drone | randomA_BBF_ensemble | 12.012382 | 20.312500 | 9.375000 |
| drone | randomA_BBF_fusedABC | 11.959218 | 20.117188 | 9.375000 |
| drone | randomA_softtoken_fusedABC | 11.090544 | 18.945312 | 9.375000 |
| drone | randomA_softtoken_geometric_fusedABC | 11.625713 | 19.921875 | 9.375000 |
| drone | randomA_softtoken_arithmetic_fusedABC | 11.693570 | 20.117188 | 9.179688 |
| drone | randomA_softtoken_shuffled_geometric_fusedABC | 7.118758 | 12.890625 | 2.343750 |
| drone | wrong_BBF_B | 11.011581 | 19.335938 | 8.007812 |
| drone | wrong_BBF_C | 11.229303 | 18.554688 | 8.398438 |
| drone | wrong_BBF_ensemble | 11.563621 | 19.531250 | 8.789062 |
| drone | wrong_BBF_fusedABC | 11.555298 | 19.335938 | 8.593750 |
| drone | wrong_softtoken_fusedABC | 10.769619 | 18.359375 | 8.789062 |
| drone | wrong_softtoken_geometric_fusedABC | 11.093736 | 18.554688 | 8.593750 |
| drone | wrong_softtoken_arithmetic_fusedABC | 11.291691 | 18.945312 | 8.593750 |
| drone | wrong_softtoken_shuffled_geometric_fusedABC | 6.411923 | 11.914062 | 1.757812 |
| drone | softtoken_A | 11.191942 | 19.140625 | 9.375000 |
| drone | softtoken_geometric_A | 11.616715 | 20.312500 | 9.179688 |
| drone | softtoken_arithmetic_A | 11.546192 | 19.921875 | 9.179688 |
| drone | softtoken_shuffled_geometric_A | 7.093355 | 12.500000 | 2.343750 |
| quad | original_source | 23.457215 | 41.992188 | 21.093750 |
| quad | prepared_frozen | 23.529157 | 43.359375 | 19.921875 |
| quad | frozen_four | 23.648277 | 43.359375 | 20.312500 |
| quad | direct_FC | 23.675184 | 44.140625 | 20.898438 |
| quad | joint3_FC | 23.695807 | 43.554688 | 20.898438 |
| quad | real_BBF_B | 23.766355 | 43.554688 | 20.703125 |
| quad | real_BBF_C | 23.081160 | 42.773438 | 19.921875 |
| quad | real_BBF_ensemble | 23.291955 | 43.164062 | 20.117188 |
| quad | real_BBF_fusedABC | 23.732826 | 43.750000 | 20.898438 |
| quad | real_softtoken_fusedABC | 22.965914 | 41.992188 | 20.703125 |
| quad | real_softtoken_geometric_fusedABC | 23.580783 | 43.750000 | 20.703125 |
| quad | real_softtoken_arithmetic_fusedABC | 23.268216 | 42.968750 | 20.507812 |
| quad | real_softtoken_shuffled_geometric_fusedABC | 7.712889 | 12.304688 | 3.125000 |
| quad | randomA_BBF_B | 22.794199 | 42.578125 | 19.726562 |
| quad | randomA_BBF_C | 22.616441 | 41.406250 | 18.945312 |
| quad | randomA_BBF_ensemble | 23.211265 | 42.773438 | 19.921875 |
| quad | randomA_BBF_fusedABC | 23.160482 | 42.578125 | 19.726562 |
| quad | randomA_softtoken_fusedABC | 22.694367 | 41.015625 | 20.117188 |
| quad | randomA_softtoken_geometric_fusedABC | 23.051502 | 42.773438 | 19.726562 |
| quad | randomA_softtoken_arithmetic_fusedABC | 22.962527 | 42.187500 | 19.726562 |
| quad | randomA_softtoken_shuffled_geometric_fusedABC | 7.260118 | 11.523438 | 2.929688 |
| quad | wrong_BBF_B | 11.423068 | 19.531250 | 8.593750 |
| quad | wrong_BBF_C | 13.633973 | 24.023438 | 10.351562 |
| quad | wrong_BBF_ensemble | 19.756834 | 35.742188 | 16.992188 |
| quad | wrong_BBF_fusedABC | 16.987888 | 30.664062 | 14.062500 |
| quad | wrong_softtoken_fusedABC | 16.072647 | 28.906250 | 13.281250 |
| quad | wrong_softtoken_geometric_fusedABC | 16.593648 | 30.273438 | 13.671875 |
| quad | wrong_softtoken_arithmetic_fusedABC | 16.194051 | 29.296875 | 13.281250 |
| quad | wrong_softtoken_shuffled_geometric_fusedABC | 5.766950 | 8.789062 | 2.929688 |
| quad | softtoken_A | 22.322428 | 40.429688 | 19.531250 |
| quad | softtoken_geometric_A | 23.292923 | 42.773438 | 19.726562 |
| quad | softtoken_arithmetic_A | 23.276278 | 42.578125 | 19.726562 |
| quad | softtoken_shuffled_geometric_A | 7.250265 | 11.718750 | 2.929688 |

## 方向与所有负比较

| Source | Target | 方法 | 对照 | Δ pp | 改善/恶化/不变 |
|---|---|---|---|---:|---:|
| waymo | drone | real_softtoken_geometric_fusedABC | original_source | -0.618313 | 25/38/193 |
| waymo | drone | real_softtoken_geometric_fusedABC | prepared_frozen | -0.469463 | 13/20/223 |
| waymo | drone | real_softtoken_geometric_fusedABC | real_BBF_fusedABC | -0.670371 | 10/16/230 |
| waymo | drone | real_softtoken_geometric_fusedABC | real_softtoken_fusedABC | +0.480125 | 23/16/217 |
| waymo | drone | real_softtoken_geometric_fusedABC | real_softtoken_arithmetic_fusedABC | -0.338793 | 5/5/246 |
| waymo | drone | real_softtoken_geometric_fusedABC | randomA_softtoken_geometric_fusedABC | -0.170409 | 4/8/244 |
| waymo | drone | real_softtoken_geometric_fusedABC | wrong_softtoken_geometric_fusedABC | +0.269109 | 10/14/232 |
| waymo | drone | real_softtoken_geometric_fusedABC | real_softtoken_shuffled_geometric_fusedABC | +4.844645 | 58/40/158 |
| waymo | drone | real_softtoken_geometric_fusedABC | real_BBF_ensemble | -0.686473 | 10/16/230 |
| waymo | drone | softtoken_geometric_A | prepared_frozen | -0.321097 | 16/15/225 |
| waymo | drone | real_BBF_fusedABC | randomA_BBF_fusedABC | +0.055369 | 7/7/242 |
| waymo | drone | real_softtoken_arithmetic_fusedABC | real_BBF_fusedABC | -0.331578 | 7/14/235 |
| waymo | quad | real_softtoken_geometric_fusedABC | original_source | +2.427431 | 40/24/192 |
| waymo | quad | real_softtoken_geometric_fusedABC | prepared_frozen | +0.646940 | 19/20/217 |
| waymo | quad | real_softtoken_geometric_fusedABC | real_BBF_fusedABC | +0.574272 | 16/8/232 |
| waymo | quad | real_softtoken_geometric_fusedABC | real_softtoken_fusedABC | +0.335787 | 12/9/235 |
| waymo | quad | real_softtoken_geometric_fusedABC | real_softtoken_arithmetic_fusedABC | +0.119884 | 4/4/248 |
| waymo | quad | real_softtoken_geometric_fusedABC | randomA_softtoken_geometric_fusedABC | +1.452776 | 20/13/223 |
| waymo | quad | real_softtoken_geometric_fusedABC | wrong_softtoken_geometric_fusedABC | +9.292861 | 68/13/175 |
| waymo | quad | real_softtoken_geometric_fusedABC | real_softtoken_shuffled_geometric_fusedABC | +23.303131 | 138/16/102 |
| waymo | quad | real_softtoken_geometric_fusedABC | real_BBF_ensemble | +0.805019 | 17/11/228 |
| waymo | quad | softtoken_geometric_A | prepared_frozen | -0.016764 | 11/11/234 |
| waymo | quad | real_BBF_fusedABC | randomA_BBF_fusedABC | +0.843208 | 18/17/221 |
| waymo | quad | real_softtoken_arithmetic_fusedABC | real_BBF_fusedABC | +0.454388 | 15/8/233 |
| drone | waymo | real_softtoken_geometric_fusedABC | original_source | -0.361978 | 44/58/154 |
| drone | waymo | real_softtoken_geometric_fusedABC | prepared_frozen | +0.019657 | 32/32/192 |
| drone | waymo | real_softtoken_geometric_fusedABC | real_BBF_fusedABC | +0.019278 | 29/30/197 |
| drone | waymo | real_softtoken_geometric_fusedABC | real_softtoken_fusedABC | -0.190629 | 20/22/214 |
| drone | waymo | real_softtoken_geometric_fusedABC | real_softtoken_arithmetic_fusedABC | +0.006163 | 12/11/233 |
| drone | waymo | real_softtoken_geometric_fusedABC | randomA_softtoken_geometric_fusedABC | -0.274664 | 3/9/244 |
| drone | waymo | real_softtoken_geometric_fusedABC | wrong_softtoken_geometric_fusedABC | +1.015731 | 18/8/230 |
| drone | waymo | real_softtoken_geometric_fusedABC | real_softtoken_shuffled_geometric_fusedABC | +4.898059 | 91/48/117 |
| drone | waymo | real_softtoken_geometric_fusedABC | real_BBF_ensemble | -0.086575 | 31/32/193 |
| drone | waymo | softtoken_geometric_A | prepared_frozen | +0.048926 | 29/30/197 |
| drone | waymo | real_BBF_fusedABC | randomA_BBF_fusedABC | -0.341578 | 3/7/246 |
| drone | waymo | real_softtoken_arithmetic_fusedABC | real_BBF_fusedABC | +0.013115 | 30/36/190 |
| drone | quad | real_softtoken_geometric_fusedABC | original_source | -2.180295 | 47/53/156 |
| drone | quad | real_softtoken_geometric_fusedABC | prepared_frozen | -0.543687 | 25/25/206 |
| drone | quad | real_softtoken_geometric_fusedABC | real_BBF_fusedABC | -0.878358 | 13/20/223 |
| drone | quad | real_softtoken_geometric_fusedABC | real_softtoken_fusedABC | +0.893952 | 21/9/226 |
| drone | quad | real_softtoken_geometric_fusedABC | real_softtoken_arithmetic_fusedABC | +0.505250 | 15/6/235 |
| drone | quad | real_softtoken_geometric_fusedABC | randomA_softtoken_geometric_fusedABC | -0.394214 | 13/15/228 |
| drone | quad | real_softtoken_geometric_fusedABC | wrong_softtoken_geometric_fusedABC | +4.681408 | 64/32/160 |
| drone | quad | real_softtoken_geometric_fusedABC | real_softtoken_shuffled_geometric_fusedABC | +8.432657 | 96/37/123 |
| drone | quad | real_softtoken_geometric_fusedABC | real_BBF_ensemble | -0.227363 | 16/16/224 |
| drone | quad | softtoken_geometric_A | prepared_frozen | -0.455703 | 14/20/222 |
| drone | quad | real_BBF_fusedABC | randomA_BBF_fusedABC | +0.301480 | 16/12/228 |
| drone | quad | real_softtoken_arithmetic_fusedABC | real_BBF_fusedABC | -1.383608 | 11/26/219 |
| quad | waymo | real_softtoken_geometric_fusedABC | original_source | -0.135981 | 32/46/178 |
| quad | waymo | real_softtoken_geometric_fusedABC | prepared_frozen | -0.043901 | 11/12/233 |
| quad | waymo | real_softtoken_geometric_fusedABC | real_BBF_fusedABC | +0.169013 | 6/8/242 |
| quad | waymo | real_softtoken_geometric_fusedABC | real_softtoken_fusedABC | +0.434600 | 6/3/247 |
| quad | waymo | real_softtoken_geometric_fusedABC | real_softtoken_arithmetic_fusedABC | -0.101026 | 0/3/253 |
| quad | waymo | real_softtoken_geometric_fusedABC | randomA_softtoken_geometric_fusedABC | -0.380497 | 5/7/244 |
| quad | waymo | real_softtoken_geometric_fusedABC | wrong_softtoken_geometric_fusedABC | +0.512840 | 8/4/244 |
| quad | waymo | real_softtoken_geometric_fusedABC | real_softtoken_shuffled_geometric_fusedABC | +8.349769 | 79/27/150 |
| quad | waymo | real_softtoken_geometric_fusedABC | real_BBF_ensemble | +0.179859 | 7/8/241 |
| quad | waymo | softtoken_geometric_A | prepared_frozen | +0.441798 | 7/7/242 |
| quad | waymo | real_BBF_fusedABC | randomA_BBF_fusedABC | -0.010231 | 5/7/244 |
| quad | waymo | real_softtoken_arithmetic_fusedABC | real_BBF_fusedABC | +0.270039 | 8/7/241 |
| quad | drone | real_softtoken_geometric_fusedABC | original_source | +0.193706 | 54/37/165 |
| quad | drone | real_softtoken_geometric_fusedABC | prepared_frozen | -0.923659 | 8/18/230 |
| quad | drone | real_softtoken_geometric_fusedABC | real_BBF_fusedABC | -0.452162 | 5/12/239 |
| quad | drone | real_softtoken_geometric_fusedABC | real_softtoken_fusedABC | +0.018937 | 8/9/239 |
| quad | drone | real_softtoken_geometric_fusedABC | real_softtoken_arithmetic_fusedABC | -0.080154 | 3/3/250 |
| quad | drone | real_softtoken_geometric_fusedABC | randomA_softtoken_geometric_fusedABC | -0.454293 | 4/10/242 |
| quad | drone | real_softtoken_geometric_fusedABC | wrong_softtoken_geometric_fusedABC | +0.170143 | 17/10/229 |
| quad | drone | real_softtoken_geometric_fusedABC | real_softtoken_shuffled_geometric_fusedABC | +3.308595 | 66/41/149 |
| quad | drone | real_softtoken_geometric_fusedABC | real_BBF_ensemble | -0.513186 | 5/15/236 |
| quad | drone | softtoken_geometric_A | prepared_frozen | -0.465319 | 7/11/238 |
| quad | drone | real_BBF_fusedABC | randomA_BBF_fusedABC | -0.224549 | 5/11/240 |
| quad | drone | real_softtoken_arithmetic_fusedABC | real_BBF_fusedABC | -0.372008 | 8/14/234 |

## 开发与扩展分别报告

[
  {
    "arm": "original_source",
    "DEV768_mIoU_percent": 16.811382998717253,
    "expansion1536_mIoU_percent": 16.144619260124063,
    "combined2304_mIoU_percent": 16.366873839655128,
    "combined_not_independent_confirmation": true
  },
  {
    "arm": "prepared_frozen",
    "DEV768_mIoU_percent": 16.22865215918868,
    "expansion1536_mIoU_percent": 16.251066577152425,
    "combined2304_mIoU_percent": 16.243595104497842,
    "combined_not_independent_confirmation": true
  },
  {
    "arm": "real_BBF_fusedABC",
    "DEV768_mIoU_percent": 16.703230292958846,
    "expansion1536_mIoU_percent": 16.238435653542936,
    "combined2304_mIoU_percent": 16.393367200014907,
    "combined_not_independent_confirmation": true
  },
  {
    "arm": "real_softtoken_fusedABC",
    "DEV768_mIoU_percent": 16.159324747746535,
    "expansion1536_mIoU_percent": 15.703252252078862,
    "combined2304_mIoU_percent": 15.85527641730142,
    "combined_not_independent_confirmation": true
  },
  {
    "arm": "real_softtoken_geometric_fusedABC",
    "DEV768_mIoU_percent": 16.88434822839348,
    "expansion1536_mIoU_percent": 16.032047615059994,
    "combined2304_mIoU_percent": 16.31614781950449,
    "combined_not_independent_confirmation": true
  },
  {
    "arm": "real_softtoken_arithmetic_fusedABC",
    "DEV768_mIoU_percent": 16.90719476802655,
    "expansion1536_mIoU_percent": 16.01349360259532,
    "combined2304_mIoU_percent": 16.311393991072396,
    "combined_not_independent_confirmation": true
  },
  {
    "arm": "randomA_BBF_fusedABC",
    "DEV768_mIoU_percent": 16.560112223081862,
    "expansion1536_mIoU_percent": 16.134485783426644,
    "combined2304_mIoU_percent": 16.276361263311717,
    "combined_not_independent_confirmation": true
  }
]

## 裁决

{
  "primary": "real_softtoken_geometric_fusedABC",
  "primary_mIoU_percent": 16.032047615059994,
  "gains_pp": {
    "original_source": -0.1125716450640688,
    "prepared_frozen": -0.21901896209243077,
    "real_BBF_fusedABC": -0.20638803848294174,
    "real_softtoken_fusedABC": 0.32879536298113266,
    "real_softtoken_arithmetic_fusedABC": 0.01855401246467281,
    "randomA_softtoken_geometric_fusedABC": -0.03688353814053258,
    "wrong_softtoken_geometric_fusedABC": 2.657015238801497,
    "real_softtoken_shuffled_geometric_fusedABC": 8.856142937019769,
    "real_BBF_ensemble": -0.08811980041779321
  },
  "native_head_complementarity_survives_expansion": false,
  "passes_strict_all_controls_gate": false,
  "robust_TTA_adoption": false,
  "TTA_run": false,
  "decision": "stop_this_native_head_signal_recipe_no_output_fitting",
  "primary_not_switched": true,
  "n": 1536,
  "FINAL_access": false,
  "historical_full_VAL_exposure": true
}

## R108既有偏好的分量归因（不改变R109方法）

{
  "source_study": "R108 sealed predictions and already exposed postseal IoUs",
  "n": 768,
  "protected_alias_fallback": 44,
  "max_replay_probability_error": 2.7755575615628914e-16,
  "source_strict_runtime_restored": true,
  "failed_attempt001_retained": true,
  "new_predictions_or_labels_created": false,
  "postseal_descriptive_not_gate_or_selector": true,
  "identity_or_background_truth_inferred": false,
  "nonnull_meaning": "one minus sourceCE no-object255; unmatched-source-query mass, NOT verified foreground/object identity",
  "groups": [
    {
      "arm": "softtoken_fusedABC",
      "group": "all",
      "n": 724,
      "nonnull_dominates_n": 97,
      "nonnull_supports_new_n": 130,
      "conditional_supports_new_n": 215,
      "mean_abs_nonnull_margin": 0.2816170433905508,
      "mean_abs_conditional_subject_margin": 0.3681347883827635
    },
    {
      "arm": "softtoken_fusedABC",
      "group": "changed",
      "n": 255,
      "nonnull_dominates_n": 97,
      "nonnull_supports_new_n": 130,
      "conditional_supports_new_n": 215,
      "mean_abs_nonnull_margin": 0.7995715271167012,
      "mean_abs_conditional_subject_margin": 1.0452140658396893
    },
    {
      "arm": "softtoken_fusedABC",
      "group": "improved",
      "n": 65,
      "nonnull_dominates_n": 31,
      "nonnull_supports_new_n": 36,
      "conditional_supports_new_n": 52,
      "mean_abs_nonnull_margin": 0.3979473562356814,
      "mean_abs_conditional_subject_margin": 0.5134449440441939
    },
    {
      "arm": "softtoken_fusedABC",
      "group": "harmed",
      "n": 79,
      "nonnull_dominates_n": 26,
      "nonnull_supports_new_n": 35,
      "conditional_supports_new_n": 66,
      "mean_abs_nonnull_margin": 1.0571807147491896,
      "mean_abs_conditional_subject_margin": 1.3500399647180326
    },
    {
      "arm": "softtoken_fusedABC",
      "group": "same_IoU_changed_query",
      "n": 111,
      "nonnull_dominates_n": 40,
      "nonnull_supports_new_n": 59,
      "conditional_supports_new_n": 97,
      "mean_abs_nonnull_margin": 0.8514133765248065,
      "mean_abs_conditional_subject_margin": 1.13966223615787
    },
    {
      "arm": "softtoken_geometric_fusedABC",
      "group": "all",
      "n": 724,
      "nonnull_dominates_n": 59,
      "nonnull_supports_new_n": 95,
      "conditional_supports_new_n": 135,
      "mean_abs_nonnull_margin": 0.06334251699947041,
      "mean_abs_conditional_subject_margin": 0.10206083362265687
    },
    {
      "arm": "softtoken_geometric_fusedABC",
      "group": "changed",
      "n": 152,
      "nonnull_dominates_n": 59,
      "nonnull_supports_new_n": 95,
      "conditional_supports_new_n": 135,
      "mean_abs_nonnull_margin": 0.30171040991853015,
      "mean_abs_conditional_subject_margin": 0.48613186541318143
    },
    {
      "arm": "softtoken_geometric_fusedABC",
      "group": "improved",
      "n": 46,
      "nonnull_dominates_n": 19,
      "nonnull_supports_new_n": 27,
      "conditional_supports_new_n": 41,
      "mean_abs_nonnull_margin": 0.22359368124806886,
      "mean_abs_conditional_subject_margin": 0.40776320544796113
    },
    {
      "arm": "softtoken_geometric_fusedABC",
      "group": "harmed",
      "n": 46,
      "nonnull_dominates_n": 17,
      "nonnull_supports_new_n": 30,
      "conditional_supports_new_n": 40,
      "mean_abs_nonnull_margin": 0.29843937815444826,
      "mean_abs_conditional_subject_margin": 0.39945831011621613
    },
    {
      "arm": "softtoken_geometric_fusedABC",
      "group": "same_IoU_changed_query",
      "n": 60,
      "nonnull_dominates_n": 23,
      "nonnull_supports_new_n": 38,
      "conditional_supports_new_n": 54,
      "mean_abs_nonnull_margin": 0.3641076929183466,
      "mean_abs_conditional_subject_margin": 0.6126642304475235
    },
    {
      "arm": "softtoken_arithmetic_fusedABC",
      "group": "all",
      "n": 724,
      "nonnull_dominates_n": 75,
      "nonnull_supports_new_n": 110,
      "conditional_supports_new_n": 135,
      "mean_abs_nonnull_margin": 0.09088812953583147,
      "mean_abs_conditional_subject_margin": 0.130689593776395
    },
    {
      "arm": "softtoken_arithmetic_fusedABC",
      "group": "changed",
      "n": 159,
      "nonnull_dominates_n": 75,
      "nonnull_supports_new_n": 110,
      "conditional_supports_new_n": 135,
      "mean_abs_nonnull_margin": 0.4138553822889433,
      "mean_abs_conditional_subject_margin": 0.5950897226044652
    },
    {
      "arm": "softtoken_arithmetic_fusedABC",
      "group": "improved",
      "n": 48,
      "nonnull_dominates_n": 23,
      "nonnull_supports_new_n": 31,
      "conditional_supports_new_n": 40,
      "mean_abs_nonnull_margin": 0.1937176721133934,
      "mean_abs_conditional_subject_margin": 0.322146801566014
    },
    {
      "arm": "softtoken_arithmetic_fusedABC",
      "group": "harmed",
      "n": 44,
      "nonnull_dominates_n": 21,
      "nonnull_supports_new_n": 32,
      "conditional_supports_new_n": 37,
      "mean_abs_nonnull_margin": 0.38428916734720175,
      "mean_abs_conditional_subject_margin": 0.43076689198497947
    },
    {
      "arm": "softtoken_arithmetic_fusedABC",
      "group": "same_IoU_changed_query",
      "n": 67,
      "nonnull_dominates_n": 31,
      "nonnull_supports_new_n": 47,
      "conditional_supports_new_n": 58,
      "mean_abs_nonnull_margin": 0.5909825993913764,
      "mean_abs_conditional_subject_margin": 0.8985444204716746
    }
  ]
}

## 主张边界

source CE头与contrastive头共享decoder和训练来源，不能称独立专家/物体身份真值。CE原生256token概率分母、当前纯文本主体mask、source eval BN/dropout保持，绝未使用标注positive_map或GT跨度。任何头之间的GT最优选择仅属诊断，不能部署。稳定静态信号不等于参数TTA；本轮没有更新权重，不能宣称已解决适应。所有负例保留；若信号不复现，不以继续调融合系数或拟合其输出来挽救。

## 实际看图与读出接口的竞争解释

{
  "generated_pages": 12,
  "pages_actually_viewed": 4,
  "viewed_aliases": [
    "CASE0001",
    "CASE0002",
    "CASE0007",
    "CASE0008"
  ],
  "not_identity_gold_annotation": true,
  "unviewed_remain_UNKNOWN": true,
  "observed_types": [
    "overlapping candidate boxes with uncertain identity",
    "visible vehicle versus elevated background selection",
    "partly out-of-view target and low A256 oracle coverage",
    "foreground person versus different person-region selection"
  ],
  "no_case_specific_filter_rule_or_policy_selection": true
}

4/12页实际查看，其余保持UNKNOWN。正例既有明显不同可见区域的选择变化，也有相互重叠框的小幅定位差异，不能全部称为实例身份修复。负例包括目标车辆转到不相交背景区域，以及丢掉冻结融合已经救回的近处行人；这些不应再一概归为query重复。采用仍由总体结果决定，图页不筛评价分母。

{
  "n": 1536,
  "postseal_descriptive": true,
  "no_new_full_ranking_or_GT_score": true,
  "interpretation": "existing probability pool averages full BBF(base+r) with independent CE, so common r enters at half strength. Restoring coefficient1 is a competing readout hypothesis, NOT a demonstrated bug, correctness label or newmethod result. Pair reversals do not establish globalTop1.",
  "groups": [
    {
      "group": "all_changed",
      "n": 316,
      "pair_might_revert_if_full_common_residual": 28,
      "mean_abs_current_margin": 0.15421143763388492,
      "mean_abs_competition_margin": 0.050203644434768185
    },
    {
      "group": "improved",
      "n": 79,
      "pair_might_revert_if_full_common_residual": 3,
      "mean_abs_current_margin": 0.15946754468242058,
      "mean_abs_competition_margin": 0.04811203885304777
    },
    {
      "group": "harmed",
      "n": 94,
      "pair_might_revert_if_full_common_residual": 8,
      "mean_abs_current_margin": 0.14817969933628883,
      "mean_abs_competition_margin": 0.04509029769774606
    },
    {
      "group": "same_IoU_changed",
      "n": 143,
      "pair_might_revert_if_full_common_residual": 17,
      "mean_abs_current_margin": 0.15527263304045635,
      "mean_abs_competition_margin": 0.0547203677511038
    }
  ]
}

几何池将包含竞争残差的完整BBF概率与CE概率平均，确实使残差的有效系数减半。但在94条IoU恶化中，额外补回半份残差仅能反转8条当前新旧候选对的相对次序，86条该候选对仍不反转；另外3条正例候选对也可能被反转。这不是全候选重评分结果，但足以撤下“单纯恢复残差就能解释大部分退化”的优先级，不据此重启残差强度扫描。
