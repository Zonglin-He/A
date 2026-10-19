# Strong基底上448与224 Action条件OPD：HC128

固定B=HiResAuxMix同时作为教师基底和学生零分布；Native单列。两Action分别完整运行TeacherDirect、DirectJoint、DirectConditional、freshFixed与Live；另同448 Wrong/Shift Live及NoAction exact。使用原130时间临时bias/40步/.1/M32/41,42,43，构造的双边界条件标签不称原生网络概率。0新模型/GPU/图像/专家/网络训练/空间update。旧448证据部署成本不能说免费；旧R24/R25关闭保持原判定。

| 比较 | 完整v增量pp [CI] |
|---|---|
| Mean_H_Live_minus_Mean_Native | +0.809812 [-0.8937073641596207, 2.580800386304195] |
| Mean_H_Live_minus_Mean_B | -0.796622 [-1.8659635513670112, 0.3354312206200941] |
| Mean_H_Live_minus_Mean_H_DirectJoint | -0.103448 [-0.24948387187834672, 0.023670542482945856] |
| Mean_H_Live_minus_Mean_H_DirectConditional | -0.103727 [-0.24820760075831774, 0.020062678742521784] |
| Mean_H_Live_minus_Mean_H_Fixed | +0.799105 [0.04026366093058353, 1.5962276206485817] |
| Mean_H_Live_minus_Mean_TeacherH | -0.082089 [-0.2545699869078486, 0.08143681149256318] |
| Mean_H_Live_minus_Mean_N_Live | +0.203308 [-1.387159449885007, 2.01391981127649] |
| Mean_H_Live_minus_Mean_WrongH_Live | +8.021706 [5.158945471109238, 11.132538331247735] |
| Mean_H_Live_minus_Mean_ShiftH_Live | +16.811183 [12.92811693921123, 20.690593576443764] |
| Mean_TeacherH_minus_Mean_B | -0.714533 [-1.7764130902030422, 0.41784894134318684] |
| Mean_TeacherN_minus_Mean_B | -1.093734 [-2.166210975607122, -0.12209389361736522] |
| Mean_TeacherWrongH_minus_Mean_B | -10.204815 [-13.669436678691268, -6.978515366419695] |
| Mean_TeacherShiftH_minus_Mean_B | -20.961591 [-25.0582340244982, -17.039528388382717] |
| Mean_H_DirectJoint_minus_Mean_B | -0.693174 [-1.7646961680691782, 0.43518195716796776] |
| Mean_H_DirectConditional_minus_Mean_B | -0.692895 [-1.7654722357776769, 0.43649520263908076] |
| Mean_H_Fixed_minus_Mean_B | -1.595727 [-3.1823222595781764, -0.0017364619584194938] |
| Mean_N_Live_minus_Mean_B | -0.999930 [-2.0318343291199614, -0.0713169983037066] |
| Mean_TeacherH_minus_Mean_TeacherN | +0.379201 [-1.2560776259989463, 2.22451065272995] |
| All9_H_Live_minus_Mean_Native | +0.809467 [-0.8940718696690902, 2.5804664281198897] |
| All9_H_Live_minus_All9_H_Fixed | +0.798962 [0.04021436468070277, 1.5960943284926914] |

| 教师质量比较 | 期望候选GTv增量pp [CI] |
|---|---|
| TeacherH_minus_Native | +0.682472 [-0.6566652135895154, 2.0509117505539978] |
| TeacherH_minus_B | +0.872707 [0.20547380169235133, 1.5674997341973993] |
| TeacherH_minus_TeacherN | -0.274813 [-1.5177010567541394, 1.0347327072359205] |
| TeacherH_minus_TeacherWrongH | +7.721098 [5.1498098231822045, 10.52840580828466] |
| TeacherH_minus_TeacherShiftH | +17.038094 [13.670166051412314, 20.49502344461709] |

三时间/空间同seed配对后父内均值为主，全9仅预登记敏感性。RC128反复曝光、全128完整预测先seal后GT，无负例/无参考/零时间样本筛除。130bias不保证完整表达目标或保留节点数边际，Direct40步不是最优投影。固定空间使全部时间臂固定GT支持sIoU及未观测实际节点sIoU相同，不虚构空间收益。区分B已有收益、448证据及Live相对同信息直接/Fixed价值。

```json
{
  "primary": "Mean_H_Live (matching seeds within parent)",
  "net_vs_Native": {
    "mean": 0.8098120887804128,
    "CI95": [
      -0.8937073641596207,
      2.580800386304195
    ],
    "parents": 128
  },
  "increment_vs_B": {
    "mean": -0.7966220184633046,
    "CI95": [
      -1.8659635513670112,
      0.3354312206200941
    ],
    "parents": 128
  },
  "versus_controls": {
    "B": {
      "mean": -0.7966220184633046,
      "CI95": [
        -1.8659635513670112,
        0.3354312206200941
      ],
      "parents": 128
    },
    "H_DirectJoint": {
      "mean": -0.10344824050870262,
      "CI95": [
        -0.24948387187834672,
        0.023670542482945856
      ],
      "parents": 128
    },
    "H_DirectConditional": {
      "mean": -0.10372702166363743,
      "CI95": [
        -0.24820760075831774,
        0.020062678742521784
      ],
      "parents": 128
    },
    "H_Fixed": {
      "mean": 0.7991053461625559,
      "CI95": [
        0.04026366093058353,
        1.5962276206485817
      ],
      "parents": 128
    },
    "TeacherH": {
      "mean": -0.08208905891058282,
      "CI95": [
        -0.2545699869078486,
        0.08143681149256318
      ],
      "parents": 128
    },
    "N_Live": {
      "mean": 0.2033084759710777,
      "CI95": [
        -1.387159449885007,
        2.01391981127649
      ],
      "parents": 128
    },
    "WrongH_Live": {
      "mean": 8.021705905340777,
      "CI95": [
        5.158945471109238,
        11.132538331247735
      ],
      "parents": 128
    },
    "ShiftH_Live": {
      "mean": 16.811182501327337,
      "CI95": [
        12.92811693921123,
        20.690593576443764
      ],
      "parents": 128
    }
  },
  "decision": "no_supported_large_independent_OPD_advantage_this_configuration",
  "no_parameter_or_base_search": true,
  "not_independent_confirmation": true,
  "goal_achieved": false
}
{
  "CPU_prediction_and_updates_seconds": 97.12221101787873,
  "fits": 2944,
  "fit_attempts": 2944,
  "actual_updates": 112640,
  "completed_fit_updates": 112640,
  "completed_trace_steps": 112640,
  "nominal_including_noop_steps": 117760,
  "scheduled_nontrivial_updates": 112640,
  "NoAction_calls": 128,
  "temporary_parameters": 130,
  "first_Fixed_Live_exact": 768,
  "max_manual_gradient_error": 5.551115123125783e-16,
  "GPU_seconds": 0,
  "new_RGB": 0,
  "new_Source_encoder": 0,
  "new_DINO": 0,
  "new_ReferDINO": 0,
  "network_training": 0,
  "spatial_updates": 0,
  "upstream448_evidence_is_not_deployment_free": true
}
{
  "status": "pass",
  "independent_Adam_gradient_sampling_steps": 112640,
  "max_gradient_state_error": 9.769962616701378e-15,
  "Live_steps_differ_from_initial_sampler": 59904,
  "first_Fixed_Live_exact": 768,
  "noAction_exact_noop_parents": 128,
  "independent_scalar_teacher_checks": 999936,
  "max_teacher_error": 4.85722573273506e-16,
  "teacher_B_node_count_preserved": true,
  "wrong448_independent_reconstruction": true,
  "old_q_interval_exact": 256,
  "frozen_spatial_arrays_exact": 512,
  "independent_original_MBR_utilities": 8955,
  "new_GT": true,
  "actual_updates": 112640,
  "official_Dense_output_checks": 14848,
  "scalar_candidate_quality_calls": 798720,
  "GT_provenance": {
    "data/hcstvg2_official_metadata/val_v2.json": "2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156"
  },
  "all128_before_GT": true,
  "all_temporal_arms_sIoU_identical": true
}
```

不是独立确认，不扫参数或挑控制晋升，空间/CURRENT/MAIN/PAPER不变。
