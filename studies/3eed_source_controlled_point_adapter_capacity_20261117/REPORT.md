# R124 — 精确pointAdapter接口的源域GT受控能力诊断

**本轮使用源域GT作临时逐输入干预，不是无标签TTA、不是source训练，也不产生部署checkpoint。** 不能把本表oracle数值作为方法性能。沿用已授权的source标签诊断；三个source100和R91所有原权重在每条输入后恢复，目标域、FINAL、外部专家及新原始观测均不进入本轮。

## 为什么需要这项不同于R121的检查

R123真实注册点InfoNCE已完成：全部786有效A/B的拟合目标下降，C对应损失也改善，正确与错配首步方向余弦约−0.71至−0.91，但选框DEV/EXP相对同信息冻结均值分别−0.1154/−0.0891个百分点。它确实使用了对应关系，却未得到任务收益。

R121只诊断RGB GroupNorm224参数，R77诊断远大于当前的模块集合，均不能回答当前pointAdapter输出4896参数的实际能力。因此本轮固定R123接口、初始化与三步预算，仅把训练信号替换为受控的source GT最佳可用候选集合。目的不是再找一层或扫描步幅，而是检查这份实际接口在相同预算下能否改变有用的排序。

## 固定条件与标签边界

384条旧sourceVAL缓存，每个source128，早已用于训练验证/模型选择，不能称独立验证。相同完整文字、原A256/六层位置、点特征与RGB标定缓存。零步使用native foundation：pointAdapter.up权重/偏置和early RGB门置零，source-trained down固定。RGB late与竞争残差来自相同sourceA原准备模型，固定不随更新漂移。

只修改Up的288×16权重与288偏置，4896参数。每步−g/||g||，半径完全沿用R123：原source整个pointAdapter范数的1%。固定三步主项；一步仅为过程记录，不按成绩选取。随机端点与对应oracle端点实际累计参数范数相同。没有回溯、源域重新训练、输出头重训或新模块。

GT loss是最大IoU候选集合的概率质量负对数，绝对1e−8认定并列；全零IoU的输入不产生GT梯度。这个目标会泄露真实目标，因而绝不作为无标签算法或目标域选择器部署。最大IoU也不是实例身份人工真值，只是固定候选上的task-controlled supervision。

## 工程、梯度与独立验证

六条工程输入的原生h/U与旧source缓存逐值相同，零步Top1复现旧R121 context；实际只写4896参数并恢复source。每条工程对GT梯度做中心有限差分，全部通过。正式384条预测和临时状态封存后，CPU独立重算1920次匹配读出、实际范数、全部归一化步骤与随机端点，核对全部source缓存IoU及总体/场景/组别。

边界需要明确：未在CPU上重复整个decoder的所有反向传播。梯度方向验证来自六条工程有限差分，全量CPU检查从保存梯度重构步进及最终独立head读出。报告不把有限的检查写成全量独立decoder反传。首次路径预检捕获历史wildcard导入覆盖H/B常量，在任何模型调用之前已停止并修复，失败代码/日志/原锁保留，不改变诊断配方。

## 全体总体与各source

| 范围 | 方法 | mIoU% | Acc25% | Acc50% |
|---|---|---:|---:|---:|
| all_three_equal | original_source | 33.248349 | 57.552083 | 34.114583 |
| all_three_equal | prepared | 35.214089 | 64.062500 | 34.895833 |
| all_three_equal | zero | 33.800991 | 59.375000 | 34.114583 |
| all_three_equal | oracle1 | 35.241465 | 62.760417 | 35.156250 |
| all_three_equal | oracle3 | 38.023572 | 67.968750 | 38.281250 |
| all_three_equal | random1 | 33.800991 | 59.375000 | 34.114583 |
| all_three_equal | random3 | 33.896849 | 59.635417 | 34.114583 |
| waymo | original_source | 45.758121 | 83.593750 | 43.750000 |
| waymo | prepared | 46.892290 | 84.375000 | 46.093750 |
| waymo | zero | 45.658939 | 82.812500 | 43.750000 |
| waymo | oracle1 | 46.084536 | 83.593750 | 44.531250 |
| waymo | oracle3 | 47.409332 | 85.156250 | 46.093750 |
| waymo | random1 | 45.658939 | 82.812500 | 43.750000 |
| waymo | random3 | 45.658939 | 82.812500 | 43.750000 |
| drone | original_source | 17.298292 | 32.031250 | 9.375000 |
| drone | prepared | 18.664499 | 41.406250 | 8.593750 |
| drone | zero | 17.632101 | 34.375000 | 8.593750 |
| drone | oracle1 | 20.092953 | 40.625000 | 9.375000 |
| drone | oracle3 | 24.290977 | 49.218750 | 14.062500 |
| drone | random1 | 17.632101 | 34.375000 | 8.593750 |
| drone | random3 | 17.899160 | 35.156250 | 8.593750 |
| quad | original_source | 36.688634 | 57.031250 | 49.218750 |
| quad | prepared | 40.085478 | 66.406250 | 50.000000 |
| quad | zero | 38.111934 | 60.937500 | 50.000000 |
| quad | oracle1 | 39.546905 | 64.062500 | 51.562500 |
| quad | oracle3 | 42.370407 | 69.531250 | 54.687500 |
| quad | random1 | 38.111934 | 60.937500 | 50.000000 |
| quad | random3 | 38.132449 | 60.937500 | 50.000000 |

## 本轮能够支持的判断

相同三步接口的source-GT干预相对自身零步为 **+4.222581 个百分点**。这属于受控源域能力，不是TTA增益。
因此，不能再笼统地把R123失败归因于“这个接口完全改不动选框”。已测预算下存在grounding有用的参数方向；但源域GT的可用方向不等于在目标域能无标签地识别这些方向。R123已实测的注册点目标没有可靠选中有益方向，不能靠source oracle成绩宣布修复。

本轮不把GT更新、梯度或最佳步数带回目标域。R123固定配方继续停止；不重新包装为一个已成功的三模态TTA方法。无标签信号仍须在相同信息直接使用、错配、随机及未拟合观测上独立证明任务增量。

## 所有改善、损伤和集中度

```json
[
  {
    "arm": "original_source",
    "n": 384,
    "gain_pp": -0.5526422323101781,
    "better": 25,
    "worse": 28,
    "same": 331,
    "largest_positive_single_input_pp": 0.18080996205985225,
    "largest_negative_single_input_pp": -0.1738506138340509,
    "positive_mass_pp": 0.9649953160653783,
    "negative_mass_pp": -1.517637548375556
  },
  {
    "arm": "prepared",
    "n": 384,
    "gain_pp": 1.4130977339262394,
    "better": 78,
    "worse": 74,
    "same": 232,
    "largest_positive_single_input_pp": 0.22304611584711495,
    "largest_negative_single_input_pp": -0.18258474298233876,
    "positive_mass_pp": 3.7234812628044067,
    "negative_mass_pp": -2.3103835288781673
  },
  {
    "arm": "zero",
    "n": 384,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0,
    "same": 384,
    "largest_positive_single_input_pp": 0.0,
    "largest_negative_single_input_pp": 0.0,
    "positive_mass_pp": 0.0,
    "negative_mass_pp": 0.0
  },
  {
    "arm": "oracle1",
    "n": 384,
    "gain_pp": 1.4404733288075002,
    "better": 29,
    "worse": 4,
    "same": 351,
    "largest_positive_single_input_pp": 0.16485281561773812,
    "largest_negative_single_input_pp": -0.07461181645292729,
    "positive_mass_pp": 1.543929951150753,
    "negative_mass_pp": -0.10345662234325283
  },
  {
    "arm": "oracle3",
    "n": 384,
    "gain_pp": 4.2225809398839935,
    "better": 70,
    "worse": 13,
    "same": 301,
    "largest_positive_single_input_pp": 0.22304611584711495,
    "largest_negative_single_input_pp": -0.11899100204943959,
    "positive_mass_pp": 4.517482672390453,
    "negative_mass_pp": -0.2949017325064582
  },
  {
    "arm": "random1",
    "n": 384,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0,
    "same": 384,
    "largest_positive_single_input_pp": 0.0,
    "largest_negative_single_input_pp": 0.0,
    "positive_mass_pp": 0.0,
    "negative_mass_pp": 0.0
  },
  {
    "arm": "random3",
    "n": 384,
    "gain_pp": 0.09585799550007054,
    "better": 4,
    "worse": 0,
    "same": 380,
    "largest_positive_single_input_pp": 0.05369331667416685,
    "largest_negative_single_input_pp": 0.0,
    "positive_mass_pp": 0.09585799550007056,
    "negative_mass_pp": 0.0
  }
]
```

全部IoU变化案例均保留在匿名逐例表。换query不直接等同于换物体；未进行私有原图身份复核的实例类别保持UNKNOWN。源域GT过拟合的good case只说明受控监督可以改变决策，不是算法正确发现目标的证据。

## 优化、范围与审计

```json
{
  "source_only": true,
  "source_GT_is_NOT_TTA": true,
  "no_deployed_weights": true,
  "no_target_followup_selected": true,
  "groups": {
    "all_three_equal": {
      "oracle1": {
        "n_eligible": 383,
        "GT_loss_falls": 383,
        "mean_GT_loss_change": -0.32618846104315796,
        "mean_parameter_L2": 0.034809654876958536,
        "mean_hidden_change_L2": 13.573929051506301
      },
      "oracle3": {
        "n_eligible": 383,
        "GT_loss_falls": 383,
        "mean_GT_loss_change": -0.8419271488403396,
        "mean_parameter_L2": 0.10178742096063886,
        "mean_hidden_change_L2": 33.893325594013106
      },
      "random1": {
        "n_eligible": 383,
        "GT_loss_falls": 184,
        "mean_GT_loss_change": 0.00035514052960933777,
        "mean_parameter_L2": 0.03480965506281492,
        "mean_hidden_change_L2": 0.8746014814775233
      },
      "random3": {
        "n_eligible": 383,
        "GT_loss_falls": 188,
        "mean_GT_loss_change": 0.002649981726062792,
        "mean_parameter_L2": 0.10178742088695289,
        "mean_hidden_change_L2": 2.7130539163596947
      }
    },
    "waymo": {
      "oracle1": {
        "n_eligible": 128,
        "GT_loss_falls": 128,
        "mean_GT_loss_change": -0.2993055315398237,
        "mean_parameter_L2": 0.039616931191824176,
        "mean_hidden_change_L2": 10.316334400326014
      },
      "oracle3": {
        "n_eligible": 128,
        "GT_loss_falls": 128,
        "mean_GT_loss_change": -0.7922691624090243,
        "mean_parameter_L2": 0.11704174063817373,
        "mean_hidden_change_L2": 27.913398653268814
      },
      "random1": {
        "n_eligible": 128,
        "GT_loss_falls": 50,
        "mean_GT_loss_change": 0.0005563771632263282,
        "mean_parameter_L2": 0.039616931105005956,
        "mean_hidden_change_L2": 1.013883930630982
      },
      "random3": {
        "n_eligible": 128,
        "GT_loss_falls": 75,
        "mean_GT_loss_change": 0.0008241752560783687,
        "mean_parameter_L2": 0.11704173987372485,
        "mean_hidden_change_L2": 3.1008404940366745
      }
    },
    "drone": {
      "oracle1": {
        "n_eligible": 127,
        "GT_loss_falls": 127,
        "mean_GT_loss_change": -0.3657085206575485,
        "mean_parameter_L2": 0.03213948383618823,
        "mean_hidden_change_L2": 15.149217930365735
      },
      "oracle3": {
        "n_eligible": 127,
        "GT_loss_falls": 127,
        "mean_GT_loss_change": -0.9542492923425943,
        "mean_parameter_L2": 0.09331696280388505,
        "mean_hidden_change_L2": 37.0126065607146
      },
      "random1": {
        "n_eligible": 127,
        "GT_loss_falls": 64,
        "mean_GT_loss_change": 0.00031150995592477577,
        "mean_parameter_L2": 0.0321394834019015,
        "mean_hidden_change_L2": 0.937096160697186
      },
      "random3": {
        "n_eligible": 127,
        "GT_loss_falls": 57,
        "mean_GT_loss_change": 0.0039015954726197614,
        "mean_parameter_L2": 0.09331696320406836,
        "mean_hidden_change_L2": 2.9777312616663654
      }
    },
    "quad": {
      "oracle1": {
        "n_eligible": 128,
        "GT_loss_falls": 128,
        "mean_GT_loss_change": -0.31386008139783916,
        "mean_parameter_L2": 0.032651688891607175,
        "mean_hidden_change_L2": 15.268541768193245
      },
      "oracle3": {
        "n_eligible": 128,
        "GT_loss_falls": 128,
        "mean_GT_loss_change": -0.7801405085155116,
        "mean_parameter_L2": 0.09493738398550822,
        "mean_hidden_change_L2": 36.77834095060825
      },
      "random1": {
        "n_eligible": 128,
        "GT_loss_falls": 70,
        "mean_GT_loss_change": 0.0001971936058199987,
        "mean_parameter_L2": 0.03265168996543642,
        "mean_hidden_change_L2": 0.6733125927858055
      },
      "random3": {
        "n_eligible": 128,
        "GT_loss_falls": 56,
        "mean_GT_loss_change": 0.00323395268188522,
        "mean_parameter_L2": 0.09493738413241792,
        "mean_hidden_change_L2": 2.0626577851362526
      }
    }
  }
}
{
  "state": "PASS_ALL_SOURCE_ZERO_BRIDGES_CPU_HEADS_NORMALIZED_STEPS_NORMS_RESTORE_AND_6_ENGINEERING_FINITE_DIFFERENCES",
  "n": 384,
  "CPU_heads": 1920,
  "states": 1920,
  "max_probability_error": 1.5298352984416397e-06,
  "max_state_arithmetic_error": 2.3283064365386963e-09,
  "max_norm_error": 1.249000902703301e-16,
  "finite_difference_n": 6,
  "finite_difference_max_relative_error": 0.008520199810955783,
  "full_decoder_gradients_independently_CPU_recomputed": false,
  "gradient_validation": "six engineering central differences; all normalized-step arithmetic replayed from saved autograd gradients, no claim all CPU decoder backprop",
  "source_GT_is_NOT_TTA": true,
  "target_data_read": false,
  "FINAL_access": false
}
{
  "state": "PASS_ALL_SOURCE_CACHED_IOU_VALUES_AND_EQUAL_SOURCE_REDUCTIONS",
  "n": 384,
  "arms": 7,
  "values": 2688,
  "fresh_GT_geometry": false,
  "target_data_read": false,
  "FINAL_access": false
}
```

三个source各自结果与所有场景、既有帧组均完整公开；不选择表现最好的source或删负例。旧sourceVAL有82个Waymo场景、Drone2个、Quad3个，场景重复和已曝光限制不因384条规模而消失。共享研究seconds不代表单方法新鲜端到端成本。公开排除原文、原图、原点云、框坐标、query索引、特征、参数状态和梯度。
