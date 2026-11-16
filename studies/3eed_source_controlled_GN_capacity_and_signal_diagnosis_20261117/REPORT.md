# R121 — 源域受控GN224能力与信号诊断

本轮只做源域受控诊断。三个epoch100源模型及现有R91双向模块不重新训练；每个输入上暂时修改224个RGB GroupNorm仿射参数，测试后恢复。源域GT仅用于oracle能力对照与事后评分，oracle状态不部署、不称为TTA。没有目标域推理或评分、外部专家、新框或FINAL访问。

## 竞争解释与范围

R120的真实GN写回和实时语义路径都已核实，但深度更新没有超过同信息冻结/错配/随机对照。缺少的区分是：在同一接口、同一单步范数下，grounding监督能否产生有益选框变化？若能而物理监督不能，主要是信号方向；若GT同预算也几乎无效，则不能只归因于监督不够准。

这不重复R77：R77测试102498个Adapter/交互参数和另一软IoU监督；这里仅RGB GN224，原生点—文字基础h/U硬固定，只让sourceA的RGB late与实际竞争残差r改变。它也是当前A源域能力代理，不能推出跨域融合F的TTA必然有效。

## 数据与固定条件

三个source各128条已暴露的source VAL，Waymo82、Drone2、Quad3个场景。它们曾用于既有模块选择，不是独立确认集。沿用现成prefix/标定后RGB数组/路由/已封存IoU；未打开原媒体或标注重新计算框。原始A256框、query身份、六层decoder位置、完整文字全部固定。没有源域B/C观测缓存，因此本轮不制造源域三采样或R59对照；目标域R59及ABC强基线的既有结果保留在R120等完整记录，不在不同数据上假比较。

一个固定全局source-frame哈希将样本分成group0/group1，同帧同组。各来源的group计数在PREFLIGHT.json。选择规则在模型结果前锁定：只用group0选P/RGB/D中总体最好的一项；group1必须超过自身零步、对应错配、等范数随机。此规则只用于源域诊断资格，不把已经曝光的source VAL包装成新测试，也不会自动部署。

## 干预

P为既有点特征恢复，RGB为互补遮挡图像恢复，D为图像单支测量稀疏深度恢复。每项原始梯度独立计算，统一归一到source GN范数1%；76001/2遮挡及query锚点保护不变。P/RGB沿用原错配点—像素关系；D将有效深度格目标循环错配。随机GN同修改范数。没有混合损失权重或步长扫描。

oracle1使用真实最佳IoU候选集合的概率质量负对数，1e-8并列范围；全零oracle输入不构造任意正标签，仍留在总分分母。oracle10为预设10次同半径GT方向尝试、9档固定Armijo回溯，累计范数上界10倍。后者只测受控容量，不是无标签验收算法。降低该代理损失也必须与实际IoU分开。

## 实现核验与失败记录

预检384条的框—seed支持、query锚点、缓存标签和形状通过。6条真实工程样本逐臂检查GN更新后native h/U仍逐值一致、作用域仅6张量224参数、逐例恢复。首次工程断言错误地比较原始每query token概率sigmoid(logodds)与跨query softmax(logodds)；实际隐藏态、文字及匹配头完全一致。修正的只是验证口径，失败代码、日志和前后lock全部保留，见ASSERTION_AMENDMENT.json。正式预测规则未因任务成绩修改。

## 全384源域任务结果

单位为百分数。all_three_equal逐source等权；由于每源128条，它与384条总体均值相同。group内仍按source等权，避免组大小不同改变来源比例。

| source | 方法 | n | mIoU% | Acc25% | Acc50% |
|---|---|---:|---:|---:|---:|
| all_three_equal | source | 384 | 33.248349 | 57.552083 | 34.114583 |
| all_three_equal | prepared | 384 | 35.214089 | 64.062500 | 34.895833 |
| all_three_equal | native | 384 | 33.248349 | 57.552083 | 34.114583 |
| all_three_equal | context_zero | 384 | 33.800991 | 59.375000 | 34.114583 |
| all_three_equal | P | 384 | 33.800991 | 59.375000 | 34.114583 |
| all_three_equal | RGB | 384 | 33.922316 | 59.635417 | 34.114583 |
| all_three_equal | D | 384 | 33.922316 | 59.635417 | 34.114583 |
| all_three_equal | wrong_P | 384 | 33.800991 | 59.375000 | 34.114583 |
| all_three_equal | wrong_RGB | 384 | 33.922316 | 59.635417 | 34.114583 |
| all_three_equal | wrong_D | 384 | 33.922316 | 59.635417 | 34.114583 |
| all_three_equal | random | 384 | 33.800991 | 59.375000 | 34.114583 |
| all_three_equal | oracle1 | 384 | 33.803873 | 59.375000 | 34.114583 |
| all_three_equal | oracle10 | 384 | 33.853610 | 59.635417 | 33.854167 |
| waymo | source | 128 | 45.758121 | 83.593750 | 43.750000 |
| waymo | prepared | 128 | 46.892290 | 84.375000 | 46.093750 |
| waymo | native | 128 | 45.758121 | 83.593750 | 43.750000 |
| waymo | context_zero | 128 | 45.658939 | 82.812500 | 43.750000 |
| waymo | P | 128 | 45.658939 | 82.812500 | 43.750000 |
| waymo | RGB | 128 | 45.667584 | 82.812500 | 43.750000 |
| waymo | D | 128 | 45.667584 | 82.812500 | 43.750000 |
| waymo | wrong_P | 128 | 45.658939 | 82.812500 | 43.750000 |
| waymo | wrong_RGB | 128 | 45.667584 | 82.812500 | 43.750000 |
| waymo | wrong_D | 128 | 45.667584 | 82.812500 | 43.750000 |
| waymo | random | 128 | 45.658939 | 82.812500 | 43.750000 |
| waymo | oracle1 | 128 | 45.667584 | 82.812500 | 43.750000 |
| waymo | oracle10 | 128 | 45.500687 | 82.812500 | 42.968750 |
| drone | source | 128 | 17.298292 | 32.031250 | 9.375000 |
| drone | prepared | 128 | 18.664499 | 41.406250 | 8.593750 |
| drone | native | 128 | 17.298292 | 32.031250 | 9.375000 |
| drone | context_zero | 128 | 17.632101 | 34.375000 | 8.593750 |
| drone | P | 128 | 17.632101 | 34.375000 | 8.593750 |
| drone | RGB | 128 | 17.987429 | 35.156250 | 8.593750 |
| drone | D | 128 | 17.987429 | 35.156250 | 8.593750 |
| drone | wrong_P | 128 | 17.632101 | 34.375000 | 8.593750 |
| drone | wrong_RGB | 128 | 17.987429 | 35.156250 | 8.593750 |
| drone | wrong_D | 128 | 17.987429 | 35.156250 | 8.593750 |
| drone | random | 128 | 17.632101 | 34.375000 | 8.593750 |
| drone | oracle1 | 128 | 17.632101 | 34.375000 | 8.593750 |
| drone | oracle10 | 128 | 17.947391 | 35.156250 | 8.593750 |
| quad | source | 128 | 36.688634 | 57.031250 | 49.218750 |
| quad | prepared | 128 | 40.085478 | 66.406250 | 50.000000 |
| quad | native | 128 | 36.688634 | 57.031250 | 49.218750 |
| quad | context_zero | 128 | 38.111934 | 60.937500 | 50.000000 |
| quad | P | 128 | 38.111934 | 60.937500 | 50.000000 |
| quad | RGB | 128 | 38.111934 | 60.937500 | 50.000000 |
| quad | D | 128 | 38.111934 | 60.937500 | 50.000000 |
| quad | wrong_P | 128 | 38.111934 | 60.937500 | 50.000000 |
| quad | wrong_RGB | 128 | 38.111934 | 60.937500 | 50.000000 |
| quad | wrong_D | 128 | 38.111934 | 60.937500 | 50.000000 |
| quad | random | 128 | 38.111934 | 60.937500 | 50.000000 |
| quad | oracle1 | 128 | 38.111934 | 60.937500 | 50.000000 |
| quad | oracle10 | 128 | 38.112752 | 60.937500 | 50.000000 |

## 源域资格与决定

```json
{
  "picked_on_predeclared_source_group0": "RGB",
  "qualification_checks": {
    "group1_above_zero": true,
    "group1_above_wrong": false,
    "group1_above_random": true
  },
  "qualified": false,
  "source_group0_scores": {
    "context_zero": 33.36736392801032,
    "P": 33.36736392801032,
    "RGB": 33.372559231966086,
    "D": 33.372559231966086,
    "random": 33.36736392801032,
    "wrong_P": 33.36736392801032,
    "wrong_RGB": 33.372559231966086,
    "wrong_D": 33.372559231966086
  },
  "source_group1_scores": {
    "context_zero": 34.45299793158569,
    "P": 34.45299793158569,
    "RGB": 34.68270486370999,
    "D": 34.68270486370999,
    "random": 34.45299793158569,
    "wrong_P": 34.45299793158569,
    "wrong_RGB": 34.68270486370999,
    "wrong_D": 34.68270486370999
  },
  "source_oracle_is_NOT_TTA": true,
  "new_source_training": false,
  "target_data_read": false,
  "FINAL_access": false
}
{
  "scope": "SOURCE_ONLY_CONTROLLED_DIAGNOSIS_NOT_UNLABELED_TTA",
  "source_labels_used": true,
  "target_data_read": false,
  "new_source_training": false,
  "deployed_source_weights_changed": false,
  "GN224_oracle1_gain_pp": 0.0028817701629648695,
  "GN224_oracle10_gain_pp": 0.05261838607481195,
  "qualified_aux_signal": null,
  "qualification_pass": false,
  "automatic_target_deployment": false,
  "TTA_adopted": false,
  "FINAL_access": false
}
```

若未通过，不把P/RGB/D依次搬到目标域继续试，也不增加GN幅度来找正数。即使通过，也需另行锁定一个有依据的目标验证；若选中D，R120已有相同信号的目标负结果，不重新运行。没有GT筛选部署伪标签、逐平台拼方法或删除负例。

## 方向与任务变化

```json
[
  {
    "source": "all_three_equal",
    "arm": "P",
    "gradient_cosine_n": 383,
    "gradient_cosine_mean": 0.024326138832763817,
    "gradient_cosine_median": -0.01066483801628517,
    "positive_gradient_cosines": 187,
    "task_surrogate_improved": 187,
    "task_surrogate_worsened": 196,
    "n": 384
  },
  {
    "source": "all_three_equal",
    "arm": "RGB",
    "gradient_cosine_n": 383,
    "gradient_cosine_mean": -0.017910830026566103,
    "gradient_cosine_median": -0.048459157255644106,
    "positive_gradient_cosines": 166,
    "task_surrogate_improved": 166,
    "task_surrogate_worsened": 217,
    "n": 384
  },
  {
    "source": "all_three_equal",
    "arm": "D",
    "gradient_cosine_n": 383,
    "gradient_cosine_mean": -0.01416208041025254,
    "gradient_cosine_median": 0.04687843098966027,
    "positive_gradient_cosines": 202,
    "task_surrogate_improved": 201,
    "task_surrogate_worsened": 182,
    "n": 384
  },
  {
    "source": "all_three_equal",
    "arm": "wrong_P",
    "gradient_cosine_n": 383,
    "gradient_cosine_mean": 0.06206691515278905,
    "gradient_cosine_median": 0.05092108578063227,
    "positive_gradient_cosines": 216,
    "task_surrogate_improved": 216,
    "task_surrogate_worsened": 167,
    "n": 384
  },
  {
    "source": "all_three_equal",
    "arm": "wrong_RGB",
    "gradient_cosine_n": 383,
    "gradient_cosine_mean": -0.016622434224278025,
    "gradient_cosine_median": -0.03204849172043882,
    "positive_gradient_cosines": 168,
    "task_surrogate_improved": 167,
    "task_surrogate_worsened": 216,
    "n": 384
  },
  {
    "source": "all_three_equal",
    "arm": "wrong_D",
    "gradient_cosine_n": 383,
    "gradient_cosine_mean": -0.01429994980955915,
    "gradient_cosine_median": 0.050577342533171414,
    "positive_gradient_cosines": 202,
    "task_surrogate_improved": 201,
    "task_surrogate_worsened": 182,
    "n": 384
  },
  {
    "source": "all_three_equal",
    "arm": "random",
    "gradient_cosine_n": 0,
    "gradient_cosine_mean": null,
    "gradient_cosine_median": null,
    "positive_gradient_cosines": 0,
    "task_surrogate_improved": 204,
    "task_surrogate_worsened": 179,
    "n": 384
  },
  {
    "source": "all_three_equal",
    "arm": "oracle1",
    "gradient_cosine_n": 383,
    "gradient_cosine_mean": 1.0,
    "gradient_cosine_median": 1.0,
    "positive_gradient_cosines": 383,
    "task_surrogate_improved": 383,
    "task_surrogate_worsened": 0,
    "n": 384
  },
  {
    "source": "all_three_equal",
    "arm": "oracle10",
    "gradient_cosine_n": 0,
    "gradient_cosine_mean": null,
    "gradient_cosine_median": null,
    "positive_gradient_cosines": 0,
    "task_surrogate_improved": 383,
    "task_surrogate_worsened": 0,
    "n": 384
  },
  {
    "source": "waymo",
    "arm": "P",
    "gradient_cosine_n": 128,
    "gradient_cosine_mean": -0.004539840160471494,
    "gradient_cosine_median": -0.009223158176371456,
    "positive_gradient_cosines": 63,
    "task_surrogate_improved": 63,
    "task_surrogate_worsened": 65,
    "n": 128
  },
  {
    "source": "waymo",
    "arm": "RGB",
    "gradient_cosine_n": 128,
    "gradient_cosine_mean": 0.002310526792470467,
    "gradient_cosine_median": 0.017083782954154393,
    "positive_gradient_cosines": 65,
    "task_surrogate_improved": 65,
    "task_surrogate_worsened": 63,
    "n": 128
  },
  {
    "source": "waymo",
    "arm": "D",
    "gradient_cosine_n": 128,
    "gradient_cosine_mean": 0.029744086344052963,
    "gradient_cosine_median": 0.180828413287704,
    "positive_gradient_cosines": 69,
    "task_surrogate_improved": 69,
    "task_surrogate_worsened": 59,
    "n": 128
  },
  {
    "source": "waymo",
    "arm": "wrong_P",
    "gradient_cosine_n": 128,
    "gradient_cosine_mean": 0.013844432378304662,
    "gradient_cosine_median": -0.01901116057223474,
    "positive_gradient_cosines": 60,
    "task_surrogate_improved": 60,
    "task_surrogate_worsened": 68,
    "n": 128
  },
  {
    "source": "waymo",
    "arm": "wrong_RGB",
    "gradient_cosine_n": 128,
    "gradient_cosine_mean": 0.0008546710797112354,
    "gradient_cosine_median": 0.019264091534954248,
    "positive_gradient_cosines": 65,
    "task_surrogate_improved": 65,
    "task_surrogate_worsened": 63,
    "n": 128
  },
  {
    "source": "waymo",
    "arm": "wrong_D",
    "gradient_cosine_n": 128,
    "gradient_cosine_mean": 0.030354690266318433,
    "gradient_cosine_median": 0.17756633235656066,
    "positive_gradient_cosines": 69,
    "task_surrogate_improved": 69,
    "task_surrogate_worsened": 59,
    "n": 128
  },
  {
    "source": "waymo",
    "arm": "random",
    "gradient_cosine_n": 0,
    "gradient_cosine_mean": null,
    "gradient_cosine_median": null,
    "positive_gradient_cosines": 0,
    "task_surrogate_improved": 70,
    "task_surrogate_worsened": 58,
    "n": 128
  },
  {
    "source": "waymo",
    "arm": "oracle1",
    "gradient_cosine_n": 128,
    "gradient_cosine_mean": 1.0,
    "gradient_cosine_median": 1.0,
    "positive_gradient_cosines": 128,
    "task_surrogate_improved": 128,
    "task_surrogate_worsened": 0,
    "n": 128
  },
  {
    "source": "waymo",
    "arm": "oracle10",
    "gradient_cosine_n": 0,
    "gradient_cosine_mean": null,
    "gradient_cosine_median": null,
    "positive_gradient_cosines": 0,
    "task_surrogate_improved": 128,
    "task_surrogate_worsened": 0,
    "n": 128
  },
  {
    "source": "drone",
    "arm": "P",
    "gradient_cosine_n": 127,
    "gradient_cosine_mean": -0.06111248582770527,
    "gradient_cosine_median": -0.06485353331521826,
    "positive_gradient_cosines": 41,
    "task_surrogate_improved": 41,
    "task_surrogate_worsened": 86,
    "n": 128
  },
  {
    "source": "drone",
    "arm": "RGB",
    "gradient_cosine_n": 127,
    "gradient_cosine_mean": -0.0035064818020236364,
    "gradient_cosine_median": -0.017858396094153434,
    "positive_gradient_cosines": 61,
    "task_surrogate_improved": 60,
    "task_surrogate_worsened": 67,
    "n": 128
  },
  {
    "source": "drone",
    "arm": "D",
    "gradient_cosine_n": 127,
    "gradient_cosine_mean": 0.1353887136504746,
    "gradient_cosine_median": 0.1196557581433209,
    "positive_gradient_cosines": 83,
    "task_surrogate_improved": 82,
    "task_surrogate_worsened": 45,
    "n": 128
  },
  {
    "source": "drone",
    "arm": "wrong_P",
    "gradient_cosine_n": 127,
    "gradient_cosine_mean": 0.03033566881953314,
    "gradient_cosine_median": 0.022118334533974868,
    "positive_gradient_cosines": 72,
    "task_surrogate_improved": 72,
    "task_surrogate_worsened": 55,
    "n": 128
  },
  {
    "source": "drone",
    "arm": "wrong_RGB",
    "gradient_cosine_n": 127,
    "gradient_cosine_mean": 0.001061501064181172,
    "gradient_cosine_median": -0.002090497013120994,
    "positive_gradient_cosines": 62,
    "task_surrogate_improved": 61,
    "task_surrogate_worsened": 66,
    "n": 128
  },
  {
    "source": "drone",
    "arm": "wrong_D",
    "gradient_cosine_n": 127,
    "gradient_cosine_mean": 0.13447448145751903,
    "gradient_cosine_median": 0.114075747074057,
    "positive_gradient_cosines": 83,
    "task_surrogate_improved": 82,
    "task_surrogate_worsened": 45,
    "n": 128
  },
  {
    "source": "drone",
    "arm": "random",
    "gradient_cosine_n": 0,
    "gradient_cosine_mean": null,
    "gradient_cosine_median": null,
    "positive_gradient_cosines": 0,
    "task_surrogate_improved": 70,
    "task_surrogate_worsened": 57,
    "n": 128
  },
  {
    "source": "drone",
    "arm": "oracle1",
    "gradient_cosine_n": 127,
    "gradient_cosine_mean": 1.0,
    "gradient_cosine_median": 1.0,
    "positive_gradient_cosines": 127,
    "task_surrogate_improved": 127,
    "task_surrogate_worsened": 0,
    "n": 128
  },
  {
    "source": "drone",
    "arm": "oracle10",
    "gradient_cosine_n": 0,
    "gradient_cosine_mean": null,
    "gradient_cosine_median": null,
    "positive_gradient_cosines": 0,
    "task_surrogate_improved": 127,
    "task_surrogate_worsened": 0,
    "n": 128
  },
  {
    "source": "quad",
    "arm": "P",
    "gradient_cosine_n": 128,
    "gradient_cosine_mean": 0.13796325323130829,
    "gradient_cosine_median": 0.21106787470043897,
    "positive_gradient_cosines": 83,
    "task_surrogate_improved": 83,
    "task_surrogate_worsened": 45,
    "n": 128
  },
  {
    "source": "quad",
    "arm": "RGB",
    "gradient_cosine_n": 128,
    "gradient_cosine_mean": -0.052424001099640884,
    "gradient_cosine_median": -0.10358161522473981,
    "positive_gradient_cosines": 40,
    "task_surrogate_improved": 41,
    "task_surrogate_worsened": 87,
    "n": 128
  },
  {
    "source": "quad",
    "arm": "D",
    "gradient_cosine_n": 128,
    "gradient_cosine_mean": -0.20645067564668573,
    "gradient_cosine_median": -0.5693579006087621,
    "positive_gradient_cosines": 50,
    "task_surrogate_improved": 50,
    "task_surrogate_worsened": 78,
    "n": 128
  },
  {
    "source": "quad",
    "arm": "wrong_P",
    "gradient_cosine_n": 128,
    "gradient_cosine_mean": 0.14177274389855082,
    "gradient_cosine_median": 0.20537852295584336,
    "positive_gradient_cosines": 84,
    "task_surrogate_improved": 84,
    "task_surrogate_worsened": 44,
    "n": 128
  },
  {
    "source": "quad",
    "arm": "wrong_RGB",
    "gradient_cosine_n": 128,
    "gradient_cosine_mean": -0.0516453190722854,
    "gradient_cosine_median": -0.09980175607721697,
    "positive_gradient_cosines": 41,
    "task_surrogate_improved": 41,
    "task_surrogate_worsened": 87,
    "n": 128
  },
  {
    "source": "quad",
    "arm": "wrong_D",
    "gradient_cosine_n": 128,
    "gradient_cosine_mean": -0.20656672090824083,
    "gradient_cosine_median": -0.5674141289030539,
    "positive_gradient_cosines": 50,
    "task_surrogate_improved": 50,
    "task_surrogate_worsened": 78,
    "n": 128
  },
  {
    "source": "quad",
    "arm": "random",
    "gradient_cosine_n": 0,
    "gradient_cosine_mean": null,
    "gradient_cosine_median": null,
    "positive_gradient_cosines": 0,
    "task_surrogate_improved": 64,
    "task_surrogate_worsened": 64,
    "n": 128
  },
  {
    "source": "quad",
    "arm": "oracle1",
    "gradient_cosine_n": 128,
    "gradient_cosine_mean": 1.0,
    "gradient_cosine_median": 1.0,
    "positive_gradient_cosines": 128,
    "task_surrogate_improved": 128,
    "task_surrogate_worsened": 0,
    "n": 128
  },
  {
    "source": "quad",
    "arm": "oracle10",
    "gradient_cosine_n": 0,
    "gradient_cosine_mean": null,
    "gradient_cosine_median": null,
    "positive_gradient_cosines": 0,
    "task_surrogate_improved": 128,
    "task_surrogate_worsened": 0,
    "n": 128
  }
]
```

梯度角度只是局部代理：正向角度或恢复损失下降不能代替Top1/IoU改善。oracle1与oracle10也不同预算，不能把后者当同成本获胜方法。

## 全部正负贡献与案例边界

```json
[
  {
    "arm": "source",
    "n": 384,
    "gain_pp": -0.5526422323101781,
    "better": 25,
    "worse": 28,
    "unchanged": 331,
    "largest_positive_contribution_pp": 0.18080996205985225,
    "largest_negative_contribution_pp": -0.1738506138340509,
    "positive_mass_pp": 0.9649953160653783,
    "negative_mass_pp": -1.517637548375556
  },
  {
    "arm": "prepared",
    "n": 384,
    "gain_pp": 1.4130977339262394,
    "better": 78,
    "worse": 74,
    "unchanged": 232,
    "largest_positive_contribution_pp": 0.22304611584711495,
    "largest_negative_contribution_pp": -0.18258474298233876,
    "positive_mass_pp": 3.7234812628044067,
    "negative_mass_pp": -2.3103835288781673
  },
  {
    "arm": "native",
    "n": 384,
    "gain_pp": -0.5526422323101781,
    "better": 25,
    "worse": 28,
    "unchanged": 331,
    "largest_positive_contribution_pp": 0.18080996205985225,
    "largest_negative_contribution_pp": -0.1738506138340509,
    "positive_mass_pp": 0.9649953160653783,
    "negative_mass_pp": -1.517637548375556
  },
  {
    "arm": "context_zero",
    "n": 384,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0,
    "unchanged": 384,
    "largest_positive_contribution_pp": 0.0,
    "largest_negative_contribution_pp": 0.0,
    "positive_mass_pp": 0.0,
    "negative_mass_pp": 0.0
  },
  {
    "arm": "P",
    "n": 384,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0,
    "unchanged": 384,
    "largest_positive_contribution_pp": 0.0,
    "largest_negative_contribution_pp": 0.0,
    "positive_mass_pp": 0.0,
    "negative_mass_pp": 0.0
  },
  {
    "arm": "RGB",
    "n": 384,
    "gain_pp": 0.12132440703956071,
    "better": 2,
    "worse": 0,
    "unchanged": 382,
    "largest_positive_contribution_pp": 0.11844263687659423,
    "largest_negative_contribution_pp": 0.0,
    "positive_mass_pp": 0.12132440703956071,
    "negative_mass_pp": 0.0
  },
  {
    "arm": "D",
    "n": 384,
    "gain_pp": 0.12132440703956071,
    "better": 2,
    "worse": 0,
    "unchanged": 382,
    "largest_positive_contribution_pp": 0.11844263687659423,
    "largest_negative_contribution_pp": 0.0,
    "positive_mass_pp": 0.12132440703956071,
    "negative_mass_pp": 0.0
  },
  {
    "arm": "wrong_P",
    "n": 384,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0,
    "unchanged": 384,
    "largest_positive_contribution_pp": 0.0,
    "largest_negative_contribution_pp": 0.0,
    "positive_mass_pp": 0.0,
    "negative_mass_pp": 0.0
  },
  {
    "arm": "wrong_RGB",
    "n": 384,
    "gain_pp": 0.12132440703956071,
    "better": 2,
    "worse": 0,
    "unchanged": 382,
    "largest_positive_contribution_pp": 0.11844263687659423,
    "largest_negative_contribution_pp": 0.0,
    "positive_mass_pp": 0.12132440703956071,
    "negative_mass_pp": 0.0
  },
  {
    "arm": "wrong_D",
    "n": 384,
    "gain_pp": 0.12132440703956071,
    "better": 2,
    "worse": 0,
    "unchanged": 382,
    "largest_positive_contribution_pp": 0.11844263687659423,
    "largest_negative_contribution_pp": 0.0,
    "positive_mass_pp": 0.12132440703956071,
    "negative_mass_pp": 0.0
  },
  {
    "arm": "random",
    "n": 384,
    "gain_pp": 0.0,
    "better": 0,
    "worse": 0,
    "unchanged": 384,
    "largest_positive_contribution_pp": 0.0,
    "largest_negative_contribution_pp": 0.0,
    "positive_mass_pp": 0.0,
    "negative_mass_pp": 0.0
  },
  {
    "arm": "oracle1",
    "n": 384,
    "gain_pp": 0.0028817701629664664,
    "better": 1,
    "worse": 0,
    "unchanged": 383,
    "largest_positive_contribution_pp": 0.002881770162966466,
    "largest_negative_contribution_pp": 0.0,
    "positive_mass_pp": 0.002881770162966466,
    "negative_mass_pp": 0.0
  },
  {
    "arm": "oracle10",
    "n": 384,
    "gain_pp": 0.052618386074819584,
    "better": 6,
    "worse": 6,
    "unchanged": 372,
    "largest_positive_contribution_pp": 0.06856765383610895,
    "largest_negative_contribution_pp": -0.06829505516349059,
    "positive_mass_pp": 0.22716445673293748,
    "negative_mass_pp": -0.17454607065811792
  }
]
```

所有IoU改善、恶化、零变化及匿名场景结果公开。good/failure按数值变化定义，未看私有图像的实例身份记为UNKNOWN；换query不自动等于换物体。最大单例贡献按完整384分母报告，不据此挑选适应输入。Drone仅2个、Quad仅3个源场景，不能用逐描述误差条冒充跨场景总体保证。

## 审计与部署边界

```json
{
  "state": "PASS_ALL_SOURCE_CPU_HEAD_GN_PHYSICAL_COMPONENT_GRADIENTS_AND_PATCHES",
  "n": 384,
  "heads": 3840,
  "GN_states": 3840,
  "aux_components": 11520,
  "physical_gradients": 2304,
  "max_probability_error": 2.1260932233646557e-06,
  "max_residual_error": 4.76837158203125e-07,
  "max_physical_loss_error": 2.384185791015625e-07,
  "max_gradient_error": 3.2782554626464844e-07,
  "max_patch_error": 7.450580596923828e-09,
  "near_ties": 0,
  "source_GT_used_for_diagnostic_only": true,
  "target_data_read": false,
  "FINAL_access": false
}
{
  "state": "PASS_ALL_SOURCE_CACHED_IOU_VALUES_AND_EQUAL_SOURCE_REDUCTIONS",
  "n": 384,
  "arms": 13,
  "values": 4992,
  "fresh_GT_geometry": false,
  "target_data_read": false,
  "FINAL_access": false
}
```

独立CPU重算末端匹配/竞争接口、实际GN状态范数、三种物理组件与六种正确/错配梯度；另程序逐值读取封存候选IoU、核对总体与成对变化。所有执行经hard09:00资源guard。逐例seconds覆盖全部实验臂的缓存后缀与诊断计算，不是单方法端到端延迟。source epoch100及已有双向模块文件SHA保持，oracle临时状态只在私有证据中，公开包排除原始媒体、文字、坐标、ID、特征、梯度和权重。

本轮的价值是对当前接口/预算与辅助方向作受控区分，不是得到新的TTA正结果。没有把source GT拟合能力或无标签辅助下降冒充跨域增益。


## 最终解释：接口影响范围与监督问题同时存在

本轮自身零步mIoU为33.800991%；P为33.800991%，RGB与D均33.922316%，oracle1为33.803873%，oracle10为33.853610%。RGB和D的+0.121324pp只由2条改善构成，最大单例贡献+0.118443pp，约占净增量97.62%；不能据此声称找到了稳定恢复规律。更关键的是，RGB与wrong_RGB、D与wrong_D、RGB与D均在全部384条上选中相同query。资格不通过的原因是对应关系控制没有形成区分，而非代码未更新。

源域GT方向在383条有非零oracle的输入上都降低了最佳候选概率质量的代理损失，但单步只改变1条选框、+0.002882pp。10步累计编辑平均达到单步半径的9.91倍，仍只有13条换query（其中1条IoU相同）、6好6坏、净+0.052618pp。原本未选中最佳IoU候选的277条中，只有4条在GT10后选中最佳候选。于是“把GN的辅助监督变正确，原幅度就能获得大量纠错”没有得到支持。

这不是数学意义的GN所有可行编辑的上界：只检验了锁定的局部梯度/范数/10步协议，并且当前A源域代理不同于跨域F。GT损失不是直接优化离散IoU，因而不能把GT10的6条损害解释为标签错了。但这一受控结果已经足以停止围绕该GN接口继续换物理监督、增大倍率或寻找接受阈值。它不是当前有证据支持的强选框纠错入口。

辅助方向本身也缺少任务一致性。P、RGB、D与GT梯度的平均cosine分别约0.0243、-0.0179、-0.0142；同一信号在很多输入上提高了GT代理损失。不能再把“GN接口弱”当作“物理监督其实正确、只差放大”的证明。两种限制同时成立于当前测试范围。

决定：不采用新的目标域TTA，不启动P/RGB/D目标网格；本轮三个source模块与原模型完全保留。R120及更早的同信息直接读出/R59/ABC、错配和所有负结果保持原样。后续若重新提出参数适应，必须同时给出能显著影响候选竞争的接口和有独立观测依据的实例监督；不能只换当前恢复损失的外观。

精确排序诊断（封存后解释，不参与方法选择）：

```json
{
  "n": 384,
  "no_new_model_or_GT": true,
  "query_difference_counts": {
    "RGB__wrong_RGB": 0,
    "D__wrong_D": 0,
    "RGB__D": 0,
    "P__context_zero": 0,
    "oracle1__context_zero": 1,
    "oracle10__context_zero": 13
  },
  "best_IoU_candidate_ranks": {
    "context_zero": {
      "n": 383,
      "mean": 19.704960835509137,
      "median": 5.0,
      "best_selected": 106
    },
    "oracle1": {
      "n": 383,
      "mean": 19.621409921671017,
      "median": 5.0,
      "best_selected": 107
    },
    "oracle10": {
      "n": 383,
      "mean": 18.98433420365535,
      "median": 5.0,
      "best_selected": 110
    }
  },
  "initial_not_best_subset_ranks": {
    "context_zero": {
      "n": 277,
      "mean": 26.862815884476536,
      "median": 10.0,
      "best_selected": 0
    },
    "oracle1": {
      "n": 277,
      "mean": 26.747292418772563,
      "median": 10.0,
      "best_selected": 1
    },
    "oracle10": {
      "n": 277,
      "mean": 25.866425992779785,
      "median": 10.0,
      "best_selected": 4
    }
  },
  "oracle10_cumulative_patch_to_single_radius": {
    "mean": 9.907553281812257,
    "median": 9.96537258591135,
    "max": 9.999058812999753
  },
  "GT_oracle_diagnostic_NOT_TTA": true
}
```
