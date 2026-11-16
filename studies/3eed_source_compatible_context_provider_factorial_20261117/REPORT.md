# R118 — 原生基础与源域兼容 RGB／竞争证据的固定读出对照

本轮固定原source100与已有R91双向模块，不重新训练，不更新参数、不增加辅助损失，不读取FINAL。候选保持原A256、六层位置、原query身份和完整文字。只替换小模块的证据来源，检验此前撤回源域Point Adapter/早期RGB路径以后，训练过的竞争头是否因输入变化失效。

## 机制与固定对照

原生点云—文字基础来自未改动3EED；NF表示native/fused观测，SF表示完整source-prepared/fused观测，SA表示完整source-prepared/同步当前A观测。最终分数仍用真实匹配MLP、归一化、主体token对其他token的logmass比，再加现有竞争残差；系数全部继承1，不按目标平台或样本调节。三个late来源×零/NF/SF/SA四个残差来源，区分模块输入条件变化与融合多时刻点特征/仅当前A RGB的失配。主项预先固定F_lSA_rSA。

额外保留A/C读出、相同SA证据作用于native A/B/C后概率平均、去残差ABC与纯native ABC。查询残差打乱与late/残差一起打乱保持数值集合且无固定点；它们检验query对应作用，不认证真实实例身份。原始source、完整preparedA/F、nativeF、修复F与lateOnlyF的旧预测和评分逐字复用。

所有证据由同一source100/已有source模块产生，读取不同缓存后缀不等于增加第二个预训练模型。但实现完整管线时需执行native和source-compatible两个后缀，不能将这里缓存CPU重放耗时冒称端到端运行成本。没有新增PointNet/注册/媒体读取。

## 数据和审计

DEV768与EXP1536总计2304定向输入，来自三目标平台、六个方向。两者都已有开发曝光，不称独立确认集；一个表达在两个source下被评估，不把它算成两份独立场景。154保护媒体别名在读取特征前prepared回退；UNKNOWN、未配准C和所有负例保留分母。24真实工程输入验证旧provider/原source/候选读出一致，全部预测封存后才读取任务GT。
独立NumPy重放40850新端点，概率最大误差1.56e-06，近平差异0；独立平台/场景/paired重算，所有43776新任务值双几何实现核验。原6参照13824值复用，合计57600任务值。

## 总体和平台

| 数据范围 | 目标平台 | 方法 | mIoU% | Acc25% | Acc50% |
|---|---|---|---:|---:|---:|
| DEV | all_six_equal | original_source | 16.811383 | 30.989583 | 14.973958 |
| DEV | all_six_equal | prepared_A | 16.228652 | 29.947917 | 12.630208 |
| DEV | all_six_equal | prepared_F | 16.703230 | 30.989583 | 13.151042 |
| DEV | all_six_equal | native_F | 16.974318 | 31.250000 | 14.843750 |
| DEV | all_six_equal | repaired_F | 17.265658 | 31.901042 | 14.322917 |
| DEV | all_six_equal | lateOnly_F | 17.180150 | 31.901042 | 14.453125 |
| DEV | all_six_equal | F_lNF_rzero | 17.180150 | 31.901042 | 14.453125 |
| DEV | all_six_equal | F_lNF_rNF | 17.265658 | 31.901042 | 14.322917 |
| DEV | all_six_equal | F_lNF_rSF | 17.283833 | 31.640625 | 14.453125 |
| DEV | all_six_equal | F_lNF_rSA | 17.334209 | 31.901042 | 14.453125 |
| DEV | all_six_equal | F_lSF_rzero | 17.228298 | 32.031250 | 14.453125 |
| DEV | all_six_equal | F_lSF_rNF | 17.271698 | 31.901042 | 14.322917 |
| DEV | all_six_equal | F_lSF_rSF | 17.323244 | 31.770833 | 14.453125 |
| DEV | all_six_equal | F_lSF_rSA | 17.329838 | 31.901042 | 14.453125 |
| DEV | all_six_equal | F_lSA_rzero | 17.228298 | 32.031250 | 14.453125 |
| DEV | all_six_equal | F_lSA_rNF | 17.260460 | 31.901042 | 14.322917 |
| DEV | all_six_equal | F_lSA_rSF | 17.349002 | 31.901042 | 14.453125 |
| DEV | all_six_equal | F_lSA_rSA | 17.354112 | 32.031250 | 14.453125 |
| DEV | all_six_equal | F_lSA_rSAshuffled | 17.151304 | 31.901042 | 14.322917 |
| DEV | all_six_equal | F_lSAshuffled_rSAshuffled | 16.928292 | 31.250000 | 14.192708 |
| DEV | all_six_equal | A_lSA_rSA | 16.917181 | 31.510417 | 14.062500 |
| DEV | all_six_equal | C_lSA_rSA | 16.492047 | 30.729167 | 13.671875 |
| DEV | all_six_equal | ABC_lSA_rSA | 17.077734 | 31.380208 | 14.453125 |
| DEV | all_six_equal | ABC_lSA_rzero | 17.033323 | 31.770833 | 14.453125 |
| DEV | all_six_equal | ABC_native | 16.734016 | 30.729167 | 14.843750 |
| DEV | waymo | original_source | 11.473631 | 23.437500 | 7.421875 |
| DEV | waymo | prepared_F | 12.569007 | 26.171875 | 5.078125 |
| DEV | waymo | native_F | 11.535583 | 23.437500 | 7.421875 |
| DEV | waymo | repaired_F | 12.304763 | 25.390625 | 7.812500 |
| DEV | waymo | lateOnly_F | 12.181059 | 25.390625 | 7.421875 |
| DEV | waymo | F_lSA_rSA | 12.262906 | 25.000000 | 7.812500 |
| DEV | waymo | ABC_lSA_rSA | 11.937716 | 23.828125 | 7.812500 |
| DEV | waymo | ABC_lSA_rzero | 11.840380 | 24.609375 | 7.421875 |
| DEV | waymo | ABC_native | 11.215742 | 22.656250 | 7.421875 |
| DEV | drone | original_source | 13.867693 | 24.609375 | 12.500000 |
| DEV | drone | prepared_F | 14.029708 | 25.000000 | 11.718750 |
| DEV | drone | native_F | 14.710633 | 26.171875 | 12.890625 |
| DEV | drone | repaired_F | 14.503438 | 26.171875 | 11.718750 |
| DEV | drone | lateOnly_F | 14.345223 | 25.781250 | 11.718750 |
| DEV | drone | F_lSA_rSA | 14.805341 | 26.953125 | 11.718750 |
| DEV | drone | ABC_lSA_rSA | 14.433565 | 26.171875 | 11.328125 |
| DEV | drone | ABC_lSA_rzero | 14.262994 | 25.390625 | 11.718750 |
| DEV | drone | ABC_native | 14.188880 | 24.609375 | 12.500000 |
| DEV | quad | original_source | 25.092825 | 44.921875 | 25.000000 |
| DEV | quad | prepared_F | 23.510976 | 41.796875 | 22.656250 |
| DEV | quad | native_F | 24.676737 | 44.140625 | 24.218750 |
| DEV | quad | repaired_F | 24.988774 | 44.140625 | 23.437500 |
| DEV | quad | lateOnly_F | 25.014169 | 44.531250 | 24.218750 |
| DEV | quad | F_lSA_rSA | 24.994087 | 44.140625 | 23.828125 |
| DEV | quad | ABC_lSA_rSA | 24.861921 | 44.140625 | 24.218750 |
| DEV | quad | ABC_lSA_rzero | 24.996594 | 45.312500 | 24.218750 |
| DEV | quad | ABC_native | 24.797425 | 44.921875 | 24.609375 |
| EXP | all_six_equal | original_source | 16.144619 | 30.598958 | 12.434896 |
| EXP | all_six_equal | prepared_A | 16.251067 | 30.598958 | 11.783854 |
| EXP | all_six_equal | prepared_F | 16.238436 | 30.598958 | 12.044271 |
| EXP | all_six_equal | native_F | 16.237263 | 30.859375 | 12.434896 |
| EXP | all_six_equal | repaired_F | 16.314985 | 30.924479 | 12.109375 |
| EXP | all_six_equal | lateOnly_F | 16.439770 | 31.184896 | 12.174479 |
| EXP | all_six_equal | F_lNF_rzero | 16.439770 | 31.184896 | 12.174479 |
| EXP | all_six_equal | F_lNF_rNF | 16.314985 | 30.924479 | 12.109375 |
| EXP | all_six_equal | F_lNF_rSF | 16.376682 | 31.054688 | 12.044271 |
| EXP | all_six_equal | F_lNF_rSA | 16.298376 | 31.054688 | 11.914062 |
| EXP | all_six_equal | F_lSF_rzero | 16.425567 | 31.119792 | 12.174479 |
| EXP | all_six_equal | F_lSF_rNF | 16.358632 | 31.054688 | 12.174479 |
| EXP | all_six_equal | F_lSF_rSF | 16.411847 | 31.054688 | 12.044271 |
| EXP | all_six_equal | F_lSF_rSA | 16.331841 | 30.989583 | 11.979167 |
| EXP | all_six_equal | F_lSA_rzero | 16.423666 | 30.989583 | 12.239583 |
| EXP | all_six_equal | F_lSA_rNF | 16.310307 | 30.859375 | 12.109375 |
| EXP | all_six_equal | F_lSA_rSF | 16.499566 | 31.119792 | 12.174479 |
| EXP | all_six_equal | F_lSA_rSA | 16.353796 | 30.989583 | 11.979167 |
| EXP | all_six_equal | F_lSA_rSAshuffled | 16.300454 | 30.989583 | 12.369792 |
| EXP | all_six_equal | F_lSAshuffled_rSAshuffled | 16.473084 | 31.640625 | 12.174479 |
| EXP | all_six_equal | A_lSA_rSA | 16.346116 | 30.729167 | 12.109375 |
| EXP | all_six_equal | C_lSA_rSA | 16.204012 | 30.859375 | 12.174479 |
| EXP | all_six_equal | ABC_lSA_rSA | 16.280616 | 30.924479 | 11.914062 |
| EXP | all_six_equal | ABC_lSA_rzero | 16.359192 | 30.989583 | 12.174479 |
| EXP | all_six_equal | ABC_native | 16.321800 | 31.250000 | 12.434896 |
| EXP | waymo | original_source | 13.450978 | 28.906250 | 6.835938 |
| EXP | waymo | prepared_F | 13.107853 | 27.734375 | 5.859375 |
| EXP | waymo | native_F | 13.085533 | 27.929688 | 6.445312 |
| EXP | waymo | repaired_F | 12.616854 | 27.148438 | 6.054688 |
| EXP | waymo | lateOnly_F | 12.839589 | 27.539062 | 6.054688 |
| EXP | waymo | F_lSA_rSA | 12.855988 | 27.343750 | 6.054688 |
| EXP | waymo | ABC_lSA_rSA | 13.024688 | 28.125000 | 6.054688 |
| EXP | waymo | ABC_lSA_rzero | 13.109440 | 28.320312 | 6.054688 |
| EXP | waymo | ABC_native | 13.399290 | 28.710938 | 6.640625 |
| EXP | drone | original_source | 11.525665 | 20.898438 | 9.375000 |
| EXP | drone | prepared_F | 11.874628 | 20.312500 | 9.375000 |
| EXP | drone | native_F | 12.067931 | 22.265625 | 9.765625 |
| EXP | drone | repaired_F | 12.240490 | 22.460938 | 9.570312 |
| EXP | drone | lateOnly_F | 12.383083 | 22.656250 | 9.570312 |
| EXP | drone | F_lSA_rSA | 12.229409 | 22.070312 | 9.570312 |
| EXP | drone | ABC_lSA_rSA | 12.009389 | 21.679688 | 9.375000 |
| EXP | drone | ABC_lSA_rzero | 11.947392 | 21.679688 | 9.179688 |
| EXP | drone | ABC_native | 11.933850 | 22.265625 | 9.570312 |
| EXP | quad | original_source | 23.457215 | 41.992188 | 21.093750 |
| EXP | quad | prepared_F | 23.732826 | 43.750000 | 20.898438 |
| EXP | quad | native_F | 23.558326 | 42.382812 | 21.093750 |
| EXP | quad | repaired_F | 24.087611 | 43.164062 | 20.703125 |
| EXP | quad | lateOnly_F | 24.096638 | 43.359375 | 20.898438 |
| EXP | quad | F_lSA_rSA | 23.975990 | 43.554688 | 20.312500 |
| EXP | quad | ABC_lSA_rSA | 23.807769 | 42.968750 | 20.312500 |
| EXP | quad | ABC_lSA_rzero | 24.020745 | 42.968750 | 21.289062 |
| EXP | quad | ABC_native | 23.632260 | 42.773438 | 21.093750 |

## 预注册裁决

```json
{
  "primary": "F_lSA_rSA",
  "cohorts": {
    "DEV": {
      "mIoU_percent": 17.354111514237356,
      "gains_pp": {
        "native_F": 0.37979400251766293,
        "prepared_F": 0.6508812212785102,
        "repaired_F": 0.08845316594408104,
        "lateOnly_F": 0.1739612164479496,
        "ABC_native": 0.6200958195745798,
        "ABC_lSA_rSA": 0.2763776591366529,
        "ABC_lSA_rzero": 0.3207888377079655,
        "F_lSA_rSAshuffled": 0.20280760670227949,
        "F_lSAshuffled_rSAshuffled": 0.4258191242903422
      },
      "passes_all_prelocked_controls": true
    },
    "EXP": {
      "mIoU_percent": 16.35379573915937,
      "gains_pp": {
        "native_F": 0.11653234097704868,
        "prepared_F": 0.11536008561643385,
        "repaired_F": 0.03881084171162286,
        "lateOnly_F": -0.08597418514381161,
        "ABC_native": 0.03199552679145512,
        "ABC_lSA_rSA": 0.07318013588611905,
        "ABC_lSA_rzero": -0.005396639234199085,
        "F_lSA_rSAshuffled": 0.053341668712874934,
        "F_lSAshuffled_rSAshuffled": -0.11928812844188386
      },
      "passes_all_prelocked_controls": false
    }
  },
  "TTA_adopted": false,
  "FINAL_access": false,
  "posthoc_primary_switch": false,
  "static_primary_gate": false
}
```

主项必须在两个范围分别超过全部既定强对照，包括同信息ABC和打乱对应；没有按结果更换主项或按平台拼接。门槛通过也只支持冻结读出的可用性，不构成TTA收益。若未通过，停止当前F/兼容来源配方，不将其输出再拟合成伪标签来挽救；某个次项正值照常保留但需要后续独立锁定评价。

## 正负尾部和组件作用

```json
[
  {
    "cohort": "DEV",
    "method": "F_lSA_rSA",
    "control": "repaired_F",
    "n": 768,
    "gain_pp": 0.088453165944076,
    "better": 10,
    "worse": 10,
    "same": 748,
    "largest_positive_single_input_pp": 0.04378231070238453,
    "largest_negative_single_input_pp": -0.027260811839140964
  },
  {
    "cohort": "DEV",
    "method": "F_lSA_rSA",
    "control": "ABC_native",
    "n": 768,
    "gain_pp": 0.6200958195745794,
    "better": 61,
    "worse": 43,
    "same": 664,
    "largest_positive_single_input_pp": 0.08699899059864145,
    "largest_negative_single_input_pp": -0.11153539380202347
  },
  {
    "cohort": "DEV",
    "method": "F_lSA_rSA",
    "control": "ABC_lSA_rSA",
    "n": 768,
    "gain_pp": 0.27637765913664947,
    "better": 22,
    "worse": 13,
    "same": 733,
    "largest_positive_single_input_pp": 0.08265650351809108,
    "largest_negative_single_input_pp": -0.0893579070700335
  },
  {
    "cohort": "DEV",
    "method": "F_lSA_rSA",
    "control": "F_lSA_rSAshuffled",
    "n": 768,
    "gain_pp": 0.20280760670227596,
    "better": 31,
    "worse": 26,
    "same": 711,
    "largest_positive_single_input_pp": 0.05663417203334555,
    "largest_negative_single_input_pp": -0.0522887786906897
  },
  {
    "cohort": "DEV",
    "method": "F_lSA_rSA",
    "control": "F_lSA_rzero",
    "n": 768,
    "gain_pp": 0.12581357139471483,
    "better": 10,
    "worse": 10,
    "same": 748,
    "largest_positive_single_input_pp": 0.08699899059864145,
    "largest_negative_single_input_pp": -0.03779742334418918
  },
  {
    "cohort": "DEV",
    "method": "F_lSA_rSA",
    "control": "F_lSF_rSF",
    "n": 768,
    "gain_pp": 0.030867435493126764,
    "better": 3,
    "worse": 1,
    "same": 764,
    "largest_positive_single_input_pp": 0.035511212473934566,
    "largest_negative_single_input_pp": -0.01123783571383455
  },
  {
    "cohort": "EXP",
    "method": "F_lSA_rSA",
    "control": "repaired_F",
    "n": 1536,
    "gain_pp": 0.03881084171162131,
    "better": 20,
    "worse": 20,
    "same": 1496,
    "largest_positive_single_input_pp": 0.03803720941588019,
    "largest_negative_single_input_pp": -0.04764303521813209
  },
  {
    "cohort": "EXP",
    "method": "F_lSA_rSA",
    "control": "ABC_native",
    "n": 1536,
    "gain_pp": 0.031995526791456085,
    "better": 106,
    "worse": 92,
    "same": 1338,
    "largest_positive_single_input_pp": 0.05015647696241018,
    "largest_negative_single_input_pp": -0.041491082509298706
  },
  {
    "cohort": "EXP",
    "method": "F_lSA_rSA",
    "control": "ABC_lSA_rSA",
    "n": 1536,
    "gain_pp": 0.07318013588612092,
    "better": 35,
    "worse": 20,
    "same": 1481,
    "largest_positive_single_input_pp": 0.04503449860016842,
    "largest_negative_single_input_pp": -0.041491082509298706
  },
  {
    "cohort": "EXP",
    "method": "F_lSA_rSA",
    "control": "F_lSA_rSAshuffled",
    "n": 1536,
    "gain_pp": 0.0533416687128787,
    "better": 54,
    "worse": 44,
    "same": 1438,
    "largest_positive_single_input_pp": 0.04729911980131544,
    "largest_negative_single_input_pp": -0.04764303521813209
  },
  {
    "cohort": "EXP",
    "method": "F_lSA_rSA",
    "control": "F_lSA_rzero",
    "n": 1536,
    "gain_pp": -0.06986999556415754,
    "better": 22,
    "worse": 22,
    "same": 1492,
    "largest_positive_single_input_pp": 0.04729911980131544,
    "largest_negative_single_input_pp": -0.04764303521813209
  },
  {
    "cohort": "EXP",
    "method": "F_lSA_rSA",
    "control": "F_lSF_rSF",
    "n": 1536,
    "gain_pp": -0.058051125416778364,
    "better": 5,
    "worse": 9,
    "same": 1522,
    "largest_positive_single_input_pp": 0.03803720941588019,
    "largest_negative_single_input_pp": -0.04272420444360952
  }
]
```

这些是任务分数变化，未逐一看图的实例身份保持UNKNOWN。source本来正确而新读出恢复原选择，与新发现目标身份是不同机制；重复框之间切换也不自动等于实例纠错。未基于good/bad案例补规则。完整正负例、每个平台和场景均随公开匿名表保留。当前原始可见RGB始终为A，多采样/多LiDAR不宣称新RGB视角。稀少Drone/Quad场景不支持总体跨场景置信区间。

## 本轮结果解释

主项DEV17.354112%，相对修复F+0.088453pp，全部预设控制通过；但EXP16.353796%，相对lateOnlyF低0.085974pp、相对去残差ABC低0.005397pp、相对late和残差联合打乱低0.119288pp。因此否定本轮“同步A兼容上下文足以稳定修复”的配方，不启动由这条静态主项触发的live-context GN拟合，不把开发通过包装成扩展成功。

按组件看，source-compatible残差SF与同步A残差SA并不等价：在相同SA late下，SF组合DEV17.349002%、EXP16.499566%，SA组合DEV17.354112%、EXP16.353796%。这项次结果完整保留，说明不能把所有上下文失效简单归因于“RGB只来自A而点来自融合”：恢复同步A没有统一改善残差。SF组合是已看过GT后才显得较好的次项，未预注册为主项、没有其匹配的SF残差打乱控制，不能直接宣布已发现可靠绑定或采纳为TTA。

原生基础+同期源模块证据确有局部总体增量，但竞争残差是否包含可迁移实例证据仍不清楚；当前结果同时允许“训练输入兼容性改善”和“对少量排序边界的数值校准”。此前训练来源记录已核验：三个source仅通用RoBERTa加随机grounding训练，没有联合三平台任务checkpoint初始化；不是混域初始权重解释当前差异。
