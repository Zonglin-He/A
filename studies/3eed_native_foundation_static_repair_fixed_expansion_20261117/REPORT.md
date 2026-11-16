# R117 — 固定原生基础修复的扩展验证

本轮是冻结架构验证，不是R115逐输入gate配方的扩展TTA。沿用R109的1536定向输入/768表达，DEV元数据路径不重叠，但历史fullVAL/R109评分曝光，不是新独立测试。Waymo/Drone/Quad各256表达、两个incoming source；110保护媒体别名仅冻结prepared回退。FINAL未访问，source100/R91参数未重训，没有专家。

## 固定配置和数据边界

主项撤回point_adapter.up和earlyRGB gateP，保留原native点—文字基础、RGB桥/late和竞争残差；另做native纯匹配、rOnly、lateOnly以及同预算ABC概率平均。原source、原preparedA/F及旧CE池均直接复用。R115用无标签DEV梯度得到的source固定门均值及等范数反向，只是冻结校准对照，不在扩展输入重新拟合、不将其称为输入特定TTA。所有原A256/六层位置固定，不新增框。

R109没有保存prefix和对齐PointNet向量，本轮只提取这些缺失前置表示，复用所有原注册/时间候选/支持阈值和sample哈希，不重新SIFT/PnP。实际旧preparedA/F隐藏态在12真实工程输入上位等，正式逐条核验原source概率、固定框、注册/采样/有效支持，与旧实验一致。当前A原生h/u与捕获原source位等。所有prefix/观测保存，后续无需再做这些前置提取。

## 全体与平台

| 范围 | 方法 | mIoU% | Acc25% | Acc50% |
|---|---|---:|---:|---:|
| all_six_equal | original_source | 16.144619 | 30.598958 | 12.434896 |
| all_six_equal | prepared_frozen | 16.251067 | 30.598958 | 11.783854 |
| all_six_equal | real_BBF_fusedABC | 16.238436 | 30.598958 | 12.044271 |
| all_six_equal | real_BBF_ensemble | 16.120167 | 30.533854 | 11.783854 |
| all_six_equal | real_softtoken_geometric_fusedABC | 16.032048 | 30.338542 | 11.979167 |
| all_six_equal | real_softtoken_arithmetic_fusedABC | 16.013494 | 30.338542 | 12.044271 |
| all_six_equal | native_A | 16.138541 | 30.664062 | 12.239583 |
| all_six_equal | native_C | 16.140763 | 30.794271 | 12.304688 |
| all_six_equal | native_fusedABC | 16.237263 | 30.859375 | 12.434896 |
| all_six_equal | rOnly_A | 16.254676 | 30.794271 | 12.369792 |
| all_six_equal | rOnly_fusedABC | 16.205650 | 30.729167 | 12.369792 |
| all_six_equal | lateOnly_A | 16.211395 | 30.598958 | 12.044271 |
| all_six_equal | lateOnly_fusedABC | 16.439770 | 31.184896 | 12.174479 |
| all_six_equal | repaired_A | 16.327146 | 30.989583 | 12.174479 |
| all_six_equal | repaired_C | 16.327471 | 31.054688 | 12.369792 |
| all_six_equal | repaired_fusedABC | 16.314985 | 30.924479 | 12.109375 |
| all_six_equal | fixed_calibration_A | 16.219253 | 30.859375 | 12.109375 |
| all_six_equal | fixed_calibration_C | 16.331662 | 31.054688 | 12.369792 |
| all_six_equal | fixed_calibration_fusedABC | 16.304593 | 30.924479 | 12.239583 |
| all_six_equal | reverse_calibration_A | 16.310094 | 31.119792 | 12.174479 |
| all_six_equal | reverse_calibration_C | 16.324800 | 31.119792 | 12.434896 |
| all_six_equal | reverse_calibration_fusedABC | 16.237096 | 30.859375 | 12.109375 |
| all_six_equal | repaired_ABC_mean | 16.416137 | 31.250000 | 12.239583 |
| waymo | prepared_frozen | 13.214120 | 28.125000 | 6.250000 |
| waymo | real_BBF_fusedABC | 13.107853 | 27.734375 | 5.859375 |
| waymo | native_fusedABC | 13.085533 | 27.929688 | 6.445312 |
| waymo | lateOnly_fusedABC | 12.839589 | 27.539062 | 6.054688 |
| waymo | repaired_fusedABC | 12.616854 | 27.148438 | 6.054688 |
| waymo | fixed_calibration_fusedABC | 12.553006 | 26.953125 | 6.054688 |
| waymo | repaired_ABC_mean | 12.841848 | 27.734375 | 6.054688 |
| drone | prepared_frozen | 12.009923 | 20.312500 | 9.179688 |
| drone | real_BBF_fusedABC | 11.874628 | 20.312500 | 9.375000 |
| drone | native_fusedABC | 12.067931 | 22.265625 | 9.765625 |
| drone | lateOnly_fusedABC | 12.383083 | 22.656250 | 9.570312 |
| drone | repaired_fusedABC | 12.240490 | 22.460938 | 9.570312 |
| drone | fixed_calibration_fusedABC | 12.226618 | 22.265625 | 9.765625 |
| drone | repaired_ABC_mean | 12.073962 | 22.265625 | 9.375000 |
| quad | prepared_frozen | 23.529157 | 43.359375 | 19.921875 |
| quad | real_BBF_fusedABC | 23.732826 | 43.750000 | 20.898438 |
| quad | native_fusedABC | 23.558326 | 42.382812 | 21.093750 |
| quad | lateOnly_fusedABC | 24.096638 | 43.359375 | 20.898438 |
| quad | repaired_fusedABC | 24.087611 | 43.164062 | 20.703125 |
| quad | fixed_calibration_fusedABC | 24.134156 | 43.554688 | 20.898438 |
| quad | repaired_ABC_mean | 24.332600 | 43.750000 | 21.289062 |

## 裁决

主修复16.314985%，高于preparedF16.238436与nativeF16.237263，但低于同信息ABC概率平均16.416137，固定主项未通过全部强对照。源域Point Adapter/earlyRGB的DEV撤回增益没有等幅保持，不能将其当普适已修复架构。固定门校准16.304593低于主修复16.314985，其DEV额外正收益同样未保留；不启动在线单门或域级校准TTA来挽救。

组件消融最明确的是，保留RGB late而去掉竞争残差达到16.439770，rOnly16.205650反而低于nativeF16.237263。与DEV中r有正贡献不同，当前竞争残差在新观测上未带来总体增量。撤回原Point Adapter后，r头的输入也改变了；它可能是残差可靠性或训练输入条件失配，尚不能仅凭分数确定。R92的旧完整架构残差漂移阴性不排除这个新条件。所有较强次项保留，不切换主项或据平台GT拼接方案。

## 完整正负尾部

```json
[
  {
    "control": "native_fusedABC",
    "n": 1536,
    "gain_pp": 0.07772149926542565,
    "better": 85,
    "worse": 81,
    "same": 1370,
    "largest_positive_single_input_pp": 0.05015647696241018,
    "largest_negative_single_input_pp": -0.03871057962902432
  },
  {
    "control": "real_BBF_fusedABC",
    "n": 1536,
    "gain_pp": 0.07654924390481001,
    "better": 228,
    "worse": 190,
    "same": 1118,
    "largest_positive_single_input_pp": 0.04764303521813209,
    "largest_negative_single_input_pp": -0.041491082509298706
  },
  {
    "control": "repaired_ABC_mean",
    "n": 1536,
    "gain_pp": -0.10115174937605448,
    "better": 33,
    "worse": 27,
    "same": 1476,
    "largest_positive_single_input_pp": 0.04503449860016842,
    "largest_negative_single_input_pp": -0.041491082509298706
  },
  {
    "control": "lateOnly_fusedABC",
    "n": 1536,
    "gain_pp": -0.12478502685543275,
    "better": 18,
    "worse": 27,
    "same": 1491,
    "largest_positive_single_input_pp": 0.04272420444360952,
    "largest_negative_single_input_pp": -0.03608102466861337
  },
  {
    "control": "fixed_calibration_fusedABC",
    "n": 1536,
    "gain_pp": 0.010391668460737791,
    "better": 5,
    "worse": 7,
    "same": 1524,
    "largest_positive_single_input_pp": 0.023608097844037824,
    "largest_negative_single_input_pp": -0.036668733750953274
  }
]
```

独立样本和场景数量没有因两个source复制同一表达而翻倍；Waymo场景较多，Drone/Quad仍是少数原场景，不做跨新场景稳定性宣称。所有别名/不可用C/负例保留总分分母，IoU变化不自动认证实例身份。

## 审计

23读出共35328任务值，17新26112与6旧9216复用；24242新CPU端点，全部新任务值由两种几何实现核验，平台/场景/paired由独立代码重算。概率误差<5e-6、近平0。12工程后仅补保存已经计算的B基础以便独立CPU审计ABC均值，不改变推理或重跑NN；元数据修订留档。所有新预测先封存再评分。

下一项只检验竞争头输入条件：在相同固定native基础上，比较原native特征读出的r、原source-compatible特征读出的r、无r及查询对应打乱。没有必要再开一轮GN层或Huber/MSE强度搜索。

## 私有图页核查与主体跨度覆盖

生成11页全图/局部/BEV/相同A256排序图，实际查看3页，其余UNKNOWN。一条正例只是从远处空间假设移向近车邻域，IoU0→0.041且oracle仅0.2498，不能叫准确grounding；一条负例中，加入r把native/lateOnly的左侧行人区域（IoU0.5542）切到另一右侧/中部区域（IoU0）。这些图解释失败类型，不决定分母或产生个案规则；也没有仅凭图认证输入失配是原因。

额外无任务GT检查沿用未修改R48纯文字parser：DEV362个非别名独立表达、EXP713个非别名表达中，仅1/2条Waymo表达与旧firstlexiconnoun mask不同，Drone/Quad全部一致。UNKNOWN沿旧mask回退，tokenizer/旧mask逐位复现。它不能解释当前Drone/Quad的总体TTA问题，未为3条输入追加词规则或新P0方法。旧R48的336条阴性范围因此得到更大纯文字覆盖，但没有把未知跨度宣称正确。

补看另一正例：lateOnly的0.5669→r读出的0.6795，两框覆盖同一个右边缘行人，原source/native也已经是0.6795。它主要恢复被RGB读出改坏的同物体候选，不能与不同实例的纠错混在一起。三页只是事后语义解释，所有其余8页身份未知，方法决策仍取1536全体。
