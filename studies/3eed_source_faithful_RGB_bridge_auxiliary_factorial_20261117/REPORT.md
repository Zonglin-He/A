# R114 — 原生grounding基础与源域条件RGB桥的分离检验

沿用三个source100和既有R91，DEV768/384表达、六方向、44保护别名冻结回退。原A256及六层位置不变，FINAL未访问，无专家和新source训练。所有24新臂先封存，再17376端点CPU独立读出与18432任务值几何双实现评分；12旧臂9216任务值复用。

## 主项和竞争解释

R113全局撤回point Adapter也改变了RGB桥的key输入及恢复图。本轮让grounding固定原生点—文字基础，RGB桥保留源域point Adapter keys，并用源完整辅助梯度更新45313桥参数；对照为native/sourceRGBkeys×native/source辅助图、同L2错配，以及gateI归零/仅更新gateI/仅更新其余45312参数。所有新提议在C读取前由当前A确定，C只是已有对齐LiDAR、没有新RGB视角；无label接受或逐例选择。固定自然clip1/0.5一步、不回溯/步数网格。全102498参数反向成本保留，不夸称只计算45313梯度。

## 结果

| 范围 | 方法 | mIoU% | Acc25% | Acc50% |
|---|---|---:|---:|---:|
| all_six_equal | original_source | 16.811383 | 30.989583 | 14.973958 |
| all_six_equal | prepared_frozen | 16.228652 | 29.947917 | 12.630208 |
| all_six_equal | real_frozen_fusedABC | 16.703230 | 30.989583 | 13.151042 |
| all_six_equal | native_real_fusedABC | 16.974318 | 31.250000 | 14.843750 |
| all_six_equal | direct_three_geometric | 16.991306 | 32.291667 | 13.671875 |
| all_six_equal | neither_correct_fusedABC | 17.265658 | 31.901042 | 14.322917 |
| all_six_equal | anchor_noLate_fusedABC | 17.120908 | 31.510417 | 14.843750 |
| all_six_equal | anchor_bridge_natural_raw_fusedABC | 17.383951 | 32.031250 | 14.453125 |
| all_six_equal | sourcekey_sourceaux_fusedABC | 17.279658 | 31.901042 | 14.322917 |
| all_six_equal | nativekey_sourceaux_fusedABC | 17.383907 | 32.031250 | 14.453125 |
| all_six_equal | sourcekey_nativeaux_fusedABC | 17.302352 | 31.901042 | 14.322917 |
| all_six_equal | sourcekey_sourceaux_wrong_fusedABC | 17.241468 | 31.770833 | 14.322917 |
| all_six_equal | native_gate_only_fusedABC | 17.383951 | 32.031250 | 14.453125 |
| all_six_equal | native_bridge_no_gate_fusedABC | 17.265658 | 31.901042 | 14.322917 |
| all_six_equal | sourcekey_frozen_fusedABC | 17.271698 | 31.901042 | 14.322917 |
| all_six_equal | gateI0_frozen_fusedABC | 16.953391 | 30.729167 | 14.192708 |
| waymo | real_frozen_fusedABC | 12.569007 | 26.171875 | 5.078125 |
| waymo | native_real_fusedABC | 11.535583 | 23.437500 | 7.421875 |
| waymo | neither_correct_fusedABC | 12.304763 | 25.390625 | 7.812500 |
| waymo | anchor_noLate_fusedABC | 11.484142 | 23.437500 | 7.421875 |
| waymo | anchor_bridge_natural_raw_fusedABC | 12.385622 | 25.390625 | 7.812500 |
| waymo | sourcekey_sourceaux_fusedABC | 12.292030 | 25.390625 | 7.812500 |
| waymo | nativekey_sourceaux_fusedABC | 12.385490 | 25.390625 | 7.812500 |
| waymo | sourcekey_nativeaux_fusedABC | 12.292030 | 25.390625 | 7.812500 |
| waymo | sourcekey_sourceaux_wrong_fusedABC | 12.292030 | 25.390625 | 7.812500 |
| waymo | native_gate_only_fusedABC | 12.385622 | 25.390625 | 7.812500 |
| waymo | native_bridge_no_gate_fusedABC | 12.304763 | 25.390625 | 7.812500 |
| waymo | sourcekey_frozen_fusedABC | 12.288986 | 25.390625 | 7.812500 |
| waymo | gateI0_frozen_fusedABC | 11.893641 | 24.218750 | 7.421875 |
| drone | real_frozen_fusedABC | 14.029708 | 25.000000 | 11.718750 |
| drone | native_real_fusedABC | 14.710633 | 26.171875 | 12.890625 |
| drone | neither_correct_fusedABC | 14.503438 | 26.171875 | 11.718750 |
| drone | anchor_noLate_fusedABC | 14.996071 | 26.953125 | 12.890625 |
| drone | anchor_bridge_natural_raw_fusedABC | 14.777456 | 26.562500 | 12.109375 |
| drone | sourcekey_sourceaux_fusedABC | 14.582912 | 26.171875 | 11.718750 |
| drone | nativekey_sourceaux_fusedABC | 14.777456 | 26.562500 | 12.109375 |
| drone | sourcekey_nativeaux_fusedABC | 14.650994 | 26.171875 | 11.718750 |
| drone | sourcekey_sourceaux_wrong_fusedABC | 14.453824 | 25.781250 | 11.718750 |
| drone | native_gate_only_fusedABC | 14.777456 | 26.562500 | 12.109375 |
| drone | native_bridge_no_gate_fusedABC | 14.503438 | 26.171875 | 11.718750 |
| drone | sourcekey_frozen_fusedABC | 14.537333 | 26.171875 | 11.718750 |
| drone | gateI0_frozen_fusedABC | 14.018964 | 24.218750 | 11.718750 |
| quad | real_frozen_fusedABC | 23.510976 | 41.796875 | 22.656250 |
| quad | native_real_fusedABC | 24.676737 | 44.140625 | 24.218750 |
| quad | neither_correct_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | anchor_noLate_fusedABC | 24.882511 | 44.140625 | 24.218750 |
| quad | anchor_bridge_natural_raw_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | sourcekey_sourceaux_fusedABC | 24.964030 | 44.140625 | 23.437500 |
| quad | nativekey_sourceaux_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | sourcekey_nativeaux_fusedABC | 24.964030 | 44.140625 | 23.437500 |
| quad | sourcekey_sourceaux_wrong_fusedABC | 24.978549 | 44.140625 | 23.437500 |
| quad | native_gate_only_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | native_bridge_no_gate_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | sourcekey_frozen_fusedABC | 24.988774 | 44.140625 | 23.437500 |
| quad | gateI0_frozen_fusedABC | 24.947567 | 43.750000 | 23.437500 |

## 裁决

主项17.279658%，只比自己的冻结hybrid高0.007960pp，低于R113自然桥17.383951%。源域条件保留没有建立更强TTA，不采用当前修复配方，不加步数/系数网格。nativekeys+sourceaux17.383907几乎复现旧次项，preparedRGBkeys下的nativeaux17.302352也不超过旧次项。辅助图条件变化是实际计算事实，不能据此宣称它是任务退化的主要原因。

更明确的受控定位：仅gateI更新17.383951%，非gate桥更新17.265658%，分别与旧全桥和零步总体完全相同。完整query相等计数如下；不能只因总体同分就假设所有对象身份相同。固定gateI=0只有16.953391%，直接删除桥有害。下一步应比较输入特定gate更新与统一/打乱gate提议，先区分动态适应和静态校准，不继续扩展注意力更新参数。

```json
[
  {
    "view": "A",
    "n": 768,
    "gateonly_vs_fullbridge_same_query": 768,
    "nongate_vs_frozen_same_query": 768
  },
  {
    "view": "C",
    "n": 768,
    "gateonly_vs_fullbridge_same_query": 768,
    "nongate_vs_frozen_same_query": 768
  },
  {
    "view": "fusedABC",
    "n": 768,
    "gateonly_vs_fullbridge_same_query": 768,
    "nongate_vs_frozen_same_query": 768
  }
]
```

## 正负尾部

```json
[
  {
    "control": "sourcekey_frozen_fusedABC",
    "n": 768,
    "gain_pp": 0.007959732090357354,
    "better": 7,
    "worse": 4,
    "same": 757,
    "largest_positive_single_input_pp": 0.04186852011328779,
    "largest_negative_single_input_pp": -0.03478109382039299
  },
  {
    "control": "neither_correct_fusedABC",
    "n": 768,
    "gain_pp": 0.013999372746648328,
    "better": 6,
    "worse": 5,
    "same": 757,
    "largest_positive_single_input_pp": 0.04186852011328779,
    "largest_negative_single_input_pp": -0.03478109382039299
  },
  {
    "control": "anchor_bridge_natural_raw_fusedABC",
    "n": 768,
    "gain_pp": -0.10429334468005176,
    "better": 4,
    "worse": 6,
    "same": 758,
    "largest_positive_single_input_pp": 0.021325276285936442,
    "largest_negative_single_input_pp": -0.08687719749109822
  }
]
```

good/bad按全固定分母记录并保留匿名完整逐例标量；本轮没有新看原图，实例身份保持UNKNOWN，不能把IoU改变自动称为目标身份纠错。历史R113图册不作为本轮新主项的语义核验。

## 工程边界

12真实输入核验实际分离main和缓存重放位等、RGB源keys冻结late与既有full路径位等、source辅助完整梯度重现及源状态恢复。两个失败attempt来自试图将单独aux梯度与原aux+初始selfkeep梯度逐位比较；极小4.486e-21舍入差仍可改变delta低位。最终使用原physical_direction91完整计算图而非放宽比较，梯度位等通过；原失败代码和日志留存，修正在任务评分前。所有新CPU概率误差<5e-6、无近平赢家歧义。

主方法及全部对照按原锁定解释，不换主项；当前结论只适用于历史暴露DEV，不是独立确认或稳定TTA成功。
