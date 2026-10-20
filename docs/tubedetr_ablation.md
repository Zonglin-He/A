# TubeDETR 两方向五行消融

用户2026-10-20明确本机执行TubeDETR五行消融；TA-STVG仍在本机任务范围之外。误建的TA准备代码、文档、临时分支和工作树已删除，没有发布TA新增代码。

## 运行与五行关系

| 设置 | 空间轨迹 | 时间区间 |
|---|---|---|
| Source | Source | Native |
| 仅空间分支 | Live | Native |
| 仅时间分支 | Source | Strong |
| 去掉空间OPD | E_Ref | Strong |
| 完整方法 | Live | Strong |

正在运行的TubeDETR已保存全部所需数组。八个位置来自原生Source；初始化只读Source和同八帧ReferDINO；空间OPD不读取Strong区间。因此这五行可以精确地从同一次完整预测读出，没有新增GPU前向或适应任务。Native保留TubeDETR原生MAP时间读出；Strong保留已冻结的高分辨率/六层后验与物理MBR。Source与Full直接复用主实验，不重复启动。

目标总体为HC1902/227和VidSTG10303/732，保持checkpoint、算法、参数、seed41、八帧、单专家、冻结网络与每query状态重置。新工具只增加离线评分，不改运行中的模型代码、配置、协议和预测回执。Source/初始化/最终轨迹及时间区间以物理帧对齐，逐query预测不重写。

## 指标与组件差值

主表按用户明确列出的5项报告：m_vIoU、vIoU@0.3、vIoU@0.5、m_tIoU、m_sIoU。补充表4项：tIoU≥0.5比例、tIoU=0比例、秒单位边界MAE、八个观测位置中有GT实际节点的sIoU。原表合计9项；未观测实际节点sIoU作为另一个诊断保存，加上它共有10项定位/分析指标。观测与未观测的GT节点分母分别保存，无GT节点时记null。

主平均按query，另存父来源等权平均、聚类bootstrap和配对统计。组件比较为WithoutOPD−Temporal（初始化）、Full−WithoutOPD（OPD）、Full−Spatial（时间）、Full−Temporal（整个空间）、Full−Source（整体）。方向与五行对应关系保持一致，不引入中间开发版本。

## 独立离线执行

```bash
.conda/tubedetr/bin/python -B tools/complete_tubedetr_ablation.py --prepare-only
.conda/tubedetr/bin/python -B -u tools/complete_tubedetr_ablation.py --wait
```

计划见 [ablation_plan.json](../results/tubedetr/ablation_plan.json)。工具在独立CPU进程中等待每个方向的完整sealed.json；核对完整总体、backbone、源权重、运行身份和封存后才启动标签读取评分。当前方向未完整时不计算或公布部分准确率。任一方向的封存身份不符则失败，保留证据，不跳过query。

本地状态在outputs/tubedetr_ablation/status.json，完整指标在各方向metrics_five_rows.json，最后生成results/tubedetr/ablation_report.md、ablation_summary.json、去除媒体标识的逐query与聚类结果及ablation_seal.json。外部TTA不重新运行，效率和Source/Ours/GT定性继续使用现有完整主实验流程。
