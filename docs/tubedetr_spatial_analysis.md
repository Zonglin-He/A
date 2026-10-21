# TubeDETR 空间诊断与有界耦合复查

2026-10-21用户要求“空间上好好分析好好调参”。这次在原两个完整方向与已经评分的Vid128候选上追加空间诊断，不改变正在运行的Vid所选全量，不从部分全量输出选参。新增tools/analyze_tubedetr_spatial.py只读完整128候选seal、score receipt、预测与同一专家缓存，校验后再读取离线标签；没有GPU调用。结果见[Vid空间诊断](../results/tubedetr_spatial_analysis/vidstg.json)。

## 实际发现

Vid选中gamma0.05/lr0.02/tau1/60步，Full相对自身初始化的m_vIoU为+0.146646pp，但dense空间IoU为−0.258231pp，未观测实际帧空间IoU为−0.353342pp；八个观测位置有GT的102条为+2.265177pp。不能把完整定位提高解释为整个空间轨迹改善。

在相同的102条观测GT有效query中，单ReferDINO观测IoU平均53.698208%，Source为51.025364%。专家优于Source的61条，60步OPD相对初始化dense sIoU为+2.022366pp、未观测实际帧+1.745906pp；专家不优于Source的41条分别−2.191035pp、−2.059146pp。另26条八个观测位置均无GT事件节点，dense损失−2.560979pp。分组依据GT，只作事后诊断，绝不据此门控或挑选推理query。

按到最近观测位置的物理秒距离分桶，在选中60步下，OPD相对初始化的dense sIoU差为：观测点+2.265177pp，0–0.5秒+0.460113pp，0.5–1秒+0.127015pp，超过1秒−0.259512pp。各桶先按query内GT帧平均，再在该桶有GT的query间等权平均，分别102/109/95/87条；不能把不同桶的绝对均值当作同一总体比较。原无外推规则保持，实际八位置指标与dense插值指标分开。

因此有三个问题需要区分：初始化的全轨迹修正强度、专家相对学生的错误风险、适应步数与传播范围的耦合。Source本已准确时的损伤、事件外观测与教师失配都不能仅靠提高学习率解决。现有回放/零状态/梯度/物理帧检查通过，不能因此声称伪标签正确或空间泛化已完成。

## HC候选开始前的修订

HC控制器原来只在CPU等待、候选0/GPU0；根代理验证身份后停止该CPU等待器，保留[原v1计划](../results/tubedetr_hc2_development/plan_v1_before_spatial_followup.json)，不动任何GPU进程。新[HC计划](../results/tubedetr_hc2_development/plan.json)在原gamma序列前补0.0125/0.025，其他学习率/温度/步数和4阶段顺序不变，GPU fit-path预算仍9×128+5。固定HC128、源权重、模型、原配置和原诊断不变。每阶段评分后自动生成HC空间诊断，报告dense/观测/未观测、尾损和教师分组；主选择目标不偷换，仍是Full query平均m_vIoU。新计划绑定空间诊断工具hash；没有在看过HC候选结果后修改计划。

## Vid追加的有限空间复查

这是一轮单独登记的后续开发，不改当前Vid选择锁，也不覆盖当前10303全量。先完成当前Vid全量和HC调参后1902全量，验证完整final seals及相应进程退出，之后才运行追加候选。计划见[空间复查plan](../results/tubedetr_vid_spatial_followup/plan.json)。

同一份输入哈希Vid128（128父来源各一条、seed41），固定tau1和alpha1，对gamma0.0125/0.025、lr0.01/0.02作四条真实适应路径，每条运行60步并保存15/30/60前缀，共12个读出。最多4×128新fit加1次原配置资格回放；不扩展预算。Source/Native/Strong和单ReferDINO证据精确缓存复用，原配置回放、网络冻结、每query状态重置、真实decoder输出与物理帧检查沿用已冻结的Vid执行代码；新worker在独立进程只重绑定计划与输出目录，原执行代码不编辑。绝不插值旧Live假装新学习率或初始化拟合。

选择规则在候选之前冻结：以此前Vid锁定配置在同128的m_vIoU和未观测实际帧sIoU为下限，在不降低这两项的候选中最大化dense m_sIoU；严格相同优先保留此前配置，再优先少步数。保留所有正负结果，原选中配置一直作为候选。若没有更好的可行配置，精确复用已完成的此前全量，报告负结果，不另跑重复GPU；若有则先锁定新profile，再做一次完整10303评价，不从新全量结果反向选参。

此规则只避免相对此前配置继续损伤，不保证超过Source。报告必须同时给出相对Source和仅时间分支的净空间贡献；若dense sIoU仍低于Source，不声称空间问题已解决。128已曝光，原完整目标划分也已用于假设诊断，后续全量不称独立确认。缓存执行成本不冒充新端到端效率。

## 执行与监控

```bash
.conda/tubedetr/bin/python -B tools/analyze_tubedetr_spatial.py --dataset vidstg
.conda/tubedetr/bin/python -B tools/tubedetr_vid_spatial_followup.py --prepare-only
.conda/tubedetr/bin/python -B -u tools/tubedetr_vid_spatial_followup.py --wait
```

新增CPU等待器runtime key vid_spatial_followup_controller，未来GPU worker key vid_spatial_followup_gpu；检查outputs/tubedetr_vid_spatial_followup/status、failure、worker_progress、logs/complete_tubedetr_vid_spatial_followup.log及阶段日志，results下plan/selected_profile/full_metrics/report/final_seal。监控继续只读，不自行修复/启动GPU。停止条件增加本轮空间复查的完整final seal（包括没有更好可行候选时的原全量复用证据），不能原两轮结束即停止。
