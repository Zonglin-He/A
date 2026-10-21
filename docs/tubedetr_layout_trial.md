# TubeDETR 单ReferDINO同帧ROI布局奖励初试

2026-10-21用户新指示：“调参先别管了，跑完全量后，我想试这个”。优先完成当前HC2→VidSTG选中配置10303全量及独立评分；候选0/GPU0的HC调参、Vid空间耦合复查CPU等待器已按用户指示停止，状态deferred_by_user_for_layout_reward_trial、automatic_resume=false，原计划/清单/结果保留，不再自动接续它们。优先级回执为runtime/tubedetr_experiment_priority.json。

此后只做一次固定Vid128的奖励初试，不默认启动新全量，不展开学习率/初始化/步数搜索。新[plan](../results/tubedetr_layout_trial/plan.json)在候选前冻结。当前正式主方法和生产代码仍是原几何OPD；试验全部放在tools/layout_trial下，不改stvg、第三方源码、原配置、当前Vid选择锁与运行身份。

## 奖励及坐标

只用现有单ReferDINO检查点。利用该模型原有八帧infer调用的forward hook缓存pixel_decoder输出（stride4、float32）；不再加载独立GroundingDINO/其他专家，不改变原生选轨。hook采集前后必须逐位复现旧同八帧教师框，教师完整权重状态hash不变。学生框和教师框都在同一帧同一教师特征图采样，不要求TubeDETR与ReferDINO特征空间对齐，也不做跨帧布局匹配。

每个ROI在6×6等距bin中心作双线性采样；每2×2平均形成有序3×3、再逐位置L2归一化。九个位置固定对应，奖励为九个余弦均值。1×1对照先平均相同36个采样点，再归一化，避免采样覆盖数量不同的混淆。原归一化cxcywh经保持比例Resize映射，stride4特征图的尾部padding用valid_ratio修正；align_corners=false。框不裁剪，图像外采样记0，零向量不产生虚假的完美相似度。

OPD反馈权重为softmax(layout_reward/tau)。原对称32候选、双尺度、噪声key、混合密度损失、解析梯度核验、Adam、8×256临时content注入和finally重置保持原实现；几何L1/GIoU不计算或混入布局反馈。对照的几何OPD与几何Direct仍使用原几何损失。候选反馈权重stop-gradient，网络权重冻结。

## 固定初试与必要对照

沿用已明确曝光的固定Vid128／128父视频／seed41，不重新按结果挑选query。共同配置使用此前锁定的gamma0.05、lr0.02、60步、alpha1和相同初始化/八帧/Source/Strong；几何OPD精确复用同128的已有选中预测，其他四条路径真实适应。

| 名称 | 评价奖励/损失 | 更新 |
|---|---|---|
| GeometryOPD | 原5×L1＋2×(1−GIoU) | 原OPD，精确参考缓存 |
| Layout1OPD | 同帧1×1池化余弦 | 原OPD |
| Layout3OPD | 同帧有序3×3布局余弦 | 原OPD |
| Layout3Direct | 1−同一3×3奖励 | 对同一临时content直接反传 |
| GeometryDirect | 原几何损失 | 对同一临时content直接反传 |

两类Direct使用相同固定Adam预算，不做独立最优学习率搜索。因此这轮只能检验同预算可行性，不能当作“公平调参后证明OPD必要”的最终结果。Cross-attention具体框评分未实现，不把mask/置信度冒充Cross-attention，不给三种方案的性能排名。没有对抗鲁棒或恢复目标身份的先验保证。

奖励量纲变化需要温度校准：在固定128的每条初始化状态，用同一32候选对布局和几何计算反馈；为3×3和1×1各选一个全局tau，使平均归一化权重熵接近原几何tau1的熵。离散tau为0.0001/0.0003/0.001/0.003/0.01/0.03/0.1/0.3/1，全程不读GT、不按query单独选温度。保存全部熵值与是否处于网格边界，不承诺该无标签校准是性能最优温度。

## 顺序、资格与标签隔离

1. 等当前10303全量final seal、引用hash正确，且当前控制器/worker退出；被用户暂缓的旧队列也不能仍存活。等待阶段无新GPU。
2. 独立教师进程取得128条八帧特征与相同框，缓存原生替代query框用于诊断；逐条证据hash与旧单专家一致，物理帧ID/RGB绑定。特征只缓存一次，3×3/1×1/Direct共用。阶段结束释放教师，再由独立学生worker工作，避免models命名空间冲突及并存模型开销。
3. 无标签温度校准，同时封存共同诊断候选：32原policy候选、教师/Source、固定中心偏移/缩放/宽高比、同一次教师原生替代query框。替代query框不冒充已知不同语义身份。
4. 学生第一条固定query须逐位复现已有几何OPD完整60步；原生Source/Native、八个物理帧与RGB核对，零状态与初始化一致。四条新路径均冻结模型、每query新状态与优化器并重置，检查有限梯度及模型前后hash。最多4×128新适应路径加1次参考资格回放。
5. 所有预测完整封存后，独立CPU评分与奖励排序诊断才读GT。报告m_vIoU、两个阈值、dense/观测/未观测sIoU、父来源平均与聚类统计。奖励排序用相同候选的GT IoU作Spearman与成对排序准确率，先帧内、再query内、再有效query等权；缺少观测GT的query单列分母。GT不回流到奖励/推理。

同一面板和原完整目标结果已曝光，不能称独立确认。特征缓存和Strong复用的执行成本单独记录，不冒充新端到端部署效率，ReferDINO特征提取成本不隐去。CPU检查或代码编译通过不等于真实GPU资格检查或定位收益。

## 与已有工作及历史Fine ROI的关系

[Fine-Grained Feature Imitation, CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Distilling_Object_Detectors_With_Fine-Grained_Feature_Imitation_CVPR_2019_paper.html)研究目标附近局部特征响应；[Focal and Global Knowledge Distillation, CVPR2022](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_Focal_and_Global_Knowledge_Distillation_for_Detectors_CVPR_2022_paper.html)研究前景/背景区别及空间关系迁移。它们支持关注区域空间信息的动机，没有验证本试验的同帧3×3奖励或OPD。这里的具体奖励、采样及效果是待验证设计。

用户已指出历史Fine ROI属于同一方向，曾有开发小收益但无稳定确认。当前目录与已检查历史未找到可直接复用的实现，因此登记为独立同方向试验，不宣称全新方案，不把旧负结果当作已解决。

## 执行与监控

```bash
.conda/tubedetr/bin/python -B tools/complete_tubedetr_layout_trial.py --prepare-only
.conda/tubedetr/bin/python -B -u tools/complete_tubedetr_layout_trial.py --wait
```

runtime key layout_trial_controller负责有限接续；layout_trial_worker记录features/calibration/adaptation当前阶段PID、cmdline和日志。outputs/tubedetr_layout_trial下status、worker_progress、failure、qualification、特征seal、无标签温度、诊断seal及适应seal是分阶段证据；results下plan、summary、reward_diagnostics、temperature_calibration、metrics、report和final_seal是最终公开证据。监控仍只读，禁止自行恢复旧调参或新建GPU工作。新停止条件为当前10303全量及本固定128初试完整终态并核对发布状态；旧暂缓任务不要求完成。
