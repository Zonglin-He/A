# 已封存 baseline 评分：根复核

本次217221逻辑行全部评分、870490项独立聚合/阈值/分解统计检查和1740870项公开匿名行复算通过；实际目检三图，数字标签重叠仅作排版修正，原图与代码保留。EATA推理及其源媒体/Fisher、新OPD其他实验均未恢复。

| 方法 | HC源→VidSTG test source vIoU | Vid源→HC2 validation source vIoU |
|---|---:|---:|
| Source Only | 12.065 | 21.072 |
| DINO-Refine | 15.372 | 21.888 |
| TENT-STVG | 9.512 | 17.437 |
| SAR-STVG | 12.065 | 20.994 |
| EATA-STVG | 未运行，用户暂停 | 21.072 |
| Target-trained reference | 21.449 | 29.895 |

主聚合先平均每query三序，再在父source内平均query，最后等权父source；官方query-macro在REPORT/TABLE中独立给出。Vid10303query/732视频；HC2 validation3482clip/query/237父movie，不能叫HC test。target-trained是监督同域参考，不是TTA或数学上界。

DINO-Refine相对Source的source Δv：Vid+3.307647pp，95%源bootstrap CI[2.909295,3.720271]；HC+0.816681pp，[0.171540,1.442783]。HC全GT支持sIoU却从49.399926%降至44.761074%，因此不宣称全轨迹空间质量普遍提高。>5pp源伤害分别16/732和18/237；完整负尾与正例一并发布。

TENT的Δv为Vid−2.552820pp、HC−3.634971pp，两个CI均低于零。主要是继承状态的时间读出损伤：inherited temporal分别−2.542248/−3.605245pp；当前query增量−0.011338/−0.008068pp。Vid138/732、HC75/237父source损失>5pp。该结果适用于冻结移植和状态轨迹，不能推广为所有TENT路线不可能有效。

SAR在Vid30909到达全部被entropy门槛过滤（92727轮），零optimizer更新，所以与Source逐值一致；HC仅15/10446到达更新（45步），Δv−0.077057pp。EATA HC仅3/10446到达更新，Δv+0.000142542pp，CI跨零；12次冗余过滤、10431次entropy过滤。不把几乎零覆盖当作算法已充分适应，也不据结果改门槛或重新调参。

Qualification独立梯度/优化器算术：TENT两真实query/集均有一步更新；现有两SAR资格query/集和两EATA HC资格query都被门槛过滤，数学审计scalar_checks=0。正式记录完整参数/预测SHA链、每100到达优化器快照一致；没有保存的正式梯度/Jacobian，所以本次未复演整个模型反传，也没有据“pass”声称这一缺失已补齐。

正负signal chain已回读。例如Vid DINO query3712 Source v=.650883→After0，admitted GT IoU0；同父source273的query3716 Source0→.792199，admitted GT IoU.953666。HC DINO query2682 .695502→.020196，与query514 .090654→.876495正例均保留。Vid TENT query4961 .791808→Before.049695→After.049695，损失几乎全部来自继承时间；query7448 .177285→.561343为相反的继承时间正例。案例是事后匿名诊断，不用于线上选择。

真实先前GPU模型wall time：TENT每到达Vid .676403s / HC .457382s；SAR1.109493s/.757449s；EATA HC .450495s。Source原native推理成本未独立保留，cached readout零调用不等于部署零成本；DINO共享观测和监督reference均按唯一query去除三序重复记账。Vid-source Fisher先前2000训练query准备wall1752.536741s单列，本次GPU新增0。CPU worker含label加载的wall与score-loop、root audit、figure/report时间在ROOT_COST_AUDIT中分开，不把wall durations称core-seconds。

quality/duration/motion/query-type/development-exposure与positive/failure/全source分布已生成；tercile ties、未知/空admission独立保留。完整目标来源已有历史曝光，不称fresh或独立confirmation。OPD主方法没有这套全量预测，不拼入开发选参分数；旧Ours已授权删除payload，不更名或重建。此有限评分阶段完成不代表原全部baselines或paper完成。
