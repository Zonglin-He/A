# R8最终机制复核：不采用当前关联与回归适配实现

已独立从全部5912条封存评分记录重算主要指标、场景分层、@25/@50覆盖路径、高r子组及正负尾保留情况。没有新增预测或IoU计算、GPU调用、阈值搜索，也没有修改G、runner、配置、原预测或评分。本复核检查的是已保存测量的归约与科学解释，不能冒充又一次独立GT几何评分。

结论：**当前R8不采用；不采用任何r门控阈值；不把最后回归层更新作为已证明的独特收益。保留Waymo-only source100冻结q0参照、全部工程证据与正负结果。** 真实更新接口已成立，但当前实现没有修好跨域定位。

## 损失首先发生在关联阶段

下表mIoU按0–1 IoU乘100展示；差值统一为百分点(pp)，不是相对百分比。

| 平台 | n | source q0 mIoU | association q1 | directG | param3 | exact kernel3 |
|---|---:|---:|---:|---:|---:|---:|
| drone | 2984 | 8.668259 | 7.033995 | 7.033814 | 7.034845 | 7.034845 |
| quad | 2928 | 18.022213 | 7.911287 | 7.945201 | 7.915799 | 7.915799 |

| 平台 | association−source(pp) | directG−association(pp) | param3−association(pp) | param3−directG(pp) | q0被覆盖 |
|---|---:|---:|---:|---:|---:|
| drone | -1.6342636 | -0.0001807 | 0.0008501 | 0.0010308 | 2624/2984 (87.9357%) |
| quad | -10.1109259 | 0.0339136 | 0.0045124 | -0.0294012 | 2572/2928 (87.8415%) |

参数与kernel3的mIoU差只有Drone3.235e−8pp、Quad8.608e−8pp量级，Acc25/50一致。这与冻结特征、固定query、最后线性层SGD的框空间等价性一致。不能把真实非零梯度本身当成独特参数适应贡献。Quad directG比param3更好，Drone两者差异也远小于关联损失；没有跨平台一致的参数独有优势。

| 平台 | source Acc25/50(%) | association Acc25/50(%) | param3 Acc25/50(%) |
|---|---:|---:|---:|
| drone | 18.7668 / 3.3847 | 10.9249 / 1.7426 | 10.9249 / 1.7091 |
| quad | 32.8552 / 15.8470 | 12.5342 / 4.4740 | 12.5683 / 4.4740 |

## 五个可用场景关联均为负

仅用伪名，不输出原始场景名。表达微平均与场景均值不能互换；尤其Drone小场景的损失很大。只有2个Drone、3个Quad场景，不可把5912条表达当5912个独立域泛化单位。

| 场景伪名 | n | association−source(pp) |
|---|---:|---:|
| drone_scene_01 | 2803 | -0.765345 |
| drone_scene_02 | 181 | -15.090505 |
| quad_scene_01 | 2109 | -4.506351 |
| quad_scene_02 | 325 | -20.377995 |
| quad_scene_03 | 494 | -27.283500 |

## 代码支持的根因解释及其边界

最终跨组rank确实为`.5*tight CLIP cosine + .5*context CLIP cosine + .02*normalized log PCsupport`，没有把原3EED grounder text score的数值作为最终跨组排序项。source q0在约88%的样本被替换；配对结果与所有场景的负delta共同支持“当前实现用新关联排序过度覆盖了更强的原grounder决策，主要损失发生在关联阶段”的解释。

但“grounder text只参与过滤和canonical化”的说法不完整：它还决定top32池、source-ranked ROI覆盖、近几何代表优先级，以及IRLS成员中心的相对权重。准确说法是：**这些上游/组内角色仍在，但跨组最终rank未保留该训练信号。** 代码和现有对照不能证明将来加回何种score融合就能修复；未执行的融合或关系建模不能写成解决方案。

另一个确定的保护边界是：q1及association_only_box在计算r与几何资格之前已赋值。r=0保护中心回归，不会恢复不确定的离散关联为q0。因此不能用“无可信伪框就不更新”暗示整条pipeline不会损害原预测。小幅中心调整也无从纠正大量已经选错group的情况。

这只否定当前实现的净效用，不是CLIP、多模态关联或TTA路线的普遍反证。RGB association优于PC-only与semantic-shuffle，说明匹配的图文证据相比这些改动排序控制仍有信息：

| 平台 | RGB−PC association(pp) | RGB−shuffle association(pp) |
|---|---:|---:|
| drone | 3.157281 | 2.775900 |
| quad | 5.673360 | 4.011833 |

但超过两个更弱的重排序控制，不能抵消对冻结source的失败，也不能证明全部空间关系或真实实例匹配正确。CLIP额外公开预训练、RGB和完整描述是新增信息与成本，不能称为不增加信息的source对照。

## 高r不构成可靠门控证据

固定的postseal `r>.5`子组，Drone143例微平均关联delta为+0.519711pp，但场景等权为−2.872769pp；Quad148例微平均仍为−3.501788pp、场景等权−4.804712pp。r是伪目标收缩的启发式强度，不是已校准的关联正确概率。本报告不推荐任何已证可靠阈值、不再扫描r，也不将研究曝光子组当新heldout验证。

## GT辅助覆盖路径：排序缺陷明显，不能改称实例身份真标签

以下离线路径按严格IoU>.25分成六个穷尽类别，合计各平台n。它检查source256→prepared→ROI→selectedgroup→canonicalanchor的预测覆盖，不表示真实对象身份/纯度；回退source可已命中而G路径缺失，路径miss不自动等于实际prediction miss。

| 平台 | source256无命中 | source有/prepared无 | prepared有/ROI无 | ROI有/selectedgroup无 | group有/anchor无 | q1anchor路径命中 |
|---|---:|---:|---:|---:|---:|---:|
| drone | 833 | 722 | 11 | 997 | 95 | 326 |
| quad | 705 | 208 | 29 | 1530 | 89 | 367 |

selectedgroup路径损失为Drone997/Quad1530，远大于ROI cap额外损失11/29；准备池过滤仍有722/208。组内canonical anchor也错过95/89个已有覆盖。这些计数支持同时区分准备过滤、跨组排序和组内选点三个环节；不能将其全部称作“错实例”，也不能把GT oracle回填成在线反馈。@50分类也已独立重算并与封存诊断一致，详见JSON。

## 正负尾、控制及最终决定

独立核验了封存signed tails的160条记录：正负号、available/retained计数、稳定排序及原saved IoU差值一致。Drone关联有681个正delta、716个负delta；Quad456正、1108负。两边的极端例都保留，正例不被抹去，坏例也不被平均收益遮盖。私有ID和原场景名不复制进本公开可读复核。

Source/association-only/directG/param/kernel各自的含义需保持：association-only隔离离散q选择，directG承载聚合伪框的直接收益，param/kernel同信息等价，PC/shuffle只检验额外图文对应是否被使用。所有成绩基于同一R8 strictFP32 fresh source pool；不与R7 cache、联合训练checkpoint或B24动态padding源VAL算方法delta。

**不采用当前R8 pipeline，不采用r阈值，也不采用“参数TTA已解决跨域缺陷”的主张。** 保留真实Waymo-only source100冻结q0、全部正负结果、协议及真实更新工程证据。该结论限于本实现、已曝光目标VAL和五个可用场景；没有宣称CLIP路线普遍失败，也没有把未来未执行的设计写成已修复。
