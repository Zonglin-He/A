# 完整框范围传播：HC64 固定配置试验

本轮针对已核实的候选范围收缩：保留 ReferDINO 原生框的连续坐标，以软 mask 加权 Source 特征对应估计框的平移和缩放，不再以粗网格阈值连通组件重新生成边界。主方法登记及默认配置不变，正式 Vid/CG 保持暂停。

## 固定差异

沿用旧 v0 的 ERef 初始化和 Strong 原物理 MBR 区间，以匹配修正前后。每帧三个槽是 ERef、前锚点完整框变换、后锚点完整框变换；前后不平均。八个观测位置直接原样保留同一个 ReferDINO query 的原生框，不做 E-NMS；Strong 事件外可以存储该候选，但策略仍强制保留 ERef，避免改变时间范围。mask 不承担决定框边界的作用。

Source 归一化特征的两向 top20 / temperature0.1 对应为 A。以 w(i,j)=mask_anchor(i) A_anchor,target(i,j) A_target,anchor(j,i) 构造软匹配权重，归一化网格中心坐标上逐轴加权最小二乘：s=Cov(x,y)/Var(x)，delta=mean(y)−s mean(x)。将拟合变换作用于完整原生框的中心和尺寸。

预先固定可靠性条件：总质量>1e−15、两端边际有效样本量各≥3、源两轴方差各>1e−6、两个正缩放均在[0.5,2]且全部有限。失败或完整框与图像无交集则该方向无效，不截断缩放、不改成仅平移、不从mask重建。置信度沿用mask回传cycle Dice，不解释为目标身份正确概率。

范围评分使用完整传播框的连续核 exp(−0.5‖[(c−c_ref)/size_ref,log(size/size_ref)]‖²)。关联评分使用候选完整框内Source特征与原生mask加权prototype的非负余弦，零prototype给0。每侧两项等权0.5/0.5，再按对应可靠度加权。关联、link和memory仍使用mask信息，但评分与最终输出读取同一个完整框；没有按小组件评分后另行扩框。保留旧0.1全轨迹几何保护及0.25连接项，不用L1＋GIoU作新反馈。

这些条件和比例是一次固定试验起点，未按新GT调参，不声称已最优或已解决教师身份/匹配错误。

## 真实比较与公平性

固定旧HC128顺序前64父来源，每父1query，seed41。两套候选/评分各运行 OPD、FixedKD、Direct，并提供无需训练的固定分数精确DP：

| 分支 | 候选与范围反馈 | 策略 |
|---|---|---|
| Old | 封存v0组件候选及旧评分 | 新运行同一OPD/FixedKD/Direct |
| Box | 完整原生框变换及范围评分 | 同一OPD/FixedKD/Direct |
| OldFixed | 旧v0已保存EvidenceDP | 无训练，精确固定分数最大化 |
| BoxFixed | 新bank的EvidenceDP | 无训练，精确固定分数最大化 |

旧v0当时在适应前停止，不能把它写成已有训练成绩。本轮用户明确授权匹配比较，旧三组适应在新独立目录实际运行，不修改原failed gate或其stage2=0历史。

原v0 policy代码以函数原样复用，不改变六权重初始化、Gumbel流、3轮×2路径、lr0.1、teacher tau0.2、prior KL0.05、clip1、prior E0.7或memory写入率0.2。Direct保留软历史/关联/记忆/教师目标的完整可用梯度，FixedKD固定两条教师历史，OPD随当前学生路径刷新。三者共享同一个bank和证据，拒绝步恢复完整Adam状态并停止。预算是相同轮数和路径数；真实计算量/耗时分别报告，不用padding伪造等耗时。

修正同时改变候选范围与unary评分，前后增量只能归于联合修正，不能单独归因于其中一项。输出都是最后接纳的真实离散策略轨迹，不是采样候选中GT最优结果或平均框。

## 封存、评价与复现

全部64查询的新旧六组真实适应、固定读出、bank、路径、receipt先封存，独立评分器之后才能读取GT。报告完整vIoU、时间、固定观测/未观测节点sIoU、全部正负与严重下降、配对父来源区间、实际更新/回滚/路径计数及缓存增量成本。保留旧几何OPD参照，不将其与六权重新状态等同为单因素奖励消融。

```bash
CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B scripts/box_propagation/run.py --run
CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B scripts/box_propagation/score.py --run-after-seal
```

配置见 [config.json](../scripts/box_propagation/config.json)。数据、原生mask、Source特征与轨迹张量仅本地使用，不上传。全流程只运行CPU缓存侧候选构建和轻量权重适应，不调用重网络、decoder、专家或RGB。HC64已用于开发，结果不能代替独立测试集确认。无GT选停止步、逐视频参数或查询删选。
