# HC64：证据接口与动作后果的 A/B 控制实验

本轮冻结已封存的完整原生框传播 bank，不改变候选、范围评分、关联特征、mask、传播、Strong 时间区间或观察位置。固定使用原 HC128 顺序前64父来源、每父一查询、seed41；全部已曝光为开发数据。新增研究模块不改变唯一注册主方法，也不恢复暂停的正式 Vid/CG。

## 共同策略接口 A

原六特征加入 `delta_u=u(a)-u(E)` 与 `c*delta_u`，得到八个临时线性权重；每查询仍从全零和原先验开始。OPD、FixedKD、Direct 同时使用此接口。prior E=.7，替代动作均分.3，inactive仅E。学习率.1、clip1、teacher tau.2、prior KL.05、memory写入率.2均不改。

保留三轮，每轮两条路径，每批最多10次内层Adam（最多30次实际更新）。OPD每轮在当前学生参数下采两条硬历史，缓存该批state/features/教师目标，再进行内层拟合；FixedKD仅在初始化采两条对应教师历史，全程复用。Direct每次内层更新重新生成两条可微软学生历史，同一轮使用固定的同一对Gumbel噪声，完整反传记忆、关联和教师目标。

每次实际Adam后都用更新后的真实greedy策略检查原bank的固定加性分数S；若下降超过1e−12，恢复全部权重和Adam状态并停止该方法。该保护与上轮相同，不使用GT，也不保证真实性能。这是统一最大更新/路径预算，实际耗时/有效更新数分别报告，不用填充计算伪造等耗时。

## 仅改变反馈的 B

B与A使用相同八特征、优化和数据预算，唯一变化是教师条件动作分数从当前r改为四节点后果Q4。参考即时分数保持：

`r(h,a)=u(a)+.25*link(previous,a)+.1*cos(memory,feature(a))-.1*G(a)`。

参考策略固定为`softmax(r/.2)`，不含学生prior或可学习权重。当前动作分别强制为三个候选；未来最多三个节点的所有有效离散参考动作精确枚举并按其参考概率求期望。奖励沿用当前r，active节点rho=r，inactive节点rho=0；窗口末端截断，不除以H、不改温度。参考记忆和previous动作转移与原策略相同，inactive不写memory但previous推进E。

B在实际学生历史上评价这份Q4。inactive起点因动作唯一且loss权重为0，约定Q全0；active起点的四节点窗口若途经inactive，仍按上述转移继续精确积分。H=1在有效起点退化为A的即时反馈。

实现按深度批量展开，单个强制当前动作最多27种未来续接，没有用平均记忆近似非线性期望。Direct接收相同Q定义，参考分支概率、各分支记忆和当前软学生历史全部保留梯度；OPD与FixedKD只在其固定批状态上停止教师目标梯度。没有删去Direct的信息，也没有人为detach其可用路径。

GKD支持研究学生自身生成状态上的分布差异，AggreVaTe类工作支持研究动作后续代价；这里是对缓存STVG策略的新试验，不能直接继承语言模型效果或无遗憾理论保证。原论文：[GKD](https://arxiv.org/abs/2306.13649)、[interactive cost-to-go imitation](https://arxiv.org/abs/1406.5979)。

## 执行、封存与评价

只运行A/B各OPD、FixedKD、Direct六组。原ERef、几何OPD、完整框固定DP和上轮三适应成绩逐位复用，固定DP与FixedKD在表中明确区分；不另加候选或新实验分支。

全部64查询、六组输出及真实训练路径/内层日志/receipt先封存，独立评价才读取GT。除完整vIoU、tIoU、固定观测/未观测sIoU和全正负/严重下降外，随本轮真实适应集成记录：初末及内层logit间距、跨越prior动作门槛的节点数、实际替换数、loss轨迹、实际更新/rollback、Q相对即时r的同状态偏好变化、参考分支数量与耗时。GT仅事后衡量最终恢复/损伤，不用于动作、参数或停止步选择。

判断：A改善说明共同表示/拟合限制缓解；B各方法改善说明后果反馈有用；只有B中OPD进一步领先且对应学生历史纠错，才支持OPD独立贡献。固定加性DP仍是强参照，Q4本身不自动证明适应必要。A同时改证据接口和内层拟合，不能把收益单独归因两者之一。

```bash
CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B scripts/evidence_policy/run.py --run
CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B scripts/evidence_policy/score.py --run-after-seal
```

配置见 [config.json](../scripts/evidence_policy/config.json)。原始bank/轨迹/mask/特征/标签/媒体只在本地，公开结果仅匿名指标和复现代码。没有新增RGB、专家、Source/decoder前向或网络训练；只适应query内八个权重。
