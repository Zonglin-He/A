# 一次性原生前景信息资格，非新方法：HC128

用户授权时间/联合自主探索，空间全部冻结。R3历史actionness投影是median/MAD、物理积分、.1logp；R14额外六层时间头静态读出+1.606pp而分布期望未提升；R17真实边界观察三配置均负。既有6.1-sol只读review要求先检查信息，而不再给同一信号改名。旧structured_temporal_v1已包含inside/outside Bernoulli代数，本轮不称新结构路线。

只读原全封存RC128 R3 Native actionness、R14既有motion/appearance时间采样logits、原p0/E/Live41/42/43坐标。源码criterion规定action正类1、负类EOS=.3，fm/fa无类权BCE；实际训练日志未知、目标域校准不保证。唯一预定反校准ActionEOS=z+log(.3)，仅理想训练最优下成立。它只改变连续mask分数的长度项，AUROC和同节点长度排序必须逐父相同，不解释成新语义。

每个原65实际节点合并两个offset字段，保存Action/ActionEOS及三个原head的同均值Flat/循环错位32（全65节点上）的控制。连续mask分数sum_{k=i..j}z_k，不叠加端点概率、温度、源训练weighted loss energy或新增先验；这里仅生成诊断分数，不做argmax/MBR/新的最终区间。Flat保留绝对水平及长度趋势；Shift保留整条分数分布但打乱时间对应。所有128字段和2080mask能量全封存、独立核验后才新GT离线诊断，不读取R17成绩选择任何规则，不使用TC109。

密集：每父原65节点，事件内标签按物理GT span，AUROC/Brier/logloss及固定.5工作点的预测前景比例/precision/recall。单类AUROC或无预测precision等未定义None，汇总明确实际定义父数，不填0。

候选：GT仅用于同一冻结空间下完整v质量；三个空间seed父内平均，不当额外独立样本。原2080同支持，Native I为参考；有益/有害是相对该父Native fullv分别>=+2pp/<=−2pp，不能在线挑类别。报告能量区分二者AUC及全部同节点长度j−i候选pair正确率（GT质量不同才定义，教师tie=.5），以及stratum内中心化rank相关。节点数相同的物理长度可差1帧，明确这一界限。Flat同长度必然tie，EOS同长度严格保留排序；总体AUC和同长度排序不能混称同一判断力。

主Action，fm/fa只作现有未加权头参照，不能从三头选赢家、调混合权重或衍生新方法。本轮无算法配置选择、无新方法准确率/输出区间、无OPD。全父保留，包括无事件内/外节点、无有益/有害候选等条件未定义者。父均/配对bootstrap10000 seed2026101909；主要比较correct−Flat/Shift及EOS−raw工作点变化。正/负与未定义全signed公开。

一次性退出：若时间特异分类/同长度候选排序不能优于控制，或只有长度/校准变化，关闭该前景重参数化分支，不再试EOS/结构读出/温度。若时间特异纠错排序明确有利，也只是一次后续能力测试资格，不自动证明>=2–3pp或Live优势；不能因AUC略高.5自动新开轮。根需结合机会大小和有益/有害排序独立裁决，无自动新模型接续。

纯CPU4线程、cap120秒（导出+新离线评分分别记录；独立数字核另记），自有非阻塞run.lock，host6GiB/disk30GiB；不为0GPU工作持有共享gpu.lock。0Source encoder/RGB/专家/网络训练/空间/学生更新，上游缓存成本非免费。数值独立核读取全部原字段/示例mask标量和EOS恒等式；评分后官方Dense逐interval核验。代码/原实际配置/全部诊断/负发现公开，空间/CURRENT/MAIN/PAPER不变。
