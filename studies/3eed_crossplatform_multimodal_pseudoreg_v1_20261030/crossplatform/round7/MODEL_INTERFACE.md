# Round7：最后中心/尺寸回归层适配接口（准备阶段）

状态：实现与合成CPU接口验证完成；真实checkpoint/model/data/G与任务收益尚未运行。只在本新目录写入文件，不修改Round6代码、锁、权重或旧实验。源模型应由协调器严格加载Round6固定epoch100、SHA256 `0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd`，并绑定真实初始化/模型与样本计划；本模块不负责构造模型或加载任何checkpoint。

## 原始模型接口事实

证据是被Round6正式计划锁定的官方代码：`models/modules.py` 的 `ThreeLayerMLP`/`ClsAgnosticPredictHead`，以及 `models/bdetr.py` 的decoder循环。官方上游commit为 `843a901f5367a6be5cc484b97c6ecad462961dfa`。

六个decoder的最后预测头为 `prediction_heads.5`，对应 `last_`。它分别有center和size两套独立MLP：

`Conv1d(288,288,1,bias=False) → BN → ReLU → Dropout(0.3) → Conv1d(288,288,1,bias=False) → BN → ReLU → Dropout(0.3) → Conv1d(288,3,1,bias=True)`。

仅最后两个Conv可更新，共1734个参数。参数名精确为：

- `prediction_heads.5.center_residual_head.net.8.weight`，形状 `[3,288,1]`。
- `prediction_heads.5.center_residual_head.net.8.bias`，形状 `[3]`。
- `prediction_heads.5.size_pred_head.net.8.weight`，形状 `[3,288,1]`。
- `prediction_heads.5.size_pred_head.net.8.bias`，形状 `[3]`。

center为`query_points_xyz + center_residual`；最后一层的base仍为原始query种子点，不是把前一decoder中心再加一次。size是直接线性输出，不是log-size、不经过softplus；负值必须记录，不能在适配中偷偷取绝对值。官方评估的尺寸clamp属于输出评估协议，应由外部统一应用于source/direct/update所有臂。该模型`heading=False`，这四个参数没有yaw表达能力；本接口仅输出6D中心/尺寸，全部yaw=0。

前5层预测的detached位置可影响最后decoder特征，但本轮不更新它们。最后回归Conv位于decoder、BBF projection和BBS分支之后，因此重放最后Conv不会重新计算候选身份或语言分数。

## API及无标签输入契约

`regression_adapter.py` 不导入官方数据loader、annotation、2D模型、teacher/EMA或G实现，不访问数据文件。它提供：

1. `capture_source(model, {'point_clouds': P[1,N,3], 'text':[actual_input_text]}, token_mask[T], query_index=None)`：一次`eval/no_grad`源前向，在两最终Conv的prehook捕获各自`[1,288,Q]`输入，缓存原始`last_center/last_pred_size`、`last_base_xyz`及纯文本BBF分数。标准Q=256。重放同两Conv必须与此源前向raw6D逐位一致，否则报错。
2. `first_noun_mask(text, offset_mapping[T,2])`：在精确模型输入上按固定词表查最早出现的名词；无词则明确unsupported，不用GT类别回填。此规则不保证等同真实类别，也不改写原query。
3. `text_bbf_scores(outputs, token_mask)`：`softmax(last_proj_queries @ proj_tokens.T / 0.07)`的指定token概率和。分母保留实际全部T个token，包括special/sentinel与实际padding；不能从tokenization以外猜偏移。query与tokens是post-cross-encoder条件化特征，不能当作独立教师。
4. 外部非学习模块生成 `PseudoTarget(fixed_source_query_index, target6, reliability6)`。`target6=[cx,cy,cz,sx,sy,sz]` 为与模型同坐标/米制的绝对框，target尺寸为正；`w6∈[0,1]^6`为逐坐标可靠性。两者进入损失前detach。G可以由协调器接入P、RGB、q、停止梯度的初始候选和已核相机K/T；本模块无RGB/相机或GT输入。不得以GT选择query、伪框、w或更新是否回滚。
5. `adapt(model, cache, pseudo, StepConfig(steps,lr,beta_metres))`：固定源query，不重排序、不生成新候选，不更新语言头；返回四个同信息臂和过程日志。lr/steps/beta无实验默认值，须由最终计划显式注册。

若外部指定`query_index`，必须来自预先声明的纯文本源读出。否则本接口使用源BBF最大分数，完全同分取第一query。槽位只在同一次源前向中稳定，不可跨重采样/视图按同index假定对象对应。

## 四臂与损失

四臂使用同一固定q及同一G/w：

- `source`：原raw6D候选池。
- `direct_G`：只在选定q的`w_j>0`坐标直接置为G坐标；`w_j=0`严格保留source。连续w数值也提供给此臂，但直接读出只将其用于支持掩码；不能称direct G使用了更少模态信息。
- `parameter_SGD`：仅更新上述四参数，用单个选定q的加权SmoothL1。
- `exact_output_space_SGD`：不更新任何参数的同信息框空间SGD重放，并保留共享Conv引起的全候选spillover。

损失定义为 `L = Σ_j w_j SmoothL1_beta(b_j − target_j) / Σ_j w_j`，直接在米制6D坐标上计算，没有隐式按尺寸归一化或新增yaw损失。使用plain SGD，无momentum、无weight decay、无裁剪；trace明确记录`clip_enabled=False, clip_factor=1`。全部w为0则所有臂返回source、零更新。负source尺寸不自动修复，loss与raw结果保持可审计；共享评估clamp另行一致应用。

## 数学同信息对照的必要解释

冻结前段后，每个中心输出坐标都是 `b_j = base_j + W_j h_center + a_j`；每个尺寸坐标是 `b_j = W_j h_size + a_j`。固定单个q、单个样本时，plain SGD的一步在选定query上严格等价于

`b_next,j = b_j − lr (||h_group(q)||² + 1) ∂L/∂b_j`。

对其余query r，输出改变为

`Δb_j(r) = −lr (h_group(r)·h_group(q) + 1) ∂L/∂b_j(q)`。

此等价在实数算术成立；Conv/SGD与框空间重放的浮点累积顺序可能造成舍入误差，合成fixture以预声明容差核验而不声称bitexact。两个输出组分别记录特征平方范数、自kernel、每步选定框、六维损失梯度、参数梯度范数、nonzero梯度参数数、参数delta和全池重放差异。最后Conv共享参数，因此未监督候选也会变化；日志单列其中心/尺寸最大漂移和改变候选数，不把`w_j=0`或未选query误称全部source不变。

当前输入上拟合G不能证明参数适应独有价值。若parameter update与框空间重放等价、或direct G更好，应照实承认。若以后要检验参数变化是否泛化到独立重采样/视图，须另锁匹配和独立输入测试；本接口没有用这种未执行测试暗示优势。

## Episodic隔离与CPU验证

每次capture/adapt使用`eval()`，冻结全部前段与BN统计、关闭dropout，仅四参数requires_grad=True；异常或正常返回均恢复四参数数值、全部buffers、全部原始gradient引用、requires_grad、每子模块training状态，以及Python/NumPy/torchCPU/已初始化CUDA RNG。非选择参数从不写入或优化，另核其版本号未变。每个样本重建空状态SGD；没有EMA/teacher/student持久状态。源cache与最终参数SHA绑定，适配前必须通过raw Conv桥。

`check_regression_adapter_cpu.py`仅构造同维度、同MLP层次的合成容器，不构造真实3EED、不读权重/数据/目标GT、不初始化CUDA。已核source重复与Conv桥bitexact、三步SGD与框空间全候选重放误差、停止梯度、零权重坐标、全零w fallback、共享Conv spillover和异常恢复。结果在`CPU_INTERFACE_FIXTURE.json`；它是工程接口证明，不是数据集或研究成绩。
