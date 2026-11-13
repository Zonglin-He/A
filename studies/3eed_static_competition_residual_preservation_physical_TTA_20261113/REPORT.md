# R92 — 固定R91主线：静态竞争残差与物理适应的分离

北京时间 2026-11-13 14:16。DEV336为历史暴露开发集，2Drone/3Quad场景。FINAL未访问。

固定主方法 static_residual_formal_bt：**22.872394%**；相对自身R91零步-0.159652pp，相对同信息表示融合-0.101785pp，相对旧动态残差BT +0.000000pp。

本轮唯一主线是R91_R76_full_disc，架构、源域已选checkpoint、文本/候选关联、输入模态和原A256保持不变。R89不另开准备/优化分支。没有源域训练、外部语义专家、EMA、额外loss或系数搜索。保留静态开发收益是研究目标，不把静态正确性当真值。

## 两项明确分开的实验

记基础分数b(theta)、冻结判别接口残差r(theta)。先在旧R91同一theta状态上比较四种读出：b0+r0、bprime+r0、b0+rprime、bprime+rprime。正确/错配物理raw与BT均保留；不看GT为每条输入挑输出。分数差严格满足二因素恒等式，mIoU经过argmax，不能把四组mIoU差值简单相加当作独立因果贡献。

固定新配方使用 **b(theta)+stopgrad(r0)**，r0来自当前输入的零步256个候选残差，不是源域平均，也不是固定Top1。基础分数仍可改变顺序。不引入第二个网络，仅缓存当前输入残差。对这套实际读出重新计算原初始预测KL和0.1query保持，并据此执行原Armijo验收；最终预测用同一套读出。旧BT验收后更换输出的版本只列为same_state诊断，不冒充正式新方法。

102498个原Adapter/双向交互/门控参数可更新；原3EED、matching、新接口权重、RGB CNN和恢复头均冻结。物理恢复0.1P+0.1RGB+0.01depth、遮挡76001/2、一步SGD alpha0.5/clip1、原九档0.5/2^k(k0..8)、Armijo c1e-4/tol1e-10全部固定。保持项在起点梯度为0；两条真实工程样本用实际计算图核对新目标梯度与旧物理方向，最大差3.8034e-21。无身份伪标签或target GT。错配仍是原物理绑定错配，BT可接受不同步长，raw同alpha比较才隔离方向。

复用R91封存的参数态与原生特征仅限完全相同theta，逐文件核验hash。新验收需要而旧搜索未保存的尺度被实际补算，未截断九档搜索或按GT回退。第三遮挡76003只在独立审计阶段测恢复，不参与接受、拟合或选参；同帧另一遮挡不称新视角/跨场景确认。

## DEV336总体结果

| 方法 | Drone mIoU % | Quad mIoU % | 等权mIoU % | 等权Acc25 % | 等权Acc50 % |
|---|---:|---:|---:|---:|---:|
| source | 15.539892 | 28.120770 | 21.830331 | 40.773810 | 21.130952 |
| R76_frozen | 17.083330 | 28.139863 | 22.611596 | 41.071429 | 21.428571 |
| ABC | 15.736332 | 28.868588 | 22.302460 | 42.261905 | 21.428571 |
| R59 | 16.385568 | 28.836664 | 22.611116 | 43.154762 | 21.726190 |
| R80_raw | 17.128930 | 28.791222 | 22.960076 | 42.261905 | 21.130952 |
| R82_bt | 17.358929 | 28.163165 | 22.761047 | 41.666667 | 21.428571 |
| R91_frozen | 17.346189 | 28.717901 | 23.032045 | 41.964286 | 21.726190 |
| R91_mean3 | 17.145428 | 28.802929 | 22.974178 | 41.964286 | 21.726190 |
| R91_feature3 | 17.145428 | 28.802929 | 22.974178 | 41.964286 | 21.726190 |
| R91_raw_dynamic | 17.025536 | 28.934628 | 22.980082 | 42.559524 | 21.428571 |
| R91_bt_dynamic | 17.073866 | 28.670921 | 22.872394 | 41.964286 | 21.726190 |
| R91_wrong_raw_dynamic | 16.729429 | 29.318644 | 23.024037 | 41.964286 | 21.726190 |
| R91_wrong_bt_dynamic | 17.348236 | 28.721194 | 23.034715 | 41.964286 | 21.726190 |
| same_state_raw_static_residual | 17.039432 | 29.069303 | 23.054367 | 42.559524 | 21.428571 |
| same_state_raw_residual_only | 17.346189 | 28.717901 | 23.032045 | 41.964286 | 21.726190 |
| same_state_bt_static_residual | 17.073866 | 28.670921 | 22.872394 | 41.964286 | 21.726190 |
| same_state_bt_residual_only | 17.346189 | 28.717901 | 23.032045 | 41.964286 | 21.726190 |
| same_state_wrong_raw_static_residual | 16.729429 | 29.057140 | 22.893285 | 41.666667 | 21.726190 |
| same_state_wrong_raw_residual_only | 17.346189 | 28.717901 | 23.032045 | 41.964286 | 21.726190 |
| same_state_wrong_bt_static_residual | 17.348236 | 28.721194 | 23.034715 | 41.964286 | 21.726190 |
| same_state_wrong_bt_residual_only | 17.346189 | 28.717901 | 23.032045 | 41.964286 | 21.726190 |
| static_residual_formal_bt | 17.073866 | 28.670921 | 22.872394 | 41.964286 | 21.726190 |
| static_residual_formal_wrong_bt | 17.348236 | 28.721194 | 23.034715 | 41.964286 | 21.726190 |

## 配对增量与全部负结果

| 比较类型 | 方法 | 对照 | 平台 | Δpp | 改善/恶化/不变 | 换query数 |
|---|---|---|---|---:|---:|---:|
| primary | static_residual_formal_bt | R91_frozen | balanced | -0.159652 | 1/3/332 | 9 |
| primary | static_residual_formal_bt | R91_frozen | drone | -0.272323 | 0/2/166 | 6 |
| primary | static_residual_formal_bt | R91_frozen | quad | -0.046980 | 1/1/166 | 3 |
| primary | static_residual_formal_bt | R91_mean3 | balanced | -0.101785 | 1/4/331 | 19 |
| primary | static_residual_formal_bt | R91_mean3 | drone | -0.071562 | 1/2/165 | 13 |
| primary | static_residual_formal_bt | R91_mean3 | quad | -0.132008 | 0/2/166 | 6 |
| primary | static_residual_formal_bt | R91_feature3 | balanced | -0.101785 | 1/4/331 | 18 |
| primary | static_residual_formal_bt | R91_feature3 | drone | -0.071562 | 1/2/165 | 13 |
| primary | static_residual_formal_bt | R91_feature3 | quad | -0.132008 | 0/2/166 | 5 |
| primary | static_residual_formal_bt | R91_bt_dynamic | balanced | +0.000000 | 0/0/336 | 0 |
| primary | static_residual_formal_bt | R91_bt_dynamic | drone | +0.000000 | 0/0/168 | 0 |
| primary | static_residual_formal_bt | R91_bt_dynamic | quad | +0.000000 | 0/0/168 | 0 |
| primary | static_residual_formal_bt | R91_raw_dynamic | balanced | -0.107688 | 11/10/315 | 45 |
| primary | static_residual_formal_bt | R91_raw_dynamic | drone | +0.048330 | 8/5/155 | 27 |
| primary | static_residual_formal_bt | R91_raw_dynamic | quad | -0.263707 | 3/5/160 | 18 |
| primary | static_residual_formal_bt | same_state_bt_static_residual | balanced | +0.000000 | 0/0/336 | 0 |
| primary | static_residual_formal_bt | same_state_bt_static_residual | drone | +0.000000 | 0/0/168 | 0 |
| primary | static_residual_formal_bt | same_state_bt_static_residual | quad | +0.000000 | 0/0/168 | 0 |
| primary | static_residual_formal_bt | static_residual_formal_wrong_bt | balanced | -0.162321 | 2/3/331 | 10 |
| primary | static_residual_formal_bt | static_residual_formal_wrong_bt | drone | -0.274369 | 1/2/165 | 7 |
| primary | static_residual_formal_bt | static_residual_formal_wrong_bt | quad | -0.050273 | 1/1/166 | 3 |
| primary | static_residual_formal_bt | R59 | balanced | +0.261278 | 44/55/237 | 182 |
| primary | static_residual_formal_bt | R59 | drone | +0.688299 | 24/35/109 | 113 |
| primary | static_residual_formal_bt | R59 | quad | -0.165743 | 20/20/128 | 69 |
| primary | static_residual_formal_bt | R80_raw | balanced | -0.087682 | 13/15/308 | 57 |
| primary | static_residual_formal_bt | R80_raw | drone | -0.055063 | 7/7/154 | 30 |
| primary | static_residual_formal_bt | R80_raw | quad | -0.120301 | 6/8/154 | 27 |
| primary | static_residual_formal_bt | R82_bt | balanced | +0.111347 | 7/8/321 | 25 |
| primary | static_residual_formal_bt | R82_bt | drone | -0.285063 | 2/5/161 | 13 |
| primary | static_residual_formal_bt | R82_bt | quad | +0.507756 | 5/3/160 | 12 |
| same_state_residual_hold | same_state_raw_static_residual | R91_raw_dynamic | balanced | +0.074286 | 3/0/333 | 6 |
| same_state_residual_hold | same_state_raw_static_residual | R91_raw_dynamic | drone | +0.013896 | 1/0/167 | 2 |
| same_state_residual_hold | same_state_raw_static_residual | R91_raw_dynamic | quad | +0.134675 | 2/0/166 | 4 |
| same_state_residual_change_only | same_state_raw_residual_only | R91_frozen | balanced | +0.000000 | 0/0/336 | 2 |
| same_state_residual_change_only | same_state_raw_residual_only | R91_frozen | drone | +0.000000 | 0/0/168 | 1 |
| same_state_residual_change_only | same_state_raw_residual_only | R91_frozen | quad | +0.000000 | 0/0/168 | 1 |
| same_state_residual_hold | same_state_bt_static_residual | R91_bt_dynamic | balanced | +0.000000 | 0/0/336 | 0 |
| same_state_residual_hold | same_state_bt_static_residual | R91_bt_dynamic | drone | +0.000000 | 0/0/168 | 0 |
| same_state_residual_hold | same_state_bt_static_residual | R91_bt_dynamic | quad | +0.000000 | 0/0/168 | 0 |
| same_state_residual_change_only | same_state_bt_residual_only | R91_frozen | balanced | +0.000000 | 0/0/336 | 0 |
| same_state_residual_change_only | same_state_bt_residual_only | R91_frozen | drone | +0.000000 | 0/0/168 | 0 |
| same_state_residual_change_only | same_state_bt_residual_only | R91_frozen | quad | +0.000000 | 0/0/168 | 0 |
| same_state_residual_hold | same_state_wrong_raw_static_residual | R91_wrong_raw_dynamic | balanced | -0.130752 | 0/1/335 | 4 |
| same_state_residual_hold | same_state_wrong_raw_static_residual | R91_wrong_raw_dynamic | drone | +0.000000 | 0/0/168 | 1 |
| same_state_residual_hold | same_state_wrong_raw_static_residual | R91_wrong_raw_dynamic | quad | -0.261504 | 0/1/167 | 3 |
| same_state_residual_change_only | same_state_wrong_raw_residual_only | R91_frozen | balanced | +0.000000 | 0/0/336 | 3 |
| same_state_residual_change_only | same_state_wrong_raw_residual_only | R91_frozen | drone | +0.000000 | 0/0/168 | 1 |
| same_state_residual_change_only | same_state_wrong_raw_residual_only | R91_frozen | quad | +0.000000 | 0/0/168 | 2 |
| same_state_residual_hold | same_state_wrong_bt_static_residual | R91_wrong_bt_dynamic | balanced | +0.000000 | 0/0/336 | 1 |
| same_state_residual_hold | same_state_wrong_bt_static_residual | R91_wrong_bt_dynamic | drone | +0.000000 | 0/0/168 | 1 |
| same_state_residual_hold | same_state_wrong_bt_static_residual | R91_wrong_bt_dynamic | quad | +0.000000 | 0/0/168 | 0 |
| same_state_residual_change_only | same_state_wrong_bt_residual_only | R91_frozen | balanced | +0.000000 | 0/0/336 | 0 |
| same_state_residual_change_only | same_state_wrong_bt_residual_only | R91_frozen | drone | +0.000000 | 0/0/168 | 0 |
| same_state_residual_change_only | same_state_wrong_bt_residual_only | R91_frozen | quad | +0.000000 | 0/0/168 | 0 |

## 竞争间隔的精确分解

对每条输入，固定零步第一名与更新后最强的其他竞争候选，计算基础、残差相对分数的变化；未发生Top1交换的输入也保留其竞争间隔。不输出真实query编号。正负号表示该固定候选对的相对作用，不等于身份真值。

| 状态 | 平台 | n | 基础间隔变化均值 | 残差间隔变化均值 | 残差中心化漂移L2均值 | 最大恒等误差 |
|---|---|---:|---:|---:|---:|---:|
| raw | all | 336 | -0.06824712 | -0.00058813 | 0.60344731 | 0 |
| raw | drone | 168 | -0.05667652 | -0.00165570 | 0.64770034 | 0 |
| raw | quad | 168 | -0.07981773 | 0.00047944 | 0.55919428 | 0 |
| bt | all | 336 | -0.00426206 | 0.00021452 | 0.13694893 | 0 |
| bt | drone | 168 | -0.00102792 | 0.00022221 | 0.15242835 | 0 |
| bt | quad | 168 | -0.00749620 | 0.00020683 | 0.12146950 | 0 |
| wrong_raw | all | 336 | -0.06329984 | -0.00041852 | 0.69950082 | 0 |
| wrong_raw | drone | 168 | -0.05595039 | -0.00142451 | 0.57621403 | 0 |
| wrong_raw | quad | 168 | -0.07064929 | 0.00058748 | 0.82278762 | 0 |
| wrong_bt | all | 336 | -0.00472346 | 0.00053921 | 0.17140635 | 0 |
| wrong_bt | drone | 168 | -0.00462864 | 0.00004756 | 0.19804207 | 0 |
| wrong_bt | quad | 168 | -0.00481829 | 0.00103085 | 0.14477062 | 0 |

## 一致验收、真实参数变化和留出恢复

| 方法 | 接受/336 | 初始/端点目标均值 | 平均alpha | 平均参数L2 | 留出恢复改善/336 |
|---|---:|---:|---:|---:|---:|
| static_residual_formal_bt | 336 | 0.03486450/0.03459221 | 0.16167341 | 0.01202150 | 335 |
| static_residual_formal_wrong_bt | 336 | 0.05293815/0.05250648 | 0.18992397 | 0.01554906 | 268 |

## 五个历史场景

保留平台和场景聚类，不把336条帧/描述当336个独立场景，不从这一历史DEV给出独立总体CI。

| 平台/场景 | n | 方法 | mIoU % |
|---|---:|---|---:|
| drone/D_S1 | 106 | R91_frozen | 13.989152 |
| drone/D_S1 | 106 | R91_bt_dynamic | 13.557545 |
| drone/D_S1 | 106 | same_state_bt_static_residual | 13.557545 |
| drone/D_S1 | 106 | same_state_bt_residual_only | 13.989152 |
| drone/D_S1 | 106 | static_residual_formal_bt | 13.557545 |
| drone/D_S1 | 106 | static_residual_formal_wrong_bt | 13.984870 |
| drone/D_S2 | 62 | R91_frozen | 23.085641 |
| drone/D_S2 | 62 | R91_bt_dynamic | 23.085641 |
| drone/D_S2 | 62 | same_state_bt_static_residual | 23.085641 |
| drone/D_S2 | 62 | same_state_bt_residual_only | 23.085641 |
| drone/D_S2 | 62 | static_residual_formal_bt | 23.085641 |
| drone/D_S2 | 62 | static_residual_formal_wrong_bt | 23.098506 |
| quad/Q_S1 | 62 | R91_frozen | 10.708624 |
| quad/Q_S1 | 62 | R91_bt_dynamic | 10.771765 |
| quad/Q_S1 | 62 | same_state_bt_static_residual | 10.771765 |
| quad/Q_S1 | 62 | same_state_bt_residual_only | 10.708624 |
| quad/Q_S1 | 62 | static_residual_formal_bt | 10.771765 |
| quad/Q_S1 | 62 | static_residual_formal_wrong_bt | 10.717546 |
| quad/Q_S2 | 49 | R91_frozen | 35.770023 |
| quad/Q_S2 | 49 | R91_bt_dynamic | 35.529054 |
| quad/Q_S2 | 49 | same_state_bt_static_residual | 35.529054 |
| quad/Q_S2 | 49 | same_state_bt_residual_only | 35.770023 |
| quad/Q_S2 | 49 | static_residual_formal_bt | 35.529054 |
| quad/Q_S2 | 49 | static_residual_formal_wrong_bt | 35.770023 |
| quad/Q_S3 | 57 | R91_frozen | 42.244590 |
| quad/Q_S3 | 57 | R91_bt_dynamic | 42.244590 |
| quad/Q_S3 | 57 | same_state_bt_static_residual | 42.244590 |
| quad/Q_S3 | 57 | same_state_bt_residual_only | 42.244590 |
| quad/Q_S3 | 57 | static_residual_formal_bt | 42.244590 |
| quad/Q_S3 | 57 | static_residual_formal_wrong_bt | 42.244590 |

## 固定回访和总体正负尾部

先固定R91_CASE001/006/010，再补入正式主方法对零步/旧BT以及同状态残差保持对旧BT的最大正/负贡献各至多两条；raw版本的最大正负尾部为封存后解释性补充，不改变任何方法、系数或评分分母。所有输入都保留在总分，案例不能成为GT部署门。原句、RGB、坐标、query索引仅留在私有图册。

| 案例 | 平台 | R91零步 | 旧BT动态 | 同状态静态残差 | 同状态仅残差变化 | 新正式BT | 旧R80raw |
|---|---|---:|---:|---:|---:|---:|---:|
| CASE001 | quad | 0.573844 | 0.573844 | 0.573844 | 0.573844 | 0.573844 | 0.000000 |
| CASE002 | drone | 0.452964 | 0.000000 | 0.000000 | 0.452964 | 0.000000 | 0.000000 |
| CASE003 | quad | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.537604 |
| CASE004 | quad | 0.233223 | 0.272370 | 0.272370 | 0.233223 | 0.272370 | 0.272370 |
| CASE005 | quad | 0.446062 | 0.327987 | 0.327987 | 0.446062 | 0.327987 | 0.327987 |
| CASE006 | quad | 0.161944 | 0.161944 | 0.161944 | 0.161944 | 0.161944 | 0.000000 |
| CASE007 | quad | 0.666919 | 0.666919 | 0.666919 | 0.666919 | 0.666919 | 0.602609 |
| CASE008 | drone | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.318500 |
| CASE009 | drone | 0.227512 | 0.227512 | 0.227512 | 0.227512 | 0.227512 | 0.227512 |

## 审计与运行成本

独立原生审计重放5040端点，执行2510次冻结后缀前向，Top1不一致0；分数/组件最大误差0，特征0，参数态0，loss0，Armijo界0。FP32容差1e-6/参数1e-7，Top1必须完全一致；与实际验收tol1e-10分开。

封存生产39.293s；复用缓存读出2501次，额外原生特征9次/物理提案9次。该成本包含历史缓存复用，不是部署冷启动端到端时延；部署仍需要当前输入零步与原物理梯度/试探前向。没有把诊断复用说成免费新观测。

本轮结果不因负值推翻已选R91架构，也不因超过较弱旧R76就称TTA成功。静态残差可能保存错误；如果保持残差后仍有损害，基础更新也是竞争解释。若仅残差变化有收益，不能永久禁止所有残差适应。后续软约束仅是待授权的同主线松弛，本轮不追加。

## 本轮最终裁决与原图核验

正式主方法与旧动态BT的最终Top1 **0/336**不同，但**28/336**的接受步长改变，概率最大变化0.0129006973。因此不是没有执行新验收；新验收确实改变连续状态/分布，但未改变离散决策。

同一旧BT状态下：bprime+r0复现旧动态BT的所有Top1；b0+rprime复现零步的所有Top1。这在该固定步长/模型/DEV范围内削弱了“主要是竞争残差漂移造成BT损害”的解释，基础分数更新已足以产生现有的主要排序变化。不等于判别残差数值不变，也不等于所有更大更新都与残差无关。

原始大步状态保留残差为23.054367%，比动态raw高+0.074286pp（3条改善、0条恶化）；相对自身零步只有+0.022322pp，11条改善/11条恶化。最大单例正贡献+0.160001pp，超过总净增益。Drone相对零步下降、Quad上升，不称稳定增益；未按它的开发分数替换已固定的正式BT主方法，也未继续选步长。

关键回访R91_CASE006（本轮CASE002）对零步候选与旧BT候选的原间隔由基础0.06009054＋残差-0.03115790组成；更新引起基础间隔-0.04336405、残差间隔-0.00015406。仅基础变化已经跨过原相对分数边界，即使把残差精确保留，仍会误选远处区域。此数值解释与原图不同物体切换相符，不作为部署门。

| 案例 | 零步→正式BT身份 | raw保持相对零步身份 | 原图观察 |
|---|---|---|---|
| CASE001 | same_candidate | same_candidate | 固定回访R91近景主体纠错：新BT、同状态两种保持和raw均保留同一主体，没有新增纠错。 |
| CASE002 | different_visible_objects | different_visible_objects | 固定回访R91_CASE006：新BT保留零步残差仍从近处车辆转向远处画面上沿。只改变残差而固定基础分数则保留原目标。 |
| CASE003 | same_candidate | different_visible_objects | 固定回访R91_CASE010：新BT仍选另一行人；raw与raw残差保持都能改到左侧主体，说明本例旧纠错被BT压回，不是残差保持新增的收益。 |
| CASE004 | same_visible_object_duplicate_boxes | same_visible_object_duplicate_boxes | 新BT的改善是同一SUV附近更合适的已有框，不是改对另一个实例。基础分数改变已足以造成这次切换。 |
| CASE005 | same_visible_object_duplicate_boxes | same_visible_object_duplicate_boxes | 同一左侧行人附近改到较差重复框；只改变残差不换框，基础更新造成退化，保持静态残差未阻止。 |
| CASE006 | same_candidate | UNKNOWN | raw残差保持避免了原raw降到零，恢复到零步的低IoU框。目标与前景车辆相互遮挡，不能根据投影/IoU判定原raw具体换了哪一实例。 |
| CASE007 | same_candidate | same_visible_object_duplicate_boxes | raw残差保持保住了画面右边缘人物附近原框，避免原raw切换到较差重复框；新BT本来就保住此框。 |
| CASE008 | same_candidate | UNKNOWN | BT仍选画面上沿错误区域，raw才回到左侧边缘目标车辆附近。目标被边缘裁切，身份细节未知；不能把raw收益归功于残差保持，因为原raw已有相同IoU。 |
| CASE009 | same_candidate | UNKNOWN | 新BT保留原有目标附近框；raw残差保持仍转到远处背景区域并丢失原重叠，具体远处物体身份未知。 |

继续固定R91_R76_full_disc为唯一研究主线，保留23.032045%的静态锚点。本轮不采纳“精确保留残差即可解决BT退化”这份配方；不重训源域、不换架构、不立即追加软保持loss。下一待检验环节应在同一模型内部的基础分数适应方向/验收，而非继续把BT损失归给残差漂移。raw结果作为范围有限的正信号保留，不能由这一历史开发集承诺泛化或正式TTA收益。

报告生成有一次CPU互斥锁正被绘图占用而未启动，失败日志保留；随后顺序重试成功。没有模型前向、预测、协议或数值容差受影响。第一版图页也保留，仅补充raw动态读出的可视面板，未重新预测或评分。
