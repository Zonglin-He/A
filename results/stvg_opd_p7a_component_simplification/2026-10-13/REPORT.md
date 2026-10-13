# P7-A / OPD 内部组件：QR × LN 历史写回

本报告覆盖原 HC-STVG-v2 与 VidSTG 开发名单，每域 32 个 parent、原两序、四臂共 512 个实际完整 fit、12800 条更新轮记录。四臂是 joint_carry、LN_carry、joint_reset、LN_reset；OPD 目标固定。原开发集已经历史 GT 暴露，本轮是匹配内部组件实验，不是独立确认结果。

I1 为 source-frozen actionness projection 主读出，I0 为同一空间 boxes 的 Native 次读出。使用原 Native Uniform4 admitted Top1 缓存证据；专家、像素、decoder grid、key/round 噪声一致，各臂保留自己的完整状态链。Query 与 Adam 每次 arrival 重置；LN-only 优化 1536 个坐标且 Query 固定零，joint 优化 1792 个。reset 从真实 source LN 重启；carry 使用 HC 1/16、Vid 1/8 的衰减写回。

参数固定：HC lr=.01 / sigma=.025 / tau=.05 / K=40；Vid .03/.1/.25/K=10。没有新 DINO、teacher、参数搜索或生产晋升。资格阶段另计 64 完整 fit、16 full capture function invocations、1600 rounds；正式阶段 128 full source-frozen capture function invocations，四臂共用每次完整函数捕获。全部正式预测及独立部署数学读回通过后才进行后验 GT 评分。

聚合先在每个 parent 内平均原两序，再等 parent paired bootstrap 10000 次，seed=20261013。以下绝对值为百分数；差值为 percentage points（pp）。R30/R50 严格使用单 query vIoU > .3/.5，边界等于阈值算未通过。m_sIoU 是固定 GT 支持上的 dense 空间均值。

## 四臂完整读出

### I1（主读出）

|目标|臂|Frozen mvIoU|Before mvIoU|After mvIoU [95% CI]|Frozen/Before/After R30|Frozen/Before/After R50|After mtIoU|After msIoU|
|---|---|---:|---:|---|---|---|---:|---:|
|HC-STVG-v2|joint_carry|29.017|30.065|32.781 [25.121, 40.670]|46.875/46.875/48.438|12.500/15.625/25.000|47.278|65.021|
|HC-STVG-v2|LN_carry|29.017|30.071|32.761 [25.209, 40.611]|46.875/48.438/50.000|12.500/15.625/25.000|47.278|65.003|
|HC-STVG-v2|joint_reset|29.017|29.017|32.415 [24.830, 40.282]|46.875/46.875/50.000|12.500/12.500/25.000|47.278|64.213|
|HC-STVG-v2|LN_reset|29.017|29.017|32.494 [24.992, 40.314]|46.875/46.875/50.000|12.500/12.500/25.000|47.278|64.283|
|VidSTG|joint_carry|12.599|13.935|16.822 [11.058, 22.994]|9.375/15.625/28.125|0.000/3.125/3.125|32.910|46.190|
|VidSTG|LN_carry|12.599|14.027|16.460 [10.656, 22.627]|9.375/17.188/28.125|0.000/3.125/3.125|32.910|45.157|
|VidSTG|joint_reset|12.599|12.599|15.996 [10.562, 21.741]|9.375/9.375/28.125|0.000/0.000/0.000|32.910|44.710|
|VidSTG|LN_reset|12.599|12.599|15.725 [10.273, 21.554]|9.375/9.375/28.125|0.000/0.000/0.000|32.910|43.871|

### I0（次读出）

|目标|臂|Frozen mvIoU|Before mvIoU|After mvIoU [95% CI]|Frozen/Before/After R30|Frozen/Before/After R50|After mtIoU|After msIoU|
|---|---|---:|---:|---|---|---|---:|---:|
|HC-STVG-v2|joint_carry|24.571|25.426|28.143 [20.639, 36.108]|34.375/37.500/37.500|9.375/12.500/20.312|38.660|65.021|
|HC-STVG-v2|LN_carry|24.571|25.446|27.921 [20.417, 35.877]|34.375/39.062/37.500|9.375/14.062/20.312|38.660|65.003|
|HC-STVG-v2|joint_reset|24.571|24.571|27.942 [20.471, 35.861]|34.375/34.375/37.500|9.375/9.375/21.875|38.660|64.213|
|HC-STVG-v2|LN_reset|24.571|24.571|27.772 [20.284, 35.721]|34.375/34.375/37.500|9.375/9.375/21.875|38.660|64.283|
|VidSTG|joint_carry|12.364|13.615|16.389 [10.832, 22.290]|9.375/15.625/28.125|3.125/3.125/0.000|31.859|46.190|
|VidSTG|LN_carry|12.364|13.719|16.085 [10.521, 22.011]|9.375/17.188/28.125|3.125/3.125/0.000|31.859|45.157|
|VidSTG|joint_reset|12.364|12.364|15.652 [10.325, 21.278]|9.375/9.375/28.125|3.125/3.125/0.000|31.859|44.710|
|VidSTG|LN_reset|12.364|12.364|15.430 [10.092, 21.105]|9.375/9.375/28.125|3.125/3.125/0.000|31.859|43.871|

## 六个预先固定的结构对比

remove_QR_carry=LN_carry−joint_carry；remove_carry_joint=joint_reset−joint_carry；remove_carry_LN=LN_reset−LN_carry；remove_QR_reset=LN_reset−joint_reset；remove_both=LN_reset−joint_carry；interaction=(LN_reset−joint_reset)−(LN_carry−joint_carry)。正值表示该明确对比下的前者较高，不预设删除组件成功。

|目标|对比|I1 After mvIoU 差值 pp [95% CI]|I0 差值 pp [95% CI]|I1 正/负/中性/未知 parent|I1 >5/>20pp 损害 parent|原 order1/order2 I1 差值 pp|
|---|---|---|---|---|---|---|
|HC-STVG-v2|remove_QR_carry|-0.019 [-1.180, +1.345]|-0.222 [-0.996, +0.417]|9/19/4/0|1/0|-0.032/-0.007|
|HC-STVG-v2|remove_carry_joint|-0.366 [-0.689, -0.145]|-0.202 [-0.334, -0.088]|8/20/4/0|0/0|-0.422/-0.310|
|HC-STVG-v2|remove_carry_LN|-0.267 [-0.485, -0.068]|-0.149 [-0.311, -0.006]|9/19/4/0|0/0|-0.310/-0.224|
|HC-STVG-v2|remove_QR_reset|+0.080 [-0.914, +1.224]|-0.169 [-0.774, +0.341]|9/19/4/0|1/0|0.080/0.080|
|HC-STVG-v2|remove_both|-0.286 [-1.306, +0.905]|-0.371 [-1.069, +0.197]|5/23/4/0|1/0|-0.342/-0.230|
|HC-STVG-v2|interaction|+0.099 [-0.195, +0.450]|+0.053 [-0.118, +0.252]|13/15/4/0|0/0|0.112/0.086|
|VidSTG|remove_QR_carry|-0.362 [-1.122, +0.177]|-0.305 [-1.016, +0.194]|11/11/10/0|1/0|-0.305/-0.419|
|VidSTG|remove_carry_joint|-0.826 [-1.603, -0.206]|-0.738 [-1.379, -0.200]|7/15/10/0|2/0|-0.578/-1.073|
|VidSTG|remove_carry_LN|-0.735 [-1.501, -0.126]|-0.655 [-1.296, -0.121]|8/14/10/0|2/0|-0.544/-0.925|
|VidSTG|remove_QR_reset|-0.271 [-1.053, +0.260]|-0.222 [-0.960, +0.279]|10/7/15/0|1/0|-0.271/-0.271|
|VidSTG|remove_both|-1.097 [-2.115, -0.261]|-0.960 [-1.865, -0.219]|8/14/10/0|3/0|-0.849/-1.344|
|VidSTG|interaction|+0.091 [-0.013, +0.204]|+0.083 [-0.023, +0.195]|13/9/10/0|0/0|0.034/0.148|

## 每序与全部 parent

|目标|order|臂|I1 After mvIoU [95% CI]|I0 After mvIoU [95% CI]|I1 total/current/inherited 增量 pp|
|---|---|---|---|---|---|
|HC-STVG-v2|order1|joint_carry|32.837 [25.167, 40.753]|28.186 [20.695, 36.137]|3.819/2.688/1.131|
|HC-STVG-v2|order1|LN_carry|32.805 [25.282, 40.640]|27.945 [20.460, 35.868]|3.787/2.609/1.178|
|HC-STVG-v2|order1|joint_reset|32.415 [24.830, 40.282]|27.942 [20.471, 35.861]|3.397/3.397/0.000|
|HC-STVG-v2|order1|LN_reset|32.494 [24.992, 40.314]|27.772 [20.284, 35.721]|3.477/3.477/0.000|
|HC-STVG-v2|order2|joint_carry|32.725 [25.053, 40.593]|28.101 [20.596, 36.076]|3.707/2.743/0.964|
|HC-STVG-v2|order2|LN_carry|32.718 [25.174, 40.599]|27.897 [20.375, 35.844]|3.701/2.771/0.930|
|HC-STVG-v2|order2|joint_reset|32.415 [24.830, 40.282]|27.942 [20.471, 35.861]|3.397/3.397/0.000|
|HC-STVG-v2|order2|LN_reset|32.494 [24.992, 40.314]|27.772 [20.284, 35.721]|3.477/3.477/0.000|
|VidSTG|order1|joint_carry|16.574 [10.912, 22.620]|16.158 [10.643, 21.967]|3.975/2.641/1.334|
|VidSTG|order1|LN_carry|16.269 [10.616, 22.266]|15.904 [10.425, 21.742]|3.671/2.332/1.338|
|VidSTG|order1|joint_reset|15.996 [10.562, 21.741]|15.652 [10.325, 21.278]|3.397/3.397/0.000|
|VidSTG|order1|LN_reset|15.725 [10.273, 21.554]|15.430 [10.092, 21.105]|3.126/3.126/0.000|
|VidSTG|order2|joint_carry|17.069 [11.178, 23.360]|16.621 [11.007, 22.577]|4.470/3.132/1.338|
|VidSTG|order2|LN_carry|16.650 [10.709, 22.987]|16.266 [10.597, 22.320]|4.051/2.532/1.519|
|VidSTG|order2|joint_reset|15.996 [10.562, 21.741]|15.652 [10.325, 21.278]|3.397/3.397/0.000|
|VidSTG|order2|LN_reset|15.725 [10.273, 21.554]|15.430 [10.092, 21.105]|3.126/3.126/0.000|

FIT_ROWS.csv 保留全部 512 匿名 arrival；PARENT_MEANS.csv 保留全部 256 parent×arm 两序均值。METRICS.csv/CONTRASTS.csv 覆盖所有 v/t/s/R30/R50、total/current/inherited 增量、严格阈值变化、各域/合并/每序，且保留所有正、负、中性和未知计数。没有根据 GT 删除 parent 或选择最佳 step。

## 实际 instrumented 成本

计量 scope 修订 capture_scope001：原始字段 actual_frozen_capture_forwards=1 保持不变，语义是一次 frozen_forward 完整 capture 函数调用（invocation），不是单次 model(...)。每次 capture 包含 offset 0/1 两个 source model forwards，records 长度为 2。因此资格 16 + 正式 128 次完整 capture 函数调用对应 32 + 256 次 source-offset model forwards。额外 decoder replay 调用不由该字段计数，不能把 32+256 称为全部 model/module calls。COSTS.csv 对原字段的 units 明确为 full_capture_function_invocations。

|目标|臂|实际 fits/rounds/backwards|空监督 fit / no-information rounds|active 坐标|instrumented fit 秒/fit|allocated capture 总秒/full function invocations|decoder cache reuse capture|
|---|---|---|---|---|---:|---|---|
|HC-STVG-v2|joint_carry|64/2560/2480|0/80|1792|2.380413|284.633485/64|1|
|HC-STVG-v2|LN_carry|64/2560/2480|0/80|1536|2.124146|0.000000/0|0|
|HC-STVG-v2|joint_reset|64/2560/2480|0/80|1792|2.423104|0.000000/0|0|
|HC-STVG-v2|LN_reset|64/2560/2480|0/80|1536|2.235673|0.000000/0|0|
|VidSTG|joint_carry|64/640/460|18/180|1792|0.559206|241.738810/64|1|
|VidSTG|LN_carry|64/640/460|18/180|1536|0.509943|0.000000/0|0|
|VidSTG|joint_reset|64/640/460|18/180|1792|0.558625|0.000000/0|0|
|VidSTG|LN_reset|64/640/460|18/180|1536|0.508812|0.000000/0|0|

正式 controller 从开始到全球 seal 的 elapsed_seconds=1332.370688；这个整体计时与各 instrumented 子项有不同范围，不能相加或替代对方。

instrumented_fit_seconds 包括原 OPD fit 及实际记录/算术检查的开销，不是未插桩部署速度。共享 frozen capture 的时间、完整函数调用计数和 decode cache reuse 仅按 capture_owner 分配一次；0 分配给其他臂表示同次捕获共用，不表示这些臂独立部署可以免费读取视频。缓存专家来自已验证旧输入，actual_new_DINO_calls=0 不计入旧专家生成成本，也不证明新视频专家成本为零。CSV 保留计量缺失的 unknown，禁止把缺失计量填零。

CUDA_peak_allocated 是每个 fit 的峰值字节数；COSTS.csv 保留每 fit 峰值的均值但不公开可加总 total。active 坐标同样是优化范围的每 fit 计量，不累加为资源成本。逐 fit 原标量保留在 FIT_ROWS.csv；缓存复用、同步与 trace/audit 插桩属于本次实际测量范围，不能由这些秒数证明最终端到端 deployment speed。

## 证据与决策边界

- 合并 64 parent，remove_QR_carry：-0.191 [-0.867, +0.512] pp；探索性 CI 跨零；未建立优势，也未建立等效。
- 合并 64 parent，remove_carry_joint：-0.596 [-1.032, -0.258] pp；此开发条件下负向，探索性 CI 全负。
- 合并 64 parent，remove_carry_LN：-0.501 [-0.915, -0.182] pp；此开发条件下负向，探索性 CI 全负。
- 合并 64 parent，remove_QR_reset：-0.096 [-0.720, +0.536] pp；探索性 CI 跨零；未建立优势，也未建立等效。
- 合并 64 parent，remove_both：-0.691 [-1.423, +0.044] pp；探索性 CI 跨零；未建立优势，也未建立等效。
- 合并 64 parent，interaction：+0.095 [-0.064, +0.276] pp；探索性 CI 跨零；未建立优势，也未建立等效。

本轮保留 OPD 核心，结构决策限定在 QR×carry 四臂。CI 跨零不等效，不足以证明组件可删；要据成本决定精简，仍需预先指定可接受任务损失的非劣界。即使某域探索性 CI 全正，也不能按测试域逐域选择结构赢家。两序仅提供顺序敏感性，不代表完整流顺序分布。正式保留/删除和后续有限调参须以统一主方法目标作出；本报告不自动删除 QR、carry 或晋升任何臂。

Before 是该臂自身状态链的进入读出，After−Before 是当前 query 写入效果，Before−Frozen 是继承影响，After−Frozen 是总影响；这些增量代数相加，不能用当前 query 改善替代累计表现。I1/I0 使用同一 boxes 且都是固定 source 时间读出，空间组件不会在本轮重新训练时间模块。GT 只用于封存后的评估，不用于监督、准入、排序、时间选择或 checkpoint/step 选择。

该开发名单有历史 GT 暴露；本轮没有使用独立确认 cohort，因此不能宣称独立泛化或等效。本轮 6 对比及大量次指标的 CI 未做多重比较校正，属于探索性区间。匿名算术读回精确复现是可复现性检查，不是新增实验或新增独立数据。公开完整匿名结果保留负结果、中性和未知；真实媒体、GT 坐标、完整数学向量、模型状态和 query 文本保持私有。

来源：protocols/stvg_opd_p7a_component_simplification_v1.md；scripts/score_stvg_opd_p7a_component_v1.py；同目录 SUMMARY.json、ROWS.jsonl.gz、PARENT_ROWS.jsonl.gz。实际部署数学和后验独立 dense/匿名统计读回分开完成。
