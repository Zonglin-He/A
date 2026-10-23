# 新版固定方法：TubeDETR隔离接入

当前LLaVA-ST两域全量已完整封存、独立评分、Root核验及最终发布后，按用户“下一轮TubeDETR也要跑”的授权执行。旧绑定代码、所有旧计划/预测/seal及旧实验结论不修改，不恢复旧调参。

方法固定为main a238705a的SimplePath及全视频单组Native2D原型。新的数值核心在tools/simplepath_tube/core中隔离保存；相对上游只改相对import，以及将已固定kappa64/beta1配置读取改为字面常量。全部函数运算保留，来源和原文件hash见results/simplepath_tubedetr/upstream_port.json。SimplePath权重.5/.5/.25、context1.5、单轮、原共识初始化gamma Vid1/HC.5，不展开任何新网格。

TubeDETR Source仍是原冻结224、stride2、原MAP时间读出及原始框。新版q0来自实际原生末层FP32逐帧起终点logits，不用六层Global6、不伪造TA双offset。Full对q0施加一次原型反馈并使用原物理时间MBR；该Full读出可改变Source时间，因此必须区分Source和Native-MBR控制，不能将其时间增益全归原型。

空间SimplePath用原生decoder输入的post-fusion视觉前缀；时间反馈ROI汇聚用实际全时间fast分支input_proj输出、进入视频/文本编码器之前的256通道网格，保留valid mask/物理帧对应。新的全视频辅助轨迹上实际pool，旧BoxFixed pooled features不复用。

缓存仅绕过视觉backbone/input_proj：冷前向观察slow/fast实际投影网格及mask/pos，回放时仍运行完整原生视频/文本编码器和decoder。每次检查两个原生调用的顺序与真实batch帧数，保存原始dtype并无损压缩；输入RGB/采样/代码/模型不同不复用。资格首先使用query级缓存。运行计划进一步冻结视频级前缀复用：前缀由不接受caption/text的原生CNN及input_proj构成，两个原query的cold/cache真实逐位检查通过；每条复用还要求相同RGB/物理采样/权重/实现和资格hash，且全部新Source逐位核对旧完整封存结果。任何不一致停止，不退回新专家。专家从既有精确RGB、文本、shape、teacher fingerprint的缓存只读加载，miss硬失败，资格不调用新专家。

第一阶段是独立有界真实资格，计划在首次新GPU前冻结。只用两域原128的首query，各一次cold、一次视觉前缀cache，共最多4次原生前向，0新专家、0GT读取。要求Source/原生时间/八位置逐位等于旧完整封存输出；cold/cache的Source、全部六层logits、q0、空间特征和耦合Full逐位一致；真实pre-text支持有限；模型state前后hash一致。失败保留输出/原因、停止、不换query/重试。CPU工程检查不能代替这些真实条件。

资格通过后，分别对两域原128固定方法验证，再按原Vid10303/HC1902总体评价，先完整预测seal后独立CPU标签评分，不依据新结果调参。后续运行计划必须单独冻结，引用本资格的hash；资格计划不是已经启动的全量计划。LLaVA后续时间候选/先验仍需单独适配，不能用单个生成时间区间伪造分布。

状态与路径：公开results/simplepath_tubedetr，私有outputs/simplepath_tubedetr，日志logs/simplepath_tubedetr_qualification.log；所有GPU任务共用runtime/gpu.lock。入口`python -B -m tools.simplepath_tube.prepare`首次冻结资格计划，然后`python -B -u -m tools.simplepath_tube.qualify`执行一次真实资格。原实验完整终态见results/llava_st_development/terminal_report.md。

2026-10-24 05:10更新：原两域首query各cold/cache真实资格通过，4次原生前向、0新专家、GT读取0；Source、时间、八位置、六层logits、q0、空间特征和Full逐位一致，模型state不变。CPU检查120项通过，包含不完整总体必须在标签访问前停止。资格结果仅为工程准入。

独立运行计划 `results/simplepath_tubedetr/run_plan.json` SHA256 `80fe566c0a238aec65028499a5a403dead1c6192e56a9a3912c963a6e0cb764c` 已在任何新128评分之前冻结；`fixed_method_lock.json`固定两域参数，无新选参。依次Vid128 GPU/CPU评分、HC128 GPU/CPU评分，写`both128_gate.json`后才可Vid10303、HC1902串行执行；128为诊断面板而非性能通过门槛，不根据均值更换query或参数。完整总体含已曝光开发query，不称未曝光独立测试。原Source/MAP、Spatial/native时间、NativeMBR/Source框、Temporal/Source框和Full五行保留，时间原型反馈另对NativeMBR比较。

入口 `python -B -m tools.simplepath_tube.run prepare`首次冻结计划，`python -B -u -m tools.simplepath_tube.run run`执行控制器。日志为`logs/complete_simplepath_tubedetr.log`及`logs/simplepath_tubedetr_<ds>_<dev128/full>_<gpu/score>.log`，runtime登记`simplepath_tubedetr_controller`/`simplepath_tubedetr_worker`，实际身份查/proc。状态、逐条进度、失败在`outputs/simplepath_tubedetr/{status,worker_progress,failure}.json`；两域预测`<ds>/<dev128/full>/sealed.json`，公开评分`results/simplepath_tubedetr/<ds>/{development_metrics,full_metrics,final_seal}.json`及总final_seal。异常保留，不自动重试或恢复。

缓存优先保存全部原生小输出及视频级CNN前缀；开发128的新post-fusion特征要求保存，全量特征在新缓存用量48GiB之前机会保存，为前缀留余量。总体新缓存64GiB、磁盘25GiB下限，无损无替换无自动删除。每条专家仅精确上下文cache命中读取。全量中与本轮128完全相同的request复用本轮预测，其余执行新的真实原生视频/文本编码器、decoder和固定新方法；旧BoxFixed full只供Source逐位回归，不当新版结果。部署冷时延与此次cache加速计时分开。
