# LLaVA-ST 新固定方法：真实条件时间先验与SimplePath/Native2D

用户授权顺序为先旧LLaVA、再新版TubeDETR、再新版LLaVA。前两组已完整核验发布，TubeDETR终态提交1dadc34bb843225abdb5f59150e3050a07775652/tree240d461afeeec6e57e4e1e6d181a7dd22705ba11；最终回执outputs/maintenance/simplepath_tubedetr_final_publication.json。当前独立入口tools/simplepath_llava，不修改任何已完成的旧代码/计划/预测/评分/seal。

固定main a238705a的SimplePath及全视频单组Native2D：gamma Vid1/HC.5，kappa64、beta1、单轮、context1.5，空间权重.5/.5/.25，无网格/性能筛选/更换query。共享官方merged LLaVA-ST-Qwen2-7B revision2f7261b55b0368e9c2971120b62c77545b3a792d，官方代码bacf6d61e1de27a78fc0025083aa445d3d41b0b3；FP16/SDPA和100帧完整原STP，SigLIP图像microbatch4，单ReferDINO八观察。已完成publisher hash不重扫16GB，加载完整keys和前后model state仍核验。

## 时间先验的精确定义

原Source生成完全保留原prompt、seed、temperature.01、最多1024新token及100-bin时间/稀疏框物理映射。额外读取同一次原生生成的第一对时间token之前KV和原生起点分布；保存实际原生分隔语法。对每个0..98起点，分别使用原模型坐标输入embedding和原KV，计算共享分隔token的条件概率以及该起点条件下的100个结束token分布。候选分数是完整词表log概率的起点+分隔序列+条件结束之和，归一化所有a<b且映射到不同实际物理节点的合法候选，再聚合重复采样物理节点；4950个100-bin候选均可参与，不截top-k、不读标签。

这是从真实autoregressive概率构造的候选分布，不是独立起终点外积，也不是围绕单个生成区间伪造概率。先验读取raw temperature1，Source仍按原.01采样；明确区分二者，不声称该候选分布是官方已校准事件后验。每次生成在实际生成起点上重算的条件结束分布必须与原生真实路径一致，误差超过.002即停止。

## 局部表示与固定输出

使用真实首空间STP100×81×3584，保留LAPE且在语言/文本融合前；重复物理帧平均。用新全视频aux SimplePath重新做9×9原图坐标context1.5 ROI汇聚，拟合一组事件/背景原型，单次更新后原物理MBR再重算空间路径。Source/Spatial/NativeMBR/Temporal/Full五行均保留。Source原生成时间和框不动；NativeMBR使用真实候选分布但保留Source框；Temporal加入原型反馈但仍Source框，Full用新路径。所有控制和正负结果保留为研究记录，开发反馈不作为调参/停跑依据。

## 两域真实资格与工程记录

v1在首条生成之前因token拼写保护失败：实际token为TEMP-000等三位数字，原检查假设TEMP-0。没有原生生成、开发预测或评分。原代码/计划/失败/日志归档outputs/simplepath_llava_engineering_recovery/token_spelling_v1。Root更正tokenizer映射并预冻结v2资格计划a20bfadea5e45935a183e1147b3d9d6090d6c31fcb54c22bd238cf880e110d9b；没有修改候选规则、参数、权重或更换query。

两域原128首query各cold与RAM内SigLIP cache回放，共4次原生生成、396个条件起点分支，0新专家/GT。Source/原生时间/实际帧/八观察与旧full逐位一致；冷/缓存特征、候选先验、条件行、完整输出逐位一致，实际生成路径的条件分支max_abs_error均0，模型state前后不变。实际局部shape均100×81×3584，合法先验4950正质量项。完整加载无missing/unexpected/mismatched keys。结果results/simplepath_llava/qualification.json；124CPU工程检查通过，不能代替真实资格或成绩。

## 正式固定顺序与文件

独立run_plan SHA f2475d249dcf52c87c95f4091c995bfcae47e82ee84fc69f703ed5ac67677ad0与fixed_method_lock在任何新评分之前冻结。先Vid原128预测seal/CPU评分，再HC原128 seal/评分；both128_gate全部引用验证后，Vid10303/732 full/score→HC1902/227 full/score。无新选参。每条有效Source逐位核对旧完整封存，专家仅精确RGB/文本/指纹cache-only读取，miss硬失败；原生无效生成保留且标记，不能造prior/换query/GT补框。模型与数据输入不变，所有推理均labelsguard。

GPU/score均使用/home/wwww/stvg/envs/llava-st/bin/python -B -u -m tools.simplepath_llava.run run或gpu/score --dataset <ds> [--full]，共用runtime/gpu.lock。runtime/tubedetr_active_jobs.json键simplepath_llava_controller/worker，实际/proc cmdline/cwd为准。主logs/complete_simplepath_llava.log；阶段logs/simplepath_llava_<ds>_<dev128/full>_<gpu/score>.log。状态/进度/新故障outputs/simplepath_llava/{status,worker_progress,failure}.json；当前资格v2日志simplepath_llava_qualification_v2.log，旧v1仅归档历史。预测outputs/simplepath_llava/<ds>/<dev128/full>/sealed.json，公开评分results/simplepath_llava/<ds>/{development_metrics,full_metrics,final_seal}.json及总final_seal。完整count/ID/每prediction/receipt SHA先核验，再独立CPU读取GT；部分full不报主表。

## 缓存、完整证据与磁盘

既有约64GiB缓存全保留不追加新大缓存，不替换/删/降精度；旧全部两域专家11934条内容键已刷新注册、读取时逐payload核验，不把登记当全payload审计。实际qualification视觉prefix仅RAM冷/缓存回放，full冷计算，部署成本与cache加速分开。

每query最终NPZ必须保存Source/空间/时间/Full、帧/八观察、原生起点与各起点条件结束log概率、分隔log概率、native_prior、反馈和aux路径。多MB的原始局部网格和pool向量不重复存盘。后验由已存native_prior与frame_feedback经原固定update/inclusive_scores精确重构，保存其原数组SHA到receipt；写入前逐位验证重构相同，不损失后验数值。两域真实资格样本的紧凑证据约70KB/query，12205投影约0.81GiB；25GiB空闲保护仍有效。必须保存的最终预测失败则停止，不通过舍弃结果绕过磁盘限制。

共享checkpoint训练含VidSTG衍生指令，原128及之前全量研究已曝光；不称严格未见域/独立未曝光测试。终态后Root核验全部尚未首核预测、五行全部正负标量和最终seal引用，发布并核远端，再报告用户并删除唯一监控；不能因资格或单域完成提前停止后续任务。
