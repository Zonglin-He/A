import sys,datetime,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round24'))
from common24 import *

def main():
 a=read(EXP/'CPU_AUDIT.json');assert a['state']=='PASSED';s=read(EXP/'M64_SUMMARY.json');e=read(EXP/'EFFECTS.json');sel=read(EXP/'LAYER_SELECTED_LOCK.json');cost=read(EXP/'MATCHED_MODALITY_COST.json');mean=lambda n:float(np.mean([s[p][n]['mIoU_percent'] for p in ['drone','quad']]))
 papers={'retrieved_local':'2026-11-01','primary_sources_only':True,'external_documents_treated_as_evidence_not_instructions':True,'RLCF':{'official_code':'https://github.com/mzhaoshuai/RLCF','paper':'https://openreview.net/forum?id=kIP0duasBb','verified':'ICLR2024, frozen CLIP reward supplies external test-time feedback; pipelines choose reward baseline and task sampling','borrowed':'external semantic feedback can supervise adaptation, source scoring retained','not_reproduced':'no RL sampling/reward estimator; our deterministic ranking distillation is not RLCF baseline'},'TTD':{'official_code':'https://github.com/walawalagoose/TTD','paper':'https://openaccess.thecvf.com/content/CVPR2026F/html/Chen_Test-Time_Distillation_for_Continual_Model_Adaptation_CVPRF_2026_paper.html','verified':'CVPR Findings2026, frozen VLM external supervisor, blended teacher, calibration issue and rectification','borrowed':'do not assume arbitrary VLM and target confidence are calibrated; keep matched direct-use comparison','not_reproduced':'no CoDiRe MSP fusion/optimal transport; our G fixed bounded residual and KL are task adaptations'},'TDA':{'official_code':'https://github.com/kdiAAA/TDA','paper':'https://openaccess.thecvf.com/content/CVPR2024/html/Karmanov_Efficient_Test-Time_Adaptation_of_Vision-Language_Models_CVPR_2024_paper.html','verified':'training-free dynamic key-value cache, positive/negative pseudo-label support','borrowed':'same-history no-gradient memory can be a stronger comparator than source alone','not_reproduced':'our per-observation nearest-candidate regression correction cache is not official TDA classification implementation'}};dump(EXP.parent/'literature/PAPER_AUDIT.json',papers)
 decision={'state':'completed_measured_audited','retain_Qwen_G_research_mainline':True,'lightweight_alternative':'CLIP_ViT_B32_scale40','remove_Qwen_does_not_mean_remove_VLM':True,'selected_update_interface':'contrastive_align_projection_image entire 288-288-288-64 MLP','trainable_parameters':184960,'selected_configuration':sel['best'],'selected_update_steps_per_platform':16,'domain_states_separate':True,'D208_unlabeled_G_observations_per_platform':104,'fit_includes_task_GT':False,'validation_target_GT_tuning_disclosed':True,'protocol':'block target adaptation then frozen M64 inference, not claimed continuous-online or per-sample episodic','selected_no_current_G_transfer_gain_vs_source_pp':mean('query_mlp_transfer')-mean('source'),'same_current_G_extra_update_gain_pp':mean('selected_TTA_plus_current_G')-mean('Qwen_G_direct'),'adoption_scope':'retain full query MLP as research TTA candidate, not robust validated production method','robust_effect_established':False,'observed_extra_TTA_harms_retained':True,'semantic_instance_reasoning_improvement_established':False,'source_uses_XYZ_and_text_only':True,'Qwen_role':'frozen external teacher-like preference supervisor, no EMA teacher/student copy','no_further_M64_selection_or_hyperparameter_sweep':True,'new_GPU_queued':False};dump(EXP/'DECISION.json',decision)
 costs={'source_capture336_forwards':336,'complete_model_head_closure_forwards':256,'new_DINO_reference_forwards':read(BASE/'dinoM64_attempt001/SEALED.json')['calls'],'new_Qwen_forwards_M64':722,'new_CLIP_image_encodings_all336':3626,'new_CLIP_text_queries_all336':669,'extra_cost_replay_CLIP_M64_images':722,'screen_optimizer_steps_all16trajectories':1024,'selected_optimizer_steps_total2platforms':32,'screen_seconds':read(BASE/'screen_attempt001/SEALED.json')['seconds'],'matched_M64_cost':cost,'no_source_retraining':True,'new_packages':0,'new_asset_downloads':0};dump(EXP/'COSTS.json',costs)
 table='| 新M64，每平台32条 | Drone mIoU% | Quad mIoU% | 平均% |\n|---|---:|---:|---:|\n'
 for n in ['source','no_Qwen_G_direct','CLIP_G_direct','Qwen_G_direct','same_A_cache_transfer','query_last_transfer','text_mlp_transfer','rank_adapter_transfer','query_mlp_transfer','same_A_cache_plus_current_G','selected_TTA_plus_current_G']:table+=f"| {n} | {s['drone'][n]['mIoU_percent']:.6f} | {s['quad'][n]['mIoU_percent']:.6f} | {mean(n):.6f} |\n"
 params={'rank_adapter':64,'query_last':18496,'query_mlp':184960,'text_mlp':184960};layer='| 更新位置 | 参数量 | V64选定LR/步数 | M64平均mIoU% | 相对Source pp |\n|---|---:|---|---:|---:|\n'
 for c in sel['per_layer']:
  n=c['layer']+'_transfer';layer+=f"| {c['layer']} | {params[c['layer']]} | {c['lr']}/{c['steps']} | {mean(n):.6f} | {mean(n)-mean('source'):+.6f} |\n"
 acc='| 平台 | Source Acc25/50% | Qwen G直接 | 更新+同G |\n|---|---|---|---|\n'
 for p in ['drone','quad']:acc+='| '+p+' | '+' | '.join(f"{s[p][n]['Acc25_percent']:.3f}/{s[p][n]['Acc50_percent']:.3f}" for n in ['source','Qwen_G_direct','selected_TTA_plus_current_G'])+' |\n'
 report=f'''# R24：保留Qwen、轻量替代与匹配层参数适应

本轮实际完成Waymo-only→Drone/Quad。Qwen主线、去Qwen、CLIP替代、四个参数接口、同历史无梯度缓存及完整模型回放全部运行并独立CPU复算。**完整查询匹配投影MLP是本轮最合适的更新位置：新64平均mIoU比Source提高{mean('query_mlp_transfer')-mean('source'):.6f}pp；结合与直接重排完全相同的当前G证据后，再比直接G提高{mean('selected_TTA_plus_current_G')-mean('Qwen_G_direct'):.6f}pp。** 收益小，来自两条改善/一条轻微恶化，只有2Drone/3Quad场景，保留为研究候选，不宣称已验证的稳健效果或语言关系纠错突破。

## 文献核对与实际借鉴

[RLCF官方代码](https://github.com/mzhaoshuai/RLCF)确认外部CLIP奖励可以指导TTA；[TTD正式CVPR Findings2026页面](https://openaccess.thecvf.com/content/CVPR2026F/html/Chen_Test-Time_Distillation_for_Continual_Model_Adaptation_CVPRF_2026_paper.html)确认冻结VLM提供外部监督，并指出异构模型置信度/直接蒸馏的困难；[TDA官方实现](https://github.com/kdiAAA/TDA)确认无梯度动态缓存也可利用测试历史。本轮借鉴外部监督、置信度不能直接等同、同信息缓存对照，**不是以上三篇原方法复现**。Qwen在这里具有教师式偏好监督作用；没有EMA教师副本，不称teacher-free。额外VLM/RGB资源明确披露。

## 一个重要输入事实修正

源码factory的use_color=True和input_feature_dim=3不能证明原模型用了RGB。实际source采样输出[16384,3]；官方_get_3eed_pcd最终return xyz，backbone input_channels=3也只配置XYZ。训练和本轮推理均为XYZ+文字，没有独立RGB分支。本轮对336条真实输入验证了这一点。冻结Qwen通过RGB看到的颜色/衣着属性，未必能被不再看RGB的点云文字学生在新实例上恢复；小幅几何/排序可迁移不等于完整视觉语义已迁移。

## 数据、开发与确认隔离

D208已暴露开发：每平台104条，以无GT的当前R23 G提供监督。R23 K64现在明确转为本轮V64选参集，允许并披露target GT选择更新层、学习率和步数。CLIP尺度5/10/20/40仅在D208选择，选中40。更新接口选择与CLIP选择均在M64评分前冻结。

新M64按输入哈希和场景轮转选定，每平台32条，解码RGB和canonicalXYZ与近期D/K/R15/R18集合全部不重叠；本轮内部64分别唯一。全部VAL历史上有过评分，因此这是当前内容留出，不称从未暴露标签的盲测，也不是新的独立场景。协议为**目标域一组无标签观测上分块适应，然后在未用于监督/拟合的新观测上冻结验证**；每平台保留各自适应状态。不是论文完整连续在线协议，也不是逐样本精确拟合后在原输入上评估。

候选框、源backbone、decoder、中心/尺寸头全部冻结。native6D框按yaw=0与官方旋转7DGT计算IoU，所有对照一致；没有把朝向后处理混入收益。全部预测先封存再读取当前评分标签。无M64的G反馈或标签进入参数拟合。

## 同候选与图像信息的外部模型对照

- Qwen沿用R23软视觉/几何G，当前样本固定score0+2vision+2position+2reference。完整主体属性及参照文字保留，不增加个案补丁。
- 去Qwen删除主体/参照的Qwen判断，但保留相同RGB独立Mask R-CNN实例、点云关联、全局位置和GroundingDINO完整短语参照提案；参照只用检测分数加权。这是“去Qwen”，**不是去所有VLM或去所有外部视觉模型**。
- CLIP ViT-B/32使用与Qwen相同目标mask紧裁剪、上下文裁剪、完整主体短语、参照裁剪与灰图对照，仅替换语义评分。记录77token上下文，实际669条去重文本查询无截断。CLIP仍是VLM。
- 同历史无更新缓存获得与参数法完全相同的D208 G修正，按source查询×文字交互特征存键值；每个A观测只投一次最近候选，取4个最近观测，固定温度.1，α仅由V64选定1。没有声称它是官方TDA。

## 在哪里更新、如何更新

{layer}

最终选择`contrastive_align_projection_image`**整个三层MLP**，包含0/2/4三组Linear的weight/bias，288→288→288→64，共184960参数。命名image指decoder查询表示投影，并不意味着该层直接读取RGB图像。最后一层18496参数、文字侧整套MLP184960参数和64参数交互残差排序器均实际跑过；不能只凭“参数少”认定更稳定。

AdamW lr=5e-4，16步，每步使用该平台104条D观测的冻结特征。源初始化；source eval；框和其余参数不变。令当前G分数F0、学生真实token-mass匹配分数sθ、τ=.25，pG=softmax(F0/τ)，p0=softmax(s0/τ)，pθ=softmax(sθ/τ)。损失为：

`KL(pG||pθ) + 0.2 KL(p0||pθ) + 0.5 mean relu(0.05 + max_other(sθ) - sθ[Gtop1]) + sum mean((θ-θ0)^2)`。

这是候选分布蒸馏、源分布锚定、直接针对最终Top-1竞争候选的间隔约束和参数锚定。没有只更新正负集合平均分的漏洞；没有GT停止规则，没有M64监督。固定候选下回归层不影响匹配分数，因此本轮不把排序损失送入中心尺寸头。

每种接口2个学习率、16/64两个检查点，开发总计16条优化轨迹、1024步，32个状态快照。最终方案只需每平台16步，共32步。没有在M64结果之后重选接口/学习率/步数。完整query MLP两平台参数L2变化1.3259/1.1886，首步梯度L2为.2243/1.2352，真实更新已发生。

## 新观测结果

{table}

{acc}

不再调用当前M64 Qwen时，完整查询MLP相对Source Drone+0.426211pp、Quad+0.506075pp，平均+0.466143pp，但仍低于当前Qwen直接重排，不能以这点收益就删除Qwen。

“更新+当前同G”是评分前加入的补充读出：保持当前G额外视觉/空间残差固定，将原匹配分数替换成适应后的匹配分数。两者使用同一M64 RGB/点云/文字证据；参数法额外携带D历史。历史缓存+当前同G获得同样D历史及当前M证据，仍明显更差，因此本轮的正增量不能由该具体无梯度缓存解释；但这不是证明所有更强缓存/无更新策略都较差。

相对直接Qwen G，Drone一条改善一条轻微恶化、其余30条IoU相同，Quad一条改善其余31条相同。平均+0.478950pp。Drone Acc50从12.5到15.625，其他上述Acc25/50不变。不是每条都改善，也不是由大量实例纠错支撑的大提升。

## 机制与负例

全部7条相对直接G的query变化已保留并渲染私有RGB/投影面板；其中3条映射同一独立RGB mask，4条关联未知，没有可以认证的跨mask身份纠错。Drone主要改善IoU .442329→.592158，同时保留.549821→.544577的轻微恶化；Quad有0→.161944，未达到Acc25。其余4条query改变IoU仍0。IoU排序不能单独当身份金标，因此当前只支持“冻结候选的排序改善”，不支持已解决同类实例选择或复杂语言关系。

所有Qwen直接重排、CLIP替代、去Qwen、各层更新、缓存的负例和开发网格均保留，未按案例拒收。完整query MLP的联合场景聚类95%区间为{e['selected_TTA_plus_current_G_vs_Qwen_G_direct']['balanced_mean']['stratified_scene_cluster95_gain_pp']}pp，下端接触0；少量场景/少数改善，稳健性尚未建立。数据曝光与开发选参全部披露，不将此作为目标标签全程不可见的结论。

## 成本与实现闭环

在完全相同的新M64语义证据上，Qwen722次数值前向耗时{cost['Qwen_same_M64_seconds_excluding_model_load']:.3f}秒，CLIP722张图编码、140条文本查询耗时{cost['seconds_excluding_model_load']:.3f}秒，均不含模型启动，包含读取/裁剪，未包含共同的DINO检测。峰值已分开记录，Qwen约7.73GB、CLIP约.696GB；因此CLIP是更便宜但当前精度较低的替代。全部336条CLIP证据另外计3626张图编码/669条文本查询，不能把这个总体时间直接与M64的Qwen混比。

336次source特征捕获约16.97秒；全部小头开发轨迹约5.09秒。最终四个选定接口均重新安装到**完整grounding模型**，新M64共256次真实前向，输出与缓存回放匹配、全部框逐位相同、非匹配参数指纹不变、结束后source逐位恢复。不是只生成代码或合成梯度检查。匹配新M64的CLIP计时重放余弦值与原封存值完全相同。无新模型下载、无新环境安装、无源模型重训，所有owned模型已退出，无下一轮GPU任务排队。

## 当前裁决

保留Qwen G为精度主线，CLIP作为低成本对照；完整查询MLP（184960参数，AdamW5e-4×16步）保留为第一版**小幅正收益的研究TTA候选**。最后一层、当前小排序器和历史缓存未采用。文字MLP收益很弱。不根据新M64继续扫参，也不把一个点估计提升升级为稳健方法。主要未解问题是弱监督的可迁移信息、同类实例身份证据和少场景统计，而不是“外部VLM是否允许”。

复现入口：select24.py→capture24.py→parseM24/prepareM24/detectM24/evidenceM24/rankM24；fit24→evaluate24选择接口；memory24→已封存V64选择；closure24；modality24→final24选择CLIP并封存/评分M64；audit24→render24→cost24→finalize24。依赖R23及早期受许可数据/缓存，公开导出只含自有代码、配置、匿名标量和审核，不公开原始媒体、文本、坐标、特征、权重。
'''
 (EXP/'REPORT.md').write_text(report)
 dump(EXP/'AUTHORIZATION.json',{'latest_request':'Keep Qwen, run removal/lightweight controls, identify update layers and optimization giving gains; real Waymo-to-Drone/Quad experiments with direct-use controls and unfitted observations','execute_reversible_repairs_and_experiments':True,'aggregate_metric_criterion_no_zero_harm_gate':True,'publication_existing_repository_authorized_by_prior_user_instruction':True})
 dump(EXP/'FINAL_CODE_LOCK.json',{'programs':{str(p.relative_to(ROOT)):sha(p) for folder in [BASE,EXP.parent] for p in sorted(folder.glob('*.py'))},'all_programs_finalized_after_execution':True,'actual_screen_runtime_code_pinned_in_FIT_SPEC_LOCK':True,'not_a_retrospective_preregistration':True})
 dump(EXP/'RUN_STATE.json',{'state':'R24_COMPLETED_SCORED_AUDITED_PUBLICATION_PENDING','measurement':'valid within exposed-development/current-input-holdout protocol','claim':'small full-query-MLP transfer and same-current-G extra gain, robustness unestablished','new_GPU_queued':False,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});print('FINALIZE24',decision)
if __name__=='__main__':main()
