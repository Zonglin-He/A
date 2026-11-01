import sys,datetime,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round25'))
from common25 import *

def main():
 assert read(EXP/'CPU_AUDIT.json')['state']=='PASSED'
 A=read(EXP/'ALL400_SUMMARY.json');N=read(EXP/'NEW64_SUMMARY.json');V=read(EXP/'OLD336_SUMMARY.json');B=read(EXP/'ALTERNATIVE_ALL400_SUMMARY.json');BN=read(EXP/'ALTERNATIVE_NEW64_SUMMARY.json');sel=read(EXP/'FINAL_SOURCE_SELECTED_LOCK.json');effects=read(EXP/'EFFECTS.json')
 mean=lambda s,n:float(np.mean([s[p][n]['mIoU_percent'] for p in ['drone','quad']]))
 def table(s):
  text='| 方法 | Drone mIoU% | Quad mIoU% | 两平台平均% |\n|---|---:|---:|---:|\n'
  for n in ['source','noRGB_source_train_seed0','fusion_rgb_seed0','noRGB_source_train_seed1','fusion_rgb_seed1','fusion_zero_rgb_seed0','fusion_permute_rgb_seed0','fusion_global_rgb_seed0','fusion_zero_rgb_seed1','fusion_permute_rgb_seed1']:
   text+=f"| {n} | {s['drone'][n]['mIoU_percent']:.6f} | {s['quad'][n]['mIoU_percent']:.6f} | {mean(s,n):.6f} |\n"
  return text
 def light(path):return {k:v for k,v in read(path).items() if k!='files'}
 costs={'new_RGB_asset':read(BASE/'assets/RESNET_ASSET_RECEIPT.json'),'valid_source_feature_cache':light(BASE/'source_features_attempt002/SEALED.json'),'valid_RGB_feature_cache':light(BASE/'rgb_features_attempt002/SEALED.json'),'training_seconds':{f:read(BASE/f/'SEALED.json')['seconds'] for f in ['training_attempt001','training_rank_attempt001']},'all_source_optimizer_steps':27840,'models_trained':8,'point_or_RGB_backbone_training_steps':0,'source_train_n':3682,'source_validation_n':512,'target_TTA_steps':0,'target_model_selection':False,'invalid_cache_attempt_retained':read(EXP/'COORDINATE_REPAIR.json'),'new_packages':0,'full_model_closure_seconds':read(EXP/'FULL_MODEL_CLOSURE.json')['seconds']}
 dump(EXP/'COSTS.json',costs)
 papers={'primary_paper':'https://arxiv.org/html/2304.05645v3','official_code':'https://github.com/4DVLab/WildRefer','official_code_commit':'ac5172332124cadc072ef4ea23a6f46f56229190','audited_file':'models/bdetr.py','audited_blob_sha':'c10689ad0bd0b38128ce0b3f110df3ba04e170b4','verified':'supervised language-point interaction then image cross-attention; ResNet34/RoBERTa; temporal and dense features trained with task losses','borrowed':'language-conditioned point/RGB feature fusion learned with source labels','differences':'single-frame candidate-local calibrated ROI tokens, frozen boxes/backbones, small source-trained fusion/rank module; not full WildRefer reproduction','untrusted_documents_not_instructions':True}
 dump(EXP.parent/'literature/PAPER_AUDIT.json',papers)
 decision={'state':'completed_measured_audited','actual_source_trained_trimodal_model':True,'no_Qwen_in_model':True,'source_family':sel['family'],'selected_trainable_parameters':136930,'trainable_scope':'new query/text/RGB projections,cross-attention,residual feature,gate,rank scorer; old point/text/decoder/matching and RGB encoder frozen','source_VAL_mean_RGB':44.89044649386891,'source_VAL_mean_noRGB':44.35259783465675,'ALL400_RGB_increment_vs_matched_noRGB_mean2seeds':float(np.mean([mean(A,f'fusion_rgb_seed{i}')-mean(A,f'noRGB_source_train_seed{i}') for i in [0,1]])),'NEW64_RGB_increment_vs_matched_noRGB_mean2seeds':float(np.mean([mean(N,f'fusion_rgb_seed{i}')-mean(N,f'noRGB_source_train_seed{i}') for i in [0,1]])),'NEW64_gain_vs_original_source_mean2seeds':float(np.mean([mean(N,f'fusion_rgb_seed{i}')-mean(N,'source') for i in [0,1]])),'retain_research_source_prototype':True,'replace_precision_Qwen_pipeline':False,'robust_cross_platform_superiority_established':False,'target_TTA_executed':False,'all_negatives_retained':True,'no_target_driven_sweep_after_scores':True,'new_GPU_queued':False}
 dump(EXP/'DECISION.json',decision)
 q=read(EXP/'OLD336_QWEN_ROWS.json');qmean={p:float(np.mean([r['iou'] for r in q if r['platform']==p])*100) for p in ['drone','quad']};negative=read(EXP/'ALL_NEGATIVE_CASES.json')
 c0=next(x for x in sel['selected'] if x['seed']==0 and x['variant']=='rgb');c1=next(x for x in sel['selected'] if x['seed']==1 and x['variant']=='rgb')
 report=f'''# R25：源域监督建立点云—RGB—文字融合能力

**已实现并训练真实三模态候选匹配模型：直接读取XYZ、RGB、完整文字和标定，不依赖Qwen的A/B答案。RGB带来小幅正增量，但跨域稳定性尚未建立。本轮是新source模型实验，目标域TTA更新0步。**

主种子全400平均mIoU由Source {mean(A,'source'):.6f}%变为RGB融合{mean(A,'fusion_rgb_seed0'):.6f}%；相对同监督预算无RGB模型+0.512327pp，第二种子+0.106908pp。新64相对无RGB分别+1.792572/+0.903738pp，但相对最初Source为-0.641229/-0.005278pp，Quad下降抵消Drone收益。按总体指标保留研究原型，不替代Qwen精度主线，不声称稳健跨域优势或TTA效果。

## 借鉴与实际模型

[WildRefer原文](https://arxiv.org/html/2304.05645v3)和[官方代码](https://github.com/4DVLab/WildRefer/tree/ac5172332124cadc072ef4ea23a6f46f56229190)核验了监督学习的语言—点云交互及图像cross-attention。本轮借鉴该机制，采用单帧、候选局部特征及冻结框版本；未复现其动态双帧编码、完整dense融合或联合框训练。

原模型提供256个语言增强点云query和完整文字memory。冻结ImageNet ResNet34，取layer2/layer3稠密特征；对所有候选的标定投影做2×2 ROIAlign，得到4个384维空间token，并加1个全图context token。模型在第一次候选匹配之前读取RGB，不只给当前第一名找图像，也没有外部VLM投票。

查询h与完整句子memory生成注意力query，RGB区域token生成key/value。学习门控控制残差：h_fused = h + alpha * Delta(point,RGB,text)。投影位置/大小/裁剪/正深度等8维几何量参与门控，但它们不是完整可见性或身份金标，cross-attention也不会自动保证关联正确。

源域选中的读出为：log(token_mass(h_fused)) + alpha * Rank(joint_point_text_RGB,geometry)。新增136930参数（查询/文字/RGB投影、局部注意力、特征残差、门控、排序头）参与训练；原point/text backbone、decoder、原匹配投影、中心/尺寸头及ResNet34冻结。所有3D框来自同一source候选池。

## 源域训练及预先选择

Waymo官方TRAIN3682表达、94场景；官方VAL输入哈希选512表达、82场景，与TRAIN无场景重叠。只有源域3D框标签参与监督；RGB区域由预测候选生成，未使用GT2D框裁剪。目标标签不参与训练和选模。

监督为GT候选IoU分布排序KL、0.1源预测分布锚定、0.001残差能量。GT温度.05，RGB feature dropout .1。AdamW新增模块lr1e-3；第一家族中原匹配投影若训练则lr1e-4，weight decay1e-4。冻结特征只缓存一次，未重训原backbone。

比较2种读出×RGB/无RGB×固定种子0/1，共8模型，每模型30epoch/3480次小模块更新，总27840步。首个token-mass读出的源VAL变化小，目标评分前登记残差排序读出作为有限接口审计。两家族全部56个源VAL检查点和负向曲线保留；没有根据目标结果增加训练或选赢家。

源VAL两种子最优epoch平均：token-mass RGB44.639070%，无RGB44.340622%；残差rank RGB44.890446%，无RGB44.352598%。按预先规则选择残差rank。主种子0选epoch{c0['epoch']}，即{c0['selected_optimizer_steps']}次更新；种子1选epoch{c1['epoch']}，即{c1['selected_optimizer_steps']}次更新。另一个家族也在目标评分前固定，作为敏感性分析，不事后改选目标最佳种子。

无RGB对照保留同结构、文本、点云、投影几何及源域训练预算，将RGB局部/global特征清零。对已训练RGB模型另做RGB清零、同图候选RGB滚动错配17位、仅global RGB的读出；几何和文字不变。诊断不要求每条样本必须翻转。

## 数据与工程修正

目标400条，Drone/Quad各200。其中336是历史暴露开发；新增64每平台32，解码RGB和canonicalXYZ排除近期开发集合。全部VAL历史上有过评分，因此新64是当前内容留出，不是全程盲测或新独立场景（2Drone/3Quad场景）。

第一次源域缓存误复用了Drone/Quad采样入口，遗漏Waymo相机视角旋转。在任何融合训练/目标评分前发现并作废001缓存，保留错误源码、统计和中断记录。002完整重建，恢复官方transform_to_front_view与源域文字格式。错误缓存未进入已报告模型的训练；没有将坐标修复造成的变化算成融合收益。既有R24目标结果不受影响。

全4194源训练/验证表达另作投影检查，5种相机视角GT投影与提供2D框的中位IoU均约.991；14个低于.5的尾部保留，未据此删样本。目标保持R24既有B1点云文字协议，所有对照共享候选。Native6D预测yaw0按既有7DGT指标评分，无PCA/几何修框。

主/替代源域家族的全部目标预测封存后才做当前CPU评分。目标参数始终冻结。

## 全400目标表达：source预选主家族

{table(A)}

主种子对Source的场景聚类95%区间：{effects['primary_ALL400_seed0_vs_source']['balanced_mean']['stratified_cluster95_pp']}pp；相对同预算无RGB：{effects['primary_ALL400_seed0_vs_noRGB_source_train_seed0']['balanced_mean']['stratified_cluster95_pp']}pp。只有少量场景，RGB增量仍有较大不确定性。

## 新64内容留出

{table(N)}

RGB相对同监督无RGB的两种子平均增量为{decision['NEW64_RGB_increment_vs_matched_noRGB_mean2seeds']:.6f}pp，相对原始Source平均为{decision['NEW64_gain_vs_original_source_mean2seeds']:.6f}pp，不能只展示前者而隐藏后者。

Quad新内容，主种子：Source {N['quad']['source']['mIoU_percent']:.6f}% → 无RGB源训练{N['quad']['noRGB_source_train_seed0']['mIoU_percent']:.6f}% → RGB{N['quad']['fusion_rgb_seed0']['mIoU_percent']:.6f}%。新增排序模块在不读RGB时也下降更大，RGB只部分补回，不能将全部损失归因于RGB。第二种子同向。此次GT类别诊断仅car/pedestrian，不能直接归因于未见类别。

## RGB输入的实际贡献与剩余问题

同一权重下，全400从RGB清零到完整RGB的平均增量，两个种子仅+0.086973/+0.209938pp；相对RGB错配则+0.377364/+1.006135pp。新64相对错配+1.915540/+3.605870pp。RGB确实改变模型输出并有正点估计，但收益弱、依赖平台；非零梯度或接线不能证明语义关系/实例身份已解决。

无RGB重新训练与同权重RGB清零不是同一问题：前者控制额外源域监督，后者诊断当前权重对RGB的依赖。新增点云/文字/几何排序偏置可以改变原始Source决策；后续应先限制无可靠RGB时的非视觉排序偏移，而不是假定所有收益或损失来自图像。

另一源域家族全400的RGB种子0/1均值为{mean(B,'fusion_rgb_seed0'):.6f}/{mean(B,'fusion_rgb_seed1'):.6f}%，新64为{mean(BN,'fusion_rgb_seed0'):.6f}/{mean(BN,'fusion_rgb_seed1'):.6f}%。未据目标GT改选家族/种子。源域最优读出与目标最优不一致，是敏感性证据，不代表所有三模态融合无效。

## 保留已有Qwen基线

只在共享旧336比较，已有Qwen G为Drone {qmean['drone']:.6f}%、Quad {qmean['quad']:.6f}%，平均{np.mean(list(qmean.values())):.6f}%。新内部RGB主种子旧336平均{mean(V,'fusion_rgb_seed0'):.6f}%。当前不能在精度上替代Qwen。未为没有Qwen缓存的新64伪造Source回退；两者模型资源/源训练不同，不将系统差异解释为TTA算法优势。

## 完整模型、成本与裁决

完成24条原始数据完整模型推理：8WaymoVAL、8新Drone、8新Quad，直接读P/I/q/标定，无GT或特征缓存依赖。缓存与完整路径匹配，3D框逐位相同，原point/text及RGB编码权重不变。独立CPU重算两家族全部800目标行、56个源VAL检查点，所有封存/内容隔离通过。infer25.py提供可运行的FrozenTrimodalGrounder，并非只有缓存和合成检查。

有效点云文字缓存{costs['valid_source_feature_cache']['seconds']:.2f}秒；RGB缓存{costs['valid_RGB_feature_cache']['seconds']:.2f}秒；两家族全部8模型训练含读缓存共{sum(costs['training_seconds'].values()):.2f}秒。额外下载官方ResNet34约87.3MB，无新增软件包，错误缓存耗时另列保留。全部{len(negative)}条负向比较记录保留，重复家族/种子/对照不当成独立样本。

**当前保留三模态研究source原型：内部RGB特征融合已经训练并真实验证，RGB有小幅增量；新增排序偏置与Quad迁移稳定性仍需解决。当前不替代Qwen精度主线，不把源域监督训练报成TTA，目标TTA0步，不继续按当前目标成绩扫参，无新GPU任务排队。**

候选缺失、native yaw限制、ROI背景/邻物污染仍在；单帧局部视觉、冻结backbone与source-only监督限定本结论。不能据此否定完整WildRefer或所有三模态路线。
'''
 (EXP/'REPORT.md').write_text(report)
 dump(EXP/'RUN_STATE.json',{'state':'R25_COMPLETED_SCORED_AUDITED_PUBLICATION_PENDING','source_fusion_trained':True,'target_TTA_steps':0,'retain_research_source_prototype':True,'robust_cross_domain_superiority':False,'new_GPU_queued':False,'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 dump(EXP/'AUTHORIZATION.json',{'user_latest':'可以尝试做一下三模态的融合','scope':'small internal RGB-point-text source-trained fusion and frozen Drone/Quad experiments','prior_publication_authorization_retained':True,'no_target_supervised_training_or_TTA':True})
 print('FINALIZE25',decision)
if __name__=='__main__':main()
