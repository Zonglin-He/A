import sys,datetime,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round23'))
from common23 import *
def main():
    audit=read(EXP/'CPU_AUDIT.json');assert audit['state']=='PASSED';dev=read(EXP/'D208_SUMMARY.json');test=read(EXP/'K64_SUMMARY.json');selection=read(EXP/'SELECTED_v3_LOCK.json');cfg=selection['selected']['configuration'];effects=read(EXP/'EFFECTS.json');mean=lambda s,a:float(np.mean([s[p][a]['mIoU_percent'] for p in ['drone','quad']]));gain=mean(test,'selected_soft_full')-mean(test,'R21_no_ref');assert gain>0
    decision={'state':'completed_measured_audited','latest_user_aggregate_policy_applied':True,'individual_harms_not_a_rejection_gate':True,'current_research_G':'R23_soft_evidence_expanded_pool','retain_as_research_baseline':True,'selected_shared_configuration':cfg,'development_net_gain_vs_R21_pp':mean(dev,'selected_soft_full')-mean(dev,'R21_no_ref'),'new64_net_gain_vs_R21_pp':gain,'new64_net_gain_vs_source_pp':mean(test,'selected_soft_full')-mean(test,'source'),'reference_extra_task_gain_established':False,'EDA_raw_is_not_reproduction_without_dense_training':True,'small_effect_few_scene_uncertainty':True,'TTA_steps':0,'TTA_extra_value_tested':False,'all_negatives_retained':True,'no_new_GPU_queued':True};dump(EXP/'DECISION.json',decision)
    table='| 新K64，每平台32条 | Drone mIoU% | Quad mIoU% | 两平台平均% |\n|---|---:|---:|---:|\n'
    for a in ['source','R21_no_ref','EDA_raw','main_attr','selected_soft_full','soft_no_ref','soft_no_vision','soft_legacy_pool','soft_no_prior']:table+=f"| {a} | {test['drone'][a]['mIoU_percent']:.6f} | {test['quad'][a]['mIoU_percent']:.6f} | {mean(test,a):.6f} |\n"
    dtable='| 开发D208 | Drone mIoU% | Quad mIoU% | 两平台平均% |\n|---|---:|---:|---:|\n'
    for a in ['source','R21_no_ref','selected_soft_full','soft_no_ref']:dtable+=f"| {a} | {dev['drone'][a]['mIoU_percent']:.6f} | {dev['quad'][a]['mIoU_percent']:.6f} | {mean(dev,a):.6f} |\n"
    acc='| K64平台 | Source Acc25/50% | R21 Acc25/50% | 新G Acc25/50% |\n|---|---|---|---|\n'
    for p in ['drone','quad']:
        acc+='| '+p+' | '+' | '.join(f"{test[p][a]['Acc25_percent']:.3f}/{test[p][a]['Acc50_percent']:.3f}" for a in ['source','R21_no_ref','selected_soft_full'])+' |\n'
    cost={'new_source_forwards':66,'source_source64_repeat_probes':2,'new_DINO_forwards':audit['actual_new_DINO_forwards'],'new_Qwen_numeric_forwards':audit['actual_new_VLM_forwards'],'reused_numeric_forwards':read(BASE/'evidence272_attempt001/SEALED.json')['reused_numeric_forwards'],'new_assets_downloaded':0,'new_packages_installed':0,'TTA_steps':0,'all_source_and_foundation_models_frozen':True};dump(EXP/'COSTS.json',cost)
    report=f'''# R23：论文机制审计后的软证据排序与实例池扩展

**按最新“总体指标优先、允许个别负例”要求，本轮不再以零错误或单个案例修好作为准入条件。新G在开发208上相对R21平均mIoU+{decision['development_net_gain_vs_R21_pp']:.6f}pp，在新64观测上+{gain:.6f}pp；Acc25/50相对R21不变。保留为当前研究G版本。增益很小，参照模块仍无额外任务收益，尚不是稳健跨平台提升或TTA效果的证明。**

## 参考论文与实现条件

- [EDA原文](https://arxiv.org/html/2209.14941v3)与[官方代码](https://github.com/yanmin-wu/EDA)：核验主对象/属性/代词/关系/其他实体的分量监督与显式推理。EDA的密集对齐在训练中实现；仅把推理公式放到3EED冻结表示上是迁移假设，不能称复现。裸分量读出在本轮新数据上退化，最终γ=0，没有把它伪装成收益来源。
- [SeeGround官方实现](https://github.com/iris0329/SeeGround)：核验target/anchor候选与实例查找表、图像和空间描述的结合。本轮借鉴扩大独立实例池、使用完整主体描述与图像证据；未复现其视角渲染、模型规模或完整流程。
- [ZSVG3D原文](https://arxiv.org/html/2311.15383v2)：核验实体定位与空间运算分开，并处理视角相关关系。本轮保留有依据的相机图像关系，未知框架为零证据，不再把观察者深度强行当所有behind的含义。软融合/边缘化是本轮自己的实现选择。

审计记录及代码blob/tree SHA见literature/PAPER_AUDIT.json。3EED作者用对象名对应token概率选框；当前source也通常选首个类别名词。这是原基线设计，不宣称作者漏掉必需trick。原source、R21对照始终不改。

## 实际改造

1. 原候选对要求q0唯一映射且最多2个同类竞争实例；现在按文字主体类别取最多6个可映射独立mask，加源映射/旧候选以保证对照覆盖。开发有实例池的观测由99条候选对入口扩展到147条；新K64为48条。Decoder重复query不充当独立实例投票，点支持/覆盖/投影构成归一化关联权重。
2. 冻结Qwen对完整主体问题的紧mask/上下文裁剪做数值Yes/No与空白对照；均值融合，不要求每个裁剪/编号同时过门。不使用完整短语相对类别词增量，不用U弃权作为主要门控。
3. 参照提案按完整短语视觉证据与定位分数形成软权重，多提案边缘化；不要求全部参照同向，不作一票否决。独立实体与目标重合的提案仍排除。未知关系不作负证据。全局图像位置与参照关系分量分开，去参照对照保留全局位置。
4. source分数、RGB证据与空间证据软相加，全部框保持source原始6D数值。最终共享参数为{cfg}；形式为source线性分数+2×有界RGB残差+2×全局位置残差+2×参照残差；token残差权重0。这里是评分，不声称校准后的对象后验。

## 开发过程与新观测隔离

D208全部已暴露，允许并披露目标开发集权重选择。先登记12组log-prior配置；首轮最佳平均22.287209，未超过R21的24.093858。随后只比较log/linear先验的24组，最佳linear平均23.825429，仍未超过。两轮均取到权重边界，因此在读取K任务标签前，登记最后12组linear、γ=0、视觉1/2/4、空间0/.25/1/2的尺度审计，选出平均24.727373。所有前轮负结果/封存/排名保留；K评分后没有继续换配置或重选控制赢家。

K64按输入哈希在本轮模型输出与开发选择前固定，每平台32，分别2Drone/3Quad场景。实际解码RGB与canonical XYZ排除近期所有开发集合，当前64内部也分别唯一。全部目标VAL历史上有过评分，因此这是本轮输入内容留出，不能称从未接触标签的盲测或独立新场景。

同一全局配置用于两平台，预测封存及选择文件哈希写入后才评分K。source权重、RGB/点云、标定、mask输入与旧基线逐条重放。主指标为两平台平均mIoU；Acc25/50、场景权重和负例全部报告。数据选择、超参开发和测试判据未使用单个案例修复要求。

{dtable}

开发Drone相对R21五条改善一条恶化；Quad三条改善两条恶化，净mIoU为正。开发Acc25提高，但Quad Acc50由37.5降至36.538，取舍未隐藏。

## 新64真实观测：冻结配置直接重排

{table}

{acc}

新G平均mIoU相对source+{decision['new64_net_gain_vs_source_pp']:.6f}pp，相对R21+{gain:.6f}pp。后者全部来自Drone的+0.513839pp，Quad不变。新Drone相对R21两例IoU改善、一例恶化：0→0.369500、0→0.167104、0.372176→0；净为正但不是每例正确。全部三个真实RGB面板均查看，图示仅为mask关联，不是额外身份金标。

## 贡献与证据边界

- 去参照在开发有2条query改变、IoU前后均为0；新64最终选择与指标相同，不能宣称关系推理有了新突破。γ=0和裸EDA退化同样保留。
- 将新方案的RGB/空间修正限制回旧候选query集合，K Drone从26.723544降到25.568855，Quad不变；扩展候选覆盖是观察到的关键差别。这个消融固定权重但改变可干预集合，不能把全部差值称纯排序算法收益。
- 去RGB的新K结果Drone24.748625/Quad30.092376，低于完整G，说明视觉信息有用；此系统使用外部冻结VLM，不能把相对纯点云文字source的全部增益归给参数适应。
- 新Drone相对R21的场景聚类95%区间为{test['drone']['selected_soft_full']['scene_cluster95_gain_pp']}pp，跨0，只有2个场景。Quad改变为0。当前是小幅总体点估计改善，不是已建立的稳健显著收益；按用户最新准则保留研究版本，不再因单个负例自动否决。
- A_quad_012从R21的0.153233进一步到0.139997；A_quad_017仍0，source为0.617932。H_quad_004也从0.597448降到0。没有写个案规则来修好这些负例，所有记录公开标量保留。

## 运行、成本与状态

{cost}

共272独立当前输入，开发在多轮网格与对照中重复计分；所有标量行不当作独立样本。新增数值VLM前向1128（另1776个旧前向证据严格复用），GroundingDINO318次，source新64+2次重复探针。无新权重下载或环境安装；source/基础模型冻结，GT不进入G运行时。CPU审核通过所有封存、新K内容隔离、旧Source/R21选择及IoU精确重放。源码、选择历史、消融和正负尾部完整保存。

当前将R23保留为后续比较的研究G，R21继续保留为共同对照。当前成果是无参数更新的直接重排；本轮未运行TTA，不能声称参数适应提供额外收益。没有新GPU实验排队。
'''
    (EXP/'REPORT.md').write_text(report);dump(EXP/'RUN_STATE.json',{'state':'R23_COMPLETED_SCORED_AUDITED_PUBLICATION_PENDING','retain_as_research_baseline':True,'TTA_steps':0,'new64_mean_gain_vs_R21_pp':gain,'reference_extra_gain_established':False,'all_negatives_kept':True,'no_new_GPU_queued':True,'client_local_date':'2026-11-01','updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});print(decision)
if __name__=='__main__':main()
