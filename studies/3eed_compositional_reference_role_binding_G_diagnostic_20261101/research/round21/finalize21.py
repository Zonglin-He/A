from pathlib import Path
import sys,datetime
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round21'))
from common21 import *
def main():
    s=read(EXP/'scored208_attempt003/SUMMARY.json');audit=read(EXP/'CPU_AUDIT.json');assert audit['state']=='PASSED';m=read(EXP/'MECHANISM_DIAGNOSIS.json');gate=read(EXP/'scored208_attempt003/TTA_GATE.json');assert not any(v['development_passed'] for v in gate.values())
    costs={'DINO_new_forwards':sum(read(BASE/f'dino208_attempt00{i}/SEALED.json')['new_calls'] for i in [1,2,3]),'Qwen_new_numeric_forwards':sum(read(BASE/f'reference208_attempt00{i}/SEALED.json')['calls'] for i in [1,2,3]),'free_generated_tokens':0,'TTA_optimizer_steps':0,'no_new_source_inference':True,'no_training':True};dump(EXP/'COSTS.json',costs)
    decision={'run':'completed','measurement':'valid_artifact_replay_passed','relationship_increment':'not_supported_in_this_208_development_panel','TTA_steps':0,'TTA_effect_tested':False,'G_adopted':False,'G_research_candidate_only':True,'reason':'Full relation vs matched repaired-appearance no-reference control has exactly zero mIoU/Acc gain on both platforms; three query changes all zero IoU. Attribute gate repair also introduces two Quad harms.','all_negatives_retained':True,'all208_historically_exposed_development':True,'independent_confirmation_launched':False,'source_boxes_frozen':True,'reference_identity_accuracy_not_established':True,'no_front_back_depth_operator':True,'no_new_GPU_queued':True,'no_hyperparameter_sweep':True};dump(EXP/'DECISION.json',decision)
    table='| 方法（全部208，每平台104） | Drone mIoU% | Quad mIoU% |\n|---|---:|---:|\n'
    for arm in s['ALL']['drone']:table+=f"| {arm} | {s['ALL']['drone'][arm]['mIoU_percent']:.6f} | {s['ALL']['quad'][arm]['mIoU_percent']:.6f} |\n"
    phase='| 已暴露子集 | 平台 | Source | R20 G | 修复属性门、无参照 | 加参照 |\n|---|---|---:|---:|---:|---:|\n'
    for ph in ['D','W','Z']:
        for p in ['drone','quad']:
            q=s[ph][p];phase+=f"| {ph} | {p} | {q['source']['mIoU_percent']:.6f} | {q['r20_fullG']['mIoU_percent']:.6f} | {q['gate_repair_no_reference']['mIoU_percent']:.6f} | {q['direct_full_relation_G']['mIoU_percent']:.6f} |\n"
    report=f'''# R21：组合参照、角色绑定与属性门因果诊断

**实际完成208条真实观测、10个直接使用对照。组合文本和视觉定位入口修复，但关系模块相对同信息去参照对照的任务增益仍为0。属性门修复提高均值同时引入2条Quad负例；不采用当前G，不启动TTA。**

## 范围和数据暴露

Waymo-only source100→Drone/Quad，原模型与256候选框保持冻结。复用R20的全部208真实观测：D112、W32、Z64；每平台104。它们在本轮之前已经有评分暴露，全部是机制开发/回归诊断，Z在R21不能重新称独立确认集。没有按本轮GT筛选输入。原始RGB、点云、标定和实例mask实际读取，旧视觉主体缓存仅在原问题、裁剪、模型与输入未变时复用。

规则、阈值和对照在新关系任务评分前固定；预测全部封存后才CPU评分。v2评分暴露后发现首句中嵌入的参照未进入图，v3仅补齐这条解析路径，保留v2全套2080任务-方法行。没有重新调阈值；v3同样属于已暴露开发，不能作为独立验证。

## 实际修改

1. 同时识别surrounding environment和surroundings关系字段。将名词的修饰语、动作和附着物保留在同一实体，保留坐在墙边、穿上衣、背包等角色属性；墙带涂鸦/墙上有涂鸦合成为墙体参照，修饰物不另当独立参照。自然描述首句里的with/逗号参照也进入图。
2. 独立全图开放词汇提案沿用已核验的GroundingDINO；只对完全相同短语复用旧观测。每参照最多3个稳定提案，用冻结Qwen固定Yes/No数值证据核验，不生成JSON/编号字符串。
3. 对每个目标假设分别排除与该目标框IoU≥0.3的参照提案；不把q0预先当正确主体。多个可信参照需要给出一致方向。角色不等约束不等于独立身份金标。
4. 原主体属性门的裁剪最小值条件改为均值≥0.5、最差裁剪≥−0.5，类别增量改为诊断值。另保留仅撤掉增量门、仅改裁剪门的因子对照。不是丢掉完整主体文字。
5. 左/右关系只在声明的相机图像框架内运算。全文相机语境继承有显式标记，严格分句坐标系对照保留；物体自身左右、前后深度和未知框架不执行。可见图像区域没有被当作完整3D物体边界。

## 正式开发结果：同信息直接使用，零参数更新

{table}

{phase}

完整关系G与去参照的mIoU、Acc25和Acc50均相同。3条最终query不同，但前后IoU全0；不能当3条纠错，亦不能凭IoU相同声称身份选择等价。原有R20 source与G评分在CPU逐条精确复现，所有新框均来自同一冻结池。

## 机制覆盖与未解决问题

{m['counts']}

实际核验{audit['reference_evidence']['proposals_evaluated']}个参照提案，语义门接收{audit['reference_evidence']['semantically_accepted']}个；这是模型接收数，不是人工定位准确率。候选关系对共{audit['relation_edges']['edges_evaluated']}条，只有{audit['relation_edges']['discriminating']}条明确区分原第一名与竞争者。大部分未知；排除了{audit['relation_edges']['role_boxes_excluded']}次目标/参照重合，但没有建立角色准确率。

私有真实面板已查看：组合墙参照落到实际涂鸦墙，没有继续把地面粉笔字当成该墙。但坐在墙边的人仍可能被“完整短语相对类别词增量≥0”的门误拒；固定数值读出避免格式失败，并不消除语义校准问题。墙后方关系没有被偷换成图像上下关系，因此墙被正确定位并不自动带来可执行的深度监督。

## 完整属性门的诊断与负例

Z_quad_026原紧裁剪偏好−0.124969、上下文+1.625015，原增量−0.250017/+1.249933；旧关系分支仍要求最差偏好≥0，因此拒绝。只移除增量门不能恢复；改裁剪容差并结合已有图像位置证据可恢复。这说明不能把漏纠错单归因于增量门。

在全部208上，修复属性门相对R20：Drone3条改善0恶化，Quad5条改善2恶化（其余改选为IoU相同）。A_quad_012由0.237646降到0.153233，A_quad_017由0.617932降到0；后者类别增量约−0.000015原本恰好触发旧门，但这不证明该微小数值是可靠身份判据。两条负例的真实图像均查看、原始记录保留。新参照没有救回它们，不能只报均值提高。

## 三条零IoU关系改选的定位上限

GT仅在封存后诊断：A_drone_033、H_drone_007、W_quad_000即使从全部256冻结框中挑最好，IoU分别只有0.158741、0、0.049514，均不能达Acc25。当前给G的候选对最好IoU都是0。这个事实限制的是当前框池在这些样本上的严格定位能力，不证明语义目标不存在，也不能把GT oracle用进运行时筛选。

## 版本、运行和成本

首次解析有两条动作短语被词表截断，以及wall has graffiti未组合；在首次新任务评分前发现，v1文本/提案/视觉记录完整保留，v2仅补算变化的短语。v3补齐首句参照；一次路径后缀迁移误改了R20对照路径，在写出任何预测前报错，失败脚本/锁/空目录保留，修正为R20 attempt002后重跑，没有重算模型证据。

{costs}

两次完整任务比较各208×10=2080行，v2/v3行重叠，不能称4160独立样本。CPU审核通过所有旧新封存、候选池、有限分数和分母；没有训练或回归更新，没有教师/EMA，没有队列中的GPU实验。

## 当前裁决

关系模块相对同信息无参照方案没有正增量，两平台均未通过预定的关系开发门；独立确认与TTA均未启动。保留组合解析/数值读出实现作为研究基础，当前融合决策不采用。未解决的具体问题是参照修饰属性的误拒、同类实例角色可辨性、前后关系的有效观测，以及部分样本冻结框池缺乏可用定位候选。
'''
    (EXP/'REPORT.md').write_text(report);dump(EXP/'RUN_STATE.json',{'state':'R21_COMPLETED_SCORED_AUDITED_PUBLICATION_PENDING','measurement':'valid','relationship_increment_pp':{'drone':0.,'quad':0.},'TTA_steps':0,'G_adopted':False,'no_new_GPU_queued':True,'report':'REPORT.md','updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});print(costs)
if __name__=='__main__':main()
