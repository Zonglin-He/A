"""Postscore report builder only; no model/GT/scoring/network entry points."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'research/round50';S=H/'independent_score/dev336_attempt001';A=H/'independent_review'
ARMS=['source','old_G','wideentry_oldW','oldentry_newW_k3','wideentry_newW_k3','oldentry_newW_k10','wideentry_newW_k10']
LABELS=dict(zip(ARMS,['Frozen source','旧 G','扩大入口＋旧 W','旧入口＋条件 W，k=3','扩大入口＋条件 W，k=3','旧入口＋条件 W，k=10','扩大入口＋条件 W，k=10']))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def main():
 auth=read(H/'experiment/ROOT_REPORT_EXPORT_PREPARATION_AUTHORIZATION.json');assert auth['state']=='ROOT_AUTHORIZED_R50_SAVED_DEV_RESULTS_REPORT_AND_LOCAL_EXPORT'
 assert auth['report_program_sha256']==sha(__file__)
 for rel,h in auth['inputs_sha256'].items():assert 'round46' not in rel and 'FINAL' not in rel and sha(ROOT/rel)==h
 seal=read(S/'SEALED.json')
 for rel,h in seal['files'].items():assert sha(S/rel)==h
 receipt=read(S/'SCORING_RECEIPT.json');assert receipt['state']=='PASS_R50_SINGLE_OLD336_SEVEN_CONFIGURATION_TASK_SCORE' and receipt['task_values']==2352
 summary=read(S/'TASK_SUMMARY.json');choice=read(S/'DEV_SELECTION_AND_FULL_GATE.json');scenes=read(S/'ALL_SCENE_SUMMARIES_PRIVATE.json');prep=read(A/'prepared_audit_attempt001/AUDIT_RESULT.json');pa=read(A/'prediction_audit_attempt002/AUDIT_RESULT.json');visual=read(ROOT/'crossplatform/round50/visual336_attempt001/COUNTS.json')
 assert prep['state'].startswith('PASS_') and pa['state'].startswith('PASS_') and choice['all_seven_unique_trials_reported']
 gate=choice['full_gate_positive_balanced_net_vs_old_G'];k=choice['selected_shared_k'];winner=choice['selected_candidate_arm'];m=summary['ALL336']['balanced_mIoU_percent'];lines=['# R50：候选入口与条件关联可靠性的开发实验','', '**裁决：'+('预定开发总体门通过；仅登记进入锁定全量对照，不宣布全量或TTA成功。**' if gate else '预定开发总体门未通过；停止该固定分支，不运行全量或TTA。**'),'', '本轮使用已多轮暴露的旧336条开发观测，Drone/Quad各168条、分别2/3个历史场景。开发标签允许在预先固定的k=3/10中选择两平台共享参数；D208/V64/M64都是旧子集，不是独立确认集。本轮DEV不读取受保护FINAL384专用工件，也不修改旧R49或恢复夜间定时任务。','', '主表保持同一Waymo-only源模型、原256个6D框、源分数、主体属性文字、Qwen模型、两种裁剪full-minus-gray、原每mask P/R关系算子与lambda=2。只改入口和关联传播；不进行参数或框更新。','', '## 全部七个预定配置','', '|配置|Drone mIoU %|Quad mIoU %|两平台平均 %|相对旧G pp|救回/破坏（相对旧G）|','|---|---:|---:|---:|---:|---:|']
 lines=[x.replace('救回/破坏（相对旧G）','IoU改善/恶化（相对旧G）') for x in lines]
 for arm in ARMS:
  d=summary['ALL336']['drone'][arm];q=summary['ALL336']['quad'][arm];lines.append(f"|{LABELS[arm]}|{d['mIoU_percent']:.6f}|{q['mIoU_percent']:.6f}|{m[arm]:.6f}|{m[arm]-m['old_G']:+.6f}|{d['improved_vs_old_G']+q['improved_vs_old_G']}/{d['worsened_vs_old_G']+q['worsened_vs_old_G']}|")
 lines+=['',f"k按扩大入口＋条件W的两平台平均mIoU选取，差值≤1e-10 pp时选10。本轮共享k={k}，主五行使用同一k；全部七个配置保留。三个新候选在此k下按同一指标选择，数值平手依次优先入口扩展＋旧W、旧入口＋新W、两者组合。选定{LABELS[winner]}，相对旧G为{choice['selected_gain_vs_old_G_pp']:+.6f} pp，相对Frozen为{choice['selected_gain_vs_source_pp']:+.6f} pp。全量门只要求相对旧G总体增益>1e-10 pp，不施加逐平台、逐个负例或零损害门。",'', ('进入全量的候选只改变入口、保留原W，其本身没有k。新条件rho方案并未胜过该覆盖入口对照，不能把入口收益写成新rho权重的有效性。' if winner=='wideentry_oldW' else '选中候选与未选候选的完整结果同时保留，不将选参后的最好值当作未筛选验证。'),'', '## 方法与工程混杂','', '旧入口粗类过滤把torchvision原生COCO标签与连续80类索引混用。扩大入口没有类别、源分或top-k筛选，因此绕过了该错误；此收益包含工程修复，不能全归给新算法。原oldG实现完整保留，没有静默修正基线。', '', f"独立准备审计确认1239个新增mask，其中{prep['new_masks_rejected_by_oldwrong_but_correct_taxonomy_compatible']}个被旧错误类门拒绝、但符合正确的预测类别家族。该计数不是目标身份正确率。新增mask使用原主体文本、原两种crop和同一冻结Qwen补齐4次numeric输入，旧记录原样复用；未追加参照物模型预算。",'', '新关联a在所选池内的独占可见点上归一，rho=池内独占点数/(框内全部可见点数+k)。池外前景、未分配、重叠歧义和k先验保留为unknown。先用a形成原锚定与clip的条件证据差，再整体乘rho；rho为0时不干预。V/P/R均通过各自W传播，源框从未用可见表面重新拟合。点支持仍共享mask/标定，只是观测支持代理，不认证完整几何或世界身份。','', '## 历史子集与场景','', '|历史子集|Frozen平均 %|旧G平均 %|选定新候选平均 %|新候选相对旧G pp|','|---|---:|---:|---:|---:|']
 for part in ('D208','V64','M64'):
  mm=summary[part]['balanced_mIoU_percent'];lines.append(f"|{part}|{mm['source']:.6f}|{mm['old_G']:.6f}|{mm[winner]:.6f}|{mm[winner]-mm['old_G']:+.6f}|")
 lines+=['', '全部场景及所有正负尾部随匿名标量公开。帧内重复或重叠query不是不同物体，少数已知场景也不是336个独立域；本轮不作独立新场景确认或总体显著性声明。','', '## 执行完整性与失败记录','', f"1239个新mask对应4956个唯一完成numeric调用，旧视觉记录逐值复用，新参照调用、grounder前向、参数更新和框更新均为0。成功视觉阶段实际记录{visual['seconds']:.3f}秒（含加载与IO，不是单方法部署延迟）。独立审计复核705024个几何单元、602112个完整分数和2352个选框结果。",'', '保留两次运行环境失败：第一次缺少accelerate而模型载入失败，0个STARTED、0个有效数值；第二次forward接口不接受logits_to_keep，保留1个rejected STARTED、0个COMPLETED。随后恢复原R15登记的语义解释器，模型/提示词/裁剪与方法均不变。额外的接口拒绝尝试没有删除，也不算有效numeric输出或任务负例。', '', '独立预测审计attempt001还因缓存a被.copy()变成C布局，与原F布局的BLAS归约不同而失败。前两条无GT诊断定位到最多1.040834e-16分数差；原a值完全相同。审计改为从计数独立重建原内存分配，继续要求完整分数、query和框逐位一致，没有放宽容差、修改封存预测或读取GT挑修复。原审计代码、失败产物、授权、日志与诊断全部保留。', '', f"全部2352个IoU只评分一次，全部{receipt['ordered_negative_comparisons']}个有序负比较保留；评分后的报告只读已保存标量，不重读GT。",'', '## 后续边界','', ('只有随后锁定全量source/oldG/newG并完成总体比较后，才决定是否注册当前输入的1/3步匹配层适应及同信息两采样无更新共识。本文只报告DEV阶段，不预填全量或TTA结果。' if gate else '按预定总体门停止当前固定分支；不追加案例规则、不按已知负例改变门槛，也不在本轮扩大到5912或TTA。此裁决仅针对本配置，不否定全部可见点关联或grounding TTA路线。')]
 (H/'REPORT_DEV.md').write_text('\n'.join(lines)+'\n')
 dump(H/'DECISION_DEV.json',{'state':'R50_DEV_GATE_PASS_FULL_PENDING' if gate else 'R50_DEV_GATE_FAIL_BRANCH_STOP','method_adopted':False,'selected_shared_k':k,'selected_candidate':winner,'gain_vs_old_G_pp':choice['selected_gain_vs_old_G_pp'],'full_gate_PASS':gate,'full_or_TTA_result_claimed':False,'old_exposeddevelopment_label_selection_disclosed':True,'all_seven_trials_and_negatives_retained':True,'technical_failures_not_scientific_negative_cases':True,'taxonomy_confound_disclosed':True})
 print(json.dumps({'state':'R50_DEV_SAVED_SCALAR_REPORT_WRITTEN','report_sha256':sha(H/'REPORT_DEV.md'),'decision_sha256':sha(H/'DECISION_DEV.json')}))
if __name__=='__main__':main()
