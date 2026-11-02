"""UNEXECUTED saved-scalar final report builder; no GT/model/network API.

Full results and a closed conditional-TTA status must exist and be pinned.
Missing future evidence is an error, never a prefilled success or zero score.
"""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[2];H=ROOT/'research/round50';E=H/'experiment'
PLAN=E/'FINAL_REPORT_EXPORT_PLAN_LOCK.json';AUTH=E/'ROOT_FINAL_REPORT_AUTHORIZATION.json'
ARMS=['source','old_G','wideentry_oldW']
DEV_ARMS=['source','old_G','wideentry_oldW','oldentry_newW_k3','wideentry_newW_k3','oldentry_newW_k10','wideentry_newW_k10']
TTA_ARMS=['source','literal_A_wideG','group_A_wideG','sameinfo_consensus0','sameinfo_consensus1','sameinfo_consensus3']
LABEL={'source':'Frozen source','old_G':'旧 G','wideentry_oldW':'扩大入口＋旧 W',
 'oldentry_newW_k3':'旧入口＋新 W，k=3','wideentry_newW_k3':'扩大入口＋新 W，k=3',
 'oldentry_newW_k10':'旧入口＋新 W，k=10','wideentry_newW_k10':'扩大入口＋新 W，k=10'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def local(rel):
 p=Path(rel);assert not p.is_absolute() and '..' not in p.parts and 'round46/final' not in rel and 'FINAL384' not in rel
 return ROOT/p
def main():
 auth=read(AUTH);plan=read(PLAN)
 assert auth['state']=='ROOT_AUTHORIZED_R50_SAVED_SCALARS_FINAL_REPORT'
 assert auth['report_program_sha256']==sha(__file__) and auth['plan_sha256']==sha(PLAN)
 def get(rel):
  assert rel in plan['inputs_sha256'] and sha(local(rel))==plan['inputs_sha256'][rel]
  return read(local(rel))
 devroot='research/round50/independent_score/dev336_attempt001/'
 dev=get(devroot+'TASK_SUMMARY.json');choice=get(devroot+'DEV_SELECTION_AND_FULL_GATE.json')
 assert choice['selected_candidate_arm']=='wideentry_oldW' and choice['selected_shared_k']==10 and choice['full_gate_positive_balanced_net_vs_old_G']
 fullroot=plan['full_score_root'].rstrip('/')+'/'
 fs=get(fullroot+'SEALED.json')
 for rel,h in fs['files'].items():assert sha(local(fullroot+rel))==h
 receipt=get(fullroot+'SCORING_RECEIPT.json');full=get(fullroot+'TASK_SUMMARY.json')
 assert receipt['state']=='PASS_R50_SINGLE_LOCKED_FULL5912_THREE_ARM_TASK_SCORE' and receipt['task_values']==17736
 assert get(plan['full_independent_audit'])['state'].startswith('PASS_')
 assert get(plan['full_saved_scalar_audit'])['state'].startswith('PASS_')
 cost=get(plan['full_evidence_cost_audit']);assert cost['state'].startswith('PASS_')
 gates=get(plan['full_gate_decision']);assert gates['full_score_seal_sha256']==sha(local(fullroot+'SEALED.json'))
 assert type(gates['full_gate_PASS']) is bool and gates['selected_candidate']=='wideentry_oldW'
 tta=get(plan['TTA_closed_status']);assert tta['full_score_seal_sha256']==sha(local(fullroot+'SEALED.json'))
 assert tta['state'] in {'NOT_RUN_FULL_GATE_FAIL','NOT_RUN_ROOT_CLOSED_RESOURCE_OR_RUNTIME_LIMIT','COMPLETED_DEV_EPISODIC_SAMEINFO','COMPLETED_DEV_VERIFIED_NO_PARAMETER_UPDATE_DIAGNOSTIC'}
 if tta['state']=='NOT_RUN_FULL_GATE_FAIL':assert gates['full_gate_PASS'] is False
 if tta['state'].startswith('COMPLETED_DEV_'):assert gates['full_gate_PASS'] is True
 tta_selection=None
 dm=dev['ALL336']['balanced_mIoU_percent'];fm=full['ALL5912']['balanced_mIoU_percent'];nfm=full['NONDEV5576']['balanced_mIoU_percent']
 lines=['# R50：扩大证据入口、关联传播与条件 TTA 总体评估','',
  f"**全量直接对照：扩大入口＋旧 W 的两平台平均 mIoU 为 {fm['wideentry_oldW']:.6f}%，相对旧 G {fm['wideentry_oldW']-fm['old_G']:+.6f} pp，相对 Frozen {fm['wideentry_oldW']-fm['source']:+.6f} pp。**",'',
  '方法与执行顺序在全量读取前冻结。DEV 只允许旧336条上的小网格选参；全量三臂只从原 Waymo-only source 的同一256候选框选择，不改变框坐标或模型参数。全量之后是否进入当前输入 TTA，另按封存结果与登记门裁决。本文不将直接重排收益写成参数适应收益。','',
  '## 开发选择与新 W 的负结果','',
  '|预定配置|Drone mIoU %|Quad mIoU %|两平台平均 %|相对旧 G pp|','|---|---:|---:|---:|---:|']
 for arm in DEV_ARMS:lines.append(f"|{LABEL[arm]}|{dev['ALL336']['drone'][arm]['mIoU_percent']:.6f}|{dev['ALL336']['quad'][arm]['mIoU_percent']:.6f}|{dm[arm]:.6f}|{dm[arm]-dm['old_G']:+.6f}|")
 lines+=['',f"按预设扩大入口＋新 W 的总体指标选择共享 k=10；三个新候选再按同一总体指标及固定平手顺序选择。胜者是扩大入口＋旧 W，DEV 相对旧 G {dm['wideentry_oldW']-dm['old_G']:+.6f} pp。这个胜者本身没有 k；新 W 的 k=10 组合反而比覆盖入口对照低 {dm['wideentry_oldW']-dm['wideentry_newW_k10']:.6f} pp。因此本轮主要开发信号来自扩大入口，不能归功于新条件可靠性传播。",'',
  '扩大入口绕过了旧代码将 torchvision 原生 COCO 类 ID 与连续80类索引混用的错误。旧 G 保持原实现，taxonomy flags 单列，未额外添加工程臂；因此收益包含绕过旧类门错误的贡献，不能全部宣称为新算法。可见点、mask 和标定是关联观测代理，不是独立世界身份真值。','',
  '## 锁定全量三臂','', '|方法|Drone mIoU %|Quad mIoU %|平衡平均 %|微平均 %|相对旧 G pp|','|---|---:|---:|---:|---:|---:|']
 for arm in ARMS:lines.append(f"|{LABEL[arm]}|{full['ALL5912']['drone'][arm]['mIoU_percent']:.6f}|{full['ALL5912']['quad'][arm]['mIoU_percent']:.6f}|{fm[arm]:.6f}|{full['ALL5912']['micro_mIoU_percent'][arm]:.6f}|{fm[arm]-fm['old_G']:+.6f}|")
 lines+=['', '|集合|条数|Frozen 平衡 mIoU %|旧 G %|扩大入口＋旧 W %|新相对旧 G pp|','|---|---:|---:|---:|---:|---:|']
 for part,n in [('ALL5912',5912),('DEV336',336),('NONDEV5576',5576)]:
  m=full[part]['balanced_mIoU_percent'];lines.append(f"|{part}|{n}|{m['source']:.6f}|{m['old_G']:.6f}|{m['wideentry_oldW']:.6f}|{m['wideentry_oldW']-m['old_G']:+.6f}|")
 lines+=['', '5912条为既有官方全量观测，Drone 2984、Quad 2928，来自2/3个已知场景。DEV336通过原三路径、完整文字、平台及场景关联，而非假定 ID/alias 相同；其余5576也属于历史暴露目标集，不叫独立新测试或新场景确认。全部场景、Acc@25/50、微平均、逐例与所有负比较另表保留；query变化和IoU变化不能单独认证语义身份纠错。', '',
  '## 条件 TTA 的实际状态','']
 if tta['state']=='NOT_RUN_FULL_GATE_FAIL':
  lines+=['全量预定门未通过，因此未运行当前输入1/3步 TTA，也没有产生其任务成绩。代码草案作为未执行条件计划保留；不以未执行配置替代负结果。']
 elif tta['state']=='NOT_RUN_ROOT_CLOSED_RESOURCE_OR_RUNTIME_LIMIT':
  lines+=['root 已关闭本轮条件 TTA，未执行原因为资源或运行限制；这不是 TTA 科学负结果，也不是成功。未填零值或推测收益。具体计量与技术状态随封存回执保留。']
 else:
  normalized=get(plan['TTA_normalized_saved_results']);assert normalized['state']=='COMPLETED_R50_DEV_EPISODIC_SAMEINFO_SAVED_RESULTS' and normalized['n']==336 and normalized['arm_names']==TTA_ARMS
  assert get(plan['TTA_independent_audit'])['state'].startswith('PASS_')
  assert get(plan['TTA_saved_scalar_audit'])['state'].startswith('PASS_')
  ts=normalized['summary'];tm=ts['balanced_mIoU_percent']
  tta_selection=normalized['selection_and_gate']
  step=tta_selection['selected_step'];selected_arm=tta_selection['selected_arm']
  assert step in (1,3) and selected_arm=='sameinfo_consensus'+str(step)
  assert step==(1 if tm['sameinfo_consensus1']>=tm['sameinfo_consensus3'] else 3)
  d0=tta_selection['gain_vs_sameinfo_consensus0_pp'];dg=tta_selection['gain_vs_literal_A_wideG_pp']
  assert abs(d0-(tm[selected_arm]-tm['sameinfo_consensus0']))<=1e-10
  assert abs(dg-(tm[selected_arm]-tm['literal_A_wideG']))<=1e-10
  assert type(tta_selection['gate_PASS']) is bool
  assert tta_selection['gate_PASS']==bool(tta_selection['accepted_episodes']>0 and tta_selection['nonzero_parameter_element_count']>0 and d0>0 and dg>0)
  lines+=['|实际 DEV336 臂|Drone mIoU %|Quad mIoU %|平衡平均 %|相对同信息0步 pp|','|---|---:|---:|---:|---:|']
  for arm in TTA_ARMS:lines.append(f"|{arm}|{ts['drone'][arm]['mIoU_percent']:.6f}|{ts['quad'][arm]['mIoU_percent']:.6f}|{tm[arm]:.6f}|{tm[arm]-tm['sameinfo_consensus0']:+.6f}|")
  lines+=['',f"按预定全体 DEV336 指标选择共同步数 {step}（精确平手选1步）。所选 {step} 步相对同信息两采样0步 {d0:+.9f} pp，相对单采样直接 G {dg:+.9f} pp；接受反馈的观测为 {tta_selection['accepted_episodes']}/336。两项增量与非零更新条件须同时满足，实际登记门 gate_PASS={str(tta_selection['gate_PASS']).lower()}。",'',
   ('该固定 TTA 配置通过开发登记门；这只是已暴露开发集上的结果，不能冒充未经执行的全量 TTA 收益。' if tta_selection['gate_PASS'] else '该固定 TTA 配置不采纳，停止且不运行全量 TTA。相对0步的正增量仍保留；它没有补足相对直接 G 的差距，不能据此宣称参数更新优于同信息直接使用。')]
  if tta['state']=='COMPLETED_DEV_VERIFIED_NO_PARAMETER_UPDATE_DIAGNOSTIC':
   assert normalized['measurement_mode']=='VERIFIED_NO_PARAMETER_UPDATE_TTA_NOT_ACTUALLY_TESTED'
   lines+=['', '独立审计已证明所有保存匹配头与源参数逐字节相同、1/3步原生输出与最终共识均等于0步。本表是明确授权的六臂纯无更新诊断；installed closure 为 NOT_APPLICABLE，不称适应成功。TTA效果未被实际检验，固定停止，不进入全量TTA。']
  else:
   assert normalized['measurement_mode']=='ACTUAL_NONZERO_EPISODIC_STATES_WITH_INSTALLED_CLOSURE'
   lines+=['', '0/1/3步使用同一A/B采样、固定视觉证据、候选组与对应、原A框及共识读出；只允许184960参数匹配MLP更新。B也参与当前输入拟合，不是未见观测验证。非零更新、零更新、拒绝、第三候选胜出和所有负例均保留；采用与否以共同总体门为准，不以拟合损失或个案判断。']
  lines+=['', 'TTA独立审计首轮在B视图投影矩阵处失败：审计器漏掉原算子对检测框的float64转换，284个值最大相差9.614442464e-9。窄诊断的首个技术尝试另因AST命名空间缺少sha而启动失败，两个原失败均保留；补齐诊断命名空间后，仅恢复float64转换即使投影、点支持、覆盖率与W逐字节复现，角点顺序对照差为零。修复以新审计文件和attempt002执行，原审计代码与生产者封存依赖未改；预测、容差、更新规则均未改变，也未因此重跑任务GT评分。']
 lines+=['', '## 测量、成本与失败保留','',
  f"全量 {receipt['task_values']} 个 IoU 仅统一评分一次，{receipt['ordered_negative_comparisons']} 个有序负比较全部保留。封存预测先经无GT独立审计，再读任务标签；报告生成只读保存标量。DEV原框/分数、池/W/几何统计和原文字graph须先通过准备桥接，最终DEV三臂端点也必须与开发缓存逐位一致。",'',
  'DEV保留两次环境故障：模型加载缺依赖（0有效输出）和forward签名拒绝（1个额外rejected STARTED、0有效输出）；随后恢复原登记语义运行环境。独立预测审计曾因C/F内存布局产生最多1.040834e-16差异，保留失败原稿，再从计数独立重建原分配；继续要求逐位一致，没有修改预测或放宽容差。', '',
  '全量文字阶段另保留计划资产路径/授权拒绝、缺少udapi、Python扩展ABI导入故障，修复仅为依赖与运行环境。旧336份graph字节复用，其他5576条使用原Stanza CPU解析。通用new_model_forwards=0不是所有模型零调用：Stanza神经前向存在、物理次数未单独计量；检测、VLM和grounder按各阶段分别统计。', '',
  '本轮是三臂共享一次缓存采集，不是三种方法独立重跑的部署延迟。物理COMPLETED调用与logical uses分别计数；加载时间和call记录时间已包含在chunk墙钟内，不重复相加。复用R8 source意味着本轮新增source前向为零，不意味着source端到端成本为零。完整计量、失败类型与冻结状态随公开摘要给出，不公开调用原文或权重。','',
  '## 裁决边界','',
  ('全量直接对照通过登记门，可支持本配置在已知全量目标集上的描述性收益；TTA是否成立仍以其实际同信息比较单独裁决。' if gates['full_gate_PASS'] else '全量直接对照没有通过登记门，停止该固定分支；开发正收益没有被推广为全量有效。'),'',
  '本轮没有修改旧R49/已关闭最终注册池，也不恢复夜间定时任务。所有全量结果保留，未因少数负例改规则、追加损失或重新选择参数。']
 (H/'REPORT.md').write_text('\n'.join(lines)+'\n')
 decision={'state':'R50_FINAL_SAVED_RESULTS_REPORTED','selected_full_configuration':'wideentry_oldW',
  'selected_full_configuration_has_no_k':True,'shared_DEV_k':10,'DEV_newW_did_not_beat_coverage_only':True,
  'full_gate_PASS':gates['full_gate_PASS'],'full_gain_vs_old_G_pp':fm['wideentry_oldW']-fm['old_G'],
  'full_gain_vs_source_pp':fm['wideentry_oldW']-fm['source'],'nonDEV_gain_vs_old_G_pp':nfm['wideentry_oldW']-nfm['old_G'],
  'TTA_actual_status':tta['state'],'method_adoption_not_inferred_from_direct_G_gain':True,
  'TTA_selected_step':None if tta_selection is None else tta_selection['selected_step'],
  'TTA_gain_vs_sameinfo_consensus0_pp':None if tta_selection is None else tta_selection['gain_vs_sameinfo_consensus0_pp'],
  'TTA_gain_vs_literal_A_wideG_pp':None if tta_selection is None else tta_selection['gain_vs_literal_A_wideG_pp'],
  'TTA_gate_PASS':None if tta_selection is None else tta_selection['gate_PASS'],
  'TTA_adoption_decision':None if tta_selection is None else ('PASS_DEVELOPMENT_GATE_ONLY' if tta_selection['gate_PASS'] else 'STOP_NOT_ADOPTED'),
  'TTA_stop_without_full_TTA':bool(tta_selection is not None and not tta_selection['gate_PASS']),
  'historical_exposure_and_taxonomy_confound_disclosed':True,'all_seven_DEV_and_three_full_arms_retained':True,
  'all_task_negatives_and_technical_failures_retained':True,'report_did_not_read_GT_or_run_models':True}
 dump(H/'DECISION.json',decision)
 dump(H/'REPORT_FINAL_RECEIPT.json',{'state':'PASS_R50_FINAL_REPORT_FROM_LOCKED_SAVED_SCALARS',
  'plan_sha256':sha(PLAN),'program_sha256':sha(__file__),
  'report_sha256':sha(H/'REPORT.md'),'decision_sha256':sha(H/'DECISION.json'),
  'full_score_seal_sha256':sha(local(fullroot+'SEALED.json')),'TTA_actual_status':tta['state']})
 print(json.dumps(decision))

if __name__=='__main__':main()
