"""Report sealed cache-only geometry/selection/refresh evidence; no GT or model IO."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_dino_residual_advantage64_v1 import D,P,verify,read,put,sha

def put_report_metadata(path,value):
 # Report/provenance products may be regenerated before remote publication.
 put(path,value,mutable=True)

def fmt(z):
 if z['mean'] is None:return 'unknown'
 lo,hi=z['CI95'];return f"{z['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}] (n={z['parents']})"

def main():
 verify(True);x=read(P/'RESULTS_PUBLIC.json');ac=read(P/'ACTUAL_NODE_DIAGNOSIS_PUBLIC.json');sel=read(P/'SELECTION_BREAKPOINT_PUBLIC.json');au=read(P/'ROOT_INDEPENDENT_AUDIT.json');au2=read(P/'REFRESH_SELECTION_AUDIT_PUBLIC.json');assert au['status']==au2['status']=='pass'
 lines=['# DINO 对 E 剩余大错误的优势与状态刷新：缓存诊断64','',
 '2026-10-19。completed / measurement audited / development diagnosis。没有新 OPD 运行、模型训练、专家查询或调参。',
 '', '## 结论', '',
 '**部分远距离、尺度和实例错配上，DINO 的正确定位证据已经存在，却未被 E 的参考选择或条件化解码保留；但尚未证明这些优势必须通过随学生状态刷新的 OPD 才能利用。**',
 '本轮将候选容量、置信度排序、E 参考选择、条件化终点与跨帧推广分开。原始候选中有正确框，不等于在线已能识别；固定最高分替换在完整面板上仍为负。',
 '旧原生共享 state 实验用的是 E 已选 DINO 参考，而非 GT 或原始最高分参考。逐张量32/32核对一致。因此它没有单独检验“先摆脱错误 E 参考，再把正确对象证据推广到全轨迹”；这不等于旧失败都由错参考造成。',
 '', '## 范围、基线与证据缺失', '',
 '同一 E+finalMBR，NH16/NV16 和 CH16/CV16，共64个已曝光开发父视频、每父1query。冻结 TA-STVG 与旧 DINO、四个计划位置、时间和原准入规则；本轮不新增候选或适应变量。正式 FH/FV64只核保护哈希，不用于本轮分层、选参或评分。源模型与继承数据来源见 MODEL_PROVENANCE_PUBLIC.json。',
 '256个计划槽位中有216次真实原始观测、191个原始接纳帧、608个保留候选。10个 Vid 父视频没有原始观测收据，40槽位明确缺失，不能计为 DINO 检测错误。原面板32另有104帧 native900、2012个实际条件查询状态；其中1242状态有相应 STVG GT、770未知。后32没有条件轨迹，不补造。',
 '全部64套 GT-free 几何读出先封存，随后才读对应 STVG/既有 VidOR 标注离线诊断。已知计划帧 HC98、Vid61，其余保持未知。不同候选/查询重复不作为独立样本；统计先帧内、再 query 内、最后按父视频汇总，配对 bootstrap10000 seed20261018。分层之间重叠。',
 '', '## 固定诊断定义', '',
 'E 差框：IoU≤0.3；远距离：差框且中心误差至少一个 GT 宽或高；尺度：差框且任一宽高比或倒数≥2；尺度近中心：再限制归一化中心误差≤0.25。恢复：教师 IoU≥0.5 且比 E 高≥0.1。它们是预锁离线诊断标准，不是在线门控。',
 'Vid 实例错配证据：E 与目标 IoU≤0.1，同时与另一个已标注实例 IoU≥0.5。只能确认占据了其他实例，不能自动解释关系词、动作语义或全部错人原因。HC 缺干扰实例标注，身份状态是 unknown；JSON 中其条件计数0不是不存在身份错误。',
 '', '## 一、错误是否集中在稀疏观测以外', '',
 '| 域/错误 | 全部已知实际节点（父视频） | 未观测实际节点（父视频） | 全部该类节点中有真实 DINO 观测 |',
 '|---|---:|---:|---:|']
 for ds,name in [('hc2','HC'),('vidstg','Vid')]:
  for st,label in [('far','远距离'),('scale','尺度'),('scale_near_center','尺度且近中心'),('instance_mismatch','实例错配证据')]:
   if ds=='hc2' and st=='instance_mismatch':lines.append('| HC/实例 | unknown | unknown | unknown |');continue
   a=ac['groups'][ds]['all_actual']['strata'][st];u=ac['groups'][ds]['unobserved_actual']['strata'][st];lines.append(f"| {name}/{label} | {a['nodes']} ({a['parents']}) | {u['nodes']} ({u['parents']}) | {a['nodes_with_actual_DINO_observation']} |")
 lines += ['', 'HC实际节点 GT 已知754、未知1326；Vid已知1330、未知1809。以上不是全视频未标注区域的错误率。多数已知大错误在未观测节点，但那里没有缓存 DINO 判断；不能把锚点上的教师优势外推为未观测帧教师已胜出。',
 '', '## 二、四帧证据中有没有超过 E 的大修正', '',
 '| 域/错误 | 已知计划帧（父） | 有原始观测 | 保留池内可恢复帧（父） | 原准入最高分可恢复帧（父） |', '|---|---:|---:|---:|---:|---:|']
 for ds,name in [('hc2','HC'),('vidstg','Vid')]:
  for st,label in [('far','远距离'),('scale','尺度'),('instance_mismatch','实例错配证据')]:
   if ds=='hc2' and st=='instance_mismatch':continue
   ff=[f for f in x['frames'] if f['dataset']==ds and f['GT_known'] and f['strata'].get(st,False)];pool=[f for f in ff if f.get('retained_recovery')];top=[f for f in ff if f['frame_admitted'] and f.get('retained_top1_iou') is not None and f['retained_top1_iou']>=.5 and f['retained_top1_iou']-f['E_IoU']>=.1]
   lines.append(f"| {name}/{label} | {len(ff)} ({len({f['alias']for f in ff})}) | {sum(f['probe_available']for f in ff)} | {len(pool)} ({len({f['alias']for f in pool})}) | {len(top)} ({len({f['alias']for f in top})}) |")
 lines += ['', '**保留池恢复是 GT 容量诊断，不是部署选择。** 池包含低于准入阈值的替代框；最高分列沿用原帧准入。不能把两列相减都解释为某一种因果损失。尺度错误可能同时远离目标，不等于纯宽高问题；Vid尺度15帧中只有3帧有真实观测，证据尤其稀疏。',
 '', '## 三、优势在哪里丢失：排序与 E 参考选择不同', '',
 '| 域 | 已知计划帧 | E已选参考可评价 | 已选参考IoU≤.3 | 池有IoU≥.5但已选≤.3 | 最高分IoU≥.5但已选≤.3 |', '|---|---:|---:|---:|---:|---:|']
 for ds,name in [('hc2','HC'),('vidstg','Vid')]:
  z=sel['groups'][ds]['all'];c=z['counts'];lines.append(f"| {name} | {z['known_frames']} | {z['selected_known']} | {c['selected_reference_bad']} | {c['good_pool_but_selected_bad']} | {c['top_good_selected_bad']} |")
 lines += ['', 'Vid的8个“池正确、已选差”帧来自4父视频，6个“最高分正确、已选差”来自 NV012/CV011 两父。HC对应池机会4帧/4父，但最高分没有通过该严格标准。选择断点为封存后机制分解，不是新在线选择器；旧 sparse_state_route32 的目标逐张量32/32等于这些 E-selected 参考。',
 '', '以下案例全部保留为事后机制例子，不用来挑部署样本：', '',
 '- **NV012/位置16：** E对目标 IoU=0，且与另一已标注实例重叠；原生最高分目标框 IoU≈0.9153。E选择另一个差参考；条件 T(E) IoU=0且被拒绝。同视频仅锚点的固定最高分替换可提高完整vIoU约1.1818pp，但未观测实际节点完全没变。此例证明未利用的对象证据确实存在，未证明跨帧转移或刷新必要性。',
 '- **NV011/位置21：** E=0、原始池中最好框≈0.9581，但最高分框与条件中心框均为0。这里连分数排序都没找对，不能仅修改“靠近E”的关联就认为已解决。',
 '- **NH006/位置36：** E=0、池最好框≈0.9116、原生最高分≈0.0023；原始帧未接纳。条件中心因原四锚框合法性检查未执行，不能写成 T(E) 实测失败。',
 '- **NH001/位置42：** E=0、尺度比≈2.16、原始最高分≈0.8446，条件中心为0且被拒绝；原E已选参考约0.3016，未落入≤.3断点计数。',
 '', '## 四、换学生状态是否带来固定查询没有的终点', '',
 '只比较相同物理帧的现存对齐条件查询。24状态均围绕初始E，并非学生已更新后真正发生的第二轮DINO调用。共同接纳子集先均衡物理帧，再均衡父视频；下表支持数为条件子集，不冒充全体64。', '',
 '| 共同接纳比较 | HC | Vid |','|---|---:|---:|']
 for k,label in [('teacher_minus_input_pp','教师−对应扰动输入IoU pp'),('teacher_minus_E_pp','教师−E IoU pp'),('teacher_minus_center_pp','教师−中心T(E) IoU pp')]:
  lines.append('| '+label+' | '+' | '.join(fmt(x['refresh_groups'][ds]['all']['both_admitted']['fields'][k]) for ds in ['hc2','vidstg'])+' |')
 for ds,name in [('hc2','HC'),('vidstg','Vid')]:
  z=x['refresh_groups'][ds]['all']['both_admitted'];lines.append(f"\n{name}：{z['sample_state_instances']}个样本状态/{z['physical_frames']}个物理帧/{z['parents']}父；输出与中心框IoU均值 {z['fields']['output_center_IoU']['mean']:.5f}。与既有固定native候选重合IoU≥.95的父均衡比例 {100*z['fields']['same_fixed_native_candidate']['mean']:.2f}%。相似不是知识相同的数学证明，但没有显示广泛新终点。")
 lines += ['', '在预锁的远距离、尺度、Vid实例错配 strata 中，没有“中心未正确接纳而新状态正确接纳”的新增恢复，也没有共同接纳下比中心高≥5IoU pp且比E高≥10pp的新增终点。HC远距离的中心查询完全未执行，其他 strata 支持也很小；因此不能把0写成所有状态刷新无效。',
 '', '保留两个正例：', '',
 '- **NH011/位置23：** 5个采样状态得到被接纳的正确输出，IoU约0.7774–0.8453；中心约0.4940未接纳，E≈0.3016。5次属于同一个物理帧/一个父，不是5个独立视频；它刚高于差框阈值0.3。4/5新输出已与固定native候选IoU≥.95，固定原生最高分本身已可用，尚不支持“必须刷新”。',
 '- **NH004/位置12：** 一个共同接纳状态从中心0.6989提高到0.7519，与固定native最近框IoU约0.8000，存在真正几何不同的更好终点；但E已为0.5860，不属于预锁大错误，且只有1帧。',
 '', '这两个例子支持“条件输出会依赖输入状态，有时有用”；没有建立“大错误上固定监督已过时，而学生更新后刷新能显著修正”的第三段证据链。',
 '', '## 五、不能直接把最高分框换上去：完整 tube 对照', '',
 '仅替换四个计划实际节点、其余实际帧保持E；时间固定MBR，插值沿用官方。所有输出都在本轮GT前封存。这里没有学生更新或新OPD。', '',
 '| 面板/读出−E，完整vIoU pp | HC | Vid |', '|---|---:|---:|']
 for panel,pl in [('original','原NH/NV32'),('selection','CH/CV32')]:
  for arm,al in [('Retained_top1_admitted','原准入最高分'),('Native900_top035','native900最高分.35'),('Conditional_center035','条件中心.35')]:
   if panel=='selection' and arm!='Retained_top1_admitted':continue
   lines.append('| '+pl+'/'+al+' | '+' | '.join(fmt(x['full_readout_groups'][ds][panel][arm]['v'])for ds in ['hc2','vidstg'])+' |')
 lines += ['', 'CH/CV没有native/条件trace，其相应保留E的0变化只表示缺失回退，不作为教师性能结论。四个读出的未观测实际节点 sIoU 变化均严格为0；完整vIoU变化来自锚点及官方插值。全量逐query的signed、unknown、位移和严重尾部均在 RESULTS_PUBLIC.json，未删无参考或零时间交集父视频。固定最高分总体负向，正确框存在与在线可靠选择之间的缺口不能跳过。',
 '', '## 决策与尚缺的一段证据', '',
 '1. **教师优势：局部存在，已举出确定的大错误正例；普遍可靠优势未建立。** 优先保留原始候选、原分数、E已选与条件终点的完整对应，而不是把原始证据先压成ROI相似度。',
 '2. **目前更具体的缺口是参考假设选择。** NV012/CV011显示正确高分参考可能被现关联舍弃；NV011/NH006又说明高分本身不够。这为后续研究“竞争对象假设/保留替代证据”提供机制依据，不等于已经有可部署的门控或新方法。',
 '3. **on-policy必要性仍未建立。** 下一项证据若要成立，须在同一父/同一帧保留固定原生池、固定中心与实际更新后状态查询，证明当前状态改变后旧目标失效、刷新纠正错误，并最终改善未观测实际节点与完整tube；现有缓存没有这类实际闭环对照。本轮不据此自动加DINO调用或启动新模型实验。',
 '4. 本轮没有修改局部OPD参数、增加专家、训练头或重新打开历史暂停分支。CURRENT/MAIN/PAPER未改；全部结果仍是曝光开发诊断，不是新来源确认，未宣称达到2–3pp或OPD贡献。',
 '', '## 工程、审计与成本', '',
 '首次缓存读取在NV003缺原观测时KeyError中断，发生在新GT访问前；前18文件和错误日志保留。工程修订显式标明10父/40槽缺失，完整canonical extracted002重新封存；没有重新调用检测器或把缺失伪造为负例。见 ENGINEERING_RECOVERY_PUBLIC.json。',
 f"成功几何提取CPU阶段 {read(P/'COST_BREAKDOWN.json')['cache_seconds']:.3f}s；离线评分/诊断 {read(P/'COST_BREAKDOWN.json')['diagnosis_seconds']:.3f}s；成功独立标量/指标审计 {au['CPU_seconds']:.3f}s。初始失败、清点、二次机制审核和发布没有完整逐项计时，不把这三项相加称全程速度。新GPU、新Source、新DINO图像编码/条件查询、新模型训练、新学生适应均0；旧缓存获取成本排除。",
 '根审计独立检查56196候选IoU、2484条件输出/输入IoU、1536指标、432分层bootstrap、61实例标记和全部原文件保护；第二份审计核240刷新聚合字段、1242状态标记、159选择断点标记。由根代理独立实现核对，不冒称另一位研究者复验或全模型重跑。',
 '实现、协议、模型来源、成本、全匿名signed/unknown和失败/正例公开；不公开原图/视频/GT或候选坐标/原父ID/句子/权重/rawcache。', '']
 (P/'REPORT.md').write_text('\n'.join(lines))
 put_report_metadata(P/'ENGINEERING_RECOVERY_PUBLIC.json',dict(status='resolved_before_new_GT',initial_failure='missing original observation at NV003 planned slot',first_partial_parents_preserved=18,missing_original_DINO_parents=['NV003','NV006','NV009','NV013','NV015','CV001','CV003','CV008','CV010','CV014'],missing_planned_slots=40,canonical_extracted='extracted002',no_model_or_teacher_rerun=True,scientific_thresholds_unchanged=True,missing_is_unknown_not_detector_failure=True,initial_runtime_sha256=sha(D/'RUNTIME_LOCK.json'),recovery_sha256=sha(D/'RECOVERY_LOCK.json')))
 put_report_metadata(P/'LOCK_PROVENANCE_PUBLIC.json',dict(status='complete',plan_sha256=sha(D/'PLAN_LOCK.json'),runtime_sha256=sha(D/'RUNTIME_LOCK.json'),recovery_sha256=sha(D/'RECOVERY_LOCK.json'),cache_barrier_sha256=sha(D/'CACHE_BARRIER.json'),all64_readouts_sealed_before_GT=True,cohort='NH16/NV16+CH16/CV16 exposed development; FH/FV excluded',baseline='E+finalMBR',not_new_OPD=True,all_original_cache_and_production_pins_preserved=True,postseal_mechanism_only=['E-selection breakpoint','actual node support','state-condition event analysis']))
 model=read(ROOT/'results/stvg_error_state/2026-10-19/coordinate_spatial_opd64/MODEL_PROVENANCE_PUBLIC.json');model['feature']='no visual feature scoring or model execution in this diagnosis; saved boxes/scores/states only';model['source_provenance']='same locked64 cached TA-STVG Source as coordinate_spatial_opd64; inherited model hashes';put_report_metadata(P/'MODEL_PROVENANCE_PUBLIC.json',model)
 (P/'ROUND_REVIEW.md').write_text('''# Root scientific and engineering review\n\nBoth saved independent audit implementations pass. No agent, model or GPU was spawned for this diagnosis.\n\nScientific finding: DINO has demonstrably correct target boxes on some E far/scale/other-instance errors, but original scoring, E-reference selection and conditional decoding do not preserve them consistently. Native Top1 globally harms this panel. Conditional state variation offers one newly admitted correct physical frame and one common-admitted novel better frame; neither establishes useful large-error closed-loop refresh. Existing sparse-state targets were E-selected references (32 exact), so previous state negatives cannot stand in for a corrected-reference experiment. They also do not disappear.\n\nBoundaries: all64 are exposed developers; identity unknown on HC; Vid co-overlap is other-instance evidence, not a semantic causal diagnosis; thresholds prelocked; selection-breakpoint analysis is postseal; missing40 slots stay unknown; conditional traces available only original32 with legality/admission attrition. Sample replicates are nested by frame/query. Four-anchor readouts leave unobserved actual nodes unchanged. No new OPD performance is claimed. Correct native candidate availability is an indispensable fixed-feedback control for any future refresh claim, not proof that deterministic teacher refresh can never help.\n\nEngineering: initial18 partial files preserved, canonical64 sealed before new annotation access, source/prediction/formal/production pins rechecked; independent scalar IoU/metrics/aggregation plus refresh/selection audit pass. No expert/model/new candidates/parameter changes/training/student updates. Public allowlist excludes raw geometry, identifiers, text, annotations, media, weights and raw features. Publication must verify both refs and exact remote bytes before closure.\n\nDecision: complete this cache diagnosis; preserve E/Fine research baselines and all negative/positive cases. No automatic follow-up or deployment promotion. The promising but unproven remaining route is sparse competing-reference/association evidence that can alter the full model trajectory, with fixed native evidence and actual state refresh separately controlled.\n''')
 print('report and public provenance complete')
if __name__=='__main__':main()
