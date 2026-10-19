"""Report completed explicitly GT-assisted finite reference capacity interventions."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_reference_capacity32_v1 import D,P,ARMS,verify,read,put,sha

def put_metadata(path,value):
 put(path,value,mutable=True)

def fmt(z):
 if z['mean']is None:return 'unknown'
 a,b=z['CI95'];return f"{z['mean']:+.4f} [{a:+.4f}, {b:+.4f}]"

def main():
 p=verify(True);x=read(P/'RESULTS_PUBLIC.json');support=read(P/'SUPPORT_AND_CAPACITY_PUBLIC.json');audit=read(P/'ROOT_INDEPENDENT_AUDIT.json');assert audit['status']=='pass';cost=read(P/'COST_BREAKDOWN.json');bar=read(D/'GLOBAL_OUTPUT_BARRIER.json');cal=read(ROOT/'artifacts/stvg_joint_robust_correction32_v1/SOURCE_CALIBRATION.json');lines=['# 参考选择、统一修正与共享 state：GT 整体干预容量32','',
 '2026-10-19。completed / measurement audited / **GT-assisted causal diagnosis only**。全部数字不是可部署成绩或新OPD收益。', '',
 '## 问题与固定范围', '',
 '同一原NH16/NV16，共32个已曝光开发父视频、每父1query；冻结跨域EMA TA-STVG、原Source/E、finalMBR与插值。四个计划节点不变。此共同集合具备原DINO pool/E/完整decoder重放缓存；不把无等价state缓存的CH/CV拼作同条件三组。正式FH/FV64仅hash保护、未用于选择或本轮评分。',
 '旧E的score-supported联合截断残差估计不是“靠近学生”筛选；本轮检验其共享(c_x,c_y,logw,logh)修正能利用多少已有证据。旧共享state用E-selected参考、旧多轨迹OPD曾探索远处而未证明身份切换，两项负结果保留。本轮不再调整局部OPD，也不新造身份标签器。', '',
 '## 干预定义', '',
 '- **Candidate_combo_oracle：** 在原E准入帧上，穷举原≤3候选或outlier，共享修正为选中参考相对Source的四维残差均值，并按原E方式应用于Source。以最终完整vIoU选最优，包含原E作为可行、同分优先的基线。它选择E已枚举的均值候选，未重新训练/调参，也不是连续四维全局上界。',
 '- **GT_E_known4：** 原四个计划位置仅在有精确标注时注入GT singleton，score1消除候选相对分数项，沿用E的source scale/cutoff/beta1/16估计与统一传播。不会把未知位置插值成GT。估计器仍可因残差不一致拒绝正确参考。',
 '- **GT_state_known4：** 用相同GT支持，沿用旧末层空间ca_qcontent_proj临时共享bias256/native RMS/E-retaining映射，Direct_anchor损失5L1sum4+2GIoU、freshAdam HC.01/Vid.03、betas.9/.999/eps1e-8，固定两步和最终step2，不按GT择优迭代。既有网络权重全部冻结，没有新网络训练。',
 '- **解释控制：** GT_mean_known4使用所有已知GT残差均值；GT_subset_oracle_known4按完整vIoU选择GT参考子集均值，区分估计器目标与该有限shared4候选表示。Local_best_mean按锚点局部IoU选择原DINO参考后求均值，不能当完整轨迹oracle。',
 '- **相同支持控制：** GT_E_matched、GT_state_matched与DINO_state_matched仅使用“原E-selected且GT已知”的共同位置。DINO与GT标签之间仅改变坐标，state/损失/优化一致。完全匹配时精确复用旧Direct；不匹配时真实两步重放。GT_known4可能在原DINO缺失或被拒位置提供额外完美证据，必须与标签质量收益分开。',
 '- 无已知GT参考时GT干预保持E；全部父视频、零时间交集、无参考与unknown保留。共享state的两步结果是该固定优化器实际达到的性能，**不是所有共享state的容量上界**。', '',
 '**GT使用边界：** 先锁协议/来源/步数/参数，再提取GT输入；GT明确参与oracle选择与state监督，不能声称预测生成GT-free。全部32最终干预输出封存后才计算汇总表；候选oracle选择本身当然已使用完整轨迹GT评分。GT/raw坐标保留私有。', '',
 '## 标注与监督支持', '', '| 域 | 父视频 | 四计划位置中GT已知帧 | 原E-selected∩GT帧 | GT已知全空query | matched全空query |', '|---|---:|---:|---:|---:|---:|']
 for ds,name in [('hc2','HC'),('vidstg','Vid')]:
  g=x['groups'][ds];lines.append(f"| {name} | {g['parents']} | {g['known_GT_anchor_frames']} | {g['matched_support_frames']} | {g['empty_GT_known_queries']} | {g['empty_matched_queries']} |")
 lines+=['', '## 主要结果：完整vIoU与未观测实际帧sIoU', '', '下表是相对同一个E的父视频平均增量，单位pp；方括号为配对父bootstrap95%区间（10000、seed20261018），各域主分母均为16。实际帧sIoU使用全部已知未观测实际decoder节点，独立于时间区间，插值收益不能代替它。', '',
 '| 干预−E | HC 完整vIoU | Vid 完整vIoU | HC 未观测实际sIoU | Vid 未观测实际sIoU |', '|---|---:|---:|---:|---:|']
 labels=[('Candidate_combo_oracle','原候选组合/full-v oracle'),('GT_E_known4','正确参考/原E估计器'),('GT_state_known4','正确参考/共享state两步'),('GT_mean_known4','正确参考/全均值'),('GT_subset_oracle_known4','正确参考/子集均值oracle'),('Local_best_mean','局部最优参考/均值'),('Fine_center','既有Fine-Center'),('Old_DINO_state','既有DINO共享state')]
 for arm,label in labels:
  vals=[fmt(x['groups'][ds]['paired'][arm+'_minus_E_MBR'][ep])for ep in ['v','unobserved_actual_s']for ds in ['hc2','vidstg']];lines.append('| '+label+' | '+' | '.join(vals)+' |')
 lines += ['', '原E完整vIoU均值HC/Vid分别为 '+ '/'.join(f"{100*x['groups'][ds]['means']['E_MBR']['v']:.4f}%"for ds in ['hc2','vidstg'])+'，同一时间分支全部干预tIoU严格不变。', '',
 '## 区分标签质量、支持覆盖和传播', '', '| 对比 | HC 完整vIoU | Vid 完整vIoU | HC 未观测实际sIoU | Vid 未观测实际sIoU |', '|---|---:|---:|---:|---:|']
 for pair,label in [('GT_state_matched_minus_DINO_state_matched','相同支持：GT state−DINO state'),('GT_state_known4_minus_GT_state_matched','GT支持扩充：known4−matched'),('GT_state_known4_minus_GT_E_known4','正确参考：state−原E'),('GT_subset_oracle_known4_minus_GT_E_known4','正确参考：子集oracle−原E'),('Candidate_combo_oracle_minus_Local_best_mean','完整轨迹选组合−局部最优均值')]:
  lines.append('| '+label+' | '+' | '.join(fmt(x['groups'][ds]['paired'][pair][ep])for ep in ['v','unobserved_actual_s']for ds in ['hc2','vidstg'])+' |')
 lines += ['', '四帧GT经原E仍然不够好时，不能立即归因为统一修正表示不足；原估计器优化的是截断残差与分数目标，并非完整vIoU。子集oracle只检验有限GT均值族，不是全连续修正最优。共享state若只拟合锚点而不传播，结论也只能限于这两步/该入口，不能排除不同接口或更充分训练。', '',
 '## 约5pp立项容量检查与影响集中', '', '| 域/干预 | 完整vIoU增量 | 均值达到5pp | 95%下界达到5pp | 改善/下降/不变query | 逐一留出均值范围pp |', '|---|---:|---|---|---|---|']
 for ds,name in [('hc2','HC'),('vidstg','Vid')]:
  for arm in ['Candidate_combo_oracle','GT_E_known4','GT_state_known4','GT_subset_oracle_known4']:
   g=x['groups'][ds]['paired'][arm+'_minus_E_MBR']['v'];rng=g['leave_one_saved_mean_range'];lines.append(f"| {name}/{arm} | {fmt(g)} | {g['mean']>=5} | {g['CI95'][0]>=5} | {g['positive']}/{g['negative']}/{g['neutral']} | [{rng[0]:+.4f}, {rng[1]:+.4f}] |")
 lines += ['', '5pp是用户提出的开发方向筛选量，不是统计显著性的替代，也不是可部署预期收益。oracle下界跨5或0时保留不确定性。没有达到5的有限干预也不能证明所有观测或传播方案不可能达到。', '',
 '## 每个query、尾部与锚点拟合', '',
 'RESULTS_PUBLIC.json保留全部32逐query指标、known/unknown支持、各干预位移、正负与<-5pp尾部、父配对区间和GT锚点loss。CANDIDATE_LANDSCAPES_PUBLIC.json保留每个有限组合的完整vIoU与匿名局部索引；不公开坐标。CAPACITY_AUDIT_ROWS_PUBLIC.json为独立审计逐query摘要。没有因为oracle案例好看而删除损伤或无支持query。', '',
 '| 域/共享state | 逻辑更新步数 | 新重放fit数 | 精确复用fit数 | 锚点损失变化均值 |', '|---|---:|---:|---:|---:|---:|']
 for ds,name in [('hc2','HC'),('vidstg','Vid')]:
  rr=[r for r in x['rows']if r['dataset']==ds]
  for arm in ['GT_state_known4','GT_state_matched','DINO_state_matched']:
   f=[r['fits'][arm]for r in rr];lines.append(f"| {name}/{arm} | {sum(v['updated_steps']for v in f)} | {sum(not v['reused_exact']for v in f)} | {sum(v['reused_exact']for v in f)} | {sum(v['loss_after']-v['loss_before']for v in f)/len(f):+.6f} |")
 lines += ['', '逻辑fit/步数含精确复用，不冒充本轮新反向次数。无支持或零梯度步无更新；每query/arm独立重置state/Adam。输出不是从采样候选直接挑框，是真实冻结decoder在最终共享state的完整轨迹。', '',
 '## 审计、工程与资源', '',
 f"新增GPU分配占用保守上界 {cost['GPU_allocation_seconds']:.3f}s，实际完整轨迹decoder调用{cost['actual_decoder_trajectory_calls']}（含零态/固定FD/支持控制），PosDecoder调用{cost['actual_PosDecoder_calls']}，反向{cost['backward_calls']}。全部Source/视觉backbone前向/DINO图像编码和条件查询/新模型训练0；旧缓存获取成本排除。CPU oracle输入阶段{cost['CPU_oracle_input_seconds']:.3f}s，汇总评分{cost['CPU_score_seconds']:.3f}s，成功独立审计{audit['CPU_seconds']:.3f}s，准备/清点/发布未完整计时，不称总运行速度。",
 '两个固定NH001/NV001平滑输出FD/AD检查通过；每query零态Source/E exact、native前5层不变、原模型hash不变。根独立CPU重算全部候选均值和完整v评分/全GT子集/最终指标、GT支持坐标、共享state Adam/损失/最终映射和父bootstrap。不是第二次全量GPU复现。共享锁/内存磁盘保护、owned16/20GiB和Swap1GiB已读取真实生效，未信号其他服务。',
 '执行前发现旧fit返回GT_used=False元数据会被继承，0GT/模型/预测阶段保存原source/runtime，并单独PREFLIGHT_METADATA_CORRECTION001覆盖标记与pin；无loss/step/label/readout科学参数变化。其它运行失败如发生单独保留。', '',
 '## 实测判断与支持覆盖分解', '',
 '现有候选组合在本面板的完整v最优增量为HC3.1763/Vid2.9676pp，均未达到预设约5pp容量筛选均值；区间仍宽，不推断其它样本或其它轨迹表示无大容量。HC去掉NH006后均值1.6603pp，Vid去掉NV012后1.4031pp，说明候选机会集中。',
 '正确GT参考通过原E为HC6.5058/Vid10.8284pp；GT全均值HC相同、Vid10.4714pp。原E在HC选择49/49正确参考，在Vid选择24/29。因而统一修正表示并没有在当前错误上耗尽大幅恢复能力；仍可能损伤个例，HC NH003/NH001分别−3.7885/−3.2290pp。GT子集oracle不包含任意当前E修正，因此它在个别query仍可低于E，不能写成含E回退的全方法上界。',
 '额外GT支持必须单独计账：GT_E_known4−GT_E_matched为HC+2.5640pp [0,7.3594]、Vid+6.8682 [0.9988,13.8115]。Vid这6.8682pp恰好全部来自原始DINO观测收据为空的5父视频（其中4父有非零贡献），不是已缓存候选的对象选择收益。原有E-selected∩GT支持下，正确GT参考经E仅+3.9418/+3.9602pp。原观测缺失不是DINO在该帧实测失败。',
 '共享state在正确标签下仍只有HC+.0593/Vid+.0427pp，原锚点loss均值仅2.38377→2.37522、3.42711→3.40892。非空GT条件下也仅+.0632/+.0569pp，不能只归咎于空支持平均。相同支持替换DINO为GT只增加+.0277/+.0074pp。两步确实更新了，但旧接口并未把完美稀疏证据变成大幅整轨迹恢复；这不是共享state数学不可达性的证明。',
 '当前筛选结果支持：更优证据质量/实际覆盖有较大诊断空间；仅把原缓存候选重排后用同一E传播，在这个面板还未达到5pp；直接复用旧两步共享state也不满足大收益目标。尚未证明对象OPD的可部署教师标签或刷新必要性，不自动扩成新方法。', '',
 '## 结论的允许范围', '',
 '这轮只回答候选选择与固定传播/适应接口的开发容量。若原候选oracle有大容量，仍缺可部署教师排序；若GT统一修正有大容量而实际E不用，需研究选择目标；若GT state仍小，只说明现接口/两步未实现大幅跨帧恢复。没有任何一项自动证明on-policy必要性。本轮不新增OPD方法、不自动继续扫参或扩大。',
 '冻结Fine/E与CURRENT/MAIN/PAPER均保留。完整代码协议/匿名signedunknown/负结果、成本、原checkpoint与审计allowlist公开；GT/DINO/轨迹坐标、私有父ID/文本/媒体/权重/raw缓存排除。', '']
 (P/'REPORT.md').write_text('\n'.join(lines));put_metadata(P/'FROZEN_CONFIG_PUBLIC.json',dict(count=32,cohort='NH16/NV16 exposed development, original cache replay',GT_assisted_diagnostic=True,arms=ARMS,state='existing final-layer transient ca_qcontent bias256',rounds=2,Adam=p['Adam'],source_scales={ds:cal['source_domains']['vidstg'if ds=='hc2'else'hc2']for ds in ['hc2','vidstg']},E_beta=1/16,temporal='unchanged finalMBR',capacity_screen_pp=5,no_target_GT_for_parameter_selection=True,GT_for_oracle_selection_and_diagnostic_supervision=True))
 (P/'MODEL_PROVENANCE_PUBLIC.json').write_bytes((ROOT/'results/stvg_error_state/2026-10-19/sparse_state_route32/MODEL_PROVENANCE.json').read_bytes());put_metadata(P/'LOCK_PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),runtime_sha256=sha(D/'RUNTIME_LOCK.json'),preflight_sha256=sha(D/'PREFLIGHT_METADATA_CORRECTION001.json'),input_barrier_sha256=sha(D/'INPUT_BARRIER.json'),output_barrier_sha256=sha(D/'GLOBAL_OUTPUT_BARRIER.json'),all32_final_outputs_sealed_before_summary_scoring=True,GT_used_before_output_barrier_for_oracle_and_supervision=True,not_deployable=True,old_source_formal64_and_production_unchanged=True))
 (P/'ROUND_REVIEW.md').write_text('''# Root scientific review\n\nExplicit GT causal intervention, not an unlabeled model improvement. Protocol and original32 roster/parameters locked before extraction. True labels only where annotations exist at the original planned physical positions; zero-time-overlap/no-support samples retained. Candidate oracle selects full fixed-time tube vIoU among original E assignment means, not per-anchor optimum or continuous four-dimensional optimum. True-reference E versus GT-subset oracle isolates scoring/association from this finite representation. Same-support GT/DINO state controls separate coordinate accuracy and additional labeled anchors. Old direct is reused only exact support/label/config matches.\n\nAll three questions are measured on same NH/NV32. GT state is fixed2 achieved optimizer performance, not a state-space upper bound or novel OPD. Original layers/weights frozen; sparse labels do not magically imply dense supervision. No automatic method admission, parameter tuning or rollout follows. Per-domain 5pp diagnostic screen is not achievable-gain proof; paired uncertainty and sample influence remain necessary. Root independent CPU geometry/metrics/Adam/aggregation audit plus fixed GPU finite-difference receipts, not a second complete GPU execution.\n\nOld negative reference/state/trajectory experiments preserved. Public export includes all signed/unknown outcomes and costs; raw annotation/box coordinates/identities/media/weights/features excluded. Both refs and actual remote bytes must be verified before closure.\n''');print('report32 complete',flush=True)
if __name__=='__main__':main()
