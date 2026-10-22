"""Render saved actual-boundary outcomes, no annotation or cache access."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def fmt(v):return f"{v['mean']:+.4f} [{v['CI95'][0]:.4f}, {v['CI95'][1]:.4f}]"if v['mean']is not None else'—'
for version in ['v1','v2']:
 out=ROOT/f'outputs/joint_boundary_readout_{version}';pub=ROOT/f'results/joint_boundary_readout_{version}';p=read(out/'plan.json');r=read(pub/'report.json');seal=read(out/'sealed.json')
 protocol={k:p[k]for k in ['schema','status','protocol','sources','formal_plan_sha256']};protocol.update(plan_sha256=r['proof']['plan_sha256'],prediction_seal_sha256=r['proof']['seal_sha256'],cohorts={'HC64':64,'HC32':32,'Vid32':32},all_predictions_before_GT=True,inputs_anonymous=[{k:item[k]for k in ['anonymous','cohort','capture_sha256','signal_sha256','pack_sha256','old_prediction_sha256']}for item in p['items']])
 for key in ['prior_attempt','ambiguity_threshold_from_unlabeled_HC64','prior_attempt_seal_sha256']:
  if key in p:protocol[key]=p[key]
 (pub/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 lines=[f'# 缓存时空联合边界读出 {version}：真实预测结果','',
 '本轮直接生成新时间区间，并在该区间运行原完整框DP；输出全128条先封存，再CPU独立评分。没有把GT错误标签当在线门控，也没有用GT oracle挑部署区间。结果未通过事先规定的采用判据，没有修改当前全量或生产配置。','',
 '## 固定机制','',
 '- 当前Global6旧hull＋原物理tIoU-MBR为主要基线；Source原生224为整体参照。原八观察、候选完整框、空间评分与DP均不改。',
 '- 区间库固定为既有12个单层/offset MBR和Global6 MBR，重复去重，平均约9个不同区间；基线始终保留，未扩大为GT建议边界。',
 '- 原后验期望tIoU为主分数。两端hidden变化的query内秩与目标ROI面积校正注意力的进入/退出正变化相乘，形成[0,1]证据bonus；物理时间积分窗口为min(区间长度/4,1秒)。',
 '- 真实/置换ROI使用原生真实attention、同一全时间辅助轨迹及保面积置换。相同候选/强度的hidden-only与置换对照分离其贡献。',
 '- 无新RGB/encoder/decoder/head/专家/GPU或更新。这里没有重新执行上一轮256次CPU decoder，直接读封存信号。',
 '- 每个新边界同时报告最终完整框选择的full vIoU，以及固定同一条基线空间轨迹、只换时间的vIoU，避免混淆空间重选与时间贡献。','']
 if version=='v1':lines+=['首轮用best expected tIoU<.5或mean view agreement<.5作为无标签含混门控，结果每组只覆盖1条，HC64的25条基线tIoU<.5查询一条也未覆盖。因此这轮几乎没有给预期纠错机制机会；保留原plan、预测和评分，另以v2明确记录一次有界覆盖修正，不覆盖本轮。','']
 else:lines+=['首轮门控覆盖不足后，仅修正一次：uncertainty=1−.5×(原最大期望tIoU＋12 view与基线的平均tIoU)，阈值锁定为HC64无标签中位数0.2970975637，HC32/Vid32使用相同阈值。门控覆盖32/14/14条，其中基线错误15/6/12条。lambda固定.10/.25，不再扫；另保留两档无门控和不使用后验的EvidenceOnly/HiddenOnly/PermutedOnly，给信号选择充分机会。该修正明确在v1结果之后，不能冒充第一次事前设定。','']
 lines+=['## 实际总体结果','', '表内为完整vIoU/tIoU百分比，不是只在错误样本上的均值。','', '| 方法 | HC64 v / t | HC32 v / t | Vid32 v / t |','|---|---:|---:|---:|']
 names=list(r['cohorts']['HC64']['means'])
 for name in names:
  cells=[]
  for c in ['HC64','HC32','Vid32']:
   m=r['cohorts'][c]['means'][name];cells.append(f"{100*m['m_vIoU']:.4f} / {100*m['m_tIoU']:.4f}")
  lines.append('| '+name+' | '+' | '.join(cells)+' |')
 lines+=['','## 全部查询配对增量与风险','', '| 方法 | 域 | ΔtIoU pp [95%CI] | Δ完整vIoU pp [95%CI] | Δ固定空间vIoU pp | Δ边界MAE秒 | 救回/损坏tIoU≥.5 | 改动查询 | v下降>5pp |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for a in p['protocol']['arms']:
  for c,g in r['cohorts'].items():
   d=g['subsets']['all']['paired'][a];lines.append(f"| {a} | {c} | {fmt(d['t'])} | {fmt(d['v'])} | {d['fixed_space_v']['mean']:+.4f} | {d['boundary_MAE_seconds']['mean']:+.4f} | {d['recovered_tIoU_half']}/{d['lost_tIoU_half']} | {g['changed'][a]} | {d['v_declined_over5pp']} |")
 lines+=['','## 原读出错误及后验含混子集','',
 'wrong使用事前固定tIoU<.5定义，correct为≥.5；仅在评分阶段确定。全部、含混、非含混、原读出错误、错误且含混、原本正确、原Source错误七个子集完整保留，不能把GT错误子集作为实际可用门控。','',
 '| 域 | 原读出错误 | 错误且含混 | 固定候选库可跨.5救回上限 | GT候选oracle ΔtIoU pp [95%CI] |','|---|---:|---:|---:|---:|']
 for c,g in r['cohorts'].items():lines.append(f"| {c} | {g['subsets']['baseline_wrong_half']['queries']} | {g['subsets']['wrong_and_ambiguous']['queries']} | {g['wrong_candidate_can_rescue_half']} | {fmt(g['candidate_oracle_delta_tIoU_pp'])} |")
 lines+=['','oracle包含基线所以差值构造上非负，只用于说明候选有多少可选空间；它不是新方法成绩。完整原始候选框/GT区间不公开。','']
 if version=='v2':
  lines+=['本轮的机制结果：','',
  '- Joint10/Joint25在三组均未把tIoU<.5救到≥.5。它们在部分查询内会减少边界误差或提高低tIoU，但没有稳定的净任务收益。',
  '- HC64 Joint10 Δt−.1175pp/Δv−.0263pp，Joint25为−.7302/−.4525pp；增加强度没有修复，后者tIoU配对区间低于零。',
  '- HC32 Joint10 Δt+.4082pp、Δv+.2664pp，但只有2条改动，t区间触及0；Vid32 Δt+.0732pp、Δv+.0028pp，仅1条改动。不能据这两个小正值跳过HC开发失败或声明双向有效。',
  '- 不用后验、只按联合证据排序的EvidenceOnly，HC64 Δt−3.5678pp[−6.3779,−.9443]，1条跨阈值救回、3条原本正确被损坏；Vid亦负。说明不只门控/先验保守，证据排序本身仍不可靠。',
  '- HiddenOnly在Vid点估较好，但HC两个组为负、Vid区间跨零。它不能被事后挑成已确认方法。',
  '- HC64置换对照在.10档反而比真实联合更好；没有建立真实目标空间信息提供额外纠错的证据。','']
 lines+=['## 采用判据与决定','',
 'HC64只在两档主联合候选中选择：Δt≥.25pp、full及fixed-space v均为正、t增量超过同档hidden-only和置换，Source相对严重v损伤不增加超过2条。通过后才有资格用两域32作冻结采用复核；未通过不换另一套参数继续试。',
 '所有128条已提前封存并报告两个复核组结果，但本轮没有通过HC64选择，因此没有从HC32/Vid32反向挑配置。',
 '决定：'+r['decision']['reason']+'。当前空间/时间方法及正式全量保持，未写入新生产profile。','',
 '## 实测缓存增量成本','',
 '| 域 | 候选库＋原utility计算ms/query | 证据特征ms/query | 输入IO ms/query | 全部对照trial ms/query |','|---|---:|---:|---:|---:|']
 for c,g in r['costs'].items():lines.append('| '+c+' | '+' | '.join(f"{1000*g[k]['mean']:.3f}"for k in ['pool_seconds','feature_seconds','input_IO_seconds','seconds'])+' |')
 lines+=['','候选库计算含12个原view MBR及已有Global6 utility；这项在部署中也有代价，不能只报微秒argmax。单个实际DP单独保存在JSON，trial会复用重复区间的DP，因此不能把全部对照的摊销数称为某一方法独立端到端速度。输入IO含缓存哈希和读取完整geometry；未新增重网络。','',
 '## 封存和核验','',
 f"{r['proof']['official_dense_checks']}项官方/dense评分最大差{r['proof']['official_dense_max_error']:.3g}。独立物理区间tIoU、门控不变、先验utility损失界、原Source/Baseline及候选特征跨两轮一致核验见verification.json，正式434源/plan和真实worker前进见formal_guard.json。8项纯合成语义测试通过。",'',
 '所有数据均历史曝光开发缓存，父隔离不能称未见测试确认。正负、无变化、严重损伤和全部七子集都在report.json/signed_queries.tsv，协议与匿名输入绑定见protocol.json。','',
 '复现入口从仓库根执行，原私有合法缓存需已存在，所有新输出拒绝覆盖：','', '```bash',
 f"CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B scripts/joint_boundary_readout/{'run'if version=='v1'else'v2_run'}.py prepare",
 f"CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B scripts/joint_boundary_readout/{'run'if version=='v1'else'v2_run'}.py execute",
 f"CUDA_VISIBLE_DEVICES='' .conda/tubedetr/bin/python -B scripts/joint_boundary_readout/{'score'if version=='v1'else'v2_score'}.py",'```']
 (pub/'report.md').write_text('\n'.join(lines)+'\n')
print('Rendered both immutable attempts')
