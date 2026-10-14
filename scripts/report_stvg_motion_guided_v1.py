"""Export all sealed anonymous measurements and a finite diagnostic report; no inference."""
import json,hashlib,statistics,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'artifacts/stvg_motion_guided_instance_opd_v1';O=ROOT/'results/stvg_motion_guided_instance_opd/2026-10-14'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
ALIASES={f'{ds}_{i:03d}':f'{prefix}{i+1:02d}'for ds,prefix in [('hc2','HC'),('vidstg','V')]for i in range(8)}
DROP={'GT_source_bindings','source_case_bindings','seals','candidate_input_binding','file_bindings','GT_span','frame_ids','interval_frame_bounds_half_open','physical_frame_id','position','actual_valid_supervision_anchor_physical_frame_ids','actual_valid_supervision_anchor_positions','anchor_physical_frame_ids','anchor_positions','original4_observation_physical_frame_ids','original4_observation_positions','prior_report_capture','report_semantics_correction'}
def anon(v):
 if isinstance(v,dict):return {ALIASES.get(k,k):anon(x)for k,x in v.items()if k not in DROP and not k.endswith('_path') and k!='path'}
 if isinstance(v,list):return [anon(x)for x in v]
 if isinstance(v,str):
  for k,a in ALIASES.items():v=v.replace(k,a)
  assert '/home/'not in v and '/attachments/'not in v,v
 return v
ARMS=[('F','Frozen'),('Projection','Projection'),('T','Temporal QA (reused)'),('teacher_direct','Spatial Direct'),('teacher_iou','Spatial IoU-OPD'),('teacher_diou','Spatial DIoU-OPD'),('T_teacher_direct','QA + Direct'),('T_teacher_iou','QA + IoU-OPD'),('T_teacher_diou','QA + DIoU-OPD')]
def main():
 z=read(B/'ROOT_POSTSEAL_REPORT.json');t=read(B/'ROOT_TEACHER_FINAL_READBACK003.json');ind=read(B/'ROOT_INDEPENDENT_METRIC_READBACK.json')
 assert ind['status']=='pass'and ind['report_sha256']==sha(B/'ROOT_POSTSEAL_REPORT.json')
 a=anon(z);a['teacher']=anon(t);save(O/'all_anonymous_results.json',a)
 assert len(a['metric_rows'])==118 and len(a['candidate_present_anchor_rows'])==83 and len(a['reference_control_locked_anchor_rows'])==3
 # Compare every retained scalar and its structure directly to its sealed original; omit only declared private bindings.
 assert read(O/'all_anonymous_results.json')==dict(anon(z),teacher=anon(t))
 oracle=[]
 for cid in ALIASES:
  rr=[r for r in z['candidate_present_anchor_rows']if r['case_id']==cid];tc=next(r for r in t['cases']if r['case_id']==cid)
  key='common_denominator_original4_GT_known_missing_as_zero_candidate_IoU';known=[r for r in rr if r[key]is not None]
  shown=[r for r in known if r['candidate_order_locked']<5]
  chosen=next((r for r in rr if r['candidate_id']==tc['base_selected_candidate_id']),None)
  oracle.append(dict(case_id=ALIASES[cid],cohort=rr[0]['cohort'],dataset=rr[0]['dataset'],scope='postseal anchor support diagnostic only; no Oracle fit, GT never used for candidate generation or selection',GT_known_anchor_count=rr[0]['GT_known_original4_anchor_count'],pool_count=len(rr),displayed_count=min(5,len(rr)),Source_IoU=rr[0][key],selected_candidate_id=tc['base_selected_candidate_id'],selected_IoU=None if chosen is None else chosen[key],pool_anchor_Oracle_IoU=max((r[key]for r in known),default=None),displayed_anchor_Oracle_IoU=max((r[key]for r in shown),default=None),missing_zero_only_for_known_GT_common_denominator=True))
 save(O/'candidate_support_oracle_diagnostic.json',dict(rows=oracle,status='pass',GT_read=True,new_fits=0))
 roots=dict(status='pass',scope='finite 16 historical-source motion/ID/spatial diagnosis; not stable method qualification',teacher_actual_calls=128,teacher_effective_calls=96,actual_fits=21,declared_rounds=105,actual_optimizer_steps=90,actual_full_final_model_readouts=21,method_no_update_cases=10,new_temporal_fits=0,new_Oracle_fits=0,metric_readouts=118,candidate_rows=83,V03_fixed_reference_rows=3,all_positive_negative_neutral_unknown_preserved=True,GT_used_for_method_update=False,independent_metric_max_abs_error=z['official_independent_metric_max_abs_error'],CURRENT_sha256=sha(ROOT/'methods/CURRENT_METHOD.json'))
 assert roots['CURRENT_sha256']=='146a2caf1633544a844ea33106c83c022c0bebad5f3f49a87bb5f81e6f8568d4'
 save(B/'ROOT_READBACK.json',roots)
 scalar=dict(status='pass',scope='complete retained anonymous scientific scalars equal sealed source results; private coordinates and provenance paths excluded',metric_rows=118,candidate_rows=83,fixed_reference_rows=3,teacher_actual_rows=128,teacher_effective_rows=96,fit_rows=21,no_fit_placeholders=30,all_positive_negative_neutral_unknown_preserved=True,private_field_exclusions=sorted(DROP),source_report_sha256=sha(B/'ROOT_POSTSEAL_REPORT.json'),source_teacher_sha256=sha(B/'ROOT_TEACHER_FINAL_READBACK003.json'),anonymous_sha256=sha(O/'all_anonymous_results.json'))
 for n in ['ROOT_ANONYMOUS_SCALAR_READBACK','PUBLIC_SCALAR_AUDIT']:save(B/(n+'.json'),scalar)
 source=read(B/'source_audit/LUNA_ACTUAL_INPUT_READBACK.json')
 save(O/'source_custody_summary.json',dict(status='pass',scope='fresh independent input decode/render/source readback; same pinned LK helper replay, not a second tracking algorithm',cases=16,fresh_RGB_frames=64,fresh_tracking_grayscale_frames=622,original_marked_video_jobs=96,corrected_marked_video_jobs=32,nonSource_trajectories=67,trajectory_positions=2541,Missing_positions=2167,transition_attempts=421,transition_accepted=307,transition_rejected=114,source_readback_sha256=sha(B/'source_audit/LUNA_ACTUAL_INPUT_READBACK.json'),GT_read=False,private_paths_and_pixels_excluded=True))
 decision=dict(status='finite_diagnosis_complete_mixed_results_no_automatic_expansion',scope='LK partial support and sparse R1 instance selection with fixed spatial updates; no stable bidomain gain established',CURRENT_promoted=False,QA_anchor_ablation_executed=False,additional8_new_unseen_confirmation=False,nonSource_full_four_anchor_count=3,nonSource_anchor_only_count=48,teacher_selected_reference_is_student_candidate=True)
 save(B/'TECHNICAL_DECISION.json',decision)
 for phase in ['method','reference_control']:
  m=read(B/phase/'ROOT_CPU_FIT_MATH_READBACK.json');save(O/(phase+'_math_summary.json'),{k:v for k,v in m.items()if isinstance(v,(bool,int,float,str))and not isinstance(v,str)or isinstance(v,str)and '/'not in v})
 for n in ['ROOT_READBACK','ROOT_ANONYMOUS_SCALAR_READBACK','PUBLIC_SCALAR_AUDIT','ROOT_INDEPENDENT_METRIC_READBACK','ROOT_PRIVATE_VISUAL_REVIEW','TECHNICAL_DECISION']:save(O/(n.lower()+'.json'),read(B/(n+'.json')))
 groups=z['method_groups_old8_vs_additional8']
 def val(cohort,ds,arm):return next((r['metrics_percent']['m_vIoU']for r in groups if r['cohort']==cohort and r['dataset']==ds and r['arm']==arm),None)
 lines=['# Motion-Guided Instance OPD：固定16例真实诊断','','## 结果与范围','','P0/P1/P2及V03三参考对照实际完成。Temporal保持原样；本轮检验局部LK候选、Object-ID置换控制与预锁Source-ID No-update。16个来源均来自历史池，旧8与新增8分开；新增8不是未见确认集，也没有旧QA输出，故不伪造其T/T+S结果。未扩大到32例，未更改CURRENT。','','## 完整Tube主结果','','单位m_vIoU (%)，每格4个来源。新增空间输出均真实full-model重新解码；旧8 T精确复用锁定QA。Direct是本轮参考来源/优化器诊断，不设论文准入门槛。','','|方法|旧8 HC2|旧8 VidSTG|新增8 HC2|新增8 VidSTG|','|---|---:|---:|---:|---:|']
 for arm,label in ARMS:
  values=[val(c,d,arm)for c in ['old8','additional8']for d in ['hc2','vidstg']];lines.append('|'+label+'|'+'|'.join('未执行'if x is None else f'{x:.6f}'for x in values)+'|')
 lines+=['','旧8 QA+IoU相对QA：HC2 -2.980399pp，VidSTG +1.396574pp。新增8 IoU-OPD相对F两域均0平均增量。正负案例均保留；不能据此宣布整个MLLM路线不可行，也没有稳定双域空间收益。','','## P0：局部视觉对应与Missing','','旧8保留原43seed（含8Source），新增8用预锁最远中心粗网格4非Source+Source，共40。两批seed规则不同，不能把旧新差异仅归因LK。LK中心位移来自候选局部特征，宽高仍沿用Source；原四观察位置保持，QA区间选锚点未执行。长边640、每4物理帧加精确观察帧，80角点、FB≤1.5px、至少4点且保留率≥.5，失踪方向不重生。未引入检测器或学习跟踪器。','','|候选支持（含Source）|旧8|新增8|','|---|---:|---:|','|全部候选|43|40|','|至少2个观察位置|18|17|','|完整4个观察位置|11|8|','|仅锚点|25|23|','|Missing位置/全部观察位置|88/172|86/160|','','67个非Source候选只有3个完整四位置、19个至少两位置、48个仅锚点。旧恒定平移43中18四帧合法，新LK完整只有11；允许Missing保留了局部证据，未提高完整轨迹支持。LK不是实例身份保证，固定Source宽高仍可能不适合新对象。','','候选质量逐位置区分：候选Missing、GT unknown、实际有效几何。报告可见且GT已知位置均值及同位置Source；另有原四位置GT已知的共同分母，候选Missing填0仅作覆盖诊断，GT unknown不填0。所有83候选和16例池内/展示池Oracle均公开；Oracle只封存后诊断，不用于补候选、选择或训练。','','## P1：稳定标记与Query控制','','128次实际R1请求，其中32次是display-order排版控制修正；有效矩阵96次=16×6。128/128格式合法，0截断。原order控制因绘制顺序改变badge位置，不能纯归因顺序；旧32原件保留，固定badge后仅重跑order/joint32。主base与预锁参考完全未变。','','|有效控制|同一几何选择/16|','|---|---:|','|只换数字ID|12|','|只换颜色|11|','|只换显示顺序，badge位置固定|14|','|联合置换|11|','','Wrong-query 16/16输出改变，其中14弃答、2改选其他非零候选，不能写成16/16改选几何。控制是提示敏感性证据，不单独证明目标身份正确。视频编码按平均帧间距近似均匀时钟，文本包含实际采样时刻；不是原生不均匀时序位置编码。','','主选择10Source→精确No-update，6非Source→Direct/IoU/DIoU。6个新参考支持分别HC04=4、HC08=2、V02/V04/V06/V07各1个位置；一帧候选不能称完整tracklet。V03本轮选择Source，保持No-update。','','## P2：实际更新与V03参考对照','','方法18实际fit、90声明轮、75实际optimizer步；3个IoU fit因均匀奖励执行0优化步但仍完成5轮检查与真实full-readout。10个Source例不调用fit。另V03固定原Source、旧R1 ID2/window4、旧ID3/window5，3IoU fit/15步。因此合21fit/105声明轮/90实际优化步/21最终完整模型读出。原1792 Query/LN、每域学习率/sigma/tau/writeback、种子与5轮保持。Direct沿原4位置分母，OPD沿有效位置均值，Missing时有效尺度差异保留披露，没有偷改loss。','','|V03固定参考|完整vIoU (%)|相对Frozen (pp)|','|---|---:|---:|']
 vg=z['V03_reference_control_groups'];f=next(r['metrics_percent']['m_vIoU']for r in vg if r['arm']=='F')
 for arm in ['F','Source_iou','oldR1ID2_iou','oldAlternativeID3_iou']:
  v=next(r['metrics_percent']['m_vIoU']for r in vg if r['arm']==arm);lines.append(f'|{arm}|{v:.6f}|{v-f:+.6f}|')
 lines+=['','V03旧ID2正增益重复出现，而Source自蒸馏和固定旧ID3下降，排除了这两个具体对照也能获得相同增益的解释；不是穷尽全部扰动，也不能证明R1已可靠识别实例。四监督位置只有1个GT已知：Source/旧ID2/旧ID3的参考IoU分别.445717/.158103/.053349，其他3处unknown。参考锚点IoU与完整Tube适应效果必须分开。','','全部118指标、21实际fit与30No-update占位、监督/未监督帧变化、参数/输出位移、每例正负neutralunknown均公开。额外8没有时间训练，本轮无新GT-Oracle fit，GT只在所有预测封存后评分。','','## 工程修正与真实读回','','学生首版误要求每fit恰好5个优化步，在7个完整case后拒绝合法零步IoU；保存旧源、接受结果和部分第8例，新002只修分派并精确续接，不改变奖励/超参。不是OOM或数值资格失败。CPU候选评分初版将单帧Missing误扩成整轨迹unknown，修正为逐位置；V03控制参考改为读取锁定旧参考而非本轮motion场景候选，原件保留。教师读回补齐128job唯一覆盖断言、响应解析及全部绑定核验。','','独立读者从原视频新解码64RGB观察帧、622灰度跟踪帧并验证hash，重建128输入；67非Source轨迹沿同pinned LK helper复现，不声称第二独立跟踪算法。奖励/权重/likelihood/Adam、Source恢复、真实full读出通过CPU数学检查。根独立118指标和83×4候选位置/3控制参考复核，官方/独立指标最大差1.2212453270876722e-15。原base/joint8contact128slots、修正版控制8contact128slots、最终16contact256slots、V03控制1contact12slots均实际根像素查看。','','## 成本和结论边界','',f"教师实际128请求：input {t['actual_cost']['input_tokens']}、visual {t['actual_cost']['visual_tokens']}、output {t['actual_cost']['output_tokens']} tokens；请求端到端合计{t['actual_cost']['end_to_end_request_seconds']:.6f}s，generate合计{t['actual_cost']['model_generate_seconds']:.6f}s，峰allocated {t['actual_cost']['GPU_peak_bytes']}bytes。含32修正请求，不含模型冷加载、资产hash与输入准备，不能当完整部署延迟。",'', '固定16历史来源、每组4个，未做显著性/总体确认；教师训练重叠未知。No-update避免了10例Source自蒸馏，但不能保证非Source选择有益。主要未解决问题是多帧候选支持与实例选择，当前结果不支持直接扩大。QA锚点、在线carry、调参、新teacher和旧队列未启动；CURRENT保持。所有私人RGB、Query、GT坐标、模型/优化器state、权重不公开。']
 lines+=['','## 六个非Source案例的实际空间增量','','单位为相对同例F的完整vIoU百分点，不加入时间QA。','','|案例|有效参考位置|Direct|IoU-OPD|DIoU-OPD|','|---|---:|---:|---:|---:|']
 for c in t['cases']:
  if c['no_update']:continue
  r=next(r for r in z['case_rows']if r['case_id']==c['case_id']and r['phase']=='method');f=r['arms']['F']['m_vIoU'];vals=[(r['arms']['teacher_'+a]['m_vIoU']-f)*100 for a in ['direct','iou','diou']];lines.append('|'+ALIASES[c['case_id']]+'|'+str(c['reference_count'])+'|'+'|'.join(f'{v:+.6f}'for v in vals)+'|')
 lines+=['','旧Vid均值由V04正增益与V02负增益共同决定；HC04大幅下降。新增HC08/V07的零完整指标变化不代表参考身份正确。17次Source冷捕获、21fit全部保存计时，以及包含失败/修正成本的阶段launch-to-exit时间见 actual_cost_all_paths.json；这些嵌套和可能重叠的时间不可简单相加当部署延迟。']
 (ROOT/'docs/STVG_MOTION_GUIDED_INSTANCE_OPD_V1.md').write_text('\n'.join(lines)+'\n')
 build_pdf(z,t,val)
 print('anonymous report and PDF exported; actual page view still required')
def build_pdf(z,t,val):
 from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak
 from reportlab.lib.styles import getSampleStyleSheet
 from reportlab.lib import colors
 from reportlab.lib.pagesizes import A4
 from reportlab.pdfbase import pdfmetrics
 from reportlab.pdfbase.ttfonts import TTFont
 pdfmetrics.registerFont(TTFont('ReportSans','/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'))
 pdfmetrics.registerFont(TTFont('ReportBold','/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'))
 styles=getSampleStyleSheet()
 for sty in styles.byName.values():sty.fontName='ReportBold' if sty.name in ('Title','Heading1','Heading2','Heading3') else 'ReportSans'
 styles['BodyText'].fontSize=9.4;styles['BodyText'].leading=14
 story=[]
 def p(txt,style='BodyText'):story.append(Paragraph(txt,styles[style]));story.append(Spacer(1,8))
 def tab(rows,widths=None):
  table=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT');table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#18354a')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,-1),'ReportSans'),('FONTNAME',(0,0),(-1,0),'ReportBold'),('FONTSIZE',(0,0),(-1,-1),8.5),('LEADING',(0,0),(-1,-1),12),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#eff4f6'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP')]));story.append(table);story.append(Spacer(1,12))
 p('Motion-Guided Instance OPD','Title');p('Finite 16-source diagnostic | 14 October 2026','Heading2')
 p('Local LK preserves partial evidence but does not resolve complete tracklet support. Spatial gains remain domain-dependent. Temporal QA and all source checkpoints stay fixed.')
 p('Complete-tube m_vIoU (%)','Heading2')
 rows=[['Method','Old HC2','Old Vid','Add HC2','Add Vid']]
 for arm,label in ARMS:
  vals=[val(c,d,arm)for c in ['old8','additional8']for d in ['hc2','vidstg']];rows.append([label]+['N/A'if v is None else f'{v:.3f}'for v in vals])
 tab(rows,[175,70,70,70,70])
 p('Four sources per cell. The additional eight come from the historical pool, not an unseen confirmation cohort. Their Temporal QA outputs do not exist; N/A is not zero. QA + spatial combines a locked old temporal interval with the new actual spatial decode.')
 p('Old QA + IoU versus QA: HC2 -2.980 pp; VidSTG +1.397 pp. Additional-eight IoU gives zero average gain in both domains. All negative, neutral and unknown observations are retained.')
 p('Execution: 128 teacher calls (96 effective controls), 21 actual fits, 105 declared rounds, 90 actual optimizer steps and 21 final full-model decodes. Ten Source selections reuse F exactly; no new temporal or GT-Oracle fits.')
 story.append(PageBreak());p('Candidate support and identity controls','Title')
 tab([['Candidate support (includes Source)','Old 8','Additional 8'],['Total candidates','43','40'],['At least 2 observation positions','18','17'],['All 4 positions','11','8'],['Anchor only','25','23'],['Missing / all observation slots','88 / 172','86 / 160']],[305,75,75])
 p('Across 67 non-Source candidates: 3 have all four positions; 19 have at least two; 48 retain only the anchor. The old constant-shift construction had 18 complete candidates among the same 43 seeds, versus 11 here. Partial retention is not improved complete-object coverage.')
 p('Old seeds reuse the original fixed union; new seeds use a fixed farthest-center grid. Candidate width and height still follow Source. Local flow is a proposal mechanism, not an identity guarantee. Missing geometry and unknown GT are separate. Present-position means and a known-GT common-denominator coverage diagnostic are both published.')
 tab([['Effective factorial control','Same geometry / 16'],['ID permutation','12'],['Color permutation','11'],['Order permutation (fixed badge positions)','14'],['Joint permutation','11'],['Wrong query: abstain / different nonzero','14 / 2']],[330,125])
 p('All 128 actual responses are valid and untruncated. Thirty-two original order/joint calls also changed badge position; they are retained as superseded controls and replaced by 32 fixed-layout calls. Base selections and locked student references are unchanged.')
 p('The six non-Source selections have 4, 2, 1, 1, 1 and 1 valid observation positions. One-anchor selection is not a complete tracklet. Control consistency does not establish semantic identity correctness.')
 story.append(PageBreak());p('V03 reference isolation and verification','Title')
 vg=z['V03_reference_control_groups'];f=next(r['metrics_percent']['m_vIoU']for r in vg if r['arm']=='F');rows=[['Fixed old reference','vIoU (%)','Delta vs F (pp)']]
 for arm in ['F','Source_iou','oldR1ID2_iou','oldAlternativeID3_iou']:
  v=next(r['metrics_percent']['m_vIoU']for r in vg if r['arm']==arm);rows.append([arm,f'{v:.6f}',f'{v-f:+.6f}'])
 tab(rows,[260,95,100])
 p('Same initialization, optimizer, key and five actual steps. Old R1 ID2 repeats its local positive effect; Source and the prelocked old ID3 decrease. This distinguishes these specific references, not every possible perturbation. The current motion-based method chooses Source for V03 and does not update it.')
 p('Only one of four reference anchors has GT: Source / old ID2 / old ID3 IoU = 0.445717 / 0.158103 / 0.053349. The other three remain unknown. Sparse reference quality and final complete-tube quality must be evaluated separately.')
 p('Independent readbacks','Heading2')
 p('Fresh decode: 64 RGB observation frames and 622 grayscale tracking frames; all 128 rendered inputs and 67 LK trajectories replayed. LK replay uses the same pinned helper. CPU fit math verifies reward weights, likelihood, Adam, source restoration and cached/full readouts. All 118 metric rows, 332 candidate positions and three old fixed references independently match; maximum metric error 1.22e-15.')
 p('Private pixel review covers original base/joint inputs, corrected order/joint controls, all final updates and V03 references. Private RGB, Query text, GT coordinates, weights and optimizer state are excluded from public output. Anonymous per-case results preserve all positive, negative, neutral and unknown values.')
 p(f"Actual teacher cost: {t['actual_cost']['input_tokens']:,} input tokens, {t['actual_cost']['visual_tokens']:,} visual tokens, {t['actual_cost']['output_tokens']:,} output tokens; {t['actual_cost']['end_to_end_request_seconds']:.3f} s summed request time. Includes repairs; excludes cold load, asset verification and preparation. This is not end-to-end deployment latency.")
 p('Finite diagnosis complete. Stable bidomain spatial benefit is not established. No automatic expansion, new temporal training, online carry, hyperparameter search or CURRENT promotion. QA-interval anchor sampling remains a separate unexecuted ablation.')
 path=ROOT/'output/pdf/stvg_motion_guided_instance_opd_v1.pdf';path.parent.mkdir(parents=True,exist_ok=True)
 def footer(c,d):c.setFont('ReportSans',8);c.setFillColor(colors.HexColor('#526572'));c.drawString(42,25,'Sealed finite diagnostic | Anonymous public evidence');c.drawRightString(A4[0]-42,25,str(d.page))
 SimpleDocTemplate(str(path),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=35,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
if __name__=='__main__':main()
