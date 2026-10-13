"""Anonymous six-arm task report; separate Oracle diagnosis and four-page figures."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import collections,csv,math,time
import numpy as np
from scripts.stvg_opd_admission_association_common_v1 import *
from scripts.score_stvg_opd_admission_association_v1 import rows_read,estimate,mean,CONTRASTS,ALL_ARMS


def fmt(e,percent=True):
 if e['mean'] is None:return 'unknown (n=0)'
 scale=100 if percent else 1;lo,hi=e['ci95']
 return f'{scale*e["mean"]:+.4f} [{scale*lo:+.4f}, {scale*hi:+.4f}]; n={e["defined_parents"]}'


def round_summary(rd,ee):
 """Mean observations/rounds within parent first; parent is the statistical unit."""
 evidence={(r['parent_id'],r['arm'],r['sample_ordinal']):r for r in ee};out={}
 for ds in ['hc2','vidstg']:
  out[ds]={}
  for a in ALL_ARMS:
   q=[r for r in rd if r['dataset']==ds and r['arm']==a];parents=sorted({r['parent_id'] for r in q});values=[]
   for p in parents:
    rr=[r for r in q if r['parent_id']==p];v=dict(parent_id=p)
    for name in ['max_reward','reward_variance','Top1_same_samples_weight_L1','mean_gradient_norm','per_frame_informative']:
     v[name]=mean([x for r in rr for x in r[name]])
    for name in ['observed_GT_delta','common_U_GT_delta','actual_U_output_movement','actual_training_output_movement','actual_parameter_displacement']:
     v[name]=mean([r[name] for r in rr if r[name] is not None])
    changed=[];improved=[]
    for r in rr:
     for j,position in enumerate(r['selected_observation_ordinals']):
      if not r['changed_Top1_observations'][j]:continue
      changed.append(r['Top1_same_samples_weight_L1'][j]);e=evidence[p,a,position]
      if e['GT_known'] and e['quality_delta_vs_Top1']>EPS:improved.append(r['Top1_same_samples_weight_L1'][j])
    v['changed_candidate_weight_L1']=mean(changed);v['GT_improved_candidate_weight_L1']=mean(improved);values.append(v)
   fields=[k for k in values[0] if k!='parent_id']
   stats={k:dict(estimate=estimate([v[k] for v in values]),unknown_parents=sum(v[k] is None for v in values)) for k in fields}
   changed_count=improved_count=improved_zero=improved_zero_reward=0
   for r in q:
    for j,pos in enumerate(r['selected_observation_ordinals']):
     if not r['changed_Top1_observations'][j]:continue
     changed_count+=1;e=evidence[r['parent_id'],a,pos]
     if e['GT_known'] and e['quality_delta_vs_Top1']>EPS:
      improved_count+=1;improved_zero+=r['Top1_same_samples_weight_L1'][j]<=EPS
      improved_zero_reward+=r['max_reward'][j]==0
   stats['counts']=dict(rounds=len(q),updated_rounds=sum(r['updated'] for r in q),empty_evidence_rounds=sum(r['empty_evidence'] for r in q),
    all_zero_reward_rounds=sum(bool(r['max_reward']) and all(x==0 for x in r['max_reward']) for r in q),
    all_uninformative_rounds=sum(bool(r['per_frame_informative']) and not any(r['per_frame_informative']) for r in q),
    changed_candidate_observation_rounds=changed_count,GT_improved_candidate_observation_rounds=improved_count,
    GT_improved_but_weight_L1_zero_observation_rounds=improved_zero,
    GT_improved_but_max_reward_zero_observation_rounds=improved_zero_reward)
   stats['parent_values']=values;out[ds][a]=stats
 return out


def source_concentration(rows,coeff,field):
 v={str(r['parent_id']):math.fsum(c*r[a+'_'+field] for a,c in coeff.items()) for r in rows if all(r[a+'_'+field] is not None for a in coeff)}
 x=np.array(list(v.values()),float);positive=np.maximum(x,0);negative=np.maximum(-x,0)
 loo=[float(np.delete(x,k).mean()) for k in range(len(x))] if len(x)>1 else []
 return dict(parent_values=v,defined_parents=len(v),positive_parents=int((x>EPS).sum()),negative_parents=int((x< -EPS).sum()),
  maximum_positive_parent_share=float(positive.max()/positive.sum()) if positive.sum()>0 else None,
  top_two_negative_parent_share=float(np.sort(negative)[-2:].sum()/negative.sum()) if negative.sum()>0 else None,
  leave_one_parent_out_mean_range=[min(loo),max(loo)] if loo else None)


def report():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.backends.backend_pdf import PdfPages
 verify();root=read(BASE/'ROOT_READBACK.json');assert root['status']=='pass'
 s=read(OUT/'SUMMARY.json');rr=rows_read(OUT/'ROWS.jsonl.gz');rd=rows_read(OUT/'ROUND_ROWS.jsonl.gz');ee=rows_read(OUT/'EVIDENCE_ROWS.jsonl.gz')
 assert len(rr)==64 and len(rd)==11200 and s['formal_arms']==ARMS
 diagnostics=round_summary(rd,ee);write(OUT/'FEEDBACK_SUMMARY.json',diagnostics)
 concentration={ds:{name:source_concentration([r for r in rr if r['dataset']==ds],coef,'I1_dense_v_after') for name,coef in CONTRASTS.items()} for ds in ['hc2','vidstg']}
 write(OUT/'SOURCE_CONCENTRATION.json',concentration)
 cases=[]
 for ds in ['hc2','vidstg']:
  q=[r for r in rr if r['dataset']==ds]
  for name,coef in CONTRASTS.items():
   for kind,chooser in [('largest_gain',max),('largest_harm',min)]:
    r=chooser(q,key=lambda r:math.fsum(c*r[a+'_I1_dense_v_after'] for a,c in coef.items()))
    cases.append(dict(dataset=ds,contrast=name,kind=kind,parent_id=r['parent_id'],
     contrast_value=math.fsum(c*r[a+'_I1_dense_v_after'] for a,c in coef.items()),row=r,
     evidence=[e for e in ee if e['parent_id']==r['parent_id'] and e['arm'] in coef],
     selection='deterministic posthoc extreme from complete fixed roster; descriptive, never an online rule'))
  for a in ['R_Top1','R_Path']:
   for kind,chooser in [('Oracle_largest_gain',max),('Oracle_largest_harm',min)]:
    r=chooser(q,key=lambda r:r['Oracle_I1_dense_v_after']-r[a+'_I1_dense_v_after'])
    cases.append(dict(dataset=ds,contrast='Oracle_minus_'+a,GT_assisted=True,kind=kind,parent_id=r['parent_id'],
     contrast_value=r['Oracle_I1_dense_v_after']-r[a+'_I1_dense_v_after'],row=r,
     selection='posthoc diagnostic only; not a student-performance upper bound'))
 write(OUT/'CASES.json',cases)
 interpretation={}
 for ds in ['hc2','vidstg']:
  x=s['datasets'][ds]['I1_dense_v_after']['contrasts']
  interpretation[ds]={k:('positive_CI' if v['estimate']['ci95'][0]>0 else 'negative_CI' if v['estimate']['ci95'][1]<0 else 'interval_crosses_zero') for k,v in x.items()}
 write(OUT/'READOUT_INTERPRETATION.json',dict(status='scoped_complete_readout',datasets=interpretation,all_contrasts_required_positive=False,
  historical_panel_not_new_confirmation=True,identity_audit_performed=False,geometric_quality_is_not_identity=True,
  Oracle_GT_assisted=True,Oracle_is_student_upper_bound=False,new_online_stream=False,
  production_promoted=False,automatic_expansion=False,monitor_recreated=False,time=time.time()))
 p=OUT/'ROWS.csv';assert not p.exists();columns=list(rr[0]);assert all(set(r)==set(columns) for r in rr)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rr)
 formal=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');oracle=read(BASE/'ORACLE_PREDICTION_BARRIER.json')
 lines=['# Admission × cross-frame association: complete six-arm matched experiment','',
  '六个无标签正式臂已在固定64历史query/Inherited-order1进入状态完成，全部384逻辑fit/9600轮封存后才构造候选池内Oracle；Oracle64逻辑fit/1600轮单独封存后统一后验任务评分。正式结果与GT-assisted诊断分别呈现。完成这轮比较不要求两个数据集五项对照全部显著，也不设代理覆盖率或零严重损害门槛。','',
  '每目标32历史父视频、每源一query；该名单曾用于开发诊断，不能称新确认集。原TA-STVG、单个Grounding DINO、完整query上下文/token occurrence、NMS/top3、最多四个原访问位置、原Box OPD、1792个Query/LN参数和进入状态保持。Query Residual清零、Adam重新初始化，无跨query写回。HC lr .01/sigma .025/tau .05/K40，Vid lr .03/sigma .10/tau .25/K10，每轮32反对称动作；没有调beta/margin/学习率/温度/名单。','',
  'H逐值复用原严格准入；R仅恢复原 ambiguous_distinct_instances 拒绝且通过原最低分数/有效几何的帧。Path为原分数加相邻box IoU、beta1；Add保留全部行列平均诱导unary；Shuffle保留unary与交互数值/范数，但置换双中心交互的一端候选对应。它不打乱视频帧、物理监督位置或原框。singleton/纯加性边的交互退化被完整保留。','',
  '选中原候选通过固定监督入口真正进入OPD，每观察训练权重1；路径分数不乘入loss。H-Top1必须精确复现旧完整OPD支持/噪声/参数/loss/输出。原H物理帧保留旧噪声，新恢复帧按query/物理frame/round固定hash，同帧各臂噪声一致；状态分叉后的动作坐标可以不同。','',
  '## 完整任务：I1主读出与I0次读出','',
  '六臂同空间轨迹统一固定原P7-A I1，另用Native I0重评分；不重推WHEN。先在parent内部平均，再对32 parents作配对10000 bootstrap seed20261013。以下单位pp，探索性次指标区间未作多重比较校正。','',
  '| Target | Contrast | I1 mvIoU [95%CI] | I0 mvIoU [95%CI] | I1 正/负/中性/未知 | I1 >5/>20pp损害 |','|---|---|---:|---:|---:|---:|']
 for ds,v in s['datasets'].items():
  for k in CONTRASTS:
   a=v['I1_dense_v_after']['contrasts'][k];b=v['I0_dense_v_after']['contrasts'][k]
   lines.append(f'| {ds} | {k} | {fmt(a["estimate"])} | {fmt(b["estimate"])} | {a["positive"]}/{a["negative"]}/{a["neutral"]}/{a["unknown"]} | {a["harm5pp"]}/{a["harm20pp"]} |')
 lines+=['','system=R-Path−H-Top1；assoc_R=R-Path−R-Top1；interaction=(R-Path−R-Top1)−(H-Path−H-Top1)；pair=R-Path−R-Add；alignment=R-Path−R-Shuffle。交互为差中之差，其负值不是某个单臂相对旧方法的实际损害。跨零表示未建立方向，不能解释为等效。','',
  '| Target | Arm | I1最终mvIoU % | I1相对进入状态增量pp | 完整GT dense sIoU增量pp |','|---|---|---:|---:|---:|']
 for ds,v in s['datasets'].items():
  for a in ARMS:lines.append(f'| {ds} | {a} | {fmt(v["I1_dense_v_after"]["arms"][a]["estimate"])} | {fmt(v["I1_dense_v_delta"]["arms"][a]["estimate"])} | {fmt(v["dense_s_delta"]["arms"][a]["estimate"])} |')
 lines+=['','## 共同未访问decoder帧：保护与纠正','',
  'U严格为有真实GT标注的decoder事件帧，排除全部实际DINO访问位置（包括原来被拒绝的位置）。初始正确为共同进入框GT IoU>=.5，错误为<.5；保持/纠正只在同一真实网格定义。无合法帧分组保留unknown及分母，不能填0。插值位置单独评价，不混入U。','',
  '| Target | Arm | U空间IoU增量pp | 正确U保持% | 错误U纠正% | 真实decoder事件增量pp | 非网格插值事件增量pp |','|---|---|---:|---:|---:|---:|---:|']
 for ds,v in s['datasets'].items():
  for a in ARMS:lines.append('|'+ds+'|'+a+'|'+'|'.join(fmt(v[f]['arms'][a]['estimate']) for f in ['own_U_delta','correct_U_retained','wrong_U_corrected','sampled_event_delta','interpolated_event_delta'])+'|')
 lines+=['','I0/I1同一轨迹的sIoU相同，tIoU空间更新增量为0；完整官方dense保持无外推与HC字面端点。这里的真实事件网格/插值诊断用半开事件span且有annotation，不把HC最后一标注帧inclusive与时间评测端点混同。','',
  '## 监督质量：准入、恢复歧义与原Top1替换','',
  '| Target | Arm | 已知所选专家GT IoU% | 恢复margin位置GT IoU% | 已知/未知/事件外/缺标位置 | 原正确Top1换差/原错误Top1换好 |','|---|---|---:|---:|---:|---:|']
 for ds,v in s['datasets'].items():
  for a in ARMS:
   q=v['selection'][a]
   lines.append(f'| {ds} | {a} | {fmt(v["expert_GT_IoU"]["arms"][a]["estimate"])} | {fmt(v["restored_expert_GT_IoU"]["arms"][a]["estimate"])} | {q["known_expert_frames"]}/{q["unknown_expert_frames"]}/{q["outside_event_expert_frames"]}/{q["missing_annotation_expert_frames"]} | {q["correct_Top1_replaced_worse"]}/{q["wrong_Top1_replaced_better"]} |')
 lines+=['','这些是几何质量，不能自动称身份正确率。H与R的监督位置数不同，各自teacher agreement也没有统一跨臂质量尺度；共同GT/U与完整tube才是共同参照。原正确Top1“换差”指已知GT IoU>=.5且新候选IoU更低；错误“换好”指原<.5且更高，跨.5纠正/破坏另存在EVIDENCE_ROWS。空候选、其他拒绝原因、context和交互退化保存在pool_coverage/匿名parent行；未观察/事件外不伪造身份标签。','',
  '## 是否真正改变OPD反馈及学生','',
  '| Target | Arm | 改候选时同samples Top1权重L1 | 平均max reward | 平均reward variance | 更新/总rounds | GT更好但L1零的观察×round |','|---|---|---:|---:|---:|---:|---:|']
 for ds,v in diagnostics.items():
  for a in ARMS:
   q=v[a];n=q['counts'];lines.append(f'| {ds} | {a} | {fmt(q["changed_candidate_weight_L1"]["estimate"],False)} | {fmt(q["max_reward"]["estimate"],False)} | {fmt(q["reward_variance"]["estimate"],False)} | {n["updated_rounds"]}/{n["rounds"]} | {n["GT_improved_but_weight_L1_zero_observation_rounds"]}/{n["GT_improved_candidate_observation_rounds"]} |')
 lines+=['','这些反馈先按parent内观察/round平均再做parent统计；观察×round计数是诊断分母，不是独立样本。Top1权重对照使用该arm同一轮同一实际动作，不拼接另一arm的旧轨迹。完整32动作/reward/weight数学witness仅存私有fit并由根复算；公开只逐观察max/variance/L1/gradient norm/informative和实际移动。','',
  '全部动作reward相等时该帧控制变量梯度可精确为0；更好参考可能位于局部探索外。其他帧有信息或优化器状态可使整体参数变化，因此“这一帧无反馈”不等于“整个query不更新”。FEEDBACK_SUMMARY、ROUND_ROWS保留实际监督GT学习、合法有标注事件U上的学习（common_U_GT_delta）、全部未访问decoder网格上的输出移动（actual_U_output_movement；此处不限定GT事件U）、参数位移、零反馈以及更好监督未改变有效权重的计数；它们帮助区分选不对、学不进去、传播不改善，不能用agreement或低drift替代任务correctness。','',
  '## Oracle：候选池内GT-assisted诊断，独立于正式六臂','',
  '仅正式384fit全局封存后，在R的原top3池、合法事件半开span且有标注的位置按GT IoU首argmax选择；缺标/事件外维持R-Top1。不新增GT框，不改变预算/OPD，不把GT选择用于部署。与某正式臂完整固定监督相同可引用GT-free canonical fit；Oracle logical仍GT-assisted，原canonical GT_used=False原值保持。它不是学生性能理论上界。','',
  '| Target | Oracle I1最终mvIoU% | Oracle−RTop1 pp | Oracle−RPath pp | Oracle U增量pp |','|---|---:|---:|---:|---:|']
 for ds,v in s['datasets'].items():
  d=v['Oracle_diagnostic'];lines.append(f'| {ds} | {fmt(d["I1_dense_v_after"]["arm"]["estimate"])} | {fmt(d["I1_dense_v_after"]["contrasts"]["R_Top1"]["estimate"])} | {fmt(d["I1_dense_v_after"]["contrasts"]["R_Path"]["estimate"])} | {fmt(d["own_U_delta"]["arm"]["estimate"])} |')
 lines+=['','## 全部来源、成本与边界','',
  f'正式384逻辑fit/9600逻辑轮实际新增 {formal["actual_complete_GPU_fits"]} 个独特完整拟合、{formal["actual_complete_rounds"]} 轮；Oracle64/1600逻辑额外新增 {oracle["actual_complete_GPU_fits"]} 个拟合、{oracle["actual_complete_rounds"]} 轮。资格60逻辑fit/1320轮、10 capture单独计；不是把逻辑臂重复时间相加作为实际花费。正式/Oracle阶段elapsed分别{formal["elapsed_seconds"]:.3f}/{oracle["elapsed_seconds"]:.3f}s，包含阶段工程开销，instrumented fit只含适配记录，不是端到端部署速度。两阶段新DINO调用为0；原专家生成及旧capture成本不因此为0。','',
  'SOURCE_CONCENTRATION保存全部父视频值、最大正源份额、前两负源份额及leave-one-out范围；CASES保留五个对照各目标最大正例和最大负例及独立Oracle反例，不删除坏视频。所有zero-intervention query、singleton/加性退化、未知分组仍进入正式固定名单。','',
  'R-Path−RTop1回答同恢复支持关联作用；Path−Add回答配对交互超出平均unary是否有用；Path−Shuffle回答原候选对应是否重要。仅几何相邻IoU可稳定选错干扰物；尚未做盲身份审核，不能宣称same referent或身份切换改善。一个条件区间跨零不是普遍路线失败，也不是等效性证明。','',
  '旧Real-Forward已完成完整64任务负结果，未重开其固定选择器；旧身份MAP同时改路径、损失与支持的开发收益不能归给单一关联。这里是新的准入×关联匹配设计，保留旧负结果。没有新增第二专家/编码器/身份网络、时间teacher、优化器、路径回归loss、完整新在线流、参数扫描或生产晋升。巡检保持关闭。','',
  '所有主/Oracle预测先seal再后验GT，独立root复核source/共同state/候选pool/固定target/噪声/动作/likelihood/Adam/密集及网格指标/parent统计；完整模型decoder Jacobian未独立重放。当前注册方法和原standby optimizer custody不变。匿名公开仅parent/sample ordinal/候选索引与真实标量；不公开物理frame ID、query/caption、媒体、GT/expert框、feature、参数状态、权重或32动作数学witness。','']
 lead=['主要任务读数（单位pp，parent配对95%CI）：']
 for ds,v in s['datasets'].items():
  x=v['I1_dense_v_after']['contrasts'];o=v['Oracle_diagnostic']['I1_dense_v_after']['contrasts']['R_Top1']
  lead.append(f'- {ds}: R-Path−H-Top1 {fmt(x["system"]["estimate"])}；R-Path−R-Top1 {fmt(x["assoc_R"]["estimate"])}；候选池Oracle−R-Top1 {fmt(o["estimate"])}。')
 lead+=['','Path、Add、Shuffle对应不同机制问题；Oracle的GT辅助正向结果只说明该历史面板仍有可被当前OPD利用的候选选择空间，不说明无标签选择器或身份关联已经解决。完整负尾和无效反馈与均值一同保留。','']
 lines[2:2]=lead
 p=OUT/'REPORT.md';assert not p.exists();p.write_text('\n'.join(lines)+'\n')
 folder=OUT/'figures';folder.mkdir(exist_ok=True);pdfpath=folder/'ADMISSION_ASSOCIATION_VALIDATION.pdf';assert not pdfpath.exists()
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'font.family':'DejaVu Sans','pdf.fonttype':42})
 pages=[]
 def finish(fig,pdf,name):
  fig.tight_layout(rect=(0,.05,1,.94));pdf.savefig(fig);fig.savefig(folder/(name+'.png'),dpi=200);plt.close(fig);pages.append(name)
 with PdfPages(pdfpath) as pdf:
  for fields,title,name in [(['I1_dense_v_after','I0_dense_v_after'],'Six-arm complete task: primary I1 and secondary I0','task_contrasts'),
                           (['own_U_delta','correct_U_retained'],'Common actual unvisited decoder frames: learning and protection','unvisited_protection')]:
   fig,axes=plt.subplots(2,2,figsize=(12,8))
   for j,ds in enumerate(['hc2','vidstg']):
    for i,field in enumerate(fields):
     ax=axes[i,j]
     for x,k in enumerate(CONTRASTS):
      e=s['datasets'][ds][field]['contrasts'][k]['estimate']
      if e['mean'] is not None:
       mu=100*e['mean'];lo,hi=np.asarray(e['ci95'])*100;ax.errorbar(x,mu,yerr=[[max(mu-lo,0)],[max(hi-mu,0)]],fmt='o',capsize=4)
     ax.axhline(0,color='gray',lw=1);ax.set_xticks(range(5),list(CONTRASTS),rotation=20);ax.set(title=ds+' / '+field,ylabel='Paired parent contrast (pp,95% CI)')
   fig.suptitle(title,fontsize=15);fig.text(.5,.015,'32 historically exposed parents per target; 10000 paired bootstrap. Unknown masks retain denominators.',ha='center',fontsize=9);finish(fig,pdf,name)
  fig,axes=plt.subplots(2,2,figsize=(12,8))
  for j,ds in enumerate(['hc2','vidstg']):
   for i,field in enumerate(['expert_GT_IoU','wrong_U_corrected']):
    ax=axes[i,j]
    for x,a in enumerate(ALL_ARMS):
     e=s['datasets'][ds][field]['arms'][a]['estimate'] if a!='Oracle' else s['datasets'][ds]['Oracle_diagnostic'][field]['arm']['estimate']
     if e['mean'] is not None:
      mu=100*e['mean'];lo,hi=np.asarray(e['ci95'])*100;ax.errorbar(x,mu,yerr=[[max(mu-lo,0)],[max(hi-mu,0)]],fmt='s' if a=='Oracle' else 'o',capsize=3,color='#b9754d' if a=='Oracle' else '#467c88')
    ax.set_xticks(range(7),ALL_ARMS,rotation=30);ax.set(title=ds+' / '+field,ylabel='Parent mean (%)');ax.grid(axis='y',alpha=.15)
  fig.suptitle('Evidence quality and actual correction; Oracle is GT-assisted diagnosis',fontsize=15);fig.text(.5,.015,'Expert IoU is geometry, not identity. Different admitted supports have different coverage; Oracle is not a student upper bound.',ha='center',fontsize=9);finish(fig,pdf,'evidence_and_Oracle')
  fig,axes=plt.subplots(2,2,figsize=(12,8))
  for j,ds in enumerate(['hc2','vidstg']):
   q=sorted([r for r in rr if r['dataset']==ds],key=lambda r:r['parent_id']);values=[100*(r['R_Path_I1_dense_v_after']-r['H_Top1_I1_dense_v_after']) for r in q]
   ax=axes[0,j];ax.bar(range(32),values,color=['#3b8c74' if x>=0 else '#b85243' for x in values]);ax.axhline(0,color='gray',lw=1);ax.set(title=ds+' / every parent system contrast',xlabel='Anonymous parent ordinal',ylabel='R Path - H Top1 I1 mvIoU (pp)')
   ax=axes[1,j];v=s['datasets'][ds];ax.bar(ARMS,[v['compute'][a]['logical_fit_seconds_mean'] for a in ARMS],color='#618b9b');ax.set_xticks(range(6),ARMS,rotation=30);ax.set(title=ds+' / per logical-arm instrumented fit',ylabel='Mean fit seconds (reuse is duplicated here)')
   pc=v['unique_physical_cost'];ax.text(.03,.96,f'Unique formal fits: {pc["formal_payloads"]}; extra Oracle: {pc["additional_Oracle_payloads"]}',transform=ax.transAxes,va='top',fontsize=9)
  fig.suptitle('All parent tails and actual/canonical compute distinction',fontsize=15);fig.text(.5,.015,'No parent removed. Sum physical costs once per canonical payload; plotted logical means are not an end-to-end speed benchmark.',ha='center',fontsize=9);finish(fig,pdf,'all_parents_and_cost')
 write(BASE/'FIGURE_EXPORT.json',dict(status='pass',PDF_files={str(pdfpath.relative_to(ROOT)):dict(sha256=sha(pdfpath),bytes=pdfpath.stat().st_size,pages=4)},
  PNG_files={str((folder/(n+'.png')).relative_to(ROOT)):sha(folder/(n+'.png')) for n in pages},PDFs=1,pages=4,
  actual_pixel_view_required=True,pixel_view_completed=False,CPU_only=True,GT_read=True,time=time.time()))
 print('ADMISSION_ASSOCIATION_REPORT_EXPORTED_FOUR_PAGES')


if __name__=='__main__':report()
