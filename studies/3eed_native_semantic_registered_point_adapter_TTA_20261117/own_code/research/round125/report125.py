"""Task-first interpretation of native-semantic registered-view adaptation."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round125';B=R/'crossplatform/round125';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==2304;rr=[r for f in sorted(O.glob('*_ALL_ROWS_PRIVATE.json')) for r in rd(f)];agg=rd(O/'AGGREGATE_ROWS.json');post=rd(O/'POSTSTRATIFIED_AGGREGATE_ROWS.json');decision=rd(H/'DECISION.json');controls=['mean_context','headmean_context','robust_context','direct_F','free3_F','wrong3_F','random3_F'];conc=[];cases=[];process=[];components=[]
 for co in ['DEV','EXP']:
  rows=[r for r in rr if r['cohort']==co]
  for ctrl in controls:
   d=np.array([r['arms']['corr3_F']['IoU']-r['arms'][ctrl]['IoU'] for r in rows]);conc.append({'cohort':co,'method':'corr3_F','control':ctrl,'n':len(rows),'gain_pp':float(d.mean()*100),'better':int((d>1e-12).sum()),'worse':int((d< -1e-12).sum()),'same':int((abs(d)<=1e-12).sum()),'query_changes':sum(r['arms']['corr3_F']['query_PRIVATE']!=r['arms'][ctrl]['query_PRIVATE'] for r in rows),'largest_positive_single_input_pp':float(max(0,d.max())*100/len(rows)),'largest_negative_single_input_pp':float(min(0,d.min())*100/len(rows))})
   for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
    for kind,sign in [('good',1),('failure',-1)]:
     part=[r for r in rows if r['source']==src and r['target']==tgt and sign*(r['arms']['corr3_F']['IoU']-r['arms'][ctrl]['IoU'])>1e-12]
     for r in sorted(part,key=lambda r:-abs(r['arms']['corr3_F']['IoU']-r['arms'][ctrl]['IoU']))[:2]:cases.append({'cohort':co,'source':src,'target':tgt,'id_PRIVATE':r['id_PRIVATE'],'kind':kind,'method':'corr3_F','control':ctrl,'delta_IoU':r['arms']['corr3_F']['IoU']-r['arms'][ctrl]['IoU'],'original_source_IoU':r['arms']['original_source']['IoU'],'zero_IoU':r['arms']['mean_context']['IoU'],'updated_IoU':r['arms']['corr3_F']['IoU'],'B_active':r['B_active'],'C_available':r['C_available'],'identity':'UNKNOWN_without_private_visual_review'})
 for r in rr:
  if r['protected_alias_fallback']:continue
  D=B/'formal'/r['cohort']/(r['source']+'_to_'+r['target']);rec=rd(D/'records_PRIVATE'/(r['id_PRIVATE']+'.json'));st=np.load(D/'states_PRIVATE'/(r['id_PRIVATE']+'.npz'));a=st['corr1_gradient'].astype(float);b=st['wrong1_gradient'].astype(float);den=np.linalg.norm(a)*np.linalg.norm(b);row={'cohort':r['cohort'],'source':r['source'],'target':r['target'],'id_PRIVATE':r['id_PRIVATE'],'active':rec['fit']['active'],'pairs':rec['fit']['fit_point_pairs'],'correct_wrong_first_gradient_cosine':float(a@b/den) if den else None,'all_arm_shared_seconds':r['seconds']}
  for setting,metrics in rec['auxiliary'].items():
   for key,v in metrics.items():row[setting+'__'+key]=v
  for kind in ['corr','wrong','free']:
   for i,step in enumerate(rec['traces'][kind]):
    row[kind+str(i+1)+'__trial_count']=len(step['trials'])
    for key,v in step.items():
     if key!='trials':row[kind+str(i+1)+'__'+key]=v
  for item in rec['writebacks']:row[item['setting']+'__actual_parameter_L2']=item['actual_parameter_L2'];row[item['setting']+'__field_L2']=item['field_L2']
  process.append(row)
 for co in ['DEV','EXP']:
  for src in ['waymo','drone','quad']:
   part=[r for r in process if r['cohort']==co and r['source']==src];active=[r for r in part if r['active']];x={'cohort':co,'source':src,'n_nonalias':len(part),'active_AB':len(active),'accepted_first_steps':sum(r['corr1__accepted'] for r in active),'accepted_steps_total':sum(r['corr'+str(t)+'__accepted'] for r in active for t in [1,2,3]),'mean_first_accepted_fraction':float(np.mean([r['corr1__fraction'] for r in active])) if active else None,'mean_actual_parameter_L2_at3':float(np.mean([r['corr3__actual_parameter_L2'] for r in active])) if active else None}
   for setting in ['corr1','corr3','wrong3','random3','free3']:
    for qty in ['fit_objective','fit_KL','heldout_C_KL']:
     key=setting+'__'+qty;zero='zero__'+qty;pp=[r for r in active if key in r and zero in r];x[setting+'_'+qty+'_n']=len(pp)
     if pp:x[setting+'_'+qty+'_mean_change']=float(np.mean([r[key]-r[zero] for r in pp]));x[setting+'_'+qty+'_decreases']=sum(r[key]<r[zero] for r in pp)
   components.append(x)
 dump(H/'CONCENTRATION.json',conc);dump(H/'GOOD_FAILURE_CASES_PRIVATE.json',cases);dump(H/'PROCESS_ROWS_PRIVATE.json',process);dump(H/'SEMANTIC_COMPONENT_SUMMARY.json',components)
 lines=['# R125 — 注册观测的任务原生语义一致性pointAdapter TTA','','R123的点描述子对比改善辅助任务却损害选框；R124的源域GT受控诊断显示相同pointUp4896/三步预算可提高源域选框4.2226个百分点，但不是无标签TTA。本轮不使用这些标签、梯度或oracle权重，固定接口，只检验真实注册观测上的原生grounding语义一致性。','','## 本轮实际改变','','教师由同一冻结3EED在注册A/B完整文字上的候选分布等权平均，停止梯度。学生重算A/B原生cross-encoder与全部decoder，最小化教师到两份学生分布的平均KL，加单位权重的归一化source点特征保持。没有硬正标签、参照物规则、颜色规则、外部专家或新source准备。','','仍从已有native foundation开始：pointAdapter输出权重/偏置与early RGB门置零，source-trained down固定；仅写输出288×16权重及288偏置4896参数。最多三步，每步归一化梯度半径上限沿用原source整个pointAdapter范数的1%，固定9级二分Armijo，仅按无标签目标接受或回退。一步是次项，不按GT选最佳端点。','','A/B使用真实配准扫描，固定原A256空间锚点与六层位置，所以不把随机采样间变化的query编号硬作同物体；但近似注册仍不认证实例身份。资格与R123相同，无充分B支持时跳过更新，零梯度记录表示程序跳过，不宣称物理导数为零。未知与所有负例计入总体。','','## 与旧实验及强对照的区别','','R57是在同一观测的采样间做近似候选传输，并更新完整匹配MLP；R75在遮挡上下文的query/token空间做恢复，也只更新完整匹配MLP。本轮并不把KL或回溯说成新原理，区别是实际注册证据、固定候选位置及进入语言推理前的真实pointAdapter更新。旧方法失败不保证新设置成功，必须直接检验。','','固定RGB late为sourceA、竞争残差为sourceMeanF，与所有新臂相同。适应本身仍是LiDAR—文字原生分布一致性；保留RGB上下文不等于本轮学到了三模态共享语义。','','除冻结点均值、head均值、R122中位数等对照外，新增完全同信息的直接语义读出：log教师概率 + 零步context logits − 零步native logits，再softmax，无网络更新。它是冻结AB语义共识加相同RGB/竞争修正，系数固定一，不另扫分数融合。','','错配对照置换教师的候选位置，保留概率集合与熵；随机对照与实际正确更新的累计参数范数相同。更强无网络更新对照直接优化A/B原始点特征变量，使用相同KL、保持与Armijo，步能量上限匹配真实Adapter各步诱发的A/B特征改变量，最后将平均修正场作用于F/C。它需要反向传播到自由变量，但不修改网络；不是零计算，也不是全局最优直接解。','','全部提案、自由特征与随机端点先确定，随后才解码C及C相关context provider。C不参与梯度或验收；F和最终RGB上下文本来允许读取C，故不是因果在线协议或新场景确认。额外报告未参与拟合C的KL与真实选框，两者不互相替代。C仅更换已注册点特征，任务仍选择A的固定256框并沿用A标注；它不是独立新帧GT或新的场景确认，动态目标的实例对应仍可能未知。','','## 数据与完整性','','DEV768/EXP1536均历史曝光，所有2304方向输入、154保护别名在分母。别名先检查、不加载特征，沿用prepared_A回退。三source100/R91均不重训，原A256坐标/全文视角/所有decoder位置不变，FINAL不动。场景后分层只用原全量场景频数；EXP缺少4个Waymo场景，仅共同支持。headmean参照不是历史R59软query传输，原DEV336 R59研究结果保留原study，不伪造跨清单同条件值。','','首次未完成运行的private序列化让同名梯度覆盖参数端点，预测计算在内存中正确但审计端点不可靠。发现时尚未读取任务GT，已中止并保留全部partial/engineering产物与原代码/锁/日志。分离_gradient键后重新完成24工程与6CPU schema探针，再正式重做；没有根据任务成绩改配置。失败尝试保留，不能删除或计作方法负结果。','','## 总体结果','','| 范围 | 方法 | mIoU% | Acc25% | Acc50% |','|---|---|---:|---:|---:|']
 for x in agg:
  if x['group']=='all_six_equal':lines.append(f"| {x['cohort']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 lines+=['','## 各目标平台','','| 范围 | 目标 | 方法 | mIoU% | Acc25% | Acc50% |','|---|---|---|---:|---:|---:|']
 for x in agg:
  if x['group']!='all_six_equal' and x['arm'] in ['original_source','prepared_A','mean_context','headmean_context','direct_F','corr1_F','corr3_F','free3_F','wrong3_F','random3_F','zero_C','corr3_C']:lines.append(f"| {x['cohort']} | {x['group']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 lines+=['','## 场景频数控制','','| 范围 | 方法 | 最小支持覆盖 | 后分层mIoU% |','|---|---|---:|---:|']
 for x in post:
  if x['group']=='all_six_equal':lines.append(f"| {x['cohort']} | {x['arm']} | {x['min_covered_full_mass']:.6f} | {x['mIoU_percent']:.6f} |")
 for n in ['OBSERVATION_INFORMATION_SUMMARY','DECISION','HELDOUT_C_TASK_ROWS','SEMANTIC_COMPONENT_SUMMARY','CONCENTRATION','NATIVE_AUDIT','SCALAR_AUDIT','GEOMETRY_AUDIT','CPU_PROBE']:
  lines+=['',f'## {n}','','```json',json.dumps(rd(H/(n+'.json')),indent=2),'```']
 lines+=['','## 本轮裁决','',('预设开发门通过；仅支持当前已暴露范围的有限候选，不把它改写为独立验证或已稳定部署。' if decision['development_gate_pass'] else '预设开发门失败，停止这份固定注册语义KL短步pointAdapter配方。不扩大步数、温度、层位置或事后选择平台/一步作为主项。'),'','判断以全部方向和同信息对照的选框净收益为主。辅助KL降低与真实写入参数不算任务成功；GT源域能力诊断也不能替无标签信号背书。全部好坏案例、拒绝、无B/C和保护回退均保留，未看私有图像认证的身份保持UNKNOWN。','','梯度验证来自工程中心有限差分，以及全量CPU步进/自由特征场/teacher/head/端点目标复算，不声称全量CPU重做decoder反向传播。27臂共62208任务值，36864新读出、25344复用既有对照；全部新预测封存后才读取任务GT，并独立检查框IoU、场景及门槛。','','seconds是多臂共享缓存的研究开销，不等于单方法端到端延迟。原始媒体、原句、ID、坐标、query索引、特征、权重及梯度不公开。']
 (H/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(decision,indent=2),flush=True)
if __name__=='__main__':main()
