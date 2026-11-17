"""Full factorial task results, all failures, task/auxiliary distinction."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round127';B=R/'crossplatform/round127';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
ST=['ab_native','ab_full','fab_native','fab_full','wrong_fab_full','random']
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==2304;rows=[r for p in sorted(O.glob('*_ALL_ROWS_PRIVATE.json')) for r in rd(p)];agg=rd(O/'AGGREGATE_ROWS.json');post=rd(O/'POSTSTRATIFIED_AGGREGATE_ROWS.json');dec=rd(H/'DECISION.json');controls=['zero_F','direct_full_F','direct_native_bridge_F','ab_native_F','ab_full_F','fab_native_F','wrong_fab_full_F','random_F'];conc=[];cases=[];process=[];factors=[]
 for co in ['DEV','EXP']:
  rr=[r for r in rows if r['cohort']==co];a={r['arm']:r['mIoU_percent'] for r in agg if r['cohort']==co and r['group']=='all_six_equal'}
  factors.append({'cohort':co,'complete_minus_native_at_AB_pp':a['ab_full_F']-a['ab_native_F'],'complete_minus_native_at_FAB_pp':a['fab_full_F']-a['fab_native_F'],'FAB_minus_AB_at_native_pp':a['fab_native_F']-a['ab_native_F'],'FAB_minus_AB_at_complete_pp':a['fab_full_F']-a['ab_full_F'],'two_factor_difference_pp':a['fab_full_F']-a['fab_native_F']-a['ab_full_F']+a['ab_native_F']})
  for c in controls:
   d=np.array([r['arms']['fab_full_F']['IoU']-r['arms'][c]['IoU'] for r in rr]);conc.append({'cohort':co,'control':c,'n':len(rr),'gain_pp':float(d.mean()*100),'better':int((d>1e-12).sum()),'worse':int((d< -1e-12).sum()),'same':int((abs(d)<=1e-12).sum()),'query_changes':sum(r['arms']['fab_full_F']['query_PRIVATE']!=r['arms'][c]['query_PRIVATE'] for r in rr),'largest_positive_single_input_pp':float(max(0,d.max())*100/len(rr)),'largest_negative_single_input_pp':float(min(0,d.min())*100/len(rr))})
  for src,tgt in sorted({(r['source'],r['target']) for r in rr}):
   for kind,sign in [('good',1),('failure',-1)]:
    part=[r for r in rr if r['source']==src and r['target']==tgt and sign*(r['arms']['fab_full_F']['IoU']-r['arms']['zero_F']['IoU'])>1e-12]
    for r in sorted(part,key=lambda r:-abs(r['arms']['fab_full_F']['IoU']-r['arms']['zero_F']['IoU']))[:2]:cases.append({'cohort':co,'source':src,'target':tgt,'id_PRIVATE':r['id_PRIVATE'],'kind':kind,'delta_IoU':r['arms']['fab_full_F']['IoU']-r['arms']['zero_F']['IoU'],'source_IoU':r['arms']['original_source']['IoU'],'zero_IoU':r['arms']['zero_F']['IoU'],'direct_IoU':r['arms']['direct_full_F']['IoU'],'updated_IoU':r['arms']['fab_full_F']['IoU'],'B_active':r['B_active'],'C_available':r['C_available'],'identity':'UNKNOWN_without_sufficient_visual_evidence'})
 for r in rows:
  if r['protected_alias_fallback']:continue
  D=B/'formal'/r['cohort']/(r['source']+'_to_'+r['target']);rec=rd(D/'records_PRIVATE'/(r['id_PRIVATE']+'.json'));st=np.load(D/'states_PRIVATE'/(r['id_PRIVATE']+'.npz'));row={k:r[k] for k in ['cohort','source','target','id_PRIVATE','B_active','C_available','seconds']}
  for k in ST[:-1]:
   tr=rec['traces'][k];row[k+'__objective_before']=tr[0]['before'];row[k+'__objective_after']=tr[-1]['after'];row[k+'__KL_after']=tr[-1]['KL_after'];row[k+'__keep_after']=tr[-1]['keep_after'];row[k+'__accepted_steps']=sum(t['accepted'] for t in tr);row[k+'__parameter_L2']=rec['writebacks'][1+ST.index(k)]['parameter_L2'];row[k+'__first_gradient_L2']=tr[0]['gradient_L2']
  g=st['fab_full_step1_gradient'].astype(float)
  for k in ['fab_native','ab_full','ab_native','wrong_fab_full']:
   v=st[k+'_step1_gradient'].astype(float);den=np.linalg.norm(g)*np.linalg.norm(v);row['first_gradient_cosine_fab_full_vs_'+k]=float(g@v/den) if den else None
  process.append(row)
 summary=[]
 for co in ['DEV','EXP']:
  active=[r for r in process if r['cohort']==co and r['B_active']]
  for k in ST[:-1]:summary.append({'cohort':co,'arm':k,'active':len(active),'objective_decreased':sum(r[k+'__objective_after']<r[k+'__objective_before'] for r in active),'accepted_steps':sum(r[k+'__accepted_steps'] for r in active),'mean_parameter_L2':float(np.mean([r[k+'__parameter_L2'] for r in active])),'mean_objective_change':float(np.mean([r[k+'__objective_after']-r[k+'__objective_before'] for r in active]))})
 for n,v in [('CONCENTRATION',conc),('GOOD_FAILURE_CASES_PRIVATE',cases),('PROCESS_ROWS_PRIVATE',process),('COMPONENT_SUMMARY',summary),('FACTORIAL_EFFECTS',factors)]:dump(H/(n+'.json'),v)
 lines=['# R127 — 拟合／部署一致的完整读出 TTA：固定四臂因子对照','','## 实际执行范围','','保留三份epoch100/source-prepared R91模型、原A256框和全部decoder位置。与R124/R125相同，native foundation的pointAdapter Up和early RGB gateP从零开始，训练过的Down固定；只更新Up权重/偏置4896参数，其他网络完全冻结。本轮不新增源域训练或专家。','','本轮只有一项有限实验：分别拟合A/B或拟合注册聚合FAB，交叉原生或完整读出监督。主项预先固定fab_full_F，三步共享预算；不按GT挑一步、平台、读出或个案。两列同时改变对应冻结教师与学生读出，不能称为只改一个KL实现位置。','','## 适应与最终计算路径','','FAB在支持有效处等权平均A/B点特征，不支持处保留A。主要输出始终是FAB，不含C。所有完整读出都采用同一份既有source-A RGB late与source-A竞争残差，数值停止梯度；但RGB向量相加后的归一化、完整文字匹配、竞争残差后的softmax进入真实更新目标。没有使用旧ABC或含C的上下文生成教师／梯度。','','原生教师为冻结原生A/B候选概率均值，完整教师为冻结完整A/B概率均值。每项最小化对应教师到学生的KL，加单位权重归一化点特征保持。分别拟合A/B时对两观测平均，拟合FAB时只用FAB。仅三步归一化SGD，半径上限为原source整个点Adapter范数的1%，固定9级二分Armijo，按该臂无标签目标接受或回退。R126只验收已有原生梯度；本轮确实从完整计算图重新生成梯度。','','这轮修正优化／部署输入和读出的一致性，仍使用同模型自共识；没有认证候选身份，也没有实现支持条件化监督或独立RGB专家。冻结竞争残差只提供分数偏置，经softmax影响更新；没有声称梯度通过其内部重新绑定计算。','','## 对照与数据边界','','四个实际更新臂：ab_native、ab_full、fab_native、fab_full。错配教师在256候选上作无固定点循环置换，保留概率集合与熵；随机参数方向匹配主项累计范数。零更新完整FAB、冻结完整AB直接概率平均、原生AB共识加零步完整读出差分均使用相同A/B信息。全部新臂固定后才解码C；C仅作描述性迁移检查，不参与梯度、回溯、配置选择或准入门。','','DEV768与EXP1536均历史曝光，共六方向2304输入、154保护别名；别名不加载特征，统一沿用prepared_A回退，其他无B证据样本零更新，全部仍在分母。B有效资格沿用R125；不按目标标签筛选。C是注册到A锚点的另一份观测描述子，使用A候选与A评分，不代表新场景GT。FINAL未访问。','','历史legacy_ABC_context额外使用C，仅作旧结果定位，不作为本轮相同信息胜负门。不得拿过去ABC+rF的绝对值与本轮FAB+rA直接归因；四臂共享新零步FAB+rA。','','## 总体结果','','| 范围 | 方法 | mIoU% | Acc25% | Acc50% |','|---|---|---:|---:|---:|']
 for x in agg:
  if x['group']=='all_six_equal':lines.append(f"| {x['cohort']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 lines+=['','## 六方向结果','','| 范围 | source→target | 方法 | mIoU% | Acc25% | Acc50% |','|---|---|---|---:|---:|---:|']
 for x in rd(O/'ALL_METHOD_ROWS.json'):
  if x['arm'] in ['original_source','zero_F','direct_full_F','ab_native_F','ab_full_F','fab_native_F','fab_full_F','wrong_fab_full_F','random_F']:lines.append(f"| {x['cohort']} | {x['source']}→{x['target']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 lines+=['','## 目标平台与场景频数控制','','| 范围 | 目标 | 方法 | mIoU% |','|---|---|---|---:|']
 for x in agg:
  if x['group']!='all_six_equal' and x['arm'] in ['zero_F','direct_full_F','fab_full_F']:lines.append(f"| {x['cohort']} | {x['group']} | {x['arm']} | {x['mIoU_percent']:.6f} |")
 lines+=['','| 范围 | 方法 | 场景支持覆盖下限 | 频数后分层mIoU% |','|---|---|---:|---:|']
 for x in post:
  if x['group']=='all_six_equal' and x['arm'].endswith('_F'):lines.append(f"| {x['cohort']} | {x['arm']} | {x['min_covered_full_mass']:.6f} | {x['mIoU_percent']:.6f} |")
 for n in ['FACTORIAL_EFFECTS','DECISION','HELDOUT_C_TASK_ROWS','CONCENTRATION','COMPONENT_SUMMARY','ENGINEERING','CPU_PROBE','NATIVE_AUDIT','SCALAR_AUDIT','GEOMETRY_AUDIT']:
  lines+=['',f'## {n}','','```json',json.dumps(rd(H/(n+'.json')),indent=2),'```']
 lines+=['','## 裁决','','主版本'+('通过' if dec['development_gate_pass'] else '未通过')+'预先锁定的同信息开发门。'+('仅保留已暴露范围候选，不能称为独立验证或生产部署。' if dec['development_gate_pass'] else '不采纳这份固定完整共识配方，不因其他臂较好而事后替换主方法。'),'','输入与读出对齐若有效，仍不等于教师知道目标身份；辅助损失下降也不等于任务改善。C单独报告，不用于挑参数。主指标保留全部输入、方向、拒绝及负例，场景后分层只用旧全量清单频数，EXP缺少4个Waymo场景仅共同支持。','','独立CPU审计复算教师、全部head读出、端点保持／KL、参数步进、provider及旧教师h/U桥接；工程样本用中心有限差分检查主完整梯度、实际权重写入与等价点注入。未声称全量CPU重做decoder反向传播。所有预测封存及无标签审计后才评分，独立几何IoU与标量审计另列。','','seconds是多臂共享缓存的研究开销，不能包装为单方法端到端延迟。私有图像、原句、数据身份、坐标、query索引、特征、权重及梯度不公开。']
 (H/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dec,indent=2),flush=True)
if __name__=='__main__':main()
