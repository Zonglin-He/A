"""Postsealed overall, unused-observation, auxiliary and all-tail interpretation."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round123';B=R/'crossplatform/round123';O=H/'scores_attempt001'
rd=lambda p:json.loads(Path(p).read_text())
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==2304
 rr=[r for f in sorted(O.glob('*_ALL_ROWS_PRIVATE.json')) for r in rd(f)];agg=rd(O/'AGGREGATE_ROWS.json');post=rd(O/'POSTSTRATIFIED_AGGREGATE_ROWS.json');decision=rd(H/'DECISION.json');concentration=[];cases=[];components=[];process=[]
 controls=['mean_context','headmean_context','robust_context','free3_F','wrong3_F','random3_F']
 for co in ['DEV','EXP']:
  rows=[r for r in rr if r['cohort']==co]
  for c in controls:
   d=np.array([r['arms']['corr3_F']['IoU']-r['arms'][c]['IoU'] for r in rows]);concentration.append({'cohort':co,'method':'corr3_F','control':c,'n':len(rows),'gain_pp':float(d.mean()*100),'better':int((d>1e-12).sum()),'worse':int((d< -1e-12).sum()),'same':int((abs(d)<=1e-12).sum()),'query_changes':sum(r['arms']['corr3_F']['query_PRIVATE']!=r['arms'][c]['query_PRIVATE'] for r in rows),'largest_positive_single_input_pp':float(max(0,d.max())*100/len(rows)),'largest_negative_single_input_pp':float(min(0,d.min())*100/len(rows))})
   for src,tgt in sorted({(r['source'],r['target']) for r in rows}):
    for kind,sign in [('good',1),('failure',-1)]:
     part=[r for r in rows if r['source']==src and r['target']==tgt and sign*(r['arms']['corr3_F']['IoU']-r['arms'][c]['IoU'])>1e-12]
     for r in sorted(part,key=lambda r:-abs(r['arms']['corr3_F']['IoU']-r['arms'][c]['IoU']))[:2]:
      cases.append({'cohort':co,'source':src,'target':tgt,'id_PRIVATE':r['id_PRIVATE'],'kind':kind,'method':'corr3_F','control':c,'delta_IoU':r['arms']['corr3_F']['IoU']-r['arms'][c]['IoU'],'original_source_IoU':r['arms']['original_source']['IoU'],'zero_IoU':r['arms']['mean_context']['IoU'],'updated_IoU':r['arms']['corr3_F']['IoU'],'B_active':r['B_active'],'C_available':r['C_available'],'identity':'UNKNOWN_without_private_visual_review'})
 for r in rr:
  if r['protected_alias_fallback']:continue
  D=B/'formal'/r['cohort']/(r['source']+'_to_'+r['target']);rec=rd(D/'records_PRIVATE'/(r['id_PRIVATE']+'.json'));states=np.load(D/'states_PRIVATE'/(r['id_PRIVATE']+'.npz'));a=states['corr1'].astype(float);b=states['wrong1'].astype(float);den=np.linalg.norm(a)*np.linalg.norm(b);row={'cohort':r['cohort'],'source':r['source'],'target':r['target'],'id_PRIVATE':r['id_PRIVATE'],'active':rec['fit']['active'],'pairs':rec['fit']['fit_point_pairs'],'directed_negatives':rec['fit']['directed_geometric_negatives'],'correct_wrong_first_step_cosine':float(a@b/den) if den else None,'all_arm_shared_seconds':r['seconds']}
  for k,v in rec['auxiliary'].items():
   for kk,vv in v.items():row[k+'__'+kk]=vv
  for k,v in rec['writebacks'][0].items():
   if k!='setting':row['zero__'+k]=v
  for kind in ['corr','wrong']:
   for i,t in enumerate(rec['traces'][kind]):
    for k,v in t.items():row[kind+str(i+1)+'__'+k]=v
  process.append(row)
 for co in ['DEV','EXP']:
  for src in ['waymo','drone','quad']:
   pp=[r for r in process if r['cohort']==co and r['source']==src];ac=[r for r in pp if r['active']];item={'cohort':co,'source':src,'n_nonalias':len(pp),'active_AB':len(ac),'mean_pairs':float(np.mean([r['pairs'] for r in pp])),'mean_actual_correct_wrong_first_step_cosine':float(np.mean([r['correct_wrong_first_step_cosine'] for r in ac])) if ac else None}
   for setting in ['corr1','corr3','wrong3','random3','free3']:
    for quantity in ['fit_objective','heldout_corr','heldout_retrieval']:
     key=setting+'__'+quantity;zero='zero__'+quantity;available=[r for r in ac if key in r and zero in r];item[setting+'_'+quantity+'_n']=len(available)
     if available:item[setting+'_'+quantity+'_mean_change']=float(np.mean([r[key]-r[zero] for r in available]));item[setting+'_'+quantity+'_improved']=sum(r[key]<r[zero] if quantity!='heldout_retrieval' else r[key]>r[zero] for r in available)
   components.append(item)
 dump(H/'CONCENTRATION.json',concentration);dump(H/'GOOD_FAILURE_CASES_PRIVATE.json',cases);dump(H/'CORRESPONDENCE_COMPONENT_SUMMARY.json',components);dump(H/'PROCESS_ROWS_PRIVATE.json',process)
 lines=['# R123 — 注册点对应驱动的真实点Adapter TTA','','本轮是实际参数适应，不是只改最终分数。仍固定三个epoch100、现有R91双向结构、完整文字、原A256与六层空间锚点；没有新source准备、外部专家、PointNet推理、原始媒体或FINAL访问。采用与否按预先锁定的全部总体对照判定。','','## 单一机制与计算路径','','R122显示打乱空间特征可显著损害选框，但精确几何中位数未提供稳定收益。本轮不拟合失败的中位数G，也不再更换GN恢复损失。直接使用真实注册扫描A/B的pre-language点特征，适应现有point_adapter.up的4896个参数，再重算原生cross-encoder/decoder的候选和文字表示。','','从已验证的native foundation开始：原source准备的pointAdapter输出权重、偏置及早期RGB门置零，固定原source down投影；不是给旧source Adapter继续叠加不明偏移。只编辑输出288×16权重及288偏置。每条输入恢复原source；三个固定归一化梯度步骤，每步半径为原source完整pointAdapter范数的1%。没有按GT/辅助loss做逐样本步长搜索。','','A/B有效支持由旧注册确定，位置相距大于3.2米才作负区域，近邻与重复支持不互斥。正对应是近似注册空间支持，不是经认证的同一物体。对称InfoNCE温度0.07加单位权重的归一化source特征保持；不生成文字目标伪标签，不使用颜色/左右规则。','','原R74是在同帧采样的postdecoder候选区域和matching MLP做对比；本轮改变实际扫描、监督单位和进入语言推理前的计算路径。PointContrast论文与官方实现是借鉴依据，但原方法长程预训练后有监督微调，不能借其结果保证当前短步TTA有效，见LITERATURE.md。','','## 同信息强对照与C边界','','全部参数/自由特征更新先固定，随后才解码C数组与旧C上下文提供者。C不参与梯度或停止规则。最终F读出允许像所有基线一样使用已有ABC观测，因此不是严格在线或全新场景验证。C区域检索和C选框作为未用于拟合的观测检查。','','对照包含：等权点特征mean、原始headmean、R122稳健聚合；同loss同预算的错误注册；与实际参数改变量等范数的随机参数；同A/B loss直接优化自由特征并逐步匹配实际Adapter引起的特征变化能量。最后一项不改网络，通过A/B平均修正场作用于F/C，是强但非全局最优的无更新对照。错误注册同时破坏正负绑定，不能狭称只测试正样本身份。','','固定RGB late取sourceA，竞争残差取sourceMeanF，均与零步及全部新臂相同。保留source双向RGB上下文，但新更新仅针对LiDAR descriptor，不能宣称本轮已经完成联合RGB参数适应。','','## 数据、工程与封存','','DEV768与EXP1536均历史曝光；154保护别名先检查且不加载特征，使用原prepared_A回退并保留分母。各方向等权平均；另用旧全量场景频数后分层，不按GT加权。EXP缺少4个Waymo场景，只报告共同支持范围。当前headmean不是历史R59的软query传输，历史DEV336 R59仍保留原研究条目，不能在不同清单伪造同条件数值。','','24条真实工程验证零步F/C候选和文字表示与旧缓存逐值相同。实际Up写回与等价点特征输入的h/U逐值相同是正常的数学等价，仅证明实现执行，不算额外收益。25臂×2304条共57600任务值，32256新预测、25344复用基线；所有预测先封存，再独立复算CPU梯度、自由特征、更新范数、head、框IoU及总体/场景/未拟合C。','','## 总体结果','','| 数据 | 方法 | mIoU% | Acc25% | Acc50% |','|---|---|---:|---:|---:|']
 for x in agg:
  if x['group']=='all_six_equal':lines.append(f"| {x['cohort']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 lines+=['','## 各目标平台','','| 数据 | 目标 | 方法 | mIoU% | Acc25% | Acc50% |','|---|---|---|---:|---:|---:|']
 for x in agg:
  if x['group']!='all_six_equal' and x['arm'] in ['original_source','prepared_A','mean_context','headmean_context','corr1_F','corr3_F','wrong3_F','random3_F','free3_F','zero_C','corr3_C']:lines.append(f"| {x['cohort']} | {x['group']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 lines+=['','## 场景频数与未拟合C','','| 数据 | 方法 | 最小场景支持覆盖 | 后分层mIoU% |','|---|---|---:|---:|']
 for x in post:
  if x['group']=='all_six_equal':lines.append(f"| {x['cohort']} | {x['arm']} | {x['min_covered_full_mass']:.6f} | {x['mIoU_percent']:.6f} |")
 for name in ['HELDOUT_C_TASK_ROWS','CORRESPONDENCE_COMPONENT_SUMMARY','CONCENTRATION','DECISION','NATIVE_AUDIT','SCALAR_AUDIT','GEOMETRY_AUDIT']:
  lines+=['',f'## {name}','','```json',json.dumps(rd(H/(name+'.json')),indent=2),'```']
 passed=decision['development_gate_pass'];lines+=['','## 执行裁决','',('预设开发门通过；这仍是历史曝光数据上的有限证据，不是独立确认。' if passed else '预设开发门未通过，停止这一固定注册点InfoNCE短步Adapter配方。不将一步、某个平台或某个场景的正次项事后升级主方法，不按本轮结果继续扫温度、半径、步数或层位置。'),'','区域对应检索改善、辅助loss下降和实际参数非零都不足以替代选框收益。只有正确对应在总体/场景频数控制下超过同信息直接使用、错误/随机，并在未拟合C仍有收益，才能支持这一参数适应机制。所有good/failure均指相对IoU变化，未访问私有图像认证的实例身份保留UNKNOWN。全量匿名逐例和场景负例公开，不删除回退、缺B/C或没有负区域的样本。','','本轮每条seconds是多臂共享缓存和多次后缀研究开销，不能当单方法全新端到端延迟。原始观测、描述、坐标、query索引、特征、权重和梯度不公开。']
 (H/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(decision,indent=2),flush=True)
if __name__=='__main__':main()
