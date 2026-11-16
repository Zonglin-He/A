"""Task/physical/context effects separated; no favorable-arm primary switch."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round120';B=R/'crossplatform/round120';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==2304;mat=rd(B/'MATRIX_SEALED.json');agg=rd(O/'AGGREGATE_ROWS.json');post=rd(O/'POSTSTRATIFIED_AGGREGATE_ROWS.json');tasks=[r for f in O.glob('*_ALL_ROWS_PRIVATE.json') for r in rd(f)];primary=mat['primary'];physical=[];context=[];conc=[];cases=[]
 for co in ['DEV','EXP']:
  tr=[r for r in tasks if r['cohort']==co];records=[]
  for r in tr:
   if r['protected_alias_fallback']:continue
   name=r['source']+'_to_'+r['target'];rec=rd(B/'formal'/co/name/'records_PRIVATE'/(r['id_PRIVATE']+'.json'));f=np.load(B/'formal'/co/name/'features_PRIVATE'/(r['id_PRIVATE']+'.npz'));grad=np.load(B/'formal'/co/name/'states_PRIVATE'/(r['id_PRIVATE']+'.npz'));g=grad['gradient'].astype(float);w=grad['wrong_gradient'].astype(float);records.append((r,rec,float(g@w/max(1e-30,np.linalg.norm(g)*np.linalg.norm(w)))))
   for st in ['depth_trust','depth_natural','wrong_depth_trust','random_GN_trust']:
    dr=f[st+'_SF_r'].astype(float)-f['zero_SF_r'].astype(float);dl=f[st+'_SA_late'].astype(float)-f['zero_SA_late'].astype(float);context.append({'cohort':co,'source':r['source'],'target':r['target'],'id_PRIVATE':r['id_PRIVATE'],'setting':st,'residual_change_L2':float(np.linalg.norm(dr)),'centered_residual_change_L2':float(np.linalg.norm(dr-dr.mean())),'late_change_L2':float(np.linalg.norm(dl)),'gradient_correct_wrong_cosine':records[-1][2]})
   f.close();grad.close()
  for st in ['depth_trust','depth_natural','wrong_depth_trust','random_GN_trust']:
   pp=[]
   for r,rec,cos in records:
    v=rec['evaluation']['physical_losses'];zero=v['zero'];new=v[st];pp.append((zero['fit_depth'],new['fit_depth'],zero['heldout_depth'],new['heldout_depth']))
   a=np.array(pp);physical.append({'cohort':co,'setting':st,'n':len(a),'zero_fit_mean':float(a[:,0].mean()),'updated_fit_mean':float(a[:,1].mean()),'zero_holdout_mean':float(a[:,2].mean()),'updated_holdout_mean':float(a[:,3].mean()),'holdout_improved':int((a[:,3]<a[:,2]-1e-12).sum()),'holdout_worsened':int((a[:,3]>a[:,2]+1e-12).sum())})
  for c in ['context_zero_F','context_zero_ABC','lateOnly_F','depth_trust_frozenR_F','depth_trust_shuffleR_F','wrong_depth_trust_live_F','random_GN_trust_live_F']:
   dd=np.array([r['arms'][primary]['IoU']-r['arms'][c]['IoU'] for r in tr]);conc.append({'cohort':co,'control':c,'n':len(tr),'gain_pp':float(dd.mean()*100),'better':int((dd>1e-12).sum()),'worse':int((dd< -1e-12).sum()),'same':int((abs(dd)<=1e-12).sum()),'largest_positive_single_input_pp':float(max(0,dd.max())*100/len(tr)),'largest_negative_single_input_pp':float(min(0,dd.min())*100/len(tr))})
   for src,tgt in sorted({(r['source'],r['target']) for r in tr}):
    for kind,sign in [('good',1),('failure',-1)]:
     tail=[r for r in tr if r['source']==src and r['target']==tgt and not r['protected_alias_fallback'] and sign*(r['arms'][primary]['IoU']-r['arms'][c]['IoU'])>1e-12]
     if tail:
      r=max(tail,key=lambda r:abs(r['arms'][primary]['IoU']-r['arms'][c]['IoU']));cases.append({'cohort':co,'source':src,'target':tgt,'id_PRIVATE':r['id_PRIVATE'],'kind':kind,'method':primary,'control':c,'delta_IoU':r['arms'][primary]['IoU']-r['arms'][c]['IoU'],'source_IoU':r['arms']['original_source']['IoU'],'zero_context_IoU':r['arms']['context_zero_F']['IoU'],'updated_IoU':r['arms'][primary]['IoU'],'identity':'UNKNOWN_without_independent_visual_review','postseal_only':True})
 summary=[]
 for co in ['DEV','EXP']:
  for st in ['depth_trust','depth_natural','wrong_depth_trust','random_GN_trust']:
   r=[x for x in context if x['cohort']==co and x['setting']==st];summary.append({'cohort':co,'setting':st,'n':len(r),**{k:float(np.mean([x[k] for x in r])) for k in ['residual_change_L2','centered_residual_change_L2','late_change_L2','gradient_correct_wrong_cosine']}})
 for n,x in [('PHYSICAL_SUMMARY',physical),('CONTEXT_EFFECT_SUMMARY',summary),('CONTEXT_EFFECT_ROWS_PRIVATE',context),('CONCENTRATION_SUMMARY',conc),('GOOD_FAILURE_CASES_PRIVATE',cases)]: (H/(n+'.json')).write_text(json.dumps(x,indent=2)+'\n')
 dec=rd(H/'DECISION.json');lines=['# R120 — RGB物理更新通过实时语义上下文的受控TTA','','本轮保持三个source100、已有源域R91双向模块、原A256/六层decoder位置/query身份不变。没有专家、新source训练或新增辅助损失。只更新既有RGB编码器三处GroupNorm的6个仿射张量，共224参数；原始点—文字grounding表示严格固定。','','## 与R116和R118的区别','','R116已经测试同一深度监督，但硬旁路earlyRGB后，其竞争残差r不随RGB更新变化，只有late能传递修改。本轮让更新后的RGB通过原source-compatible RGB→point→文字→竞争接口重算r，原生基础h/U仍冻结。对同一GN参数状态，比较实时r、冻结r、冻结late以及query打乱r。它检验一个实际计算路径，而非再搜索更新层或放大步长。','','冻结基座使用R118的sourceA late/sourceF残差次项。它是在已经曝光的开发结果中被选出，不是R118已采用主项，也不是独立确认；R118同步A主项仍判失败，没有把其输出当伪标签拟合。现有GN更新直接来自实测稀疏深度。','','主项一次原始梯度归一到source GN范数的1%，保留此前自然步、错配训练深度同范数、随机GN同范数控制；不回溯、不搜索步长。DEV复用R116已保存RAW方向，工程重算正确/错配梯度位等；EXP只做一次当前A梯度。75%有效深度格拟合、25%物理格留出，76001/2互补遮挡原样。所有方向固定后才读B/C和留出误差。深度头输入仅RGB，不能复制输入深度。','','sourceA当前同步观测提供late，sourceF融合点上下文提供竞争残差；nativeA/B/C/F基础独立固定。最终F/C读出会使用融合上下文中的C，因此C没有参与梯度，但不能称不含C上下文的信息隔离验证，更不是新RGB视角。完整部署可能需要同一3EED的两个共享参数后缀，不声称缓存重放速度是端到端速度。','','## 数据与技术核验','','DEV768和EXP1536均已历史曝光，154保护别名在读取特征前prepared回退；全部负例、UNKNOWN、无变化和无C样本在总分分母。24真实工程输入验证旧sourceA/F隐藏态、文字、late、竞争残差位等，当前nativeA的h/U在实际GN写回后严格不变；作用域仅GN224。所有新预测先封存，再独立CPU读出/接口末端、真实GN写回范数和物理拟合/留出误差核验，最后评分。','', '## 全体与平台','','| 数据 | 目标 | 方法 | mIoU% | Acc25% | Acc50% |','|---|---|---|---:|---:|---:|']
 keep=['original_source','prepared_A','prepared_F','native_F','repaired_F','lateOnly_F','context_zero_F','context_zero_ABC','native_ABC',primary,'depth_trust_frozenR_F','depth_trust_frozenLate_F','depth_trust_shuffleR_F','wrong_depth_trust_live_F','random_GN_trust_live_F','depth_natural_live_F']
 for x in agg:
  if x['group']=='all_six_equal' or x['arm'] in keep:lines.append(f"| {x['cohort']} | {x['group']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 lines+=['','## 场景比例敏感性','','同R119，权重只取完整18620行的场景频数，各方法相同。EXP缺4个Waymo场景，按共同支持报告，不能冒充实际全量新推理。所有这些数据都已开发曝光；后分层不产生独立确认。','','| 数据 | 方法 | 最小完整场景质量覆盖 | 后分层mIoU% |','|---|---|---:|---:|']
 for x in post:
  if x['group']=='all_six_equal' and x['arm'] in keep:lines.append(f"| {x['cohort']} | {x['arm']} | {x['min_covered_full_mass']:.6f} | {x['mIoU_percent']:.6f} |")
 lines+=['','## 预设裁决','','```json',json.dumps(dec,indent=2),'```','','这里的gate仅是开发原型准入，即使通过也不自动替换生产方法或声称全量稳定TTA。没有将较有利的自然步/ABC读出事后改为主项，没有按平台或案例拼参数。若主项未优于同信息零步及错配/随机/冻结r控制，停止当前实时GN语义路径配方，不追加步长、层或损失网格。','','## 物理恢复与语义路径分别报告','','```json',json.dumps(physical,indent=2),json.dumps(summary,indent=2),'```','','同一参数状态的实时r与冻结r比较控制了物理更新和观测；辅助误差一样不能代替最终选框收益。正确与错配深度保留相同目标值集合，故相似结果只否定空间对应的独立贡献，不能断言其所有全局深度统计都无效。没有把非零r变化或损失下降写成任务成功。','','## 全部正负尾部','','```json',json.dumps(conc,indent=2),'```','','所列good/failure按IoU差异定义，未独立看图的物体身份保持UNKNOWN；原source已正确的恢复和新身份纠错不同。所有匿名逐例、场景、参数写回与物理误差行公开，原图/文字/坐标/ID/特征/梯度/权重仅留本地。固定框在Acc50的上限限制沿用R119诊断，不因一轮更新改变。','','## 文献与成本边界','','文献适用条件与官方实现核对见LITERATURE.md。借鉴了有限参数适应和保留任务路径的条件，不是ProxyTTA复现。已有source-prepared模块不等于源域辅助更新已与grounding对齐；这项联系由本轮任务对照检验。每条seconds覆盖全部实验臂的缓存后缀与共享计算，不是单方法端到端延迟。'];lines+=['','## 零步数值后端与决策核验','','主项DEV17.313490%低于自身零步17.349002%，EXP16.499566%与零步完全相同；正确、错配与随机方向均无总体新收益。硬固定native基础后，GN的实时r路径已接通，但没有表现为额外任务收益。','', '```json',json.dumps(rd(H/'DECISION_EQUIVALENCE_SUMMARY.json'),indent=2),json.dumps(rd(H/'ZERO_NUMERICS_SUMMARY.json'),indent=2),'```','','零步GPU头与原CPU基线在所有2150非别名输入的Top1一致，CPU逐值重放误差为0；因此这轮DEV单例退化不能归因于零步后端差异。主更新相对零步DEV换1个query且恶化1条，EXP换2个query但IoU不变。这个后核验不修改主结果、预测或更新规则。实时r与冻结r的总体分数相同不自动表示概率相同，精确query差异见上表。','','物理留出仍显著改善：DEV705/724、EXP1404/1426，但不决定哪一个候选才是文字目标。保留自然步等全部次项，不继续加步、放大或切换归一化层来挽救。当前深度监督/实时GN语义路径配方不采纳；不据此否定一切无专家TTA。'];(H/'REPORT.md').write_text('\n'.join(lines)+'\n');print('REPORT120_COMPLETE',flush=True)
if __name__=='__main__':main()
