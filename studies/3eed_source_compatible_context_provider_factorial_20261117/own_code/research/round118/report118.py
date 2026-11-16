from core118 import R,H,B,rd,dump,PRIMARY
import numpy as np

def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==2304;agg=rd(H/'scores_attempt001/AGGREGATE_ROWS.json');rr=[r for f in (H/'scores_attempt001').glob('*_ALL_ROWS_PRIVATE.json') for r in rd(f)];conc=[];cases=[]
 for co in ['DEV','EXP']:
  cr=[r for r in rr if r['cohort']==co]
  for p,c in [(PRIMARY,'repaired_F'),(PRIMARY,'ABC_native'),(PRIMARY,'ABC_lSA_rSA'),(PRIMARY,'F_lSA_rSAshuffled'),('F_lSA_rSA','F_lSA_rzero'),('F_lSA_rSA','F_lSF_rSF')]:
   dd=np.array([r['arms'][p]['IoU']-r['arms'][c]['IoU'] for r in cr]);conc.append({'cohort':co,'method':p,'control':c,'n':len(cr),'gain_pp':float(dd.mean()*100),'better':int((dd>1e-12).sum()),'worse':int((dd< -1e-12).sum()),'same':int((abs(dd)<=1e-12).sum()),'largest_positive_single_input_pp':float(max(0,dd.max())*100/len(cr)),'largest_negative_single_input_pp':float(min(0,dd.min())*100/len(cr))})
   for s,t in sorted({(r['source'],r['target']) for r in cr}):
    for kind,sgn in [('good',1),('failure',-1)]:
     rows=[r for r in cr if r['source']==s and r['target']==t and sgn*(r['arms'][p]['IoU']-r['arms'][c]['IoU'])>1e-12 and not r['protected_alias_fallback']]
     if rows:
      x=max(rows,key=lambda r:abs(r['arms'][p]['IoU']-r['arms'][c]['IoU']));cases.append({'cohort':co,'source':s,'target':t,'id_PRIVATE':x['id_PRIVATE'],'kind':kind,'method':p,'control':c,'delta_IoU':x['arms'][p]['IoU']-x['arms'][c]['IoU'],'original_source_IoU':x['arms']['original_source']['IoU'],'candidate_identity':'UNKNOWN_without_independent_visual_review','postseal_diagnostic_only':True})
 dump(H/'CONCENTRATION_SUMMARY.json',conc);dump(H/'GOOD_FAILURE_CASES_PRIVATE.json',cases)
 decision=rd(H/'DECISION.json');natives=rd(H/'NATIVE_AUDIT.json');lines=['# R118 — 原生基础与源域兼容 RGB／竞争证据的固定读出对照','','本轮固定原source100与已有R91双向模块，不重新训练，不更新参数、不增加辅助损失，不读取FINAL。候选保持原A256、六层位置、原query身份和完整文字。只替换小模块的证据来源，检验此前撤回源域Point Adapter/早期RGB路径以后，训练过的竞争头是否因输入变化失效。','','## 机制与固定对照','','原生点云—文字基础来自未改动3EED；NF表示native/fused观测，SF表示完整source-prepared/fused观测，SA表示完整source-prepared/同步当前A观测。最终分数仍用真实匹配MLP、归一化、主体token对其他token的logmass比，再加现有竞争残差；系数全部继承1，不按目标平台或样本调节。三个late来源×零/NF/SF/SA四个残差来源，区分模块输入条件变化与融合多时刻点特征/仅当前A RGB的失配。主项预先固定F_lSA_rSA。','','额外保留A/C读出、相同SA证据作用于native A/B/C后概率平均、去残差ABC与纯native ABC。查询残差打乱与late/残差一起打乱保持数值集合且无固定点；它们检验query对应作用，不认证真实实例身份。原始source、完整preparedA/F、nativeF、修复F与lateOnlyF的旧预测和评分逐字复用。','','所有证据由同一source100/已有source模块产生，读取不同缓存后缀不等于增加第二个预训练模型。但实现完整管线时需执行native和source-compatible两个后缀，不能将这里缓存CPU重放耗时冒称端到端运行成本。没有新增PointNet/注册/媒体读取。','','## 数据和审计','','DEV768与EXP1536总计2304定向输入，来自三目标平台、六个方向。两者都已有开发曝光，不称独立确认集；一个表达在两个source下被评估，不把它算成两份独立场景。154保护媒体别名在读取特征前prepared回退；UNKNOWN、未配准C和所有负例保留分母。24真实工程输入验证旧provider/原source/候选读出一致，全部预测封存后才读取任务GT。',f"独立NumPy重放{natives['new_CPU_endpoints']}新端点，概率最大误差{natives['max_probability_error']:.3g}，近平差异{natives['CPU_near_ties']}；独立平台/场景/paired重算，所有43776新任务值双几何实现核验。原6参照13824值复用，合计57600任务值。",'','## 总体和平台','','| 数据范围 | 目标平台 | 方法 | mIoU% | Acc25% | Acc50% |','|---|---|---|---:|---:|---:|']
 for x in agg:
  if x['group']=='all_six_equal' or x['arm'] in ['original_source','prepared_F','native_F','repaired_F','lateOnly_F',PRIMARY,'ABC_native','ABC_lSA_rSA','ABC_lSA_rzero']:lines.append(f"| {x['cohort']} | {x['group']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 lines+=['','## 预注册裁决','','```json',__import__('json').dumps(decision,indent=2),'```','','主项必须在两个范围分别超过全部既定强对照，包括同信息ABC和打乱对应；没有按结果更换主项或按平台拼接。门槛通过也只支持冻结读出的可用性，不构成TTA收益。若未通过，停止当前F/兼容来源配方，不将其输出再拟合成伪标签来挽救；某个次项正值照常保留但需要后续独立锁定评价。','','## 正负尾部和组件作用','','```json',__import__('json').dumps(conc,indent=2),'```','','这些是任务分数变化，未逐一看图的实例身份保持UNKNOWN。source本来正确而新读出恢复原选择，与新发现目标身份是不同机制；重复框之间切换也不自动等于实例纠错。未基于good/bad案例补规则。完整正负例、每个平台和场景均随公开匿名表保留。当前原始可见RGB始终为A，多采样/多LiDAR不宣称新RGB视角。稀少Drone/Quad场景不支持总体跨场景置信区间。']
 (H/'REPORT.md').write_text('\n'.join(lines)+'\n');print('REPORT118_COMPLETE',flush=True)
if __name__=='__main__':main()
