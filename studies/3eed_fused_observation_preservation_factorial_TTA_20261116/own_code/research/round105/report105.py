"""All directions, all negatives, matched direct-use gates; no posthoc method change."""
from pathlib import Path
import json,numpy as np,collections
R=Path(__file__).resolve().parents[2];H=R/'research/round105';B=R/'crossplatform/round105';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==768;agg=rd(O/'AGGREGATE_ROWS.json');ss={x['arm']:x for x in agg if x['group']=='all_six_equal'};v=lambda k:ss[k]['mIoU_percent'];primary='TTA_jointABC';controls=['prepared_frozen','frozen_jointABC','frozen_ensemble3','teacher_shuffle_jointABC','wrong_TTA_jointABC','joint3_FC'];gains={k:v(primary)-v(k) for k in controls};gate=all(gains[k]>0 for k in controls[:5]);rr=[r for p in O.glob('*_ALL_ROWS_PRIVATE.json') for r in rd(p)];cases=[];subsets=[]
 for name in rd(B/'MATRIX_SEALED.json')['directions']:
  a=[r for r in rr if r['source']+'_to_'+r['target']==name]
  for subset,part in [('all',a),('update_and_real_C',[r for r in a if r['status'].get('TTA_output_enabled',False)]),('fallback',[r for r in a if not r['status'].get('TTA_output_enabled',False)])]:
   for arm in controls+[primary,'frozen_AC','TTA_AC']:
    subsets.append({'direction':name,'subset':subset,'n':len(part),'arm':arm,'mIoU_percent':float(np.mean([r['arms'][arm]['IoU'] for r in part])*100) if part else None})
  for ctrl in ['prepared_frozen','frozen_jointABC','frozen_ensemble3']:
   for sign in [-1,1]:
    a0=sorted([r for r in a if (r['arms'][primary]['IoU']-r['arms'][ctrl]['IoU'])*sign>1e-12],key=lambda r:abs(r['arms'][primary]['IoU']-r['arms'][ctrl]['IoU']),reverse=True)
    for i in sorted({0,min(1,len(a0)-1),len(a0)//2} if a0 else set()):
     r=a0[i];cases.append({'source':r['source'],'target':r['target'],'row_PRIVATE':r['id_PRIVATE'],'control':ctrl,'kind':'good' if sign>0 else 'failure','delta_IoU':r['arms'][primary]['IoU']-r['arms'][ctrl]['IoU'],'arms':r['arms'],'status':r['status'],'selection':'two_extremes_and_median_per_direction_control_sign'})
 fits=[]
 for label in ['real','teacher_shuffle','wrong']:
  z=[r['fits'][label] for r in rr if label in r['fits']];fits.append({'fit':label,'n':len(z),**{k:float(np.mean([a[k] for a in z])) if z else None for k in ['initial_error','final_error','relative_error','accepted_steps','attempted_steps','parameter_L2']}})
 conc=[]
 for c in controls:
  ds=np.array([r['arms'][primary]['IoU']-r['arms'][c]['IoU'] for r in rr]);conc.append({'control':c,'better':int((ds>1e-12).sum()),'worse':int((ds< -1e-12).sum()),'same':int((abs(ds)<=1e-12).sum()),'net_pp':float(ds.mean()*100),'largest_positive_single_input_pp':float(max(0,ds.max())*100/768),'largest_negative_single_input_pp':float(min(0,ds.min())*100/768)})
 decision={'primary':primary,'mIoU_percent':v(primary),'gains_pp':gains,'passes_development_gates':gate,'robust_adoption':False,'decision':'candidate_requires_fixed_expansion' if gate else 'do_not_adopt_this_multiview_TTA_recipe','new_setting':'actual multiple RGB observations on canonical A LiDAR; offline/exposed scenes','all_inputs':768,'FINAL_access':False,'no_experts':True,'source_training':False,'primary_not_switched':True};dump(H/'DECISION.json',decision);dump(H/'GOOD_FAILURE_CASES_PRIVATE.json',cases);dump(H/'SUBSET_RESULTS.json',subsets);dump(H/'FIT_SUMMARY.json',fits);dump(H/'CONCENTRATION_SUMMARY.json',conc)
 lines=['# R105 — 实际额外观测下的原生双向融合与匹配适应','','本轮改变证据获取：同一A坐标、原A256、完整文字与原source准备参数不变，按固定传感器顺序引入真实B/C图像。没有新的预训练网络；source CNN对每张图分别编码，经过已训练的稀疏RGB↔LiDAR交互读取。额外点支持由传感器PnP/反向循环、实测深度及邻近回波约束，失败保持UNKNOWN。','','这是新的离线多观测setting。计数器不被当作秒，未证明连续视频或因果在线；不是新场景测试，也不能将收益回写为旧单观测R95有效。当前开发数据多轮暴露，FINAL没有访问。','','## 预检与实施验证','',json.dumps(rd(H/'FULL_REGISTRATION_SUMMARY.json'),indent=2),'','原24条错场景预检全部被拒绝，但不能据此认证正确实例或校准错误率。保护帧以及保护JPEG内容别名在解码前排除。对内容别名anchor仅复用既有冻结预测；所有768条仍留在分母。','',json.dumps(rd(H/'COVERAGE_SUMMARY.json'),indent=2),'','工程验证包括原始A稀疏图逐项一致、额外但未使用RGB bank不影响输出、重复A bank数值等价、真实B图像的原生前向、实际写参数后的全模型重放与显式Jacobian。首个工程attempt因在no_grad中调用导数检查失败，完整保存；检查器显式enable_grad后重试，不涉及方法或标签选择。','','## 固定方法','','q=(p_A+p_AB)/2来自同一冻结模型，停止梯度。相同q监督A和A+B，只有末端18496参数query投影更新；LM仍为10次、每次最多6次阻尼验收。每个视图使用自己的原生竞争残差，该残差不依赖更新的投影，冻结它不偷换任务分数。C的网络预测在全部拟合结束后才计算，不用于目标、验收或停止。B不可用不拟合；C不可用则记录已发生拟合与成本，但最终回到同信息冻结读出。','','主端点为更新后jointABC，必须与冻结jointABC及冻结概率集成比较；只超过原A或历史单观测结果不够。额外绑定打乱只改变新增图像的可见点归属，保留A图和可见性；另有q中B候选身份打乱控制。历史R100方法仅作上下文，输入信息不同。','','## 全部输入结果','','| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |','|---|---|---:|---:|---:|']
 for x in agg:lines.append(f"| {x['group']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 lines+=['','## 所有正负变化与集中度','',json.dumps(conc,indent=2),'','## 裁决','',json.dumps(decision,indent=2),'','辅助拟合下降不能证明实体身份正确；几何对应通过也不是语言判断真值。good/failure按每方向、每对照、符号分别取两个极端和中位用于事后机制诊断，不改变评价分母。场景聚类结果完整保留，不把共享场景/描述视为独立样本，不为单场景给总体CI。若未过同信息与错配控制，不加步幅/层位置网格挽救。文献借鉴与边界见LITERATURE.md。'];(H/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
