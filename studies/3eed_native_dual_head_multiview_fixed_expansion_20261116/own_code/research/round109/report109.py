"""Fixed non-DEV expansion and exposed DEV side by side, no parameter adaptation."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round109';O=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==1536;agg=rd(O/'AGGREGATE_ROWS.json');aa={x['arm']:x for x in agg if x['group']=='all_six_equal'};val=lambda k:aa[k]['mIoU_percent'];primary='real_softtoken_geometric_fusedABC';controls=['original_source','prepared_frozen','real_BBF_fusedABC','real_softtoken_fusedABC','real_softtoken_arithmetic_fusedABC','randomA_softtoken_geometric_fusedABC','wrong_softtoken_geometric_fusedABC','real_softtoken_shuffled_geometric_fusedABC','real_BBF_ensemble'];rows=[r for f in O.glob('*_ALL_ROWS_PRIVATE.json') for r in rd(f)];conc=[];sub=[]
 for c in controls:
  d=np.array([r['arms'][primary]['IoU']-r['arms'][c]['IoU'] for r in rows]);conc.append({'control':c,'n':len(rows),'gain_pp':float(d.mean()*100),'better':int((d>1e-12).sum()),'worse':int((d< -1e-12).sum()),'same':int((abs(d)<=1e-12).sum()),'largest_positive_single_input_pp':float(max(0,d.max())*100/1536),'largest_negative_single_input_pp':float(min(0,d.min())*100/1536)})
 for label in ['all','any_real_extra','no_real_extra_nonalias','protected_alias']:
  rr=[r for r in rows if label=='all' or (label=='protected_alias' and r['status']['protected_alias_fallback']) or (not r['status']['protected_alias_fallback'] and ((label=='any_real_extra' and any(r['status']['families']['real'][v+'_available'] for v in ['B','C'])) or (label=='no_real_extra_nonalias' and not any(r['status']['families']['real'][v+'_available'] for v in ['B','C']))))]
  if rr:
   for arm in [primary]+controls+['softtoken_geometric_A','softtoken_arithmetic_A']:sub.append({'subset':label,'n':len(rr),'arm':arm,'mIoU_percent':float(np.mean([r['arms'][arm]['IoU'] for r in rr])*100)})
 cases=[]
 for direction in rd(R/'crossplatform/round109/MATRIX_SEALED.json')['directions']:
  rr=[r for r in rows if r['source']+'_to_'+r['target']==direction]
  for sign in [-1,1]:
   xs=[r for r in rr if sign*(r['arms'][primary]['IoU']-r['arms']['real_BBF_fusedABC']['IoU'])>1e-12]
   if xs:
    r=max(xs,key=lambda r:abs(r['arms'][primary]['IoU']-r['arms']['real_BBF_fusedABC']['IoU']));cases.append({'source':r['source'],'target':r['target'],'row_PRIVATE':r['id_PRIVATE'],'kind':'good' if sign>0 else 'failure','control':'real_BBF_fusedABC','delta_IoU':r['arms'][primary]['IoU']-r['arms']['real_BBF_fusedABC']['IoU'],'postseal_selection':'largest absolute sign-specific change per direction, descriptive only'})
 dump(H/'GOOD_FAILURE_CASES_PRIVATE.json',cases)
 mapping={'original_source':'original_source','prepared_frozen':'prepared_frozen','real_frozen_fusedABC':'real_BBF_fusedABC','softtoken_fusedABC':'real_softtoken_fusedABC','softtoken_geometric_fusedABC':primary,'softtoken_arithmetic_fusedABC':'real_softtoken_arithmetic_fusedABC','randomA_frozen_fusedABC':'randomA_BBF_fusedABC'};dev=rd(R/'research/round108/scores_attempt001/AGGREGATE_ROWS.json');dev={x['arm']:x for x in dev if x['group']=='all_six_equal'};ranges=[]
 for old,new in mapping.items():
  ranges.append({'arm':new,'DEV768_mIoU_percent':dev[old]['mIoU_percent'],'expansion1536_mIoU_percent':val(new),'combined2304_mIoU_percent':float((dev[old]['mIoU_percent']+2*val(new))/3),'combined_not_independent_confirmation':True})
 # Logical signal gate is separated from the stricter primary-versus-all-controls adoption gate.
 gate=all(val(primary)>val(k) for k in controls);signal=val(primary)>val('real_BBF_fusedABC');decision={'primary':primary,'primary_mIoU_percent':val(primary),'gains_pp':{k:val(primary)-val(k) for k in controls},'native_head_complementarity_survives_expansion':signal,'passes_strict_all_controls_gate':gate,'robust_TTA_adoption':False,'TTA_run':False,'decision':'frozen_signal_retained_not_TTA_and_requires_error_quality_diagnostic' if signal else 'stop_this_native_head_signal_recipe_no_output_fitting','primary_not_switched':True,'n':1536,'FINAL_access':False,'historical_full_VAL_exposure':True};dump(H/'DECISION.json',decision);dump(H/'CONCENTRATION_SUMMARY.json',conc);dump(H/'SUBSET_RESULTS.json',sub);dump(H/'RANGE_COMPARISON.json',ranges)
 lines=['# R109 — 原生双头与真实点观测的固定配置扩展验证','','先验证监督来源，不新增适应损失、参数更新、BN校准、模型或source训练。固定R106实测点特征对齐和R108独立soft-token头的等权读出；保持同一A256、坐标、文字视角、RGB与R91双向结构。主读出仍是几何池，不按结果换成算术池。','','## 清单与观测边界','',json.dumps(rd(H/'INPUT_SUMMARY.json'),indent=2),'','768新增表达/物理路径，两个incoming sources形成1536定向输入，排除当前DEV锚点路径。所有拒绝/未知/保护别名均留在分母。全VAL有历史模型和标签暴露，这不是全新独立测试；Waymo/Drone/Quad场景分别89/2/3，不按帧IID给总体CI。相邻观测由固定计数器顺序检索，时间单位未知，不宣称因果在线视频。','','## 传感器对应','',json.dumps(rd(H/'REGISTRATION_SUMMARY.json'),indent=2),'','复用原SIFT+独立实测深度PnP数值验证、反向注册与留出循环阈值；无实例mask/外部缓存/提供位姿。先锁定B/C，再检查几何，不在失败后搜索更有利邻帧。已注册FINAL路径不进入frame白名单，受保护JPEG内容别名不解码。初次注册启动缺少父目录，在读取任何传感器前失败；保留attempt001脚本/日志，仅修parents=True后重启，未重跑任何有效注册。','','## 全体结果','','| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |','|---|---|---:|---:|---:|']
 for x in agg:lines.append(f"| {x['group']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 lines+=['','## 方向与所有负比较','','| Source | Target | 方法 | 对照 | Δ pp | 改善/恶化/不变 |','|---|---|---|---|---:|---:|']
 for x in rd(O/'ALL_PAIRED_ROWS.json'):lines.append(f"| {x['source']} | {x['target']} | {x['method']} | {x['control']} | {x['gain_pp']:+.6f} | {x['better']}/{x['worse']}/{x['same']} |")
 lines+=['','## 开发与扩展分别报告','',json.dumps(ranges,indent=2),'','## 裁决','',json.dumps(decision,indent=2),'','## R108既有偏好的分量归因（不改变R109方法）','',json.dumps(rd(H/'CE_COMPONENT_DIAGNOSTIC.json'),indent=2),'','## 主张边界','','source CE头与contrastive头共享decoder和训练来源，不能称独立专家/物体身份真值。CE原生256token概率分母、当前纯文本主体mask、source eval BN/dropout保持，绝未使用标注positive_map或GT跨度。任何头之间的GT最优选择仅属诊断，不能部署。稳定静态信号不等于参数TTA；本轮没有更新权重，不能宣称已解决适应。所有负例保留；若信号不复现，不以继续调融合系数或拟合其输出来挽救。'];(H/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
