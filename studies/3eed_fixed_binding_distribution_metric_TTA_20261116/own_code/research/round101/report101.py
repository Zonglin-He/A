"""All outcomes decide the registered primary; no secondary endpoint promotion."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round101';S=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==768;rows=rd(S/'ALL_METHOD_ROWS.json');agg=rd(S/'AGGREGATE_ROWS.json');pairs=rd(S/'ALL_PAIRED_ROWS.json');aa={x['arm']:x for x in agg if x['group']=='all_six_equal'};v=lambda k:aa[k]['mIoU_percent'];controls=['prepared_frozen','joint3_FC','direct_FC','direct_four','kl3_shuffle_FC'];gains={k:v('kl3_FC')-v(k) for k in controls};decision={'primary':'kl3_FC','primary_mIoU_percent':v('kl3_FC'),'gains_pp':gains,'passes_primary_comparisons':all(x>0 for x in gains.values()),'n':768,'historically_exposed_development':True,'independent_confirmation':False,'robust_adoption':False,'one_changed_factor':'KL objective and matched Fisher metric','no_external_experts':True,'no_source_retraining':True,'FINAL_access':False,'C_new_scene':False};(H/'DECISION.json').write_text(json.dumps(decision,indent=2)+'\n')
 lines=['# R101 — 固定绑定监督的分布度量检验','','R100固定全量扩展未保持R99的开发优势。本轮固定原三源、A256、RGB–点云绑定差分G、三拟合视图、18,496个末端线性参数和10×6优化预算，仅将均匀分数L2及其Gauss–Newton度量替换为精确候选KL及匹配Fisher度量。真实目标、打乱目标均运行。源模型逐输入重置，C在两种拟合完成后才读取，不参与验收。768条来自R99已暴露开发集；不宣称独立泛化。','','| 方向 | n | 冻结 | 直接FC | L2三视图FC | KL三视图FC | KL打乱FC |','|---|---:|---:|---:|---:|---:|---:|']
 for src in ['waymo','drone','quad']:
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   z={x['arm']:x for x in rows if x['source']==src and x['target']==tgt};lines.append('| '+src+'→'+tgt+' | 128 | '+' | '.join(f"{z[k]['mIoU_percent']:.6f}" for k in ['prepared_frozen','direct_FC','joint3_FC','kl3_FC','kl3_shuffle_FC'])+' |')
 lines+=['','## 等权六方向及各目标平台','','| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |','|---|---|---:|---:|---:|']
 for x in agg:
  if x['arm'] in ['original_source','prepared_frozen','binding_direct','direct_FC','direct_four','lm_FC','joint3_FC','joint3_C','kl3_F','kl3_C','kl3_FC','kl3_shuffle_FC']:lines.append(f"| {x['group']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 lines+=['','## 全部预设比较','','| 方向 | 方法 | 对照 | Δ pp | 改善/恶化/不变 |','|---|---|---|---:|---:|']
 for x in pairs:lines.append(f"| {x['source']}→{x['target']} | {x['method']} | {x['control']} | {x['gain_pp']:+.6f} | {x['better']}/{x['worse']}/{x['same']} |")
 case=[];fit=[]
 for f in sorted(S.glob('*_ALL_ROWS_PRIVATE.json')):
  rr=rd(f)
  for i,r in enumerate(rr):
   case.append({'source':r['source'],'target':r['target'],'id_PRIVATE':r['id_PRIVATE'],'row':i+1,'delta_vs_L2':100*(r['arms']['kl3_FC']['IoU']-r['arms']['joint3_FC']['IoU']),'delta_vs_direct':100*(r['arms']['kl3_FC']['IoU']-r['arms']['direct_FC']['IoU']),'arms':r['arms']})
   for name,x in r['fits'].items():fit.append({'source':r['source'],'target':r['target'],'fit':name,**x})
 cases=[]
 for src in ['waymo','drone','quad']:
  for tgt in ['waymo','drone','quad']:
   z=[x for x in case if x['source']==src and x['target']==tgt]
   for sign in [-1,1]:
    tail=sorted([x for x in z if x['delta_vs_L2']*sign>1e-10],key=lambda x:x['delta_vs_L2']*sign,reverse=True);cases.extend(tail[:2])
 (H/'GOOD_FAILURE_CASES_PRIVATE.json').write_text(json.dumps(cases,indent=2)+'\n');fs=[]
 for name in ['kl3','kl3_shuffle']:
  z=[x for x in fit if x['fit']==name];fs.append({'fit':name,'n':len(z),**{k:float(np.mean([x[k] for x in z])) for k in ['initial_error','final_error','relative_error','parameter_L2','accepted_steps','attempted_steps']},'no_accepted_step_n':sum(x['accepted_steps']==0 for x in z)})
 (H/'FIT_SUMMARY.json').write_text(json.dumps(fs,indent=2)+'\n');lines+=['','## 机制与边界','','Fisher双对偶矩阵已与显式参数Jacobian核验，KL梯度与自动求导比较；全部预测和每个接受/拒绝trial均独立重放。C为同一物理观测的新遮挡而非新相机或新场景。新方法复用的特征缓存有R100原生前向审计及逐文件哈希；没有新增图像专家。','','好坏尾部保留于私有案例索引，公开表保留所有匿名结果。IoU变化只作为事后效果，不当作实例身份真值或更新筛选条件。对梯度与损失度量的假设不能替代grounding指标。','','## 裁决','',json.dumps(decision,indent=2),'','若KL未超过同信息直接读出与L2对照，停止以同一G单纯更换度量作为性能主线，不扩大步数。若开发全部比较通过，也只作为候选，需固定后扩大验证；不能据开发集宣称稳健TTA。']
 (H/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps(decision))
if __name__=='__main__':main()
