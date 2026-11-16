"""Report all fixed controls; auxiliary decrease is not an adoption criterion."""
from pathlib import Path
import json,numpy as np
R=Path(__file__).resolve().parents[2];H=R/'research/round103';S=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==768;agg=rd(S/'AGGREGATE_ROWS.json');a={x['arm']:x for x in agg if x['group']=='all_six_equal'};v=lambda k:a[k]['mIoU_percent'];ctrl=['prepared_frozen','signcommon_FC','direct_FC','joint3_FC','scaledmean3_FC','shufflecommon3_FC'];g={k:v('common3_FC')-v(k) for k in ctrl};gates=rd(H/'experiment/DECISION_RULE_LOCK.json')['primary_gates'];decision={'primary':'common3_FC','primary_mIoU_percent':v('common3_FC'),'gains_pp':g,'passes_prelocked_primary_gates':all(g[k]>0 for k in gates),'robust_adoption':False,'n':768,'historical_development_exposure':True,'no_experts':True,'no_source_training':True,'FINAL_access':False,'primary_not_switched':True,'no_more_steps_or_parameter_grid':True};(H/'DECISION.json').write_text(json.dumps(decision,indent=2)+'\n');lines=['# R103 — 同一共识绑定证据的参数适应检验','','固定R102共识G，使用原R99三视图L2–LM更新末端18,496参数。没有新增损失、扩大更新层或调参。以同信息共识直接读出为强对照；同范数旧G和打乱共识都以相同预算拟合。沿用768条开发输入、原A256及原样本，C在全部拟合后读取。不是独立确认；没有FINAL。','','| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |','|---|---|---:|---:|---:|']
 for x in agg:lines.append(f"| {x['group']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 pairs=rd(S/'ALL_PAIRED_ROWS.json');lines+=['','## 六方向全部成对结果','','| 方向 | 方法 | 对照 | Δ pp | 改善/恶化/不变 |','|---|---|---|---:|---:|']
 for x in pairs:lines.append(f"| {x['source']}→{x['target']} | {x['method']} | {x['control']} | {x['gain_pp']:+.6f} | {x['better']}/{x['worse']}/{x['same']} |")
 rr=[r for f in sorted(S.glob('*_ALL_ROWS_PRIVATE.json')) for r in rd(f)];fit=[]
 for label in ['common3','scaledmean3','shufflecommon3']:
  z=[r['fits'][label] for r in rr];fit.append({'fit':label,'n':len(z),**{k:float(np.mean([x[k] for x in z])) for k in ['initial_error','final_error','relative_error','parameter_L2','accepted_steps','attempted_steps']}})
 (H/'FIT_SUMMARY.json').write_text(json.dumps(fit,indent=2)+'\n');cases=[]
 for src in ['waymo','drone','quad']:
  for tgt in ['waymo','drone','quad']:
   for control in ['prepared_frozen','signcommon_FC']:
    z=[r for r in rr if r['source']==src and r['target']==tgt]
    for sign in [-1,1]:
     tail=sorted([r for r in z if (r['arms']['common3_FC']['IoU']-r['arms'][control]['IoU'])*sign>1e-12],key=lambda r:(r['arms']['common3_FC']['IoU']-r['arms'][control]['IoU'])*sign,reverse=True)
     if tail:
      for ix in sorted({0,min(1,len(tail)-1),len(tail)//2}):
       r=tail[ix];cases.append({'source':src,'target':tgt,'row_PRIVATE':r['id_PRIVATE'],'control':control,'kind':'good' if sign>0 else 'failure','change_IoU':r['arms']['common3_FC']['IoU']-r['arms'][control]['IoU'],'all_arms':r['arms'],'fits':r['fits'],'selection':'two_extremes_and_median_per_direction_control_sign'})
 (H/'GOOD_FAILURE_CASES_PRIVATE.json').write_text(json.dumps(cases,indent=2)+'\n');lines+=['','## 解释边界','','G、所有弱证据和拟合目标在更新前冻结；不按GT筛反馈、选择步幅或停止更新。保留全部负例、UNKNOWN和零IoU。F/C/FC均为预设端点，主指标固定FC。原始source、prepared_frozen、旧G的三视图LM和同信息零更新都保留。旧G的R99开发优势在R100全量未保持，因此本轮即便开发通过，也不能宣称稳定有效；需固定后扩大检验。辅助误差下降、参数非零和偏好落实不替代任务指标。','','## 裁决','',json.dumps(decision,indent=2)]
 (H/'REPORT.md').write_text('\n'.join(lines)+'\n');print(decision)
if __name__=='__main__':main()
