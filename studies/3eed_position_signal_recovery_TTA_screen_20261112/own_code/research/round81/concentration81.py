"""Descriptive contribution concentration; never filters deployed updates or scores."""
from core81 import *
import collections
rows=read(H81/'SIGNAL_CONTROL_TASK_ROWS.json');out=[]
for part in ['VAL1280','TRAIN1024']:
 rr=[r for r in rows if r['cohort']==part];n=collections.Counter(r['platform'] for r in rr)
 for ref in ['frozen_bi','frozen_direct3','all__ALL__shuffle']:
  delta=[r['all__ALL_IoU']-r[ref+'_IoU'] for r in rr];c=[v*50/n[r['platform']] for v,r in zip(delta,rr)];out.append({'cohort':part,'reference':ref,'gain_pp':float(sum(c)),'improved':sum(v>1e-10 for v in delta),'worsened':sum(v<-1e-10 for v in delta),'largest_positive_contribution_pp':float(max(c)),'gain_excluding_largest_positive_original_denominator_pp':float(sum(c)-max(c))})
dump(H81/'CANDIDATE_CONCENTRATION_ROWS.json',out);candidate=read(H81/'EXPLORATORY_FULL_FUSION_CANDIDATE.json');candidate['same_information_increment_concentration']=out;candidate['same_information_increment_fragile']=True;dump(H81/'EXPLORATORY_FULL_FUSION_CANDIDATE.json',candidate);d=read(H81/'DECISION.json');d['exploratory_full_combined_backtracking']=candidate;dump(H81/'DECISION.json',d);r=read(H81/'REPORT_NUMERIC_AUDIT.json');r['descriptive_candidate_concentration']=out;dump(H81/'REPORT_NUMERIC_AUDIT.json',r)
with (H81/'REPORT.md').open('a') as f:
 f.write('\n\n## 增量集中度：完整均值保留，尾部只作描述\n\n| 范围 | 对照 | 完整增量 pp | 改善/恶化 | 最大正贡献 pp | 去掉最大正贡献后其余合计 pp（原分母） |\n|---|---|---:|---:|---:|---:|\n')
 for r in out:f.write(f"| {r['cohort']} | {r['reference']} | {r['gain_pp']:+.6f} | {r['improved']}/{r['worsened']} | {r['largest_positive_contribution_pp']:.6f} | {r['gain_excluding_largest_positive_original_denominator_pp']:+.6f} |\n")
 f.write('\n相对冻结模型的正增量在移去最大正贡献后仍为正；相对强零更新直接平均则在两组都变负，说明额外参数更新收益很小且集中。不是删除样本重报主成绩，也不以所有个例必须上涨作门槛，更没有针对尾部修规则；全部样本的原总体结果仍为主结果。当前可以说有弱TTA入口，不能说已经找到足够稳健、幅度充分的实例纠错信号。\n')
print(out)
