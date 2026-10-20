"""Render all saved-parameter results without additional GT evaluation."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.evaluate_stvg_vid_saved_parameters128_v1 import D,P,read,put,ARMS
def main():
 r=read(P/'RESULTS_PUBLIC.json');assert read(P/'ROOT_AUDIT_PUBLIC.json')['status']=='pass'
 cases={x['alias']:x for x in r['rows']};impact={}
 for arm in ['Step20_A100','Step30_A100']:
  delta=np.asarray([100*(x['values'][arm]['v']-x['values']['Step40_A100']['v'])for x in r['rows']]);j=next(i for i,x in enumerate(r['rows'])if x['alias']=='BV041')
  impact[arm]=dict(all128_mean_pp=float(delta.mean()),BV041_contribution_to_all128_mean_pp=float(delta[j]/128),other127_mean_pp=float(np.delete(delta,j).mean()),scope='concentration explanation only; no sample removed from reported method')
 put(P/'CASE_INTERPRETATION_PUBLIC.json',dict(concentration=impact,cases={a:cases[a]for a in ['BV041','BV021','BV030','BV067','BV106']},all_parents_kept=True))
 best=r['development_ranking'][0]
 put(P/'DECISION_PUBLIC.json',dict(status='completed_finite_development_comparison',best_development_configuration=best,
  relative_to_E=r['paired'][best+'_minus_E_Strong']['v'],relative_to_original40=r['paired'][best+'_minus_Step40_A100']['v'],
  independent_confirmation=False,stable_gain_not_established=True,two_to_three_pp_gain_not_observed=True,
  dataset_specific_parameters_allowed=True,HC_configuration_unchanged=True,production_registry_promotion=False,
  no_parameter_choice_per_video=True,alpha_is_post_adaptation_output_parameter=True,
  learning_rate_temperature_sampling_scale_trials='not executed: matching complete Vid decoder-input cache not located and new RGB/Source forward forbidden',
  no_additional_grid_expansion=True))
 lines=['# Vid128：保存步数与输出写回强度统一评估','',
 '完成：原HC2→VidSTG冻结TA-STVG、完整128父/每父1查询、seed41。统一使用各自原Strong区间。仅离线提取/变换已保存预测并评分，0模型/decoder/专家/RGB前向、0新学生更新、0网络训练。全部正负平样本保留，不按视频选择停止步或参数。','',
 '本轮用户允许步数之外的参数及分数据集配置。当前Vid完整面板未保存可重放的decoder输入；同批旧输入也只含预测/观测。因此新lr/tau/采样尺度没有合法的同条件全128缓存重放，不冒称已经试验。实际新增参数为输出写回alpha：中心残差线性缩放、宽高对数残差缩放。它是优化结束后的输出映射，不是重新运行更小学习率，也没有让学生按减弱后的状态重新采样。','',
 '预先固定 steps={20,30,40} × alpha={0.25,0.5,1}，九组+E。alpha1精确复原原保存框，alpha0精确E；其他参数不动。所有1280输出先封存再读取GT。原40/alpha1和E分数逐项精确复现。','',
 '## 全矩阵','',
 '| 步数 | 输出alpha | 完整vIoU % | 相对E pp [95% CI] | 相对原40步 pp | 未观察实际节点sIoU % | 相对E下降>5pp父数 |',
 '|---|---:|---:|---|---:|---:|---:|']
 e=r['means']['E_Strong'];lines.append(f"| E基线 | — | {100*e['v']:.6f} | — | — | {100*e['unobserved_actual_sIoU']:.6f} | — |")
 for a in ARMS[1:]:
  v=r['means'][a];d=r['paired'][a+'_minus_E_Strong'];m=d['v'];vs40=0. if a=='Step40_A100'else r['paired'][a+'_minus_Step40_A100']['v']['mean'];lines.append(f"| {int(a[4:6])} | {int(a[-3:])/100:g} | {100*v['v']:.6f} | {m['mean']:+.6f} [{m['CI95'][0]:+.6f},{m['CI95'][1]:+.6f}] | {vs40:+.6f} | {100*v['unobserved_actual_sIoU']:.6f} | {d['tail']['below_minus5']} |")
 lines+=['',f"全部设置tIoU相同：{100*e['t']:.6f}%。因此本轮差异完全来自空间轨迹/写回，而非时间区间变更。",'',
 '## 原问题：减少更新能否保住E+Strong？','',
 '20步/alpha1完整v相对E−0.036991pp CI[−0.418804,+0.327591]；30步+0.004219pp CI[−0.454115,+0.479140]；40步−0.277910pp CI[−1.150458,+0.431887]。30步相对40步+0.282130pp CI[−0.096107,+0.952765]，20步+0.240919pp CI[−0.216175,+0.986374]。均未建立确定差异。','',
 '减少更新把点均值带回E附近，并缓解最坏尾部；没有建立稳定超越E，也没有预设非劣容忍界，因此不能声称已统计证明“不损伤E”。开发最大均值为30/alpha1，不自动晋升最终方法。HC不要求改成30步。','',
 'alpha<1的六组点均值均低于E。20/30步的减弱写回均低于同一步数alpha1；40步alpha0.25相对原40步仅+0.017282pp，区间[−0.514402,+0.619729]，alpha0.5则更低。不能把“缩回E能缓解某个坏例”推广为统一写回已产生净增量；同样，这不排除尚未重跑的学习率/温度可能有用。','',
 '## 个例和集中度','',
 '| 样本 | E+Strong v % | 20步/alpha1 | 30步/alpha1 | 40步/alpha1 |','|---|---:|---:|---:|---:|']
 for a in ['BV041','BV021','BV030','BV067','BV106']:
  v=cases[a]['values'];lines.append('| '+a+' | '+' | '.join(f"{100*v[k]['v']:.6f}"for k in ['E_Strong','Step20_A100','Step30_A100','Step40_A100'])+' |')
 lines+=['',
 'BV041的实际评分确认了上轮几何界限指向的后期损伤：实际节点sIoU为E90.1526%、20步89.9270%、30步86.5517%、40步14.6354%；原时间区间保持相同。减少更新能保住这例绝大部分E效果。','',
 '但BV021在20步已经降至2.6725%完整v，仍远低于E18.9111%；BV030的20步4.3649%反而低于40步4.6615%。它们不能由“只在最后十步更新过度”统一解释。','',
 '正例也会失去后期恢复：BV067从40步24.2486%变为20步17.6964%；BV106从5.9269%变为20步0%。所以停止更早会同时减少损伤和有效修正。','',
 '30步相对40步全128均值+0.282130pp，其中BV041单例对全均值贡献+0.310635pp，其他127例自身均值−0.028730pp；20步相对40步+0.240919pp，其中BV041贡献+0.324387pp，其他127例均值−0.084125pp。这只是集中度分析，原128分母和全部样本都保留。当前改善主要是防止一个灾难负例，不能称为广泛提升。','',
 '## 传播检查','',
 '| 设置 | 有GT的观察节点sIoU % | 未观察实际节点sIoU % |','|---|---:|---:|']
 for a in ['E_Strong','Step20_A100','Step30_A100','Step40_A100']:
  v=r['means'][a];lines.append(f"| {a} | {100*v['observed_actual_sIoU']:.6f} | {100*v['unobserved_actual_sIoU']:.6f} |")
 lines+=['',
 '三种原轨迹预算在观察位置的均值都高于E，而未观察实际节点均低于E。该模式支持继续将跨帧转移和教师监督适用范围列为竞争原因；不等于证明某一个参考锚点单独导致损伤。观察节点和未观察节点是固定但不同分母的统计，不能相加。','',
 '## 数据、核验、成本与决定','',
 'Vid128已暴露于原40步评估和失败分析，本轮用于预算/输出参数开发选择，不是独立确认。所有128父只有seed41；种子、父视频及查询均未重选。HC沿用原冻结版本；允许今后按数据集固定不同配置，不用双域同参数作为门槛。','',
 '官方/Dense指标独核1280次，最大差2.22e−15；原E/40指标身份检查1536项。独立读取保存节点/密集IoU并重聚合4960项、均值70项、父级bootstrap90项，最大差2.22e−15。旧128预测+128trace、方法与注册保持原SHA。GT派生逐节点诊断只存私有目录，公开匿名逐父指标及所有参数结果。','',
 f"本轮评分CPU wall {r['costs']['scoring_seconds']:.6f}s；提取、根复核、公开、档案另计。没有新模型或学生计算，不能把这个时间当20/30步完整部署延迟；原Source/Strong/两空间专家成本仍由原完整链路承担。",'',
 '决定：保留30步/alpha1为开发候选及原40步作对照；没有2–3pp新增收益或独立验证，不自动改正式配置。关闭本轮九组有限检查，不据此宣称所有参数无效。新lr/tau/尺度实验仍受缺失decoder输入和禁止新增RGB前向的边界限制；未执行就不报成绩。','',
 '证据：[完整逐父/全矩阵](RESULTS_PUBLIC.json)、[独立根审计](ROOT_AUDIT_PUBLIC.json)、[缓存资格](CACHE_QUALIFICATION_PUBLIC.json)、[集中度与正负个例](CASE_INTERPRETATION_PUBLIC.json)、[固定协议](../../../../protocols/stvg_vid_saved_parameters128_v1.md)。']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
 print('REPORT_WRITTEN',best)
if __name__=='__main__':main()
