"""Fixed-configuration expansion, no promotion based on the full labels."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[2];H=R/'research/round100';S=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==18620;rows=rd(S/'ALL_METHOD_ROWS.json');agg=rd(S/'AGGREGATE_ROWS.json');pairs=rd(S/'ALL_PAIRED_ROWS.json');aa={x['arm']:x for x in agg if x['group']=='all_six_equal'};v=lambda k:aa[k]['mIoU_percent'];decision={'primary':'joint3_FC','primary_mIoU_percent':v('joint3_FC'),'gains_pp':{k:v('joint3_FC')-v(k) for k in ['original_source','prepared_frozen','lm_FC','direct_FC','direct_four','joint3_shuffle_FC']},'heldout_C_gain_vs_oldC_pp':v('joint3_C')-v('lm_C'),'F_gain_vs_direct_pp':v('joint3_F')-v('binding_direct'),'passes_main_comparisons':all(v('joint3_FC')>v(k) for k in ['prepared_frozen','lm_FC','direct_FC','direct_four','joint3_shuffle_FC']),'n':18620,'unchanged_R99_method':True,'FINAL_access':False,'expert_added':False,'source_retraining':False,'historical_exposure':True,'robust_adoption':False};(H/'DECISION.json').write_text(json.dumps(decision,indent=2)+'\n')
 lines=['# R100 — R99固定配方的六方向全量扩展','','R99开发768条出现正信号后，保持G、三视图拟合、18,496参数、原始A256、十次LM与每次六次阻尼、原source状态和C后读取规则不变。没有继续挑参数。原768条输出经哈希核验复用，新增输入使用完全相同程序；全部20臂、负例、UNKNOWN和零IoU保留。全量18620来自此前R98历史暴露范围，属于固定配置扩展，不能称全新独立测试。','','| 方向 | n | 原始 | 冻结 | 旧单视图FC | 直接FC | 直接四视图 | 新三视图FC | 打乱三视图FC |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for src in ['waymo','drone','quad']:
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   z={x['arm']:x for x in rows if x['source']==src and x['target']==tgt};lines.append('| '+src+'→'+tgt+' | '+str(z['joint3_FC']['n'])+' | '+' | '.join(f"{z[k]['mIoU_percent']:.6f}" for k in ['original_source','prepared_frozen','lm_FC','direct_FC','direct_four','joint3_FC','joint3_shuffle_FC'])+' |')
 lines+=['','## 开发部分与扩大部分分开看','','原768条用于选择这一候选；其余17852条是本算法未用于开发输出的扩大部分，但数据本身存在R98历史暴露，仍不能称新独立测试。以下按方向等权列主要差值，不让复用的开发正例掩盖扩大部分。','', '| 部分 | 对照 | 等权Δ pp | 改善/恶化 |','|---|---|---:|---:|']
 sp=rd(S/'STRATUM_PAIRED_ROWS.json')
 for st in ['R99_DEV128','EXPANSION_ONLY']:
  for control in ['prepared_frozen','lm_FC','direct_FC','direct_four','joint3_shuffle_FC']:
   z=[x for x in sp if x['stratum']==st and x['control']==control];lines.append(f"| {st} | {control} | {sum(x['gain_pp'] for x in z)/len(z):+.6f} | {sum(x['better'] for x in z)}/{sum(x['worse'] for x in z)} |")
 lines+=['','## 全部平台和读出','','| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |','|---|---|---:|---:|---:|']
 for x in agg:lines.append(f"| {x['group']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 lines+=['','## 全部预设成对比较','','| 方向 | 方法 | 对照 | Δ pp | 改善/恶化/不变 |','|---|---|---|---:|---:|']
 for x in pairs:lines.append(f"| {x['source']}→{x['target']} | {x['method']} | {x['control']} | {x['gain_pp']:+.6f} | {x['better']}/{x['worse']}/{x['same']} |")
 lines+=['','## 辅助诊断与范围','','对R99全部768输入的CPU头重放显示：三视图拟合减小A/B/C分数目标误差，但7条改善和3条恶化的C拟合误差都下降；22次FC选框变化中18次没有改变F或C各自Top1。CPU/GPU概率最大差约1.21e-6且无Top1差，数值差与初始过严格阈值失败均保留，不冒称逐位一致。全部正式成绩仍来自原CUDA封存预测。','','另对R98全部18620条初始误差分解：源Top10承载38.68%概率，仅占7.50%等权分数误差；这不是参数梯度份额的结论。按相同头Jacobian实际投影，分数L2与分布KL初始参数梯度在768开发输入平均余弦.5578。这里只建立度量差异，KL未在本轮执行或选优，不能声称其必然更好。相关诊断与已实际读取的Hinton原文sec2.1见diagnostics；没有引入外部teacher。','','本轮按hash复用R98固定G缓存（真实/错配三视图分数）和原768预测，再重算其余输入的三视图头拟合。所有新视图正确绑定基础分数与原缓存核对，source权重及原始框均固定。执行耗时包含缓存复用与多个控制，不是完整端到端部署延迟；原始G缓存生成也有成本。','','## 判断','',json.dumps(decision,indent=2),'', '候选方法必须超过同信息直接读出及错配控制，不能只超过裸source。F和C是预设诊断端点，不在看到全量标签后替换主端点。优化下降与非零参数不是任务成功。若开发优势未保持，降低该配方优先级；不要无依据增加步数或按少数案例补门。下一步按夜间状态和剩余时间选择最小区分实验，保持无外部专家、无源训练、FINAL受保护；2026-11-16北京时间09:30停止新增模型、训练和评分。']
 nondev=[x for x in sp if x['stratum']=='EXPANSION_ONLY' and x['control']=='lm_FC'];nondev_gain=sum(x['gain_pp'] for x in nondev)/len(nondev);lines+=['',f"固定扩展裁决：开发768条相对旧LM FC为+0.205804pp，扩大17852条为{nondev_gain:+.6f}pp，完整矩阵相对自身冻结{decision['gains_pp']['prepared_frozen']:+.6f}pp。开发优势未保持，因此当前三视图L2配方不采纳为稳定TTA。保留全部有益与有害变化；不通过改主端点、删Quad方向或增加步数挽救。下一项若执行，应单独锁定并比较候选分布度量，与本固定矩阵区分。"]
 (H/'REPORT.md').write_text('\n'.join(lines)+'\n');print(decision)
if __name__=='__main__':main()
