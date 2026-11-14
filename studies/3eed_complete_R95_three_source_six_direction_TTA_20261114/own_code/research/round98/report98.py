"""Fixed complete-R95 six-direction results, never promote a secondary endpoint."""
from pathlib import Path
import json
R=Path(__file__).resolve().parents[2];H=R/'research/round98';S=H/'scores_attempt001';B=R/'crossplatform/round98';rd=lambda p:json.loads(Path(p).read_text())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['state'].startswith('PASS_')
 rows=rd(S/'ALL_METHOD_ROWS.json');agg=rd(S/'AGGREGATE_ROWS.json');pairs=rd(S/'ALL_PAIRED_ROWS.json')
 maincontrols=['prepared_frozen','frozen_FC','direct_FC','direct_four'];overall={x['arm']:x for x in agg if x['target_group']=='all_six_equal'};val=lambda a:overall[a]['mIoU_percent']
 gains={a:val('lm_FC')-val(a) for a in maincontrols};wins={a:sum(p['gain_pp']>0 for p in pairs if p['control']==a) for a in maincontrols}
 decision={'method':'user_confirmed_complete_R95_lm_FC','architecture':'R91_R76_full_disc','primary_mIoU_percent':val('lm_FC'),'six_direction_equal_gains_pp':gains,'positive_directions':wins,'beats_all_fixed_main_controls_in_macro':all(v>0 for v in gains.values()),'new_source_training':False,'target_hyperparameter_search':False,'FINAL_access':False,'independent_confirmation':False,'all_negatives_retained':True,'secondary_not_promoted':True,'adoption':'measured_fixed_recipe_only_not_robust_adoption'}
 (H/'DECISION.json').write_text(json.dumps(decision,indent=2)+'\n')
 lines=['# R98 — 用户确认的R95完整方法：三个source、六个跨平台方向','','本轮执行用户明确确认的“R95完整版本（当前登记）”。R97采用R91结构与旧物理恢复回溯，属于历史对照，不能代表本轮R95。保留R91双向RGB–LiDAR与主体／参照物接口，实际适应采用真实／错配绑定稠密差分监督、末端匹配线性投影LM更新和未参与拟合的C遮挡验证。没有重新训练source，没有新增算法、目标域选参或访问FINAL。','','## 主结果：原始A256不变','','所有数值为mIoU百分数，差值单位为百分点。每个方向保留全部资格清单，缺证据、UNKNOWN、零IoU与失败更新都计入分母。','', '| 方向 | n | 原始3EED | 准备后冻结 | 同视图零修正F/C | 同信息直接F/C | 直接四视图 | 完整R95 lm_FC | Δ冻结 | Δ直接F/C |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for src in ['waymo','drone','quad']:
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   z={x['arm']:x for x in rows if x['source']==src and x['target']==tgt};v=lambda a:z[a]['mIoU_percent']
   lines.append(f"| {src}→{tgt} | {z['lm_FC']['n']} | {v('original_source'):.6f} | {v('prepared_frozen'):.6f} | {v('frozen_FC'):.6f} | {v('direct_FC'):.6f} | {v('direct_four'):.6f} | {v('lm_FC'):.6f} | {v('lm_FC')-v('prepared_frozen'):+.6f} | {v('lm_FC')-v('direct_FC'):+.6f} |")
 lines+=['','## 固定实现与公平对照','','三个epoch100 source模型及其对应R91准备模块均原样复用R96/R97，不把Waymo模块加载给Drone或Quad。沿用R97逐表达式点采样、原始256框及原生query anchor；几何缓存按点采样／RGB／标定哈希验证后只读复用。参照物解析与接口保持原实现。','','完整R95：真实绑定下完整输入与遮挡76001/76002的基础分数，减去两种保持可见性的错配绑定在相同三视图的分数均值；对差分去候选均值，叠加真实绑定静态竞争残差，构造停止梯度的候选分布。仅更新末端64×288权重及64偏置，共18,496参数，其余冻结。每条从对应source状态恢复。','','沿用lm95原始数值程序：中心化log概率平方误差，最多10次重线性化，每次最多6次阻尼试探；实际误差下降且gain ratio>0.1才接受。主方法为不设范数帽的lm。mixture、shuffle与capped三种旧控制完整保留，不能根据本轮GT替换主方法。','','所有四种拟合与步长／停止决定完成后，才计算遮挡C（76003）。用拟合后的投影分别读取原完整F及C，主端点是其概率平均lm_FC。C是同一物理观测的未拟合遮挡，不是新场景、相机或独立实例；静态事实残差r0按原R95复用于C。因此本轮不能宣称跨场景泛化。','','直接F/C、直接四视图、混合直接读出与零差分F/C控制使用同样观测和固定证据但不改参数。R97概率／表示平均与旧物理回溯仅为历史参照；它们不能替代同R95信息的直接控制。另保留原R94 Adam/GN与四种拟合的F、C、FC读出，共29个预设实验臂。','','## 总体与平台汇总','','六方向等权，各目标平台的两个入向source等权；Waymo样本较多不会在主指标中掩盖其他平台。','', '| 范围 | 方法 | mIoU % | Acc25 % | Acc50 % |','|---|---|---:|---:|---:|']
 for x in agg:lines.append(f"| {x['target_group']} | {x['arm']} | {x['mIoU_percent']:.6f} | {x['Acc25_percent']:.6f} | {x['Acc50_percent']:.6f} |")
 lines+=['','## 全部预设成对比较','','| 方向 | 对照 | ΔmIoU pp | 改善/恶化/不变 | Acc25救回/破坏 |','|---|---|---:|---:|---:|']
 for p in pairs:lines.append(f"| {p['source']}→{p['target']} | {p['control']} | {p['gain_pp']:+.6f} | {p['better']}/{p['worse']}/{p['same']} | {p['Acc25_rescue']}/{p['Acc25_break']} |")
 lines+=['','## 结果解释：静态能力、证据与更新分别看','',f"完整系统相对原始3EED为{val('lm_FC')-val('original_source'):+.6f}pp，但准备后冻结模型本来就有{val('prepared_frozen')-val('original_source'):+.6f}pp。不能把全部系统差异称作TTA增量。F端点相对冻结F为{val('lm_F')-val('prepared_frozen'):+.6f}pp；未拟合C相对冻结C为{val('lm_C')-val('frozen_C'):+.6f}pp。原始F证据直接读出binding_direct为{val('binding_direct'):.6f}%，匹配更新F为{val('lm_F'):.6f}%。这些是预设端点的描述性比较，不将F替换为新的主方法。",'', '完整R95在四个方向相对冻结版改善，但Quad-only两个出向方向退化，抵消总体收益。参数更新相对直接F/C有很小的正差，尚未恢复到冻结双向模型水平；未拟合C也未显示稳定正增量。因此本轮不支持把当前配方认定为稳定的跨平台TTA。该结论限定于本方法、固定预算和既定清单，不推断所有TTA或双向结构无效。']
 lines+=['','## 优化、成本与审计','','| 方向 | fit | 初始/最终平均误差 | 平均接受步数 | 非零更新输入 |','|---|---|---:|---:|---:|']
 for x in rd(H/'FIT_SUMMARY_ROWS.json'):lines.append(f"| {x['source']}→{x['target']} | {x['fit']} | {x['initial_error_mean']:.8g}/{x['final_error_mean']:.8g} | {x['accepted_steps_mean']:.4f} | {x['nonzero_actual_parameter_updates']}/{x['n']} |")
 lines+=['','误差下降与非零参数变化只说明求解器执行，不代表选框改善。所有接受与拒绝的试探标量、最终四组权重和F/C表示留在本地；公开所有任务标量和每输入四组优化端点统计，不公开权重、特征、媒体、原句、坐标或原始ID。','', '| 方向 | 整体执行秒 | GPU峰值bytes |','|---|---:|---:|']
 for direction in rd(B/'MATRIX_SEALED.json')['directions']:
  x=rd(B/direction/'SEALED.json');lines.append(f"| {direction} | {x['seconds']:.3f} | {x['peak_GPU_bytes']} |")
 lines+=['','上表计入29个实验臂、诊断和存盘的整体耗时与峰值，不是主方法单独部署的延迟。评价沿用R96/R97与官方数据加载器逐位核对的GT7清单；通用3D交核验的是同一份几何，不引入另一套目标定义。']
 audit=rd(H/'NATIVE_AUDIT.json');lines+=['',f"历史R95四拟合十二端点及R94 GN参考实现精确重放通过。全部六方向预测先封存，之后核验全部文件哈希、差分证据、{audit['endpoints']}个拟合端点、所有接受条件标量及18个新原生完整重放案例，再解封任务评分。18个案例重算完整拟合过程；没有宣称独立重算每个输入的所有中间矩阵。主端点每条另以通用3D凸体交核对IoU，独立程序复核539,980个读出与全部成对负比较。",'', '首个部分预测尝试的旧R94 GN参考包装没有清理前一Adam对照的梯度；在任何目标评分前发现并修复，保留失败尝试后正式重跑。R95主监督与解析LM没有使用该梯度，重叠32条十二LM端点逐位相同，修复后的GN与原记录精确一致。','','Waymo目标含93场景，Drone2场景，Quad3场景；18,620个有向任务来自9,310条目标表达式，不是独立场景数量。本清单含历史暴露范围，不称为新独立测试，不据此给跨场景总体显著性结论。原100epoch来源、各source准备差异详见保留的R96/R97报告。','','## 裁决','',f"固定主方法六方向等权mIoU={val('lm_FC'):.6f}%。相对准备后冻结{gains['prepared_frozen']:+.6f}pp，相对同信息直接F/C {gains['direct_FC']:+.6f}pp，相对直接四视图{gains['direct_four']:+.6f}pp。同时超过全部固定主要控制={decision['beats_all_fixed_main_controls_in_macro']}。不把次要拟合、单方向最佳或错配控制升级为主方法；本轮给出固定完整R95的实际表现，未继续调参或访问FINAL。",'']
 (H/'REPORT.md').write_text('\n'.join(lines));print(decision)
if __name__=='__main__':main()
