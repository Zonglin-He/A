"""Allowlisted anonymous A/G evidence; never publish private tensors or inputs."""
import sys,time,shutil,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_selected_rollout_common_v1 import *
PUBLIC=Path('/home/wwww/visual-grounding-public-A');REL='results/tastvg_selected_rollout/2026-10-02'
def put(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def pp(m):return f"{100*m['mean']:+.4f} [{100*m['ci95'][0]:+.4f}, {100*m['ci95'][1]:+.4f}]"
def run():
 assert read(BASE/'ROOT_READOUT.json')['status']=='completed_pending_root_public_audit'
 dst=PUBLIC/REL;costs={};audits={};total=0;decisions={};obs=[]
 for ds in DATASETS:
  m=read(BASE/ds/'PAIRED.json')['G']['corruption']['nonexpert']['metrics']['delta_m_vIoU'];obs.append(ds+' G−A '+pp(m)+' pp')
 report=['# Selected Student Rollout：完整参数梯度实现专家选中的学生 tube','',
  '**实际结果（未来 corrupt nonexpert）**：'+'；'.join(obs)+'。区间为95%配对source-bootstrap；裁决限于本次曝光开发面板。','',
  '本轮仅A/G。A原Rank-RKL；G由专家奖励first-argmax选一个student-generated tube，用停止梯度的该tube与central之间native L1+GIoU求完整1792参数VJP。G的更新长度仍取自己当前状态的lr×原RKL梯度范数，方向换成负selected-target梯度；不沿probe轴，不复制旧A轨迹范数。central被选中或exact flat时G精确no-op。','',
  '各数据集原32历史开发来源、一query/源、双序、clean+五类5%瞬时部署corruption、25%专家；384到达/臂/数据集，共1536。官方同域checkpoint、原Paper48帧/像素、两offset、H与Sa2VA/UniversalVTG缓存、九probe、1792空间参数不变。Vid K1/HC K8和各自封存lr/teacher温度保留；空间state流内继承，当前输出先于更新，时间Fast rerank不变。','',
  '全部四stream预测共同封存后才CPU读已曝光GT。10000次配对source-bootstrap，seed20261001；不当作fresh全量。A每个384到达逐值复现前轮A。first exact argmax处理并列，flat精确spread=0；RKL旧rank tie tolerance1e-12未改，二者含义分开。','']
 for ds in DATASETS:
  p=verify(ds);out=dst/ds;decision=read(BASE/ds/'DEVELOPMENT_DECISION.json');decisions[ds]=decision['arm'];radius=read(BASE/ds/'A/SUPPORT.json')['spec']['radius']
  put(out/'CONFIG.json',dict(dataset=ds,sources=32,queries=32,orders=p['splits']['search']['orders'],conditions=p['conditions'],arrivals_per_arm=384,completed_arms=['A','G'],params=p['params'],probe_radius=radius,expert_fraction=.25,spatial_parameters=1792,historical_exposure=True,checkpoint_state_sha256=read(POOL/ds/'CAPTURE_BARRIER.json')['checkpoint_state_sha256']))
  for name in ['PAIRED.json','DEVELOPMENT_DECISION.json','GROSS_FUTURE_VS_A.json','DIRECTION_DIAGNOSIS.json','NEGATIVE_TAILS.json','CASES.json','SMOKE.json','ROOT_PREDICTION_READBACK.json']:put(out/name,read(BASE/ds/name))
  report.extend([f'## {ds}','',f'封存参数：`{json.dumps(p["params"],ensure_ascii=False)}`。本轮下一步开发优先臂 **{decision["arm"]}**，只按预登记未来nonexpert均值规则，未晋升生产。','',
   '| 臂 | 全部corrupt ΔvIoU vs Frozen (pp,95%CI) | 未来nonexpert ΔvIoU vs Frozen | 未来 ΔvIoU vs A |','|---|---:|---:|---:|'])
  for arm in ['A','G']:
   a=BASE/ds/arm;dest=out/arm
   for name in ['REQUEST.json','ROWS.json','SPATIAL_STEP_ROWS.json','SUMMARY.json','AUDIT.json','COMPLETION.json']:put(dest/name,read(a/name))
   s=read(a/'SUMMARY.json');contrast='—' if arm=='A' else pp(read(BASE/ds/'PAIRED.json')[arm]['corruption']['nonexpert']['metrics']['delta_m_vIoU'])
   report.append(f'| {arm} | {pp(s["corruption"]["all"]["metrics"]["delta_m_vIoU"])} | {pp(s["corruption"]["nonexpert"]["metrics"]["delta_m_vIoU"])} | {contrast} |')
   audit=read(a/'AUDIT.json');costs[f'{ds}/{arm}']={**audit,'optimization_backbone_forwards':0,'new_full_validation_forwards':0,'new_expert_inference':0};audits[f'{ds}/{arm}']=sha(a/'AUDIT.json');total+=384
  dg=read(BASE/ds/'DIRECTION_DIAGNOSIS.json');gross=read(BASE/ds/'GROSS_FUTURE_VS_A.json');tail=read(BASE/ds/'NEGATIVE_TAILS.json')
  report.extend(['','| 臂 | 相对A未来gross gain/loss (pp) | 相对Frozen未来gross gain/loss (pp) | 相对A未来arrival损害>5pp | 首步唯一有益首选但更新受损 |','|---|---:|---:|---:|---:|'])
  for arm in ['A','G']:
   d=dg[arm]['corruption'];g=gross[arm]['metrics'];report.append(f'| {arm} | {100*g["gross_gain_vs_A"]["mean"]:.4f}/{100*g["gross_loss_vs_A"]["mean"]:.4f} | {100*g["gross_gain_vs_Frozen"]["mean"]:.4f}/{100*g["gross_loss_vs_Frozen"]["mean"]:.4f} | {tail[arm]["harm_over5pp"]}/{tail[arm]["arrivals"]} | {d["unique_useful_top_harm_first"]}/{d["first_eligible_steps"]} |')
  report.extend(['','| 臂 | 平均框坐标绝对变化 | output-space cosine平均/正/负/未定义 | selected距离下降/eligible | 平均step固定时间GT ΔvIoU(pp) |','|---|---:|---:|---:|---:|'])
  for arm in ['A','G']:
   d=dg[arm]['corruption'];cos=d['output_cosine'];cv='null' if cos is None else f'{cos["mean"]:+.6f}'
   report.append(f'| {arm} | {d["functional_box_movement"]["mean"]:.8f} | {cv}/{d["cosine_positive"]}/{d["cosine_negative"]}/{d["cosine_undefined"]} | {d["selected_distance_decreases"]}/{d["eligible_steps"]} | {100*d["post_fixed_time_delta"]["mean"]:+.6f} |')
  report.extend(['',f'G no-op原因计数：`{json.dumps(dg["G"]["corruption"]["no_op_reasons"],ensure_ascii=False)}`；幅度可匹配step {dg["G"]["corruption"]["matched_steps"]}/{dg["G"]["corruption"]["eligible_steps"]}。零output或零target方向余弦记null并另报，不置零混进均值。HC八步都保存动态候选/梯度/目标，非空no-op不早停。',''])
 report.extend(['## 本轮机制判断','',
  '两集开发优先仍是A。G−A在Vid为负均值且95%区间跨0；HC为负均值且本面板95%配对区间在0以下。不能将曝光开发的区间当作fresh验证或排除所有selected-rollout方法。','',
  'Vid G64次有效移动全部positive cosine（平均.73056），但只7次缩短selected几何距离，57次增大；A为5次下降。方向为正仍可能超过近邻目标或沿非欧氏损失路径移动，不能由cosine单独认定已实现有益的tube纠正。G平均局部GT gain降到+.41028pp（A+.93770），局部gross loss .55201pp（A .18354），并新增1个>5pp局部负步；future相对A4/240到达损害>5pp。','',
  'HC G582/600步positive cosine、570/600步selected距离下降（A559/600），但固定时间GT受损167→261，selected目标GT有益而执行有害19→104；首步唯一有益首选却更新受损2→12。selected距离下降同时GT损害143→242。框运动并未像E/F缩小13–18倍，平均movement .00112855对A .00108483。因此本结果不支持把问题只归结为四轴低敏感性或无法移动decoder输出；匹配RKL幅度、完整clip几何与稀疏事件支持、critic正确性/共享state转移仍是竞争解释。','',
  '上述局部计数来自各自不同演化轨迹，不是同一teacher实例的因果替换。若下一步研究event-conditioned证据，需另锁专家取帧/时间支持及对照；本轮没有据GT选择在线frame、gate或自动新作业。','',
  'primary未来子集各240到达，覆盖Vid32来源、HC30来源；HC另2来源在两序均处于expert位置，不进future主指标，仍保留在全组结果。',''])
 report.extend(['## 测量含义与判断边界','',
  'box movement为封存normalized cxcywh逐帧四坐标的FP64平均绝对差；cosine比较post−central与selected−central。独立CPU重建全部FP32参数执行、raw rewards/ranks、两个几何目标、functional scalars、状态链与dense指标。selected D下降/方向为正仍不等于GT改善；固定时间GT只解释局部动作，不替换当前pre-update线上输出。','',
  'G保留专家选择的student hard target，不直接拟合Sa2VA box，但仍是自生成target学习。两条轨迹分叉后自己的RKL范数不同，central/flat no-op又改变更新次数；胜出只能支持这整个执行规则，不能唯一归因为方向或声称跨臂每step长度完全相同。负结果也不排除其他尺度、函数参数化或证据支持方式。','',
  '完整1792D VJP避免四轴限制，但有限步长、L1/GIoU非光滑、native routing及critic误选仍可能使输出方向和任务效用分离。坏步减少需与gross gain/loss和负尾共同看；不以单个坏例或条件GT分组构造线上阈值。','',
  '两集均为原多轮历史曝光32来源的小开发面板，未加独立确认或全量。clean/control及所有arrival匿名结果、正负cases、K8每步和整个arrival效用均保留。优先臂选择按strict primary mean（1e-12 tie保留A），不等于统计显著性或论文普适结论。','',
  'G仍计算RKL counterfactual以及eligible noncentral selected-target额外backward，没有backward效率宣称。零新backbone/专家推理；完整replay/provider/两类backward和worker wall见COSTS.json。wall含载入、I/O和检查，不称纯GPU kernel。','',
  '无GT smoke首轮diagnostic日志先FP32相减产生独立FP64核对差异，保存functional_logging_001并仅改日志精度，正式预测前revision001重新pin，模型/目标/更新无改变。','',
  '文献联系仅为机制动机：[KL-free OPD](https://arxiv.org/abs/2609.33791)分析LLM token更新方向；[Best-of-N teacher rollout](https://arxiv.org/abs/2605.09725)使用teacher轨迹correctness优先选择并含GT recovery。本轮student tube selector不继承其保证，也不使用GT在线过滤。','',
  '本轮不自动启动事件条件Sa2VA取帧、top2、时间head、critic训练、超参grid、新模型/baseline或旧fullquery队列；保持生产注册与所有旧证据。',''])
 put(dst/'COSTS.json',costs);txt='\n'.join(report)+'\n';(ROOT/'docs/TA_SELECTED_ROLLOUT_REVIEW.md').write_text(txt);(PUBLIC/'docs/TA_SELECTED_ROLLOUT_REVIEW.md').write_text(txt);(dst/'README.md').write_text(txt)
 code=set(read(BASE/'RUNTIME_LOCK.json')['pins'])|set(read(BASE/'SCORING_RUNTIME_LOCK.json')['pins'])|{f'scripts/{stem}_tastvg_selected_rollout_v1.py' for stem in ['export','readout','draw','finish']}
 code.discard('scripts/verify_remote_tastvg_selected_rollout_v1.py');code|={'scripts/manifest_tastvg_selected_rollout_public_v1.py','scripts/audit_tastvg_selected_rollout_public_v1.py','scripts/verify_tastvg_selected_rollout_remote_v1.py','scripts/audit_tastvg_selected_rollout_predictions_v1.py','scripts/test_tastvg_selected_rollout_statistics_v1.py'}
 for rel in sorted(code):target=PUBLIC/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
 for name in ['RUNTIME_LOCK.json','RUNTIME_REVISION001.json','SCORING_RUNTIME_LOCK.json','SMOKE_ROOT_ACCEPTANCE.json','ROOT_READOUT.json','ROOT_REQUEST_VERIFICATION.json','CPU_TESTS.json','STATISTICS_CPU_TEST.json','LITERATURE_SOURCES.json','PUBLIC_SOURCE_PREFLIGHT.json']:
  put(dst/name,read(BASE/name))
 put(dst/'ENGINEERING_PRELAUNCH.json',read(BASE/'recovery/functional_logging_001/ERROR.json'))
 put(dst/'ENGINEERING_CLOSEOUT.json',{k:read(BASE/'recovery'/k/'ERROR.json') for k in ['export_path_002','public_audit_shadow_003']})
 put(BASE/'ROOT_EXPORT.json',dict(status='exported_pending_public_audit_and_remote',total_arrivals=total,result_relative_path=REL,code_paths=sorted(code),audits=audits,development_decisions=decisions,time=time.time()))
 print('Exported',total,'anonymous A/G arrivals')
if __name__=='__main__':run()
