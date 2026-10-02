"""Publish anonymous P0 measurements without media, labels, weights or tensors."""
import sys,time,shutil,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_directional_common_v1 import *
from scripts.readout_tastvg_directional_v1 import paired
PUBLIC=Path('/home/wwww/visual-grounding-public-A')
REL='results/tastvg_directional_preference/2026-10-02'
def put(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def pp(m):return f"{100*m['mean']:+.4f} [{100*m['ci95'][0]:+.4f}, {100*m['ci95'][1]:+.4f}]"
def run():
 assert read(BASE/'ROOT_READOUT.json')['status']=='completed_pending_root_public_audit'
 dst=PUBLIC/REL;costs={};audits={};total=0
 report=['# P0：保留同状态 RKL 幅度，替换参数更新方向','',
  '本轮实际执行 A/E/F 三臂：A 原 rank-RKL SGD；E 四组正负 probe 的 rank contrast 加权方向；F 最大绝对 rank contrast 的单轴方向。E/F 在自己的当前状态计算原 RKL 梯度，以 lr×全参数梯度范数确定位移幅度，不复制旧 A 轨迹的范数。原 teacher 温度和几何 KL 仍用于这一个幅度参照与诊断。','',
  'VidSTG 与 HC-STVG-v2 各原32个历史开发来源、一query/源、两序、clean+五类5%瞬时部署corruption、25%专家，每臂384到达，总2304。官方同域checkpoint、原Paper48像素和采样、原九个probe、原1792空间参数和两offset不变。Vid K1、HC K8、lr/temperature使用v3封存值。原Sa2VA、UniversalVTG与H缓存全部复用。当前预测在更新前封存；空间更新只服务后续arrival，时间Fast rerank不变。','',
  '全部六stream预测共同封存之后才读取已历史曝光的开发GT，完成独立参数算术/状态链/教师奖励/指标核验。这里的置信区间是10000次配对source-bootstrap，seed20261001；不把重复开发曝光结果写成fresh测试。','']
 decisions={}
 for ds in DATASETS:
  p=verify(ds);out=dst/ds;decision=read(BASE/ds/'DEVELOPMENT_DECISION.json');decisions[ds]=decision['arm']
  radius=read(BASE/ds/'A/SUPPORT.json')['spec']['radius']
  put(out/'CONFIG.json',dict(dataset=ds,sources=32,queries=32,orders=p['splits']['search']['orders'],conditions=p['conditions'],arrivals_per_arm=384,completed_arms=['A','E','F'],params=p['params'],probe_radius=radius,expert_fraction=.25,spatial_parameters=1792,historical_exposure=True,checkpoint_state_sha256=read(POOL/ds/'CAPTURE_BARRIER.json')['checkpoint_state_sha256']))
  for name in ['PAIRED.json','DEVELOPMENT_DECISION.json','GROSS_FUTURE_VS_A.json','DIRECTION_DIAGNOSIS.json','NEGATIVE_TAILS.json','CASES.json','SMOKE.json']:put(out/name,read(BASE/ds/name))
  report.extend([f'## {ds}','',f'封存参数：`{json.dumps(p["params"],ensure_ascii=False)}`。本轮开发下一步优先臂 **{decision["arm"]}**，只按未来corrupt nonexpert源宏平均，未晋升生产。','',
   '| 臂 | 全部corrupt ΔvIoU vs Frozen (pp, 95%CI) | 未来nonexpert ΔvIoU vs Frozen | 未来 ΔvIoU vs A |','|---|---:|---:|---:|---:|'])
  for arm in ['A','E','F']:
   a=BASE/ds/arm;dest=out/arm
   for name in ['REQUEST.json','ROWS.json','SPATIAL_STEP_ROWS.json','SUMMARY.json','AUDIT.json','COMPLETION.json']:put(dest/name,read(a/name))
   s=read(a/'SUMMARY.json');whole=s['corruption']['all']['metrics']['delta_m_vIoU'];future=s['corruption']['nonexpert']['metrics']['delta_m_vIoU'];contrast='—' if arm=='A' else pp(paired(ds,arm)['corruption']['nonexpert']['metrics']['delta_m_vIoU'])
   report.append(f'| {arm} | {pp(whole)} | {pp(future)} | {contrast} |')
   costs[f'{ds}/{arm}']={**read(a/'AUDIT.json'),'optimization_backbone_forwards':0,'new_full_validation_forwards':0,'new_expert_inference':0};audits[f'{ds}/{arm}']=sha(a/'AUDIT.json');total+=384
  report.extend(['','| 臂 | 相对A未来gross gain/loss (pp) | 相对A未来arrival损害>5pp | 首步唯一有益teacher首选但更新GT受损 | 无rank方向/eligible步 |','|---|---:|---:|---:|---:|'])
  diag=read(BASE/ds/'DIRECTION_DIAGNOSIS.json');gross=read(BASE/ds/'GROSS_FUTURE_VS_A.json');tail=read(BASE/ds/'NEGATIVE_TAILS.json')
  for arm in ['A','E','F']:
   d=diag[arm]['corruption'];g=gross[arm]['metrics'];report.append(f'| {arm} | {100*g["gross_gain_vs_A"]["mean"]:.4f} / {100*g["gross_loss_vs_A"]["mean"]:.4f} | {tail[arm]["harm_over5pp"]} / {tail[arm]["arrivals"]} | {d["unique_useful_top_harm_first"]} / {d["first_eligible_steps"]} | {d["zero_direction_steps"]} / {d["eligible_steps"]} |')
  report.extend(['',
   '几何RKL与偏好方向的余弦、四轴投影、实际/参照位移、probe半径比、GT配对方向正反/中性、每一步及整个arrival固定时间空间效用、loss下降但GT受损、正负cases和完整计算量均公开。GT-neutral的正负probe效用并列不计作critic误选。','',
   f'绝对probe半径 {radius:.12g}。零rank方向时E/F严格no-op，此时无法满足非零RKL幅度匹配，单独报告；A仍保留旧RKL的熵项行为。E/F轨迹分叉后其RKL参照梯度也不同，不能将此实验称为跨臂每个arrival完全相同的实际位移分布。',''])
 observations=[]
 for ds in DATASETS:
  ct=read(BASE/ds/'PAIRED.json');observations.append(ds+'：E−A '+pp(ct['E']['corruption']['nonexpert']['metrics']['delta_m_vIoU'])+' pp，F−A '+pp(ct['F']['corruption']['nonexpert']['metrics']['delta_m_vIoU'])+' pp，开发优先臂 '+decisions[ds])
 report[2:2]=['**实际结果（future corrupt nonexpert）**：'+'；'.join(observations)+'。区间为95%配对source-bootstrap；方法决定限于此曝光开发面板。','']
 movement=read(BASE/'FUNCTIONAL_MOVEMENT_AUDITED.json')['datasets']
 report.extend(['## 为什么减少坏步仍未获得未来收益','', 'Vid首步唯一有益teacher首选却更新GT受损，A为3次、E/F为0次；HC全部步GT受损A为167次，E/F为78/119次。坏步减少也伴随有益变化减弱，不能只用坏例计数决定方法。','', '| 数据集/臂 | 每step平均fixed-time GT ΔvIoU (pp) | 平均normalized框坐标绝对变化 | 平均参数step L2 |','|---|---:|---:|---:|'])
 for ds in DATASETS:
  dg=read(BASE/ds/'DIRECTION_DIAGNOSIS.json')
  for arm in ['A','E','F']:
   mv=movement[ds][arm]['statistics'];d=dg[arm]['corruption'];report.append(f'| {ds}/{arm} | {100*d["post_fixed_time_delta"]["mean"]:+.6f} | {mv["box_mean_absolute_coordinate_change"]["mean"]:.8f} | {mv["actual_parameter_step"]["mean"]:.8f} |')
 report.extend(['', '同状态的参数位移匹配，不等于框输出变化匹配。新方向平均框坐标变化比A小约13–18倍，原RKL范数仅约6.6–6.7%在四轴投影中；符合四轴子空间函数敏感性较弱的解释，但该posthoc观察没有单独建立因果。它阻止将此负结果解读为已排除所有direction机制，亦不证明事件参考采样已成为唯一瓶颈。框变化是封存normalized cxcywh的逐帧四坐标平均绝对差，无额外forward/GT读取。FUNCTIONAL_STEP_ROWS与独立标量核验可重建全部mean/median。',''])
 report.extend(['## 解释边界与成本','',
  'E/F的改动同时将更新约束到原四个probe轴；因此胜出可以支持“参数空间偏好方向有用”，不能单独证明所有坏例唯一来自KL的方向翻译。F按专家正负轴rank contrast选轴，不是teacher/student高disagreement选择。E合成方向并没有被专家直接评价；F沿轴的执行长度也未必等于probe长度，均无GT安全保证。','',
  '负均值且区间跨零只表明此锁定实现没有建立优于A的证据，不排除其他方向估计/幅度机制；正开发均值也不自动成为论文全量结论。保留原A正控、全部阴性和负例，不追加温度/学习率/预算搜索，不按本轮GT修改线上gate。','',
  '本轮零新backbone forward、零新expert inference。每个inner step复用缓存H，经TA-STVG后半段产生九候选、原RKL backward与post诊断；E/F没有省掉RKL的幅度计算。所有实际replay/provider/backward计数和worker wall time见COSTS.json，wall包含模型载入与保存/检查，不称纯GPU kernel时间。原批次的96次full parity forward不计入本轮新执行。','',
  '语言模型原论文仅提供方向信息检查的动机，本方法是直接参数偏好执行，不是原生token policy OPD的复现：[KL-free OPD](https://arxiv.org/abs/2609.33791)。[Decomposed OPD](https://proceedings.mlr.press/v306/yoon26f.html)研究VLM视觉/语言梯度分量，不提供本STVG配置的理论保证。','',
  '事件条件专家取帧与定位质量critic是尚未执行的后续问题，本轮没有自动加新专家输入、时间参数学习、模型、baseline或全query队列。生产注册保持原值。',''])
 put(dst/'COSTS.json',costs)
 text='\n'.join(report)+'\n';(ROOT/'docs/TA_DIRECTIONAL_PREFERENCE_REVIEW.md').write_text(text);(PUBLIC/'docs/TA_DIRECTIONAL_PREFERENCE_REVIEW.md').write_text(text);(dst/'README.md').write_text(text)
 code=set(read(BASE/'RUNTIME_LOCK.json')['pins'])|set(read(BASE/'SCORING_RUNTIME_LOCK.json')['pins'])|{'scripts/export_tastvg_directional_v1.py','scripts/audit_tastvg_directional_public_v1.py','scripts/draw_tastvg_directional_v1.py','scripts/readout_tastvg_directional_v1.py','scripts/finish_tastvg_directional_v1.py','scripts/manifest_tastvg_directional_public_v1.py','scripts/verify_tastvg_directional_remote_v1.py','scripts/diagnose_tastvg_directional_functional_v1.py'}
 for rel in sorted(code):
  target=PUBLIC/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
 for name in ['RUNTIME_LOCK.json','SCORING_RUNTIME_LOCK.json','SMOKE_ROOT_ACCEPTANCE.json','ROOT_READOUT.json','ROOT_REQUEST_VERIFICATION.json','CPU_TESTS.json','LITERATURE_SOURCES.json','PUBLIC_SOURCE_PREFLIGHT.json','FUNCTIONAL_MOVEMENT.json','FUNCTIONAL_MOVEMENT_AUDITED.json','FUNCTIONAL_STEP_ROWS.json']:
  put(dst/name,read(BASE/name))
 put(dst/'ENGINEERING_PRELAUNCH.json',dict(cpu_test_001=read(BASE/'recovery/cpu_test_001/ERROR.json'),scope='Test harness reused overwritten local vector; repaired before all model smoke; method unchanged.'))
 put(BASE/'ROOT_EXPORT.json',dict(status='exported_pending_public_audit_and_remote',total_arrivals=total,result_relative_path=REL,code_paths=sorted(code),audits=audits,development_decisions=decisions,time=time.time()))
 print('Exported',total,'anonymous P0 arrivals')
if __name__=='__main__':run()
