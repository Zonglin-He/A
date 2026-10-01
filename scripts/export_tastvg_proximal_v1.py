"""Anonymous completed outputs and a reviewer-readable development report."""
import sys,time,shutil,json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_proximal_common_v1 import *
PUBLIC=Path('/home/wwww/visual-grounding-public-A')
REL='results/tastvg_proximal_ablation/2026-10-02'
def export_write(path,value):
 path=Path(path)
 assert path.is_relative_to(PUBLIC/REL) or path==BASE/'ROOT_EXPORT.json'
 path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.export-tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
 tmp.replace(path)
def pp(metric):
 m=metric;return f"{100*m['mean']:+.4f} [{100*m['ci95'][0]:+.4f}, {100*m['ci95'][1]:+.4f}]"
def run():
 import torch,numpy as np
 from scripts.readout_tastvg_proximal_v1 import paired
 from scripts.score_tastvg_best_quick_v1 import source_summary
 assert read(BASE/'FOLLOWTHROUGH_STATUS.json')['status']=='completed_pending_root_audit_publication'
 dst=PUBLIC/REL;report=['# 近端目标、有界空间更新与原生时间头：匹配开发实验','',
  '结论：本轮空间B/C/D的未来nonexpert平均收益均未超过A，所有对应配对95%置信区间跨零；原生时间头T的未来vIoU变化接近零。两数据集保留原封存配置A，不接入新目标、位移预算或时间头更新。固定混合对照的触发条件两集均为false，实际未运行E_fixed。这个结论只针对本轮已锁起始配置，不外推为所有近端目标或时间参数学习无效。','',
  '原 VidSTG / HC-STVG-v2 各32个调参来源，一query/源、原双序、clean与五类5%瞬时corruption、25%专家；每臂384到达。全部历史曝光，只是开发机制对照，不是新的独立测试或全query评估。原官方同域checkpoint、原Paper48采样与像素、两offset、缓存专家及H均相同。当前输出在arrival更新前封存，空间状态和时间head用于未来到达。','',
  'A=原rank teacher+SGD；B=证据强度近端teacher+SGD；C=原rank teacher+整个arrival位移上限；D=B+C。有界半径是绝对probe半径的0.1，不是0.1乘相对rho，也不是每个inner step重新领取预算。B/D精确flat reward为no-op。','',
  's_ref仅由A精确复现v3selected流后记录的首步raw reward spread非零中位数得到，未读GT。目标在每步当前student几何分布与rank teacher间做概率空间算术混合，随后detach；post-step KL仍对同一个冻结目标计算。p仍是geometry compatibility，不能称为原生tube policy。','',
  'TOP-D提供概率空间proximal teacher的构造启发；本实验参数L2投影并非原文的policy-ratio-clipped优化器，不援引其语言模型理论作为STVG安全保证。[TOP-D原文§3](https://arxiv.org/html/2607.04751#S3)','']
 costs={};rootchecks={};total=0
 for ds in DATASETS:
  p=verify(ds);selected=read(BASE/ds/'SPATIAL_SELECTION.json')['arm'];arms=['A','B','C','D']+(['E_fixed'] if (BASE/ds/'E_fixed/COMPLETION.json').exists() else [])+['T']
  out=dst/ds;out.mkdir(parents=True,exist_ok=True);config=dict(dataset=ds,sources=32,queries=32,orders=p['splits']['search']['orders'],conditions=p['conditions'],arrivals_per_arm=384,completed_arms=arms,params=p['params'],expert_fraction=.25,spatial_parameters=1792,temporal_parameters=514,historical_exposure=True,checkpoint_state_sha256=read(POOL/ds/'CAPTURE_BARRIER.json')['checkpoint_state_sha256'])
  export_write(out/'CONFIG.json',config)
  for name in ['CALIBRATION.json','TEMPORAL_CALIBRATION.json','CONTROL_TRIGGER.json','SPATIAL_SELECTION.json','PAIRED_SPATIAL.json']:
   j=read(BASE/ds/name)
   if name=='CALIBRATION.json':j={k:v for k,v in j.items() if k!='receipts'}
   export_write(out/name,j)
  if (BASE/ds/'PAIRED_FIXED_CONTROL.json').exists():shutil.copy2(BASE/ds/'PAIRED_FIXED_CONTROL.json',out/'PAIRED_FIXED_CONTROL.json')
  tp=paired(ds,'T',selected);export_write(out/'PAIRED_NATIVE_TEMPORAL.json',tp)
  report.extend([f'## {ds}','',f'封存参数：`{json.dumps(p["params"],ensure_ascii=False)}`。空间选择 `{selected}`，仅按四臂未来nonexpert腐蚀vIoU；E_fixed为机制对照，不参与四臂选择。','',
   '| 臂 | 全部corrupt ΔvIoU vs Frozen (pp, 95%CI) | 未来nonexpert ΔvIoU vs Frozen | 未来 ΔvIoU vs A |','|---|---:|---:|---:|'])
  for arm in arms:
   a=BASE/ds/arm;dest=out/arm;dest.mkdir(exist_ok=True)
   for name in ['REQUEST.json','ROWS.json','SPATIAL_STEP_ROWS.json','SUMMARY.json','AUDIT.json','COMPLETION.json']:
    shutil.copy2(a/name,dest/name)
   if arm=='T':
    for name in ['TEMPORAL_STEP_ROWS.json','TEMPORAL_AUDIT.json','HEAD_SUPPORT.json','SMOKE.json']:shutil.copy2(a/name,dest/name)
   s=read(a/'SUMMARY.json');contrast=None if arm=='A' else paired(ds,arm)
   whole=s['corruption']['all']['metrics']['delta_m_vIoU'];future=s['corruption']['nonexpert']['metrics']['delta_m_vIoU'];pairtext='—' if arm=='A' else pp(contrast['corruption']['nonexpert']['metrics']['delta_m_vIoU'])
   report.append(f'| {arm} | {pp(whole)} | {pp(future)} | {pairtext} |')
   normrows=[]
   for cond in p['conditions']:
    for order,seq in p['splits']['search']['orders'].items():
     for at in range(0,len(seq),4):
      xx=load(a/'online'/cond/order/f'{at:05}.pt')
      for step_index,st in enumerate(xx['update_steps']):
       delta=float(torch.sqrt(sum((st['post_state'][n]-v).double().square().sum() for n,v in st['pre_state'].items())))
       uu=st['update'];normrows.append(dict(condition=cond,order=order,arrival=at,step=step_index,actual_step_L2=delta,eligible=uu is not None,strength=float(uu['strength']) if uu else 0.))
   export_write(dest/'ACTUAL_STEP_NORMS.json',normrows)
   export_write(dest/'STEP_STATISTICS.json',dict(eligible_steps=sum(r['eligible'] for r in normrows),strength_eligible_mean=float(np.mean([r['strength'] for r in normrows if r['eligible']])) if any(r['eligible'] for r in normrows) else 0.,actual_step_L2_mean=float(np.mean([r['actual_step_L2'] for r in normrows])),actual_step_L2_max=float(max(r['actual_step_L2'] for r in normrows))))
   costs[f'{ds}/{arm}']={**read(a/'AUDIT.json'),'validation_full_input_checks':24 if arm=='A' else 0,'validation_full_offset_forwards':48 if arm=='A' else 0,'optimization_backbone_forwards':0,'new_expert_inference':0};rootchecks[f'{ds}/{arm}']=sha(a/'AUDIT.json');total+=384
  report.extend(['',f'原生时间头 T 相对选定空间 `{selected}`：未来corrupt ΔvIoU **{pp(tp["corruption"]["nonexpert"]["metrics"]["delta_m_vIoU"])} pp**；ΔtIoU **{pp(tp["corruption"]["nonexpert"]["metrics"]["delta_m_tIoU"])} pp**。保留全部区间和负例，不按测试表现继续调这个head。','',
   'T只更新原生start/end最后一层514参数，采用两offset合法span概率的乘积在保留候选pair上再归一化，明确是受限native span分布。固定一步SGD、原封存lr/teacher温度，独立无GT时间reward spread尺度、近端目标，初始source-head L2的0.005预算。预算是开发起点，不是专家已探测的head区域。两数据集T预测先共同封存再GT评分。空间参数状态与框全部逐值匹配相应无T基线。','',
   '| 臂 | corrupt首步“唯一有益首选但实际更新GT受损” | loss下降且GT受损（全部步） | flat精确no-op | projection触发/有效步 |','|---|---:|---:|---:|---:|'])
  for arm in arms:
   st=[r for r in read(BASE/ds/arm/'SPATIAL_STEP_ROWS.json') if r['condition']!='clean'];first=[r for r in st if r['step']==0];valid=[r for r in st if 'projection_factor' in r]
   report.append(f'| {arm} | {sum(r["unique_useful_top_harm"] for r in first)} / {len(first)} | {sum(r["loss_decreased"] and r["delta_update"]< -1e-12 for r in st)} / {len(st)} | {sum(r.get("flat_noop",False) for r in st)} | {sum(r["projection_factor"]<1 for r in valid)} / {len(valid)} |')
  diagnosis={};cases={};base_rows=read(BASE/ds/'A/ROWS.json')
  for arm in arms:
   rr=read(BASE/ds/arm/'ROWS.json');pr=[]
   for x,y in zip(rr,base_rows):pr.append({**x,'delta_vs_A':x['Ours_m_vIoU']-y['Ours_m_vIoU'],'gross_gain_vs_A':max(0,x['Ours_m_vIoU']-y['Ours_m_vIoU']),'gross_loss_vs_A':max(0,y['Ours_m_vIoU']-x['Ours_m_vIoU'])})
   ss=[r for r in pr if r['condition']!='clean' and not r['expert_scheduled']];diagnosis[arm]=source_summary(ss,['delta_vs_A','gross_gain_vs_A','gross_loss_vs_A']);cases[arm]={'positive':sorted(ss,key=lambda r:-r['delta_vs_A'])[:8],'negative':sorted(ss,key=lambda r:r['delta_vs_A'])[:8]}
  export_write(out/'GROSS_FUTURE_VS_A.json',diagnosis);export_write(out/'CASES.json',cases)
  cal=read(BASE/ds/'CALIBRATION.json');cap=.1*read(BASE/ds/'A/SUPPORT.json')['spec']['radius']
  report.extend(['',f'无GT空间校准 s_ref={cal["s_ref"]:.17g}；整个arrival L2预算={cap:.17g}。预登记固定混合系数={cal["fixed_lambda"]:.17g}，但本轮不满足运行条件，未执行固定混合臂。',
   '上述“唯一有益首选但更新受损”计数包含任意大于数值容差的损害；四空间臂在本开发集的这类首步案例均没有超过5pp的更新损害。因此不能将计数变化写成已经修复旧快速面板的严重负例。',
   'gross gain/loss、全组/clean/expert/nonexpert、每步GT效用、实际step位移/强度统计、正负匿名cases和计算量都在对应JSON中；机制计数仅解释开发结果，不替代未来指标。固定混合系数来自预先冻结的A首步校准平均（空证据计0）；实际轨迹的eligible强度均值、梯度与位移仍不同，该对照不等于匹配实际更新预算。',''])
 export_write(dst/'COSTS.json',costs)
 report.extend(['## 验证与边界','',
  f'实际完成 {total} 次新到达（四臂3072、native head768；固定混合对照未触发）。没有新expert推理，优化步全部复用缓存H；A为验证真实回插额外执行每集24输入×2offset，共96次完整forward，包含backbone，不将这部分错误记为零。后半段replay、provider读取和backward逐臂计数保存。worker wall time含载入/保存/检查，不能称纯GPU kernel时间。',
  'A两数据集全部预测、梯度、pre/post状态精确复现封存v3 selected流。每个新臂独立核验raw reward、rank、geometry分布、近端target、实际SGD与arrival projection算术、状态链、双dense metric和预测先于GT；native head额外核验合法offset配对、受限概率、514参数更新及空间stream不变。公开匿名标量另用自包含auditor重建聚合/paired bootstrap/校准/选择规则。',
  '来源是反复曝光的原开发集，置信区间反映源间差异，不能作为未经曝光的确认结论。没有恢复全query、改变corruption或reference采样、加入GT线上gate、anchor/memory/EMA或外部baseline。生产方法未自动晋升。',''])
 text='\n'.join(report)+'\n';(ROOT/'docs/TA_PROXIMAL_ABLATION_REVIEW.md').write_text(text);(PUBLIC/'docs/TA_PROXIMAL_ABLATION_REVIEW.md').write_text(text);(dst/'README.md').write_text(text)
 code=set(read(BASE/'RUNTIME_LOCK.json')['pins'])|set(read(BASE/'FOLLOWTHROUGH_RUNTIME_LOCK.json')['pins'])|{'scripts/export_tastvg_proximal_v1.py','scripts/audit_tastvg_proximal_public_v1.py','docs/tastvg_proximal_ablation_v1/EXECUTION.md'}
 for rel in sorted(code):
  target=PUBLIC/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
 for name in ['BACKBONE_COST_CORRECTION.json','ENGINEERING_PRELAUNCH.json','GT_EXPOSURE_METADATA_CORRECTION.json']:
  if (BASE/name).exists():shutil.copy2(BASE/name,dst/name)
 export_write(BASE/'ROOT_EXPORT.json',dict(status='exported_pending_public_scalar_audit_and_remote',total_arrivals=total,result_relative_path=REL,code_paths=sorted(code),audit_files=rootchecks,time=time.time()))
 print('Exported',total,'anonymous arrivals to',REL)
if __name__=='__main__':run()
