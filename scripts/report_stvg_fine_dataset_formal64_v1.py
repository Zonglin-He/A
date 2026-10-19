"""All prespecified contrasts for a frozen dataset-specific prospective parent evaluation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_fine_dataset_formal64_v1 import D,P,CODE,verify
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha

def main():
 import numpy as np
 from scripts.score_stvg_joint_robust_correction32_v1 import paired
 p=verify();z=read(P/'RESULTS_PUBLIC.json');a=read(P/'ROOT_INDEPENDENT_AUDIT.json');inp=read(D/'INPUT_DEPENDENCY_AUDIT.json');assert a['status']==inp['status']=='pass';bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');gpu=read(D/'GPU_STOP.json');assert read(D/'FIT_SCHEDULE_GUARD.json')['status']=='pass';rows=[];groups={}
 for r in p['records']:
  x=load(D/'features'/(r['alias']+'.pt'));o=load(D/'predictions'/(r['alias']+'.pt'));E=x['E'].double();allowed=o['allowed'];fits={}
  for m,f in o['fits'].items():fits[m]=dict(updated_nodes=[t['updated_frames']for t in f['rounds']],nonzero_gradient_nodes=[t['nonzero_gradient_frames']for t in f['rounds']],normalized_output_shift_mean=float(((f['final'][:,:2]-E[:,:2])/E[:,2:]).norm(dim=-1)[allowed].mean())if allowed.any()else 0.,nonzero_output_nodes=int((f['final']!=E).any(-1).sum()),teacher_KL=[t['KL']for t in f['rounds']])
  rows.append(dict(alias=r['alias'],dataset=r['dataset'],step=p['configuration']['step_size_by_target_dataset'][r['dataset']],active_reference=bool(x['positions']),retained_anchors=len(x['positions']),allowed_nodes=int(allowed.sum()),fits=fits))
 for ds in ['hc2','vidstg']:
  rr=[r for r in rows if r['dataset']==ds];groups[ds]=dict(count=len(rr),reference_active_parents=sum(r['active_reference']for r in rr),retained_anchors=sum(r['retained_anchors']for r in rr),mean_normalized_shift={m:float(np.mean([r['fits'][m]['normalized_output_shift_mean']for r in rr]))for m in ['Fixed','Live']})
 put(P/'ACTUAL_UPDATES.json',dict(status='complete',count=64,groups=groups,rows=rows))
 put(P/'CONFIGURATION.json',{k:v for k,v in p.items()if k not in ['records','created_epoch','deadline_epoch']})
 put(P/'MODEL_PROVENANCE.json',dict(backbone='original frozen TA-STVG cross-domain EMA',target_hc2_uses='TASTVG_VidSTG.pth',target_vidstg_uses='TASTVG_HCSTVG2.pth',checkpoints={f:read(D/'RUNTIME_LOCK.json')['pins'][f]for f in ['checkpoints/TASTVG_VidSTG.pth','checkpoints/TASTVG_HCSTVG2.pth']},actual_model_state_hashes=inp['models'],feature='true raw ResNet layer3 stride16/1024 forward hook plus original pixelmask valid',weights_frozen=True,new_Source_captures=64,physical_original_offset_forwards=128,new_DINO_experts_training=0))
 put(P/'COST_BREAKDOWN.json',dict(total_Source_captures_completed=64,physical_original_offset_forward_passes=128,final_worker_DINO_encodings_completed=0,engineering_DINO_encodings_completed=0,new_original_DINO_detection_calls=0,new_training_runs=0,new_heads=0,new_GPU_seconds=gpu['GPU_seconds'],GPU_cap_seconds=900,CPU_fit_prediction_seconds=bar['CPU_seconds'],actual_fits=128,round_slots=256,old_Source_E_DINO_cost_excluded=True,audit_score_publication_cost_separate=True,not_first_deployment_latency=True))
 def sign(v):return 'positive_interval'if v['CI95'][0]>0 else('negative_interval'if v['CI95'][1]<0 else'uncertain')
 decision={ds:{k:{ep:sign(v)for ep,v in c.items()}for k,c in g['paired'].items()}for ds,g in z['groups'].items()}
 put(P/'EVALUATION_STATUS.json',dict(status='complete',count=64,frozen_config_path=p['frozen_configuration'],frozen_config_sha256=p['frozen_configuration_sha256'],candidate_json_is_immutable_pre_evaluation_snapshot=True,per_domain_intervals=decision,no_new_parameters_selected=True,no_production_promotion=True,no_extra_steps_or_expansion=True))
 put(P/'EXPOSURE_AND_ENGINEERING.json',dict(status='complete',count=64,eligible_parent_counts=p['eligible_counts'],excluded_manifests=len(p['excluded_manifests']),excluded_old_NH_NV_and_CH_CV=True,no_parameter_panel_parent_overlap=True,roster_sha256=read(D/'RUNTIME_LOCK.json')['roster_sha256'],selection='fixed seed SHA identities before Source/fits/new GT, not scores/reference/cachequality',historical_Source_E_expanded_full_evaluation=True,not_globally_unseen=True,new_frozen_Fine_config_not_previously_evaluated_on_these_parents=True,all_rows_unknown_no_reference_and_zero_changes_preserved=True,all_foundation_dependencies_and_checkpoint_pinned_before_capture=True,input_audit_pass_before_fit=True,all64_outputs_sealed_before_GT=True,actual_Source_time_logits_exact_runtime=True,root_audit_not_independent_GPU_replay=True,shared_gpu_lease_and_owned_watchdog900=True,other_queues_heartbeat_not_resumed=True))
 ci=lambda v:'unknown'if v['mean']is None else f"{v['mean']:+.6f} [{v['CI95'][0]:+.6f},{v['CI95'][1]:+.6f}]"
 lines=['# 分数据集冻结 Fine-Center 正式评估：HC32＋Vid32','','本轮在固定配置之后锁定新的64父视频，每父一条query。HC步长1、Vid步长.5；tau.1/scale.25/两轮17/final2/原FineROI分数平均/E+finalMBR/宽高及四计划观察节点/纯插值scope全部固定。E、Fixed、Live三组；同域Fixed和Live使用相同参数与噪声，独立从零开始，首步exact，Live第二轮围绕实际更新state重新采样。没有参数搜索、GT选步或按结果换配置。','','本轮结果：Live相对E的完整vIoU与未观察actual sIoU四个区间均跨0，尚未建立超过E的稳定收益。HC的Live−Fixed在两项指标上有小幅名义正区间，Vid未确认；这项相对Fixed的增量不等于已经改善E。所有结果与负例保留，不根据本轮再换参数。','','## 样本与证据边界','',f"身份SHA预锁各32，合格池HC{p['eligible_counts']['hc2']}/Vid{p['eligible_counts']['vidstg']}，排除{len(p['excluded_manifests'])}份历史小面板名单，包括原NH/NV与参与选参的CH/CV，parent无重叠。未按E收益、参考质量或GT挑样，不替换无参考/零变化/失败。此前存在Source/E expanded/full评价，因此是未参与当前Fine方法和参数选择的冻结配置评估，不称全项目从未曝光测试。冻结前两面板仍属于开发证据。本轮结果不用于再选本版参数。",'', 'GT只在全部64×3输出与候选/概率/gradient/state路径封存后读取，用于统一离线评分与审计。主结果为父视频宏平均完整vIoU和原全部已知未观察actual节点sIoU，不以active-only或in-scope分母替代；unknown保持null。预测时间区间三组一致，插值评测帧没有新增独立视觉监督。','','| 方法 | HC完整vIoU% | Vid完整vIoU% | HC未观察actual sIoU% | Vid未观察actual sIoU% |','|---|---:|---:|---:|---:|']
 for m in p['arms']:lines.append('| '+m+' | '+' | '.join(f"{100*z['groups'][ds]['metrics'][m][ep]:.6f}"for ds,ep in [('hc2','v'),('vidstg','v'),('hc2','unobserved_actual_s'),('vidstg','unobserved_actual_s')])+' |')
 for ds,g in z['groups'].items():
  lines+=['',f'## {ds}：预定配对差值，pp','','| 对比 | 完整vIoU［95%CI］ | 未观察actual sIoU［95%CI］ | v改善/下降/不变 | v低于−5pp数 |','|---|---|---|---|---|']
  for k,c in g['paired'].items():v=c['v'];lines.append(f"| {k} | {ci(v)} | {ci(c['unobserved_actual_s'])} | {v['positive']}/{v['negative']}/{v['neutral']} | {v['severe_loss_below_minus5pp']} |")
  lines+=['',f"有参考父视频{groups[ds]['reference_active_parents']}/32，保留锚点{groups[ds]['retained_anchors']}；无参考仍在主分母。所有区间为预定固定种子10000次父级配对bootstrap、未多重性校正；跨零保持不确定，不能以点均值自动宣称优势。"]
  for ep in ['v','unobserved_actual_s']:
   diff=[(r['alias'],100*(r['values']['Live'][ep]-r['values']['E_MBR'][ep]))for r in z['rows']if r['dataset']==ds and r['values']['Live'][ep]is not None and r['values']['E_MBR'][ep]is not None];diff.sort(key=lambda x:x[1]);lines.append(f"Live−E {ep} 最低三例 {diff[:3]}；最高三例 {diff[-3:]}。仅列明尾部，不删除任何父视频。")
 lines+=['','## 解释范围与资源','', '本轮直接检验各域冻结配置是否改善E，以及同配置Live是否超过Fixed；没有HC step.5额外对照，不能从本轮再宣称HC step1优于.5。没有新增Shuffle/Direct，故不能单凭本轮证明全部视觉因果或优于搜索；历史对照保留。逐域按上述CI与负尾判断，不要求同一数值步长，不在结果后调整温度/尺度/轮数/锚点融合或再挑样本。冻结配置JSON是冻结时快照，当前完成状态另见EVALUATION_STATUS，不修改其hash或生产注册。','',f"实际64次冻结Source捕获，原offset forward128，新增GPU{gpu['GPU_seconds']:.6f}s/900s；原4帧DINO缓存复用，新DINO/专家/头/训练均0。layer3是真实中间特征，Source框及两时间logits逐例exact；foundation代码和两checkpoint事前pins，原共享gpu.lock与本项900s watchdog，未信号其它服务。128fit/256轮CPU构造{bar['CPU_seconds']:.6f}s，评分/审计/公开及旧Source/E/DINO成本另计。",'', 'root独立NumPy ROI/候选/概率/梯度/state/readout/官方指标/父bootstrap核验通过；它没有独立重跑GPU。GT不参与在线选择，原全部signed/unknown结果公开。现有GPT6.1Sol/max只读审查见ROUND_REVIEW。CURRENT/MAIN/PAPER和其它队列/heartbeat不变，无自动晋升或追加实验。','','## 全部逐样本差值，pp','','| Alias | Dataset | Fixed−E完整v | Live−E完整v | Live−Fixed完整v | Live−E未观察actual s |','|---|---|---:|---:|---:|---:|']
 for r in z['rows']:
  v=r['values'];vals=[100*(v[a][ep]-v[b][ep])if v[a][ep]is not None and v[b][ep]is not None else None for a,b,ep in [('Fixed','E_MBR','v'),('Live','E_MBR','v'),('Live','Fixed','v'),('Live','E_MBR','unobserved_actual_s')]];lines.append('| '+r['alias']+' | '+r['dataset']+' | '+' | '.join('unknown'if x is None else f'{x:+.6f}'for x in vals)+' |')
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n')
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(9,3.6))
 for ax,ds in zip(axs,['hc2','vidstg']):
  for i,k in enumerate(['Fixed_minus_E_MBR','Live_minus_E_MBR','Live_minus_Fixed']):
   v=z['groups'][ds]['paired'][k]['v'];m=v['mean'];lo,hi=v['CI95'];ax.errorbar(i,m,yerr=[[m-lo],[hi-m]],fmt='o',capsize=4)
  ax.axhline(0,color='k',lw=.7);ax.set_xticks(range(3),['Fixed-E','Live-E','Live-Fixed']);ax.set(title=ds,ylabel='Full vIoU difference (pp)')
 fig.tight_layout();fig.savefig(P/'COMPARISON.svg');fig.savefig(D/'COMPARISON.png',dpi=120);plt.close(fig)
 put(D/'PUBLIC_FILES_FINAL.json',dict(files=CODE+['scripts/continue_stvg_fine_dataset_formal64_v1.py','scripts/audit_stvg_fine_dataset_formal64_inputs_v1.py','scripts/audit_stvg_fine_dataset_formal64_v1.py','scripts/report_stvg_fine_dataset_formal64_v1.py','configs/stvg_fine_center_dataset_candidate_v1.json']+[str(f.relative_to(ROOT))for f in P.iterdir()if f.is_file()]))
 print('report complete',flush=True)
if __name__=='__main__':main()
