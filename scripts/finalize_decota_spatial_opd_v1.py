"""Post-completion root aggregation audit, anonymous figures and review draft.

No GPU, new model inference, hyperparameter selection or automatic promotion.
Visual inspection and remote publication are separate actual root actions.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys, json, gzip, time, collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_spatial_opd_common_v1 import *
import numpy as np
ARMS=['on_policy','frozen_rollout','shuffled_feedback']

def rows(path):
 with gzip.open(path,'rt',encoding='utf-8') as f:return [json.loads(s) for s in f]

def independent_statistics(items,field):
 values=collections.defaultdict(list)
 for r in items:values[r['source_id']].append(r[field])
 a=np.asarray([sum(v)/len(v) for _,v in sorted(values.items())],float)
 rng=np.random.default_rng(20261006);samples=np.empty(10000)
 for j in range(10000):samples[j]=np.mean(a[rng.integers(len(a),size=len(a))])
 return dict(mean=float(a.mean()),ci95=np.quantile(samples,[.025,.975]).tolist(),parents=len(a),
  gross_gain_pp=float(np.maximum(a,0).mean()*100),gross_loss_pp=float(-np.minimum(a,0).mean()*100),
  harm_gt5pp_parents=int((a<-.05).sum()),harm_gt20pp_parents=int((a<-.20).sum()),
  harm_gt5pp_cells=sum(r[field]<-.05 for r in items),harm_gt20pp_cells=sum(r[field]<-.20 for r in items))

def fstat(s,arm,field):return s['arms'][arm]['metrics'][field]
def pp(m):return f"{100*m['mean']:+.2f} [{100*m['ci95'][0]:+.2f}, {100*m['ci95'][1]:+.2f}]"

def run():
 import torch
 torch.set_num_threads(2)
 verify();complete=read(BASE/'GPU_CPU_COMPLETION.json');assert complete['adapted_arrivals']==2880
 design=read(BASE/'DESIGN_LOCK.json');data={};summaries={};diagnoses={};checks=0;total=0
 for name,stage in design['stages'].items():
  dest=BASE/'stages'/name;completion=read(dest/'CPU_COMPLETION.json');barrier=read(dest/'PREDICTION_BARRIER.json')
  assert barrier['status']=='sealed' and completion['GT_after_all_stage_predictions']
  assert completion['time']>=barrier['time']
  for p,h in completion['files'].items():assert sha(ROOT/p)==h,p
  for p,h in barrier['files'].items():assert sha(BASE/p)==h,p
  ss=read(PUB/name/'SUMMARY.json');rr=rows(PUB/name/'ROWS.jsonl.gz');dd=rows(PUB/name/'SAMPLE_DIAGNOSIS.jsonl.gz')
  expected=sum(map(len,stage['orders'].values()))*len(stage['conditions'])*3
  assert len(rr)==len(dd)==completion['rows']==barrier['cells']==expected;total+=expected
  assert len(stage['parents'])==len(set(stage['parents']))
  grouped=collections.defaultdict(dict)
  for r in rr:
   key=(r['source_id'],r['order'],r['condition']);assert r['arm'] not in grouped[key];grouped[key][r['arm']]=r
   assert abs(r['delta_total_v']-r['delta_inherited_v']-r['delta_current_v'])<1e-12
   assert abs(r['After_v']-r['Frozen_v']-r['delta_total_v'])<1e-12
   assert r['gradient_calls']<=10
  assert all(set(v)==set(ARMS) for v in grouped.values())
  for cell in grouped.values():
   assert len({v['Frozen_v'] for v in cell.values()})==1
   for v in cell.values():
    assert abs(v['on_policy_minus_frozen_rollout_v']-cell['on_policy']['After_v']+cell['frozen_rollout']['After_v'])<1e-12
    assert abs(v['on_policy_minus_shuffled_v']-cell['on_policy']['After_v']+cell['shuffled_feedback']['After_v'])<1e-12
  for arm in ARMS:
   armrows=[r for r in rr if r['arm']==arm]
   for field in ['delta_total_v','delta_inherited_v','delta_current_v','delta_total_s','on_policy_minus_frozen_rollout_v','on_policy_minus_shuffled_v']:
    computed=independent_statistics(armrows,field);saved=fstat(ss,arm,field)
    assert abs(computed['mean']-saved['mean'])<1e-12 and np.max(np.abs(np.asarray(computed['ci95'])-saved['ci95']))<1e-12,(name,arm,field)
    if field.startswith('delta_'):
     assert computed['harm_gt20pp_cells']==saved['harm_gt20pp_cells'] and computed['harm_gt5pp_cells']==saved['harm_gt5pp_cells']
    checks+=1
  # Real no-update output parity, using the qualified source-native interface.
  for qual in ['dev_hc2','dev_vidstg'] if name=='dev_hc2' else []:
   b=read(BASE/'qualification'/qual/'PREDICTION_BARRIER.json')
   for p in b['files']:
    z=load(BASE/p);inp=load(BASE/z['input']['path'])
    if z['arrival']==0:assert torch.equal(z['fit']['before'],inp['native_boxes']);checks+=1
  data[name]=rr;summaries[name]=ss;diagnoses[name]=dd
  print('OPD_ROOT_AGGREGATION',name,expected,flush=True)
 assert total==2880
 extras={}
 for name,rr in data.items():
  extras[name]={}
  for arm in ARMS:
   dd=[d for r in diagnoses[name] if r['arm']==arm for d in r['details']]
   aa=[r for r in rr if r['arm']==arm]
   extras[name][arm]=dict(known_sample_frame_rounds=len(dd),unknown_sample_frames=sum(r['sample_diagnosis_unknown_frames'] for r in aa),
    sample_headroom_gt5pp_fraction=float(np.mean([d['sample_best_gt_iou']>d['central_before_gt_iou']+.05 for d in dd])) if dd else None,
    weight_improves_gt_fraction=float(np.mean([d['teacher_vs_uniform_gt_iou']>0 for d in dd])) if dd else None,
    weighted_GT_minus_uniform=float(np.mean([d['teacher_vs_uniform_gt_iou'] for d in dd])) if dd else None,
    mean_central_observed_GT_delta=float(np.mean([d['central_gt_delta'] for d in dd])) if dd else None,
    central_GT_wrong_direction_fraction=float(np.mean([d['teacher_vs_uniform_gt_iou']>0 and d['central_gt_delta']<0 for d in dd])) if dd else None,
    GPU_fit_wall_seconds=sum(r['compute']['fit_GPU_seconds'] for r in aa),CPU_math_seconds=sum(r['compute']['CPU_math_seconds'] for r in aa),
    capture_wall_seconds=sum(r['compute']['capture_seconds'] for r in aa),new_DINO_calls=sum(r['compute']['new_DINO_calls'] for r in aa),
    backward_steps=sum(r['gradient_calls'] for r in aa),CUDA_peak_allocated_bytes=max(r['compute']['CUDA_peak_allocated'] for r in aa),
    empty_queries=sum(r['empty'] for r in aa))
 confirmation=[summaries['confirm_hc2'],summaries['confirm_vidstg']]
 decision=dict(dual_dataset_after_frozen_lowerCI_positive=all(fstat(s,'on_policy','delta_total_v')['ci95'][0]>0 for s in confirmation),
  dual_dataset_true_over_shuffled_lowerCI_positive=all(fstat(s,'on_policy','on_policy_minus_shuffled_v')['ci95'][0]>0 for s in confirmation),
  dual_dataset_refresh_over_frozen_lowerCI_positive=all(fstat(s,'on_policy','on_policy_minus_frozen_rollout_v')['ci95'][0]>0 for s in confirmation),
  automatic_promotion=False)
 rootaudit=dict(status='pass',adapted_arrivals=total,independent_aggregate_statistics=checks,decision=decision,signal_chain=extras,
  historical_exposure=True,old_saved_resume=read(PAPER/'user_opd_pause_20261006/EXACT_RESUME_RECEIPT.json'),time=time.time())
 write(PUB/'ROOT_AUDIT.json',rootaudit)
 figs=PUB/'figures';figs.mkdir(exist_ok=True)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors=['#1565C0','#E69F00','#7B1FA2']
 fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
 for ax,name in zip(axes,['confirm_vidstg','confirm_hc2']):
  for j,arm in enumerate(ARMS):
   m=fstat(summaries[name],arm,'delta_total_v');y=m['mean']*100;lo,hi=np.array(m['ci95'])*100
   ax.errorbar(j,y,yerr=[[y-lo],[hi-y]],fmt='o',capsize=5,color=colors[j])
  ax.axhline(0,color='gray',linewidth=.8);ax.set_xticks(range(3),['Current rollout','Frozen rollout','Shuffled feedback'],rotation=15)
  ax.set_title(name.replace('confirm_','').upper()+': 128 parent sources, two orders');ax.set_ylabel('After - Frozen vIoU (pp)')
 fig.savefig(figs/'confirmation_intervals.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
 for ax,name in zip(axes,['confirm_vidstg','confirm_hc2']):
  for j,arm in enumerate(ARMS):
   r=[v for v in data[name] if v['arm']==arm];g=collections.defaultdict(list)
   for v in r:g[v['source_id']].append(v['delta_total_v'])
   a=np.sort([np.mean(v)*100 for v in g.values()]);ax.step(a,np.arange(1,len(a)+1)/len(a),where='post',label=arm,color=colors[j])
  ax.axvline(-20,color='red',linestyle=':',label='-20 pp');ax.axvline(0,color='gray',linewidth=.8);ax.legend(fontsize=8)
  ax.set_title(name);ax.set_xlabel('Parent-mean After - Frozen vIoU (pp)');ax.set_ylabel('Cumulative fraction')
 fig.savefig(figs/'confirmation_tails.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True);cases=[]
 for j,name in enumerate(['confirm_vidstg','confirm_hc2']):
  candidates=[r for r in data[name] if r['arm']=='on_policy' and any(d['source_id']==r['source_id'] and d['order']==r['order'] and d['details'] for d in diagnoses[name])]
  assert candidates,'No GT-supported case; preserve missingness instead of inventing a plot'
  for k,r in enumerate([max(candidates,key=lambda r:r['delta_total_v']),min(candidates,key=lambda r:r['delta_total_v'])]):
   dd=next(d['details'] for d in diagnoses[name] if d['source_id']==r['source_id'] and d['order']==r['order'] and d['arm']=='on_policy')
   ax=axes[j,k];means=lambda field:[np.mean([d[field] for d in dd if d['round']==t])*100 for t in range(10)]
   ax.plot(range(10),means('sample_best_gt_iou'),label='Sample best',linestyle=':')
   ax.plot(range(10),means('teacher_weighted_gt_iou'),label='Weighted samples')
   ax.plot(range(10),means('central_after_gt_iou'),label='Student central after')
   ax.set_title(f"{name}: parent {r['source_id']}, {r['order']}, delta {r['delta_total_v']*100:+.2f} pp")
   ax.set_xlabel('Round (zero based)');ax.set_ylabel('Observed annotated-frame IoU (%)');ax.legend(fontsize=8)
   cases.append(dict(stage=name,source_id=r['source_id'],order=r['order'],delta_total_v=r['delta_total_v'],observed_GT_frames=r['observed_GT_frames'],unobserved_GT_frames=r['unobserved_GT_frames']))
 fig.savefig(figs/'best_and_worst_signal_chain.png',dpi=180);plt.close(fig);write(PUB/'CASES.json',cases)
 report=['# Explicit spatial-policy OPD R1 review','',f'实际完成：{total}个适应到达，三臂、各32开发及128来源互斥确认，clean跨域双序与同域五类5%复核；未改CURRENT。','',
  '## 固定末轮任务结果','', '| Panel | Arm | ΔvIoU (pp), source bootstrap CI | Inherited | Current | >20pp cells |','|---|---|---:|---:|---:|---:|']
 for name,s in summaries.items():
  for arm in ARMS:report.append(f"| {name} | {arm} | {pp(fstat(s,arm,'delta_total_v'))} | {pp(fstat(s,arm,'delta_inherited_v'))} | {pp(fstat(s,arm,'delta_current_v'))} | {fstat(s,arm,'delta_total_v')['harm_gt20pp_cells']} |")
 report += ['', '## 确认集的部件证据','', '| Panel | Current minus frozen-rollout | True minus shuffled |','|---|---:|---:|']
 for name in ['confirm_vidstg','confirm_hc2']:report.append(f"| {name} | {pp(fstat(summaries[name],'on_policy','on_policy_minus_frozen_rollout_v'))} | {pp(fstat(summaries[name],'on_policy','on_policy_minus_shuffled_v'))} |")
 report += ['', '判定只作用于这套预锁工作点：', '', '```json',json.dumps(decision,ensure_ascii=False,indent=2),'```','',
  '双集After−Frozen为正、真反馈胜过打乱、刷新胜过固定rollout是三个独立问题；未通过的一项不能由sample oracle或加权样本质量替代。','',
  '## 信号链与执行','', '| Panel / arm | Better sample (>5pp) fraction | Weight improves GT fraction | Weighted−uniform GT (pp) | Central observed step Δ (pp) | Known frame-rounds |','|---|---:|---:|---:|---:|---:|']
 for name,arms in extras.items():
  for arm,d in arms.items():
   fmt=lambda v,scale=1:'missing' if v is None else f'{v*scale:.3f}'
   report.append(f"| {name}/{arm} | {fmt(d['sample_headroom_gt5pp_fraction'])} | {fmt(d['weight_improves_gt_fraction'])} | {fmt(d['weighted_GT_minus_uniform'],100)} | {fmt(d['mean_central_observed_GT_delta'],100)} | {d['known_sample_frame_rounds']} |")
 report += ['', '未知GT观察帧保留缺失；sample-frame-round平均是诊断口径，不是source-macro任务指标。共享decoder的未观察帧变化和>5/>20pp负尾在SUMMARY/ROWS中独立报告；观察改善不等于完整tube改善。', '',
  '## 审计、成本与边界','',
  f'{checks}项独立父source聚合/配对10000-bootstrap与任务分解复核通过；阶段CPU已有全1792 Adam/反馈梯度/状态链以及独立dense审计。独立NumPy审核不声称独立实现了整个decoder Jacobian。', '',
  'GPU_fit_wall_seconds为同步fit范围的实际壁钟，CPU_math_seconds另列。capture成本为每臂对应的共享输入开销，整实验求总成本时只计算一次共享capture；DINO调用只记在第一个适应臂。全部真实成本在ROOT_AUDIT/ROWS中。', '',
  'Gaussian尺度.25是探索尺度，非校准不确定性；同分反馈跳过Adam，梯度只经学生likelihood。固定10轮中央输出，末轮LNdelta1/16；无best-state或best-sample选择。有限小尺度几何反馈的一阶方向可能接近几何优化，detach不是新增知识源。', '',
  '确认集合与本轮开发来源互斥，未用于本轮选择，但整个官方池已有历史曝光，不写fresh。没有与旧best-state在线链作伪matched比较；旧paper工作点只作历史内部参照。', '',
  '第一次资格加载器工程失败与原件保存在recovery/loader_isolation_001；只隔离进程，未放松GT/像素/数学断言或改科学配置。', '',
  '图表生成不等于根已目检，匿名代码结果还需独立分支GitHub逐字节验证。所有此项关闭后接续旧TENT11791保存前缀；旧HC媒体/Table2单位依赖继续保留。', '',
  '![Confirmation](../results/decota_spatial_opd/2026-10-06/figures/confirmation_intervals.png)',
  '![Tails](../results/decota_spatial_opd/2026-10-06/figures/confirmation_tails.png)',
  '![Signal chain](../results/decota_spatial_opd/2026-10-06/figures/best_and_worst_signal_chain.png)']
 (ROOT/'docs/TA_DECOTA_SPATIAL_OPD_R1_REVIEW.md').write_text('\n'.join(report)+'\n')
 write(BASE/'ROOT_AUDIT_REPORT_COMPLETION.json',dict(status='complete_pending_actual_visual_review_and_remote_publication',adapted_arrivals=2880,
  report='docs/TA_DECOTA_SPATIAL_OPD_R1_REVIEW.md',figures=[str(f.relative_to(ROOT)) for f in sorted(figs.glob('*.png'))],audit_sha256=sha(PUB/'ROOT_AUDIT.json'),automatic_promotion=False,time=time.time()))
 print('OPD_ROOT_REPORT_READY',flush=True)

if __name__=='__main__':run()
