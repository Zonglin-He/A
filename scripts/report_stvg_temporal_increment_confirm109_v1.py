"""Independent input/algebra audit and frozen-confirmation report, no inference."""
from pathlib import Path
import sys,json,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_increment_confirm109_v1 import D,P,DEV

def main():
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,independent_cdf
 from vg_tta.stvg_where_when_event128_v1 import interval_scores
 from scipy.special import logsumexp
 p=read(D/'PLAN_LOCK.json');res=read(P/'RESULTS_PUBLIC.json');assert len(res['rows'])==109 and res['selected_lambda']==.5
 checks=dict(prior_CDF=0,feedback_integrals=0,posterior=0,lambda0_exact=0,space_exact=0,token_max=0,legacy_selected_exact=0,old_inputs_unchanged=0);maxcdf=maxpost=0.;reuse=updates=0
 for r in p['records']:
  a=r['alias'];z=load(D/'predictions'/f'{a}.pt');prior=load(D/'inputs'/f'{a}.pt');sp=load(D/'spatial'/f'{a}.pt');sem=load(D/'semantics'/f'{a}.pt');q=z['q0'];ids=np.array(z['frame_ids']);err=float(abs(independent_cdf(probabilities(prior['records'],prior['logits']),ids.tolist())-q).max());assert err<1e-12;maxcdf=max(maxcdf,err);checks['prior_CDF']+=1
  assert __import__('torch').equal(sem['premax'].max(-1,keepdim=True).values,sem['all_logits']);checks['token_max']+=1;checks['legacy_selected_exact']+=int(sem['legacy_selected_exact']);assert sem['frame_ids8']==[prior['frame_ids'][i]for i in prior['planned_positions']]
  expected=dict(E=prior['E'],**sp['outputs'])
  for arm,b in z['trajectories'].items():assert np.array_equal(b,expected[arm]);checks['space_exact']+=1
  ii,jj=np.triu_indices(len(ids),1);cs=np.stack((ids[ii],ids[jj]+1),1).tolist()
  for arm in expected:
   curve=z['curves'][arm];raw=interval_scores(curve,np.array(z['edges']),cs);valid=np.array([x is not None for x in raw]);v=np.array([x for x in raw if x is not None]);f=np.zeros(len(ii))
   if curve is not None and len(v)and v.std()>1e-12 and np.ptp(curve)>1e-12:f[valid]=(v-v.mean())/v.std()
   for lam in [0.,.5]:
    o=z['outputs'][f'{arm}_L{lam:g}'];post=o['posterior'];pos=q[ii,jj]>0;lp=np.log(q[ii,jj][pos])+lam*f[pos];exp=np.zeros(len(ii));exp[pos]=np.exp(lp-logsumexp(lp));err=float(abs(exp-post[ii,jj]).max());assert err<1e-11 and np.all(post[q==0]==0);maxpost=max(maxpost,err);checks['posterior']+=1
    if lam==0:assert np.array_equal(post,q)and o['interval']==prior['interval'];checks['lambda0_exact']+=1
   checks['feedback_integrals']+=len(raw)
  assert sha(R/r['old_input'])==r['old_input_sha256']and sha(R/r['subject_path'])==r['subject_sha256'];checks['old_inputs_unchanged']+=1
  if r['reuse_spatial']:
   v=r['reuse_spatial'];assert sha(R/v['input'])==v['input_sha256']and sha(R/v['prediction'])==v['prediction_sha256'];old=load(R/v['prediction']);assert all(np.array_equal(sp['outputs'][f'Live{s}'],old['outputs'][f'Live_{s}'])for s in p['seeds']);reuse+=1
  else:
   assert sp['actual_updates']==120;updates+=sp['actual_updates']
 assert all(sha(R/f)==h for f,h in p['pins'].items());assert sha(DEV/'SELECTION_LOCK.json')==p['selection_lock_sha256']
 cost=res['cost'];newref=sum(x.get('new_ReferDINO',0)for x in cost);assert newref==109;seconds=sum(x['GPU_seconds']for x in cost);assert seconds<1800
 audit=dict(status='pass',checks=checks,max_prior_CDF_error=maxcdf,max_reweight_error=maxpost,old_spatial_reused=reuse,new_spatial_baselines=109-reuse,actual_temporary_state_updates=updates,new_ReferDINO_video_forwards=newref,new_ReferDINO_image_frames=8*newref,new_DINO=0,network_weight_updates=0,temporal_OPD_updates=0,GPU_lease_seconds=seconds,selected_lambda_unchanged=.5,all109_sealed_before_new_GT=True,all_parent_disjoint_from_development=True)
 put(P/'ROOT_AUDIT.json',audit)
 dev=read(R/'results/stvg_error_state/2026-10-19/temporal_increment128/RESULTS_PUBLIC.json');means=res['means'];comp=res['paired']['Live_seedmean_L0.5_minus_Live_seedmean_L0'];dv=np.array([100*(r['values']['Live_seedmean_L0.5']['v']-r['values']['Live_seedmean_L0']['v'])for r in res['rows']]);order=np.argsort(dv)[::-1];leave=[float(np.delete(dv,i).mean())for i in range(len(dv))];diag=dict(top_positive=[dict(alias=res['rows'][int(i)]['alias'],delta_pp=float(dv[i]),contribution_to_all109_pp=float(dv[i]/109))for i in order[:8]],leave_one_out_mean_range_pp=[min(leave),max(leave)],each_seed_full_v_delta_pp={str(s):float(np.mean([100*(r['values'][f'Live{s}_L0.5']['v']-r['values'][f'Live{s}_L0']['v'])for r in res['rows']]))for s in p['seeds']});put(P/'POSTSEAL_DIAGNOSTICS.json',diag)
 lines=['# 保留原分布的事件重加权：冻结lambda0.5的HC109确认','','日期：2026-10-19。HC128选择后冻结lambda0.5；全部109个父与本轮lambda开发集不交叉。历史Source/旧方法及RH/RF/NH曝光逐项公开；不是完全未使用过的方法测试集。每父一个预锁query，三seed41/42/43先父内平均，109父等权。确认数据不改配置。','','## 主结果：空间完全固定','','| 面板 | 原MBR完整vIoU% | 重加权完整vIoU% | 增量pp及95%配对区间 |','|---|---:|---:|---|']
 for label,x in [('HC128开发',dev),('HC109确认',res)]:
  v=x['paired']['Live_seedmean_L0.5_minus_Live_seedmean_L0']['v'];lines.append(f'| {label} | {100*x["means"]["Live_seedmean_L0"]["v"]:.6f} | {100*x["means"]["Live_seedmean_L0.5"]["v"]:.6f} | {v["mean"]:+.6f} [{v["CI95"][0]:+.6f}, {v["CI95"][1]:+.6f}] |')
 lines+=['','| HC109固定空间 | lambda | vIoU% | tIoU% | 开始误差frame | 结束误差frame | 平均边界误差frame |','|---|---:|---:|---:|---:|---:|---:|']
 for space in ['Live_seedmean','E']:
  for lam in [0.,.5]:
   v=means[f'{space}_L{lam:g}'];lines.append(f'| {space} | {lam:g} | {100*v["v"]:.6f} | {100*v["t"]:.6f} | {v["start_absolute_error_frames"]:.6f} | {v["end_absolute_error_frames"]:.6f} | {v["mean_boundary_error_frames"]:.6f} |')
 lines+=['','tIoU增量及区间：'+json.dumps(comp['t'],ensure_ascii=False),'','全部正负/零变化和严重退化：','```json',json.dumps(comp['tails'],ensure_ascii=False,indent=2),'```','','## 事后错误分解','','GT只在全部109预测全局封存后使用，分组来自原区间；不进入在线选择、门控、参数或回退。每行组内父等权。原时间损失=同空间GT时间−原时间；容量=原MBR全2080决策支持GT最优−原时间；不能当作教师可兑现成绩。','','| 原错误 | 父数 | 原时间损失pp | 原MBR支持容量pp | 冻结lambda实际恢复pp |','|---|---:|---:|---:|---:|']
 for k,v in res['error_decomposition'].items():
  nums=['unknown'if v[n]is None else f'{v[n]:+.6f}'for n in ['original_temporal_loss_pp','candidate_capacity_pp','actual_selected_recovery_pp']];lines.append('| '+k+' | '+str(v['parents'])+' | '+' | '.join(nums)+' |')
 lines+=['','## 执行、成本与核验','','用户接受109而非原要求至少128，允许确认最多128次八帧ReferDINO。本轮实际109次视频前向/872图，同一原八位置。每次同时保存原生选中轨迹和token-max前语义，32条旧原生选中输出bitexact；109条max-token与原生logits精确。没有额外DINO、条件teacher查询或新模型训练。','','固定空间算法tau0.2/尺度0.05,0.25/40步/lr0.01/32候选；32父×3seed精确复用旧RF32，77父×3seed只生成已冻结空间基线，9240次临时state更新。时间对照共享完全相同空间数组，不将这些更新当作新temporal OPD。新增Source分布/特征导出、ReferDINO和空间基线成本分别计数：','```json',json.dumps(audit,ensure_ascii=False,indent=2),'```','', '完整stage成本：','```json',json.dumps(cost,ensure_ascii=False,indent=2),'```','','p0、候选支持、finalMBR及反馈定义与开发一致。65物理端点/2080strict-index决策包括零mass决策；零质量不复活，lambda0精确原矩阵与原MBR。R为同旧八帧事件词组/全局单轨关联/duration内外contrast，视频内全legal定义支持标准化，fullspan无outside增量0，平坦缺失不变。缓存八点不是独立逐帧观测，token分数不是校准事件概率。','','## 决策','','本次仅检验原分布与既定事件反馈的互补性，不是temporal OPD。'+('确认均值没有兑现开发正增量，停止对该反馈继续调lambda/温度或以原分布被丢弃解释失败；当前配置不支持2–3pp以上收益，不接入生产、不启动temporal OPD。'if comp['v']['mean']<=0 else'确认增量应连同区间、尾部和集中度判断；不得因开发选中或单个正均值宣称2–3pp已建立。')+' 冻结的是比较，不因结果更改空间配置或选择新lambda。研究/生产登记保持不变。','','全量逐父×seed signed/unknown数值在RESULTS_PUBLIC.json，曝光在EXPOSURE_PUBLIC.json；无删样本、按错误类型挑参数或事后换配置。新增接口/下一轮不自动启动。','']
 (P/'REPORT.md').write_text('\n'.join(lines));put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),selection_sha256=p['selection_lock_sha256'],code_pins=p['pins'],all109_predictions_before_GT=True,development_commit=read(DEV/'publication/FINAL_GITHUB_RECEIPT.json')['commit'],original_roster_plan_sha256=p['original_roster_plan_sha256'],private_excluded=['media','captions','parent identifiers','GT coordinates','weights','raw features/logits/posteriors/trajectories']))
 print(json.dumps(dict(audit=audit,primary_delta=comp['v'],t_delta=comp['t'],diagnostics=diag),indent=2))
if __name__=='__main__':main()
