"""Post-seal same-action-universe readout and complete distribution quality."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,fcntl,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_plausibility128_v1 import D,P,OLD,PRE,FIN
from scripts.score_stvg_where_when_event128_v1 import avg,paired

def _main():
 import torch
 from main_method.runtime import resource_guard
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
 torch.set_num_threads(4);beg=time.perf_counter();p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_before_GT']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json')and all(sha(R/f)==h for f,h in p['pins'].items());assert all(sha(R/f['path'])==f['sha256']for f in bar['files']);spaces=oldexact=algebra=utilities=0;maxerr=0.;sizes=[];outside={k:0 for k in ['NativeHead','SquareHead','ContrastHead']};ii,jj=np.triu_indices(65,1)
 for r in p['records']:
  resource_guard();assert time.perf_counter()-beg<900;a=r['alias'];assert sha(OLD/'predictions'/f'{a}.pt')==r['counterfactual_sha256'];old=load(OLD/'predictions'/f'{a}.pt');z=load(D/'predictions'/f'{a}.pt');q=old['outputs']['Native']['q'];m=old['outputs']['MeanMotion']['q'];h=(q>0)&(q>=.1*q.max());assert np.array_equal(h,z['H']);sizes.append(int(h.sum()))
  for name,lv in [('NativeHead',np.log(q[h])),('SquareHead',2*np.log(q[h])),('ContrastHead',2*np.log(q[h])-np.log(m[h]))]:
   ex=np.zeros_like(q);v=np.exp(lv-lv.max());ex[h]=v/v.sum();actual=z['outputs'][name]['q'];err=float(abs(ex-actual).max());assert err<2e-14 and np.array_equal(actual>0,h);maxerr=max(maxerr,err);algebra+=int(h.sum());dec=final_mbr(z['frame_ids'],actual);assert dec['interval']==z['outputs'][name]['interval'];outside[name]+=not bool(h[tuple(dec['indices'])]);pick=np.unique(np.r_[np.linspace(0,len(ii)-1,19).astype(int),np.where((ii==dec['indices'][0])&(jj==dec['indices'][1]))[0]]);u=brute_utility(z['frame_ids'],actual,pick);err=float(abs(u-dec['utility_grid'][ii[pick],jj[pick]]).max());assert err<1e-11;maxerr=max(maxerr,err);utilities+=len(pick)
  for name,o in old['outputs'].items():assert np.array_equal(o['q'],z['outputs'][name]['q'])and o['interval']==z['outputs'][name]['interval'];oldexact+=1
  for name,b in old['trajectories'].items():assert np.array_equal(b,z['trajectories'][name]);spaces+=1
 audit=dict(status='pass',independent_H_parents=128,independent_normalization_scalar_checks=algebra,same_H_support_all_three=True,old_posteriors_intervals_exact=oldexact,frozen_spatial_arrays_exact=spaces,independent_MBR_original_action_utilities=utilities,max_error=maxerr,H_count_mean=float(np.mean(sizes)),H_count_min=min(sizes),H_count_max=max(sizes),MBR_outside_H_counts=outside,MBR_actions_unrestricted=True,new_GT=False);put(D/'AUDIT_BEFORE_GT.json',audit)
 resource_guard();assert time.perf_counter()-beg<900 and all(sha(R/f)==h for f,h in p['pins'].items());put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_before_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')))
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];prev={r['alias']:r for r in read(R/'results/stvg_error_state/2026-10-20/temporal_motion_counterfactual128/RESULTS_PUBLIC.json')['rows']};rows=[];officialchecks=scalarcalls=0
 for r in p['records']:
  resource_guard();assert time.perf_counter()-beg<900;a=r['alias'];z=load(D/'predictions'/f'{a}.pt');ids=np.array(z['frame_ids']);values={};curves={k:[]for k in ['v','t']}
  for space,boxes in z['trajectories'].items():
   tube=DenseTube(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True)
   for name,o in z['outputs'].items():
    assert final_mbr(ids,o['q'])['interval']==o['interval'];iv=o['interval'];v=tube.score(iv);off=official(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;officialchecks+=1;values[space+'_'+name]=dict(**v,boundary_MAE=(abs(iv[0]-spans[r['query_ordinal']][0])+abs(iv[1]-spans[r['query_ordinal']][1]))/2)
    if name not in ['NativeHead','SquareHead','ContrastHead']:assert all(abs(v[k]-prev[a]['values'][space+'_'+name][k])<2e-10 for k in ['v','t','s'])
   if space!='E':
    vals=[tube.score([int(ids[i]),int(ids[j])+1])for i,j in zip(ii,jj)];scalarcalls+=len(vals)
    for k in curves:curves[k].append(np.array([v[k]for v in vals]))
  for name in p['arms']:values['Mean_'+name]={k:avg([values[f'Live{s}_{name}'][k]for s in p['seeds']])for k in ['v','t','s','boundary_MAE']}
  quality={k:np.mean(v,0)for k,v in curves.items()};expected={name:{k:float(o['q'][ii,jj]@curve)for k,curve in quality.items()}for name,o in z['outputs'].items()};rows.append(dict(alias=a,values=values,expected_candidate_quality=expected,diagnostics=z['diagnostics']));print('score_plausibility',a,flush=True)
 means={key:{k:avg([r['values'][key][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for key in ['Mean_'+a for a in p['arms']]};ds={};expected={};pairs=[('ContrastHead',k)for k in ['Native','NativeHead','SquareHead','HiResAuxMix','Contrast']]+[('NativeHead','Native'),('SquareHead','Native')]
 for a,b in pairs:
  key=a+'_minus_'+b;delta=np.array([100*(r['values']['Mean_'+a]['v']-r['values']['Mean_'+b]['v'])for r in rows]);ds[key]={k:paired([(100 if k in ['v','t','s']else 1)*(r['values']['Mean_'+a][k]-r['values']['Mean_'+b][k])for r in rows])for k in ['v','t','s','boundary_MAE']};ds[key]['tails']=dict(positive=int((delta>1e-10).sum()),negative=int((delta< -1e-10).sum()),flat=int((abs(delta)<=1e-10).sum()),below_minus3=int((delta< -3).sum()),below_minus5=int((delta< -5).sum()),leave_one_out_mean_range_pp=[float(((delta.sum()-delta)/127).min()),float(((delta.sum()-delta)/127).max())],worst=sorted([dict(alias=r['alias'],delta_pp=float(v))for r,v in zip(rows,delta)],key=lambda x:x['delta_pp'])[:8]);expected[key]={k:paired([100*(r['expected_candidate_quality'][a][k]-r['expected_candidate_quality'][b][k])for r in rows])for k in ['v','t']}
 resource_guard();assert time.perf_counter()-beg<900 and all(sha(R/f)==h for f,h in p['pins'].items());cost=read(D/'COST.json');put(P/'RESULTS_PUBLIC.json',dict(status='completed_exposed_supporting_condition_capability_not_OPD',parents=128,rows=rows,means=means,paired=ds,expected_quality=expected,cost=cost,spatial_frozen=True));dist=expected['ContrastHead_minus_Native']['v'];hard=ds['ContrastHead_minus_Native']['v'];controls=['NativeHead','SquareHead','HiResAuxMix'];go_dist=dist['mean']>=2 and dist['CI95'][0]>0 and all(expected['ContrastHead_minus_'+k]['v']['CI95'][0]>0 for k in controls);go_hard=hard['mean']>=2 and hard['CI95'][0]>0 and all(ds['ContrastHead_minus_'+k]['v']['CI95'][0]>0 for k in controls);decision=dict(primary_hard=hard,primary_expected=dist,decision='qualifies_one_actual_controlled_transfer'if go_dist else'eligible_static_component_confirmation_only'if go_hard else'close_current_mean_motion_quotient_family_in_this_information_budget',not_OPD=True,not_independent_confirmation=True,no_beta_coefficient_norm_search=True,goal_achieved=False);put(P/'DECISION_PUBLIC.json',decision)
 audit.update(new_GT=True,official_Dense_output_checks=officialchecks,scalar_candidate_quality_calls=scalarcalls,GT_provenance=prov,all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']),old_metrics_exact_previous=True);assert audit['all128_before_GT'];put(P/'ROOT_AUDIT.json',audit);put(P/'INDEPENDENT_AUDIT.json',audit);sc=dict(CPU_audit_and_scoring_seconds=time.perf_counter()-beg,GPU_seconds=0,publication_archive_cost_excluded=True);put(D/'SCORING_COST.json',sc);put(P/'SCORING_COST_PUBLIC.json',sc)
 lines=['# 固定合理性条件：HC128纯缓存能力检查','','R22负结果保留，此轮不是工程修复或原官方LVLM算法复现。按原Native固定H={I:p0>=.1maxp0,p0>0}，三Head共用H，限制belief不限制原MBR动作集合。词表合理性不保证事件区间正确；旧Native、Square、Contrast、MeanMotion、HiResAuxMix原样保存。只有ContrastHead为候选，不以控制赢家事后晋升。','','| 对比 | 完整v增量pp [CI] | 候选期望GTv增量pp [CI] |','|---|---|---|']
 for name,x in ds.items():lines.append(f'| {name} | {x["v"]["mean"]:+.6f} {x["v"]["CI95"]} | {expected[name]["v"]["mean"]:+.6f} {expected[name]["v"]["CI95"]} |')
 lines+=['','旧曝光RC128/三冻结空间seed父内平均，所有新输出seal后GT，TC109未选参。只beta=.1一次预锁，无新模型/图像/专家/训练/学生或空间update。超过旧Contrast不等于净收益；三Head共同改善不等于motion知识增量。','','```json',json.dumps(decision,indent=2),json.dumps(cost,indent=2),json.dumps(audit,indent=2),'```','','hard与分布质量分别判定，非OPD/独立确认，不扫beta/系数/范数/均值范围。CURRENT/MAIN/PAPER与空间保持不变。']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],exposure='RC128 repeatedly exposed, TC109 not used',primary_sources=['https://arxiv.org/pdf/2311.16922','https://github.com/DAMO-NLP-SG/VCD/blob/master/vcd_utils/vcd_sample.py'],not_official_VCD_reproduction=True,excluded=['captions','parentIDs','media','GT','weights','raw caches']))
 for f in P.glob('*.json'):f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 resource_guard();assert time.perf_counter()-beg<900 and all(sha(R/f)==h for f,h in p['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in bar['files'])
 print(json.dumps(decision,indent=2),flush=True)
def main():
 lock=open(D/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:_main()
 finally:lock.close()
if __name__=='__main__':main()
