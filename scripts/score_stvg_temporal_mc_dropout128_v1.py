"""Post-seal same-action-universe readout and complete distribution quality."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,fcntl,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_mc_dropout128_v1 import D,P,MEM,PRE,FIN,STAGE,DEV
from scripts.score_stvg_where_when_event128_v1 import avg,paired

def _main():
 import torch
 from main_method.runtime import resource_guard
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table,independent_cdf
 torch.set_num_threads(4);beg=time.perf_counter();p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_before_GT']and bar['weights_and_flags_unchanged']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json')and all(sha(R/f)==h for f,h in p['pins'].items());assert all(sha(R/f['path'])==f['sha256']for f in bar['files']);spaces=oldexact=cdfs=maskchecks=utilities=0;maxerr=0.;ii,jj=np.triu_indices(65,1);diagnostics={}
 for r in p['records']:
  resource_guard();assert time.perf_counter()-beg<900;a=r['alias']
  for path,h in [(STAGE/'predictions'/f'{a}.pt',r['native_stage_sha256']),(MEM/'temporal_inputs'/f'{a}.pt',r['memory_sha256']),(PRE/'inputs'/f'{a}.pt',r['input_sha256']),(FIN/'Live/predictions'/f'{a}.pt',r['spatial_sha256']),(DEV/'priors'/f'{a}.pt',r['prior_sha256'])]:assert sha(path)==h
  old=load(STAGE/'predictions'/f'{a}.pt');prior=load(DEV/'priors'/f'{a}.pt');z=load(D/'predictions'/f'{a}.pt');mask=z['outputs']['Native']['q']>0;ps={k:[]for k in ['FullMC','HeadOnlyMC','OutputOnlyMC']};assert len(z['draws'])==48
  for d in z['draws']:
   for off,v in enumerate(d['offsets']):
    want=d['stream']*10**10+r['query_ordinal']*10**6+off*10**5+d['draw']*1000;assert v['seed']==want and v['calls']==dict(MHA=12,Dropout=24)and v['head_masks']==v['head_control_masks']
    m=v['output_mask'].numpy();assert np.isfinite(m).all()and np.all((m==0)|(abs(m-1/.7)<2e-7))and v['max_head_mask_scaled_error']<=16*np.finfo(np.float32).eps
    for arm,pre in [('FullMC','full_pre_output'),('HeadOnlyMC','head_pre_output')]:assert np.array_equal(v[arm].numpy(),v[pre].numpy()*m);maskchecks+=1
    assert np.array_equal(v['OutputOnlyMC'].numpy(),z['native_logits'][off][-1].numpy()*m);maskchecks+=1
   for arm in ps:
    g=probabilities(z['records'],[v[arm]for v in d['offsets']]);ids,q=mass_table(g);ex=independent_cdf(g,ids);err=float(abs(ex-q).max());assert err<2e-13;maxerr=max(maxerr,err);cdfs+=1;assert ids==z['frame_ids'];q[~mask]=0;q/=q.sum();ps[arm].append(q)
  for arm,draws in ps.items():
   ex=np.mean(draws,0);actual=z['outputs'][arm]['q'];assert np.array_equal(ex,actual);dec=final_mbr(z['frame_ids'],actual);assert dec['interval']==z['outputs'][arm]['interval'];pick=np.unique(np.r_[np.linspace(0,len(ii)-1,19).astype(int),np.where((ii==dec['indices'][0])&(jj==dec['indices'][1]))[0]]);u=brute_utility(z['frame_ids'],actual,pick);err=float(abs(u-dec['utility_grid'][ii[pick],jj[pick]]).max());assert err<1e-11;maxerr=max(maxerr,err);utilities+=len(pick)
  for name in ['Native','HiResAuxMix']:assert np.array_equal(z['outputs'][name]['q'],old['outputs'][name]['q'])and z['outputs'][name]['interval']==old['outputs'][name]['interval'];oldexact+=1
  for off,nl in enumerate(z['native_logits']):assert torch.equal(nl[-1],prior['logits'][off])
  for name,b in old['trajectories'].items():assert np.array_equal(b,z['trajectories'][name]);spaces+=1
  diagnostics[a]=dict(TV_Full_vs_Native=float(abs(z['outputs']['FullMC']['q']-z['outputs']['Native']['q']).sum()/2),TV_Full_vs_HeadOnly=float(abs(z['outputs']['FullMC']['q']-z['outputs']['HeadOnlyMC']['q']).sum()/2),stream_TV_to_Full=[float(abs(np.mean(ps['FullMC'][i*16:(i+1)*16],0)-z['outputs']['FullMC']['q']).sum()/2)for i in range(3)],removed_mass=z['removed_mass'])
 audit=dict(status='pass',independent_CDF_draw_checks=cdfs,paired_head_and_output_mask_multiplication_checks=maskchecks,old_posteriors_intervals_exact=oldexact,frozen_spatial_arrays_exact=spaces,independent_MBR_original_action_utilities=utilities,max_error=maxerr,all_48_draw_means_exact=True,native_logits_exact=True,new_GT=False);put(D/'AUDIT_BEFORE_GT.json',audit)
 resource_guard();assert time.perf_counter()-beg<900 and all(sha(R/f)==h for f,h in p['pins'].items());put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_before_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')))
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];rows=[];officialchecks=scalarcalls=0
 for r in p['records']:
  resource_guard();assert time.perf_counter()-beg<900;a=r['alias'];z=load(D/'predictions'/f'{a}.pt');ids=np.array(z['frame_ids']);values={};curves={k:[]for k in ['v','t']}
  for space,boxes in z['trajectories'].items():
   tube=DenseTube(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True)
   for name,o in z['outputs'].items():
    assert final_mbr(ids,o['q'])['interval']==o['interval'];iv=o['interval'];v=tube.score(iv);off=official(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;officialchecks+=1;values[space+'_'+name]=dict(**v,boundary_MAE=(abs(iv[0]-spans[r['query_ordinal']][0])+abs(iv[1]-spans[r['query_ordinal']][1]))/2)
   if space!='E':
    vals=[tube.score([int(ids[i]),int(ids[j])+1])for i,j in zip(ii,jj)];scalarcalls+=len(vals)
    for k in curves:curves[k].append(np.array([v[k]for v in vals]))
  for name in p['arms']:values['Mean_'+name]={k:avg([values[f'Live{s}_{name}'][k]for s in p['seeds']])for k in ['v','t','s','boundary_MAE']}
  quality={k:np.mean(v,0)for k,v in curves.items()};expected={name:{k:float(o['q'][ii,jj]@curve)for k,curve in quality.items()}for name,o in z['outputs'].items()};rows.append(dict(alias=a,values=values,expected_candidate_quality=expected,diagnostics=diagnostics[a]));print('score_conditional_MC',a,flush=True)
 means={key:{k:avg([r['values'][key][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for key in ['Mean_'+a for a in p['arms']]};ds={};expected={};pairs=[('FullMC',k)for k in ['Native','HeadOnlyMC','OutputOnlyMC','HiResAuxMix']]+[('HeadOnlyMC','Native'),('OutputOnlyMC','Native')]
 for a,b in pairs:
  key=a+'_minus_'+b;delta=np.array([100*(r['values']['Mean_'+a]['v']-r['values']['Mean_'+b]['v'])for r in rows]);ds[key]={k:paired([(100 if k in ['v','t','s']else 1)*(r['values']['Mean_'+a][k]-r['values']['Mean_'+b][k])for r in rows])for k in ['v','t','s','boundary_MAE']};ds[key]['tails']=dict(positive=int((delta>1e-10).sum()),negative=int((delta< -1e-10).sum()),flat=int((abs(delta)<=1e-10).sum()),below_minus3=int((delta< -3).sum()),below_minus5=int((delta< -5).sum()),leave_one_out_mean_range_pp=[float(((delta.sum()-delta)/127).min()),float(((delta.sum()-delta)/127).max())],worst=sorted([dict(alias=r['alias'],delta_pp=float(v))for r,v in zip(rows,delta)],key=lambda x:x['delta_pp'])[:8]);expected[key]={k:paired([100*(r['expected_candidate_quality'][a][k]-r['expected_candidate_quality'][b][k])for r in rows])for k in ['v','t']}
 resource_guard();assert time.perf_counter()-beg<900 and all(sha(R/f)==h for f,h in p['pins'].items());cost=read(D/'COST.json');put(P/'RESULTS_PUBLIC.json',dict(status='completed_exposed_conditional_MC_capability_not_OPD',parents=128,rows=rows,means=means,paired=ds,expected_quality=expected,cost=cost,spatial_frozen=True));dist=expected['FullMC_minus_Native']['v'];hard=ds['FullMC_minus_Native']['v'];controls=['HeadOnlyMC','OutputOnlyMC','HiResAuxMix'];go_dist=dist['mean']>=2 and dist['CI95'][0]>0 and all(expected['FullMC_minus_'+k]['v']['CI95'][0]>0 for k in controls);go_hard=hard['mean']>=2 and hard['CI95'][0]>0 and all(ds['FullMC_minus_'+k]['v']['CI95'][0]>0 for k in controls);decision=dict(primary_hard=hard,primary_expected=dist,decision='qualifies_one_actual_controlled_transfer'if go_dist else'eligible_static_component_confirmation_only'if go_hard else'close_this_conditional_MC_configuration',not_OPD=True,not_independent_confirmation=True,no_rate_draw_seed_mixture_search=True,goal_achieved=False);put(P/'DECISION_PUBLIC.json',decision)
 audit.update(new_GT=True,official_Dense_output_checks=officialchecks,scalar_candidate_quality_calls=scalarcalls,GT_provenance=prov,all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']));assert audit['all128_before_GT'];put(P/'ROOT_AUDIT.json',audit);put(P/'INDEPENDENT_AUDIT.json',audit);sc=dict(CPU_audit_and_scoring_seconds=time.perf_counter()-beg,GPU_seconds=0,publication_archive_cost_excluded=True);put(D/'SCORING_COST.json',sc);put(P/'SCORING_COST_PUBLIC.json',sc)
 lines=['# 冻结时间模块条件MC积分：HC128能力检查','','同Native memory和第二pass query，48完整后验积分；只保留原TimeDecoder .1和原两次head .3随机位置。HeadOnly/OutputOnly共享head/output masks。固定网络参数，没有model.train/新训练/新图像/专家/空间或学生update。三16draw流不是独立父样本，不选最优种子。','','| 对比 | 完整v增量pp [CI] | 候选期望GTv增量pp [CI] |','|---|---|---|']
 for name,x in ds.items():lines.append(f'| {name} | {x["v"]["mean"]:+.6f} {x["v"]["CI95"]} | {expected[name]["v"]["mean"]:+.6f} {expected[name]["v"]["CI95"]} |')
 lines+=['','RC128反复曝光、三冻结空间seed父内平均、全128预测封存后GT。TC109不选参。条件MC并非完整Bayesian推断，实际checkpoint训练日志缺失；不能以简单噪声平滑为新事件知识。本轮非OPD/独立确认，不以hard失败否定所有蒸馏，不选控制晋升。原2080动作MBR不变。','','```json',json.dumps(decision,indent=2),json.dumps(cost,indent=2),json.dumps(audit,indent=2),'```','','Gal&Ghahramani2016原始依据：https://proceedings.mlr.press/v48/gal16.html；该理论不保证本冻结checkpoint校准或STVG收益。CURRENT/MAIN/PAPER与空间保持不变。']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],exposure='RC128 repeatedly exposed, TC109 not used',primary_sources=['https://proceedings.mlr.press/v48/gal16.html'],conditional_not_full_pipeline_Bayesian=True,excluded=['captions','parentIDs','media','GT','weights','raw caches']))
 for f in P.glob('*.json'):f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 resource_guard();assert time.perf_counter()-beg<900 and all(sha(R/f)==h for f,h in p['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in bar['files'])
 print(json.dumps(decision,indent=2),flush=True)
def main():
 lock=open(D/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:_main()
 finally:lock.close()
if __name__=='__main__':main()
