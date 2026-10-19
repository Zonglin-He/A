"""Post-seal same-action-universe readout and complete distribution quality."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,fcntl,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_duration_condition128_v1 import D,P,OLD,PRE,FIN,STRONG,WRONG
from scripts.score_stvg_where_when_event128_v1 import avg,paired

def _main():
 import torch,math
 from main_method.runtime import resource_guard
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
 torch.set_num_threads(4);beg=time.perf_counter();p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_before_GT']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json')and all(sha(R/f)==h for f,h in p['pins'].items());assert all(sha(R/f['path'])==f['sha256']for f in bar['files']);spaces=oldexact=algebra=utilities=0;maxerr=0.;ii,jj=np.triu_indices(65,1)
 for r in p['records']:
  resource_guard();assert time.perf_counter()-beg<900;a=r['alias'];assert sha(OLD/'predictions'/f'{a}.pt')==r['raw_prediction_sha256']and sha(STRONG/'predictions'/f'{a}.pt')==r['strong_prediction_sha256'];old=load(OLD/'predictions'/f'{a}.pt');strong=load(STRONG/'predictions'/f'{a}.pt');assert sha(WRONG/f'{a}.pt')==r['wrong_control_sha256'];wrong=load(WRONG/f'{a}.pt');z=load(D/'predictions'/f'{a}.pt');q=old['q0'];x=z['Action'];mask=q>0
  assert np.array_equal(z['WrongAction'],np.asarray(wrong['field'],dtype=np.float64))and np.array_equal(x,np.asarray(old['Action'],dtype=np.float64))and np.array_equal(z['energy'][ii,jj],old['teacher_energy'][ii,jj])
  for name,curve in [('GlobalFuse',x),('DurationConditional',x),('DurationShift32',np.roll(x,32)),('DurationWrongParent',z['WrongAction'])]:
   ex=np.zeros_like(q);groups={}
   for i,j in zip(ii,jj):
    if q[i,j]>0:groups.setdefault(j-i+1 if name!='GlobalFuse'else 0,[]).append((i,j,math.log(q[i,j])+sum(float(v)for v in curve[i:j+1])))
   for k,group in groups.items():
    mx=max(v for i,j,v in group);vs=[math.exp(v-mx)for i,j,v in group];den=sum(vs);mass=sum(q[i,j]for i,j,v in group)if name!='GlobalFuse'else 1.
    for (i,j,v),w in zip(group,vs):ex[i,j]=mass*w/den
   actual=z['outputs'][name]['q'];err=float(abs(ex-actual).max());assert err<3e-13 and np.array_equal(actual>0,mask);maxerr=max(maxerr,err);algebra+=int(mask.sum())
   if name!='GlobalFuse':
    for k in range(2,66):
     h=(jj-ii+1)==k;assert abs(actual[ii[h],jj[h]].sum()-q[ii[h],jj[h]].sum())<3e-13
   dec=final_mbr(z['frame_ids'],actual);assert dec['interval']==z['outputs'][name]['interval'];pick=np.unique(np.r_[np.linspace(0,len(ii)-1,19).astype(int),np.where((ii==dec['indices'][0])&(jj==dec['indices'][1]))[0]]);u=brute_utility(z['frame_ids'],actual,pick);err=float(abs(u-dec['utility_grid'][ii[pick],jj[pick]]).max());assert err<1e-11;maxerr=max(maxerr,err);utilities+=len(pick)
  for name,other in [('Native',old['outputs']['Native']),('RawTeacher',old['outputs']['TeacherDirect']),('HiResAuxMix',strong['outputs']['HiResAuxMix'])]:assert np.array_equal(z['outputs'][name]['q'],other['q'])and z['outputs'][name]['interval']==other['interval'];oldexact+=1
  for name,b in old['trajectories'].items():assert np.array_equal(b,z['trajectories'][name]);spaces+=1
 audit=dict(status='pass',independent_scalar_normalization_checks=algebra,old_posteriors_intervals_exact=oldexact,frozen_spatial_arrays_exact=spaces,independent_MBR_original_action_utilities=utilities,max_error=maxerr,node_count_marginal_preserved_all128=True,old_raw_energy_exact=True,MBR_actions_unrestricted=True,new_GT=False);put(D/'AUDIT_BEFORE_GT.json',audit)
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
  quality={k:np.mean(v,0)for k,v in curves.items()};expected={name:{k:float(o['q'][ii,jj]@curve)for k,curve in quality.items()}for name,o in z['outputs'].items()};rows.append(dict(alias=a,values=values,expected_candidate_quality=expected,diagnostics=z['diagnostics']));print('score_duration',a,flush=True)
 means={key:{k:avg([r['values'][key][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for key in ['Mean_'+a for a in p['arms']]};ds={};expected={};pairs=[('DurationConditional',k)for k in ['Native','GlobalFuse','DurationShift32','DurationWrongParent','HiResAuxMix','RawTeacher']]+[('GlobalFuse','Native'),('DurationShift32','Native')]
 for a,b in pairs:
  key=a+'_minus_'+b;delta=np.array([100*(r['values']['Mean_'+a]['v']-r['values']['Mean_'+b]['v'])for r in rows]);ds[key]={k:paired([(100 if k in ['v','t','s']else 1)*(r['values']['Mean_'+a][k]-r['values']['Mean_'+b][k])for r in rows])for k in ['v','t','s','boundary_MAE']};ds[key]['tails']=dict(positive=int((delta>1e-10).sum()),negative=int((delta< -1e-10).sum()),flat=int((abs(delta)<=1e-10).sum()),below_minus3=int((delta< -3).sum()),below_minus5=int((delta< -5).sum()),leave_one_out_mean_range_pp=[float(((delta.sum()-delta)/127).min()),float(((delta.sum()-delta)/127).max())],worst=sorted([dict(alias=r['alias'],delta_pp=float(v))for r,v in zip(rows,delta)],key=lambda x:x['delta_pp'])[:8]);expected[key]={k:paired([100*(r['expected_candidate_quality'][a][k]-r['expected_candidate_quality'][b][k])for r in rows])for k in ['v','t']}
 resource_guard();assert time.perf_counter()-beg<900 and all(sha(R/f)==h for f,h in p['pins'].items());cost=read(D/'COST.json');put(P/'RESULTS_PUBLIC.json',dict(status='completed_exposed_duration_conditional_capability_not_OPD',parents=128,rows=rows,means=means,paired=ds,expected_quality=expected,cost=cost,spatial_frozen=True));dist=expected['DurationConditional_minus_Native']['v'];hard=ds['DurationConditional_minus_Native']['v'];controls=['GlobalFuse','DurationShift32','DurationWrongParent','HiResAuxMix'];go_dist=dist['mean']>=2 and dist['CI95'][0]>0 and all(expected['DurationConditional_minus_'+k]['v']['CI95'][0]>0 for k in controls);go_hard=hard['mean']>=2 and hard['CI95'][0]>0 and all(ds['DurationConditional_minus_'+k]['v']['CI95'][0]>0 for k in controls);decision=dict(primary_hard=hard,primary_expected=dist,decision='qualifies_one_actual_controlled_transfer'if go_dist else'eligible_static_component_confirmation_only'if go_hard else'close_this_duration_conditional_configuration',not_OPD=True,not_independent_confirmation=True,no_temperature_duration_weight_step_search=True,goal_achieved=False);put(P/'DECISION_PUBLIC.json',decision)
 audit.update(new_GT=True,official_Dense_output_checks=officialchecks,scalar_candidate_quality_calls=scalarcalls,GT_provenance=prov,all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']));assert audit['all128_before_GT'];put(P/'ROOT_AUDIT.json',audit);put(P/'INDEPENDENT_AUDIT.json',audit);sc=dict(CPU_audit_and_scoring_seconds=time.perf_counter()-beg,GPU_seconds=0,publication_archive_cost_excluded=True);put(D/'SCORING_COST.json',sc);put(P/'SCORING_COST_PUBLIC.json',sc)
 lines=['# 保留节点数边际的时间证据融合：HC128缓存能力检查','','旧Raw负结果保留；本轮利用原Native完整长度边际和同长度事件位置证据。节点数不是精确物理持续时间，belief边际保持也不保证MBR最终区间长度保持。唯一DurationConditional，不挑控制晋升；无新专家/模型/训练/学生或空间更新。','','| 对比 | 完整v增量pp [CI] | 候选期望GTv增量pp [CI] |','|---|---|---|']
 for name,x in ds.items():lines.append(f'| {name} | {x["v"]["mean"]:+.6f} {x["v"]["CI95"]} | {expected[name]["v"]["mean"]:+.6f} {expected[name]["v"]["CI95"]} |')
 lines+=['','RC128反复曝光、三冻结空间seed父内平均、全父全128预测封存后GT。TC109不选参。本轮非OPD/独立确认，不以hard失败否定所有蒸馏；也不以KL/边际守恒或旧Raw改善充当真实净收益。','','```json',json.dumps(decision,indent=2),json.dumps(cost,indent=2),json.dumps(audit,indent=2),'```','','Source自身特征可能重复计入，保留错误长度会限制位置修复，事件logits并非校准概率。CURRENT/MAIN/PAPER与空间保持不变。']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],exposure='RC128 repeatedly exposed, TC109 not used',new_external_theory_claim=False,excluded=['captions','parentIDs','media','GT','weights','raw caches']))
 for f in P.glob('*.json'):f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 resource_guard();assert time.perf_counter()-beg<900 and all(sha(R/f)==h for f,h in p['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in bar['files'])
 print(json.dumps(decision,indent=2),flush=True)
def main():
 lock=open(D/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:_main()
 finally:lock.close()
if __name__=='__main__':main()
