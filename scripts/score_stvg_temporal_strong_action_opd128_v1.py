"""Independent teacher/NumPy optimizer/sampling audit, then globally sealed GT."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,math,fcntl,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_strong_action_opd128_v1 import D,P,OLD,HR,STAGE,WRONG,CONDITIONS,SEEDS,STATIC,FIT_ARMS
from scripts.score_stvg_temporal_foreground_opd128_v1 import audit_fit
from scripts.score_stvg_where_when_event128_v1 import avg,paired

def independent_teacher(B,x):
 n=len(x);out=np.zeros_like(B)
 for k in range(2,n+1):
  ix=[(i,i+k-1)for i in range(n-k+1)if B[i,i+k-1]>0]
  if not ix:continue
  raw=[math.log(B[i,j])+sum(float(v)for v in x[i:j+1])for i,j in ix];mx=max(raw);v=[math.exp(w-mx)for w in raw];den=sum(v);mass=sum(B[i,j]for i,j in ix)
  for (i,j),w in zip(ix,v):out[i,j]=mass*w/den
 return out

def _main():
 import torch
 from main_method.runtime import resource_guard
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
 torch.set_num_threads(4);beg=time.perf_counter();p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_before_GT']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json');steps=changes=first=noops=spchecks=oldchecks=targetchecks=utilities=0;maxerr=targeterr=0.;ii,jj=np.triu_indices(65,1)
 def check():
  resource_guard();assert time.perf_counter()-beg+10<900
 def pins():assert all(sha(R/f)==h for f,h in p['pins'].items())
 pins();assert all(sha(R/f['path'])==f['sha256']for f in bar['files'])
 for r in p['records']:
  check();a=r['alias']
  for base,key in [(OLD,'raw_sha256'),(HR,'high_sha256'),(STAGE,'base_sha256')]:assert sha(base/'predictions'/f'{a}.pt')==r[key]
  assert sha(WRONG/f'{a}.pt')==r['wrong_mapping_sha256'];z=load(D/'predictions'/f'{a}.pt');old=load(OLD/'predictions'/f'{a}.pt');strong=load(STAGE/'predictions'/f'{a}.pt');h=load(HR/'predictions'/f'{a}.pt');B=strong['outputs']['HiResAuxMix']['q'];assert np.array_equal(z['q0'],B);curve=np.empty(65)
  for off,v in enumerate(h['observation']['actions']):curve[off::2]=v.numpy().reshape(-1)
  assert np.array_equal(curve,z['curves']['H'])and np.array_equal(z['curves']['N'],old['Action'])and np.array_equal(z['curves']['ShiftH'],np.roll(curve,32))
  mapping=load(WRONG/f'{a}.pt');assert mapping['donor']==r['donor']==z['donor'];dr=next(v for v in p['records']if v['alias']==r['donor']);assert sha(HR/'predictions'/f"{dr['alias']}.pt")==dr['high_sha256'];dh=load(HR/'predictions'/f"{dr['alias']}.pt");dx=np.empty(65)
  for off,v in enumerate(dh['observation']['actions']):dx[off::2]=v.numpy().reshape(-1)
  ids=np.asarray(z['frame_ids'],float);di=np.asarray(dh['frame_ids'],float);interp=np.interp((ids-ids[0])/(ids[-1]-ids[0]),(di-di[0])/(di[-1]-di[0]),dx);sd=curve.std();ds=interp.std();wrong=np.full(65,curve.mean())if sd<=1e-12 or ds<=1e-12 else curve.mean()+(interp-interp.mean())*sd/ds;assert abs(wrong-z['curves']['WrongH']).max()<2e-12
  for kind in CONDITIONS:
   t=z['teachers'][kind];ex=independent_teacher(B,z['curves'][kind]);err=float(abs(t-ex).max());assert err<2e-12 and np.array_equal(t>0,B>0);targeterr=max(targeterr,err);targetchecks+=int((B>0).sum())
   for k in range(2,66):
    mask=jj-ii+1==k;assert abs(t[ii[mask],jj[mask]].sum()-B[ii[mask],jj[mask]].sum())<3e-13
  for name,f in z['outputs'].items():
   if 'trace'not in f:continue
   t=B if name=='NoActionLive41'else z['teachers'][f['teacher']];aud=audit_fit(B,t,f);steps+=aud['steps'];changes+=aud['Live_steps_differ_from_initial_sampler'];maxerr=max(maxerr,aud['maxerror'])
   if name=='NoActionLive41':assert np.array_equal(f['q'],B)and f['actual_updates']==0;noops+=1
   else:
    mask=t>0;diag=f['fit_diagnostics'];kl=float((t[mask]*np.log(t[mask]/f['q'][mask])).sum());assert abs(kl-diag['final_joint_KL'])<2e-12
  for kind in ['H','N']:
   for seed in SEEDS:
    f,l=z['outputs'][f'{kind}_Fixed{seed}'],z['outputs'][f'{kind}_Live{seed}'];assert len(f['trace'])==len(l['trace'])
    if f['trace']:
     for key in ['sampled_pairs','gradient','eta_after','uniforms']:assert np.array_equal(f['trace'][0][key],l['trace'][0][key])
    first+=1
  for name,ref in [('Native',old['outputs']['Native']),('B',strong['outputs']['HiResAuxMix'])]:assert np.array_equal(z['outputs'][name]['q'],ref['q'])and z['outputs'][name]['interval']==ref['interval'];oldchecks+=1
  for name,b in old['trajectories'].items():assert np.array_equal(b,z['trajectories'][name])and np.array_equal(b,strong['trajectories'][name]);spchecks+=1
  for name in ['TeacherH','H_DirectJoint','H_Live41','H_Live42','H_Live43']:
   o=z['outputs'][name];dec=final_mbr(z['frame_ids'],o['q']);assert dec['interval']==o['interval'];pick=np.unique(np.r_[np.linspace(0,len(ii)-1,13).astype(int),np.flatnonzero((ii==dec['indices'][0])&(jj==dec['indices'][1]))]);u=brute_utility(z['frame_ids'],o['q'],pick);assert abs(u-dec['utility_grid'][ii[pick],jj[pick]]).max()<1e-11;utilities+=len(pick)
 audit=dict(status='pass',independent_Adam_gradient_sampling_steps=steps,max_gradient_state_error=maxerr,Live_steps_differ_from_initial_sampler=changes,first_Fixed_Live_exact=first,noAction_exact_noop_parents=noops,independent_scalar_teacher_checks=targetchecks,max_teacher_error=targeterr,teacher_B_node_count_preserved=True,wrong448_independent_reconstruction=True,old_q_interval_exact=oldchecks,frozen_spatial_arrays_exact=spchecks,independent_original_MBR_utilities=utilities,new_GT=False);put(D/'AUDIT_BEFORE_GT.json',audit);check();pins();put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_before_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')))
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];rows=[];officialchecks=scalarcalls=updates=0
 for r in p['records']:
  check();a=r['alias'];z=load(D/'predictions'/f'{a}.pt');ids=np.array(z['frame_ids']);values={};curves={k:[]for k in ['v','t']};state={};fixed_actual={}
  for name,o in z['outputs'].items():
   assert final_mbr(ids,o['q'])['interval']==o['interval']and np.array_equal(o['q']>0,z['q0']>0)
   if 'trace'in o:updates+=o['actual_updates'];state[name]=dict(actual_updates=o['actual_updates'],eta_norm=float(np.linalg.norm(o['eta'])),**o.get('fit_diagnostics',{}))
  for space,boxes in z['trajectories'].items():
   tube=DenseTube(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True);selected=np.isin(tube.fids,np.array(sorted(set(ids)-set(r['physical_frame_ids8']))));fixed_actual[space]=dict(unobserved_actual_nodes=int(selected.sum()),sIoU=float(tube.iou[selected].mean())if selected.any()else None,all_temporal_arms_identical=True)
   for name,o in z['outputs'].items():
    iv=o['interval'];v=tube.score(iv);off=official(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;officialchecks+=1;values[space+'_'+name]=dict(**v,boundary_MAE=(abs(iv[0]-spans[r['query_ordinal']][0])+abs(iv[1]-spans[r['query_ordinal']][1]))/2)
   assert len({values[space+'_'+name]['s']for name in z['outputs']})==1
   if space!='E':
    vv=[tube.score([int(ids[i]),int(ids[j])+1])for i,j in zip(ii,jj)];scalarcalls+=len(vv)
    for k in curves:curves[k].append(np.array([v[k]for v in vv]))
  for name in STATIC+['NoActionLive41','H_DirectJoint','H_DirectConditional','N_DirectJoint','N_DirectConditional']:values['Mean_'+name]={k:avg([values[f'Live{s}_{name}'][k]for s in SEEDS])for k in ['v','t','s','boundary_MAE']}
  for kind in CONDITIONS:
   for mode in ['Fixed','Live']if kind in ['H','N']else['Live']:
    values[f'Mean_{kind}_{mode}']={k:avg([values[f'Live{s}_{kind}_{mode}{s}'][k]for s in SEEDS])for k in ['v','t','s','boundary_MAE']};values[f'All9_{kind}_{mode}']={k:avg([values[f'Live{s}_{kind}_{mode}{t}'][k]for s in SEEDS for t in SEEDS])for k in ['v','t','s','boundary_MAE']}
  quality={k:np.mean(v,0)for k,v in curves.items()};expected={name:{k:float(o['q'][ii,jj]@curve)for k,curve in quality.items()}for name,o in z['outputs'].items()};rows.append(dict(alias=a,values=values,expected_candidate_quality=expected,policy_state=state,fixed_unobserved_actual_sIoU=fixed_actual,diagnostics=z['diagnostics']));print('score_strong_action',a,flush=True)
 keys=['Mean_'+x for x in STATIC]+['Mean_'+x for x in ['H_DirectJoint','H_DirectConditional','N_DirectJoint','N_DirectConditional','H_Fixed','H_Live','N_Fixed','N_Live','WrongH_Live','ShiftH_Live']];means={key:{k:avg([r['values'][key][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for key in keys};pairs=[('Mean_H_Live','Mean_'+x)for x in ['Native','B','H_DirectJoint','H_DirectConditional','H_Fixed','TeacherH','N_Live','WrongH_Live','ShiftH_Live']]+[('Mean_'+x,'Mean_B')for x in ['TeacherH','TeacherN','TeacherWrongH','TeacherShiftH','H_DirectJoint','H_DirectConditional','H_Fixed','N_Live']]+[('Mean_TeacherH','Mean_TeacherN'),('All9_H_Live','Mean_Native'),('All9_H_Live','All9_H_Fixed')];ds={}
 for x,y in pairs:
  delta=np.array([100*(r['values'][x]['v']-r['values'][y]['v'])for r in rows]);d={k:paired([(100 if k in ['v','t','s']else 1)*(r['values'][x][k]-r['values'][y][k])for r in rows])for k in ['v','t','s','boundary_MAE']};d['tails']=dict(positive=int((delta>1e-10).sum()),negative=int((delta< -1e-10).sum()),flat=int((abs(delta)<=1e-10).sum()),below_minus3=int((delta< -3).sum()),below_minus5=int((delta< -5).sum()),positive_contribution_pp=float(np.maximum(delta,0).mean()),negative_contribution_pp=float(np.minimum(delta,0).mean()),LOO_mean_range_pp=[float(((delta.sum()-delta)/127).min()),float(((delta.sum()-delta)/127).max())],worst=sorted([dict(alias=r['alias'],delta_pp=float(v))for r,v in zip(rows,delta)],key=lambda x:x['delta_pp'])[:8]);ds[x+'_minus_'+y]=d
 expected={x+'_minus_'+y:{k:paired([100*(r['expected_candidate_quality'][x][k]-r['expected_candidate_quality'][y][k])for r in rows])for k in ['v','t']}for x,y in [('TeacherH',x)for x in ['Native','B','TeacherN','TeacherWrongH','TeacherShiftH']]};check();pins();cost=read(D/'COST.json');assert updates==cost['actual_updates']==steps
 net=ds['Mean_H_Live_minus_Mean_Native']['v'];controls={x:ds['Mean_H_Live_minus_Mean_'+x]['v']for x in ['B','H_DirectJoint','H_DirectConditional','H_Fixed','TeacherH','N_Live','WrongH_Live','ShiftH_Live']};go=net['mean']>=2 and net['CI95'][0]>0 and all(v['CI95'][0]>0 for v in controls.values());decision=dict(primary='Mean_H_Live (matching seeds within parent)',net_vs_Native=net,increment_vs_B=controls['B'],versus_controls=controls,decision='freeze_one_parent_disjoint_confirmation'if go else'no_supported_large_independent_OPD_advantage_this_configuration',no_parameter_or_base_search=True,not_independent_confirmation=True,goal_achieved=False)
 put(P/'RESULTS_PUBLIC.json',dict(status='completed_exposed_development_actual_conditional_OPD',parents=128,means=means,paired=ds,teacher_expected_quality=expected,rows=rows,cost=cost,spatial_frozen=True,new_network_training=0,primary_aggregation='three matching temporal/spatial seeds per parent; all9 secondary only'));put(P/'DECISION_PUBLIC.json',decision)
 audit.update(new_GT=True,actual_updates=updates,official_Dense_output_checks=officialchecks,scalar_candidate_quality_calls=scalarcalls,GT_provenance=prov,all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']),all_temporal_arms_sIoU_identical=True);assert audit['all128_before_GT'];put(P/'ROOT_AUDIT.json',audit);put(P/'INDEPENDENT_AUDIT.json',audit);sc=dict(CPU_audit_and_scoring_seconds=time.perf_counter()-beg,GPU_seconds=0);put(D/'SCORING_COST.json',sc);put(P/'SCORING_COST_PUBLIC.json',sc)
 lines=['# Strong基底上448与224 Action条件OPD：HC128','','固定B=HiResAuxMix同时作为教师基底和学生零分布；Native单列。两Action分别完整运行TeacherDirect、DirectJoint、DirectConditional、freshFixed与Live；另同448 Wrong/Shift Live及NoAction exact。使用原130时间临时bias/40步/.1/M32/41,42,43，构造的双边界条件标签不称原生网络概率。0新模型/GPU/图像/专家/网络训练/空间update。旧448证据部署成本不能说免费；旧R24/R25关闭保持原判定。','','| 比较 | 完整v增量pp [CI] |','|---|---|']
 for key,v in ds.items():lines.append(f'| {key} | {v["v"]["mean"]:+.6f} {v["v"]["CI95"]} |')
 lines+=['','| 教师质量比较 | 期望候选GTv增量pp [CI] |','|---|---|']
 for key,v in expected.items():lines.append(f'| {key} | {v["v"]["mean"]:+.6f} {v["v"]["CI95"]} |')
 lines+=['','三时间/空间同seed配对后父内均值为主，全9仅预登记敏感性。RC128反复曝光、全128完整预测先seal后GT，无负例/无参考/零时间样本筛除。130bias不保证完整表达目标或保留节点数边际，Direct40步不是最优投影。固定空间使全部时间臂固定GT支持sIoU及未观测实际节点sIoU相同，不虚构空间收益。区分B已有收益、448证据及Live相对同信息直接/Fixed价值。','','```json',json.dumps(decision,indent=2),json.dumps(cost,indent=2),json.dumps(audit,indent=2),'```','','不是独立确认，不扫参数或挑控制晋升，空间/CURRENT/MAIN/PAPER不变。']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],upstream448='R6 actual cached Source448 RGB second-pass Action; old HiResFull/Fluct used it',exposure='RC128 repeatedly exposed; TC109 not selected',excluded=['captions','parentIDs','media','GT','weights','raw caches and adaptation traces']))
 for f in P.glob('*.json'):f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 check();pins();assert all(sha(R/f['path'])==f['sha256']for f in bar['files']);print(json.dumps(decision,indent=2),flush=True)

def main():
 lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:_main()
 finally:lock.close()
if __name__=='__main__':main()
