"""Independent NumPy state/gradient/Adam audit followed by sealed GT scores."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,hashlib,numpy as np
from scipy.special import softmax
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_conditional_opd128_v1 import D,P,HR
from scripts.score_stvg_where_when_event128_v1 import avg,paired

def audit_fit(q0,t,fit):
 n=len(q0);mask=q0>0;i,j=np.where(mask);lf=np.full(q0.shape,-1e30);lf[mask]=np.log(q0[mask]);tmS=t.sum(1);tmE=t.sum(0);cs=np.divide(t.T,tmE[:,None],out=np.zeros_like(t),where=tmE[:,None]>0);ce=np.divide(t,tmS[:,None],out=np.zeros_like(t),where=tmS[:,None]>0);eta=np.zeros((2,n));mom=np.zeros_like(eta);var=np.zeros_like(eta);rng=np.random.default_rng(fit['seed']);maxerr=0.;changed=0
 for k,tr in enumerate(fit['trace']):
  assert k==tr['step'];err=float(abs(eta-tr['eta_before']).max());assert err<2e-10;maxerr=max(maxerr,err);q=np.zeros_like(q0);q[mask]=softmax(lf[mask]+eta[0,i]+eta[1,j]);mode=fit['mode'];pairs=None
  if mode in ['Fixed','Live']:
   uniform=rng.random(32);assert np.array_equal(uniform,tr['uniforms']);probs=q0[mask]if mode=='Fixed'or k==0 else q[mask];probs/=probs.sum();cdf=np.cumsum(probs);cdf[-1]=1.;indices=np.searchsorted(cdf,uniform,side='right');pairs=np.column_stack([i[indices],j[indices]]);assert np.array_equal(pairs,tr['sampled_pairs']);assert float(abs(probs[indices]-tr['selected_sampling_probability']).max())<2e-10;digest=hashlib.sha256(cs[pairs[:,1]].tobytes()+ce[pairs[:,0]].tobytes()).hexdigest();assert digest==tr['teacher_conditional_digest'];changed+=int(k>0 and not np.array_equal(pairs,np.column_stack([i[np.searchsorted(np.cumsum(q0[mask]/q0.sum()),uniform,side='right')],j[np.searchsorted(np.cumsum(q0[mask]/q0.sum()),uniform,side='right')]])))if mode=='Live'else 0
  if mode=='DirectJoint':gradient=.5*np.stack([q.sum(1)-tmS,q.sum(0)-tmE])
  else:
   ps=softmax(lf.T+eta[0][None],axis=1);pe=softmax(lf+eta[1][None],axis=1)
   if mode=='DirectConditional':ws,we=tmE,tmS
   else:ws=np.bincount(pairs[:,1],minlength=n)/len(pairs);we=np.bincount(pairs[:,0],minlength=n)/len(pairs)
   gradient=.5*np.stack([ws@(ps-cs),we@(pe-ce)])
  err=float(abs(gradient-tr['gradient']).max());assert err<2e-10;maxerr=max(maxerr,err);mom=.9*mom+.1*tr['gradient'];var=.999*var+.001*tr['gradient']**2;eta-=.1/(1-.9**(k+1))*mom/(np.sqrt(var)/np.sqrt(1-.999**(k+1))+1e-8);eta-=eta.mean(1,keepdims=True);err=float(abs(eta-tr['eta_after']).max());assert err<2e-10;maxerr=max(maxerr,err)
 assert float(abs(eta-fit['eta']).max())<2e-10;q=np.zeros_like(q0);q[mask]=softmax(lf[mask]+eta[0,i]+eta[1,j]);assert float(abs(q-fit['q']).max())<2e-10
 return dict(steps=len(fit['trace']),maxerror=maxerr,Live_steps_differ_from_initial_sampler=changed)

def main():
 import torch
 torch.set_num_threads(4);p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_before_GT']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items())
 for f in bar['files']:assert sha(R/f['path'])==f['sha256']
 # Audit inference without target GT. This does not re-fit a network or policy.
 audited_steps=0;maxerr=0.;changed=0;first=0
 for r in p['records']:
  z=load(D/'predictions'/f"{r['alias']}.pt");assert sha(HR/'predictions'/f"{r['alias']}.pt")==r['teacher_prediction_sha256']
  for name,f in z['outputs'].items():
   if 'trace'not in f:continue
   a=audit_fit(z['q0'],z['teacher'],f);audited_steps+=a['steps'];maxerr=max(maxerr,a['maxerror']);changed+=a['Live_steps_differ_from_initial_sampler']
  for seed in p['seeds']:
   a,b=z['outputs'][f'Fixed{seed}'],z['outputs'][f'Live{seed}']
   if a['trace']:
    for k in ['sampled_pairs','gradient','eta_after']:assert np.array_equal(a['trace'][0][k],b['trace'][0][k])
   first+=1
 put(D/'POLICY_AUDIT_BEFORE_GT.json',dict(status='pass',independent_numpy_Adam_gradient_sampling_steps=audited_steps,maxerror=maxerr,first_Fixed_Live_exact=first,Live_steps_differ_from_initial_sampler=changed,new_models=0,new_GT=False))
 put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_before_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')))
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];oldrows={r['alias']:r for r in read(R/'results/stvg_error_state/2026-10-20/temporal_hires128/RESULTS_PUBLIC.json')['rows']};rows=[];checks=spacechecks=updates=0
 for r in p['records']:
  a=r['alias'];z=load(D/'predictions'/f'{a}.pt');old=load(HR/'predictions'/f'{a}.pt');values={};state_summary={}
  for name,o in z['outputs'].items():
   assert abs(o['q'].sum()-1)<1e-10 and np.array_equal(o['q']>0,z['q0']>0);assert final_mbr(z['frame_ids'],o['q'])['interval']==o['interval']
   if 'trace'in o:updates+=o['actual_updates'];state_summary[name]=dict(updates=o['actual_updates'],eta_norm=float(np.linalg.norm(o['eta'])),first_loss=o['trace'][0]['loss']if o['trace']else 0,last_loss=o['trace'][-1]['loss']if o['trace']else 0)
  for space,boxes in z['trajectories'].items():
   assert np.array_equal(boxes,old['trajectories'][space]);spacechecks+=1;tube=DenseTube(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True)
   for name,o in z['outputs'].items():
    iv=o['interval'];v=tube.score(iv);off=official(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;checks+=1
    if name in ['Native','TeacherDirect']:assert max(abs(v[k]-oldrows[a]['values'][space+'_'+('Native'if name=='Native'else'HiResMix')][k])for k in ['v','t','s'])<2e-10
    values[space+'_'+name]=dict(**v,interval=iv,boundary_MAE=(abs(iv[0]-spans[r['query_ordinal']][0])+abs(iv[1]-spans[r['query_ordinal']][1]))/2)
  for name in ['Native','TeacherDirect','DirectJoint','DirectConditional']:values['Mean_'+name]={k:avg([values[f'Live{s}_{name}'][k]for s in p['seeds']])for k in ['v','t','s','boundary_MAE']}
  for name in ['Fixed','Live']:
   values['Mean_'+name]={k:avg([values[f'Live{s}_{name}{s}'][k]for s in p['seeds']])for k in ['v','t','s','boundary_MAE']};values['EMean_'+name]={k:avg([values[f'E_{name}{s}'][k]for s in p['seeds']])for k in ['v','t','s','boundary_MAE']}
  rows.append(dict(alias=a,values=values,policy_updates=state_summary));print('scored',a,flush=True)
 keys=['Mean_'+a for a in ['Native','TeacherDirect','DirectJoint','DirectConditional','Fixed','Live']]+['E_Native','EMean_Live'];means={key:{k:avg([r['values'][key][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for key in keys};contrasts=[('Mean_'+a,'Mean_Native')for a in ['TeacherDirect','DirectJoint','DirectConditional','Fixed','Live']]+[('Mean_Live','Mean_'+a)for a in ['TeacherDirect','DirectJoint','DirectConditional','Fixed']]+[('Mean_DirectConditional','Mean_DirectJoint'),('EMean_Live','E_Native')]+[(f'Live{s}_Live{s}',f'Live{s}_Fixed{s}')for s in p['seeds']];ds={}
 for a,b in contrasts:
  d={k:paired([(100 if k in ['v','t','s']else 1)*(r['values'][a][k]-r['values'][b][k])for r in rows])for k in ['v','t','s','boundary_MAE']};v=np.array([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rows]);d['tails']=dict(positive=int((v>1e-10).sum()),negative=int((v< -1e-10).sum()),flat=int((abs(v)<=1e-10).sum()),below_minus3=int((v< -3).sum()),below_minus5=int((v< -5).sum()),positive_contribution_pp=float(np.maximum(v,0).mean()),negative_contribution_pp=float(np.minimum(v,0).mean()),worst=sorted([dict(alias=r['alias'],delta_pp=float(x))for r,x in zip(rows,v)],key=lambda x:x['delta_pp'])[:8]);ds[a+'_minus_'+b]=d
 cost=read(D/'COST.json');assert updates==cost['actual_updates'];res=dict(status='completed_exposed_development_actual_conditional_OPD',parents=128,seeds=p['seeds'],means=means,paired=ds,rows=rows,cost=cost,spatial_frozen=True,teacher_visual_refresh=False,teacher_query_states_refresh_in_Live=True,all_new_network_training=0);put(P/'RESULTS_PUBLIC.json',res)
 audit=dict(status='pass',**{k:v for k,v in read(D/'POLICY_AUDIT_BEFORE_GT.json').items()if k not in ['status','new_GT']},new_target_GT_scored=True,official_DenseTube_checks=checks,spatial_arrays_exact=spacechecks,updates=updates,Native_TeacherDirect_exact_previous=True,all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']),GT_provenance=prov);assert audit['all128_before_GT'];put(P/'INDEPENDENT_AUDIT.json',audit);put(P/'ROOT_AUDIT.json',audit)
 gain=ds['Mean_Live_minus_Mean_Native']['v'];vs={a:ds['Mean_Live_minus_Mean_'+a]['v']for a in ['TeacherDirect','DirectJoint','DirectConditional','Fixed']};go=gain['mean']>=2 and gain['CI95'][0]>0 and all(x['CI95'][0]>0 for x in vs.values());decision=dict(primary='paired-seed Mean_Live',net=gain,versus_controls=vs,decision='freeze_for_parent_disjoint_confirmation'if go else'no_supported_large_independent_OPD_advantage_in_this_configuration',not_independent_confirmation=True,no_parameter_search_authorized_by_this_round=True);put(P/'DECISION_PUBLIC.json',decision)
 lines=['# 缓存教师下的双边界条件OPD：HC128','','教师T=.5旧224+.5新448 Source后验固定。只优化每视频新初始化的130临时时间logit偏置，不训练网络/定位头/评价器，不动空间。Native/TeacherDirect完整复用；DirectJoint为完整教师KL，DirectConditional为teacher-forced状态同条件损失，Fixed每步fresh初始分布采样，Live每步fresh当前分布采样，paired41/42/43。两边界条件标签来自构造的T，不是原生Source条件decoder概率。','','| 输出 | 完整vIoU% | tIoU% | v增量pp [CI95] |','|---|---:|---:|---|']
 for name in ['Native','TeacherDirect','DirectJoint','DirectConditional','Fixed','Live']:
  v=means['Mean_'+name];c=ds.get('Mean_'+name+'_minus_Mean_Native');d='0'if c is None else f'{c["v"]["mean"]:+.6f} [{c["v"]["CI95"][0]:+.6f},{c["v"]["CI95"][1]:+.6f}]';lines.append(f'| {name} | {100*v["v"]:.6f} | {100*v["t"]:.6f} | {d} |')
 lines+=['','主Mean Fixed/Live只取同seed时间/空间配对再父内均值；其他crossseed值不当独立样本。Direct确定性时间在三个冻结空间上平均。全部父及负尾保留；RC128反复曝光，任何开发正值不独立证明。没有重查RGB，Live刷新的是当前学生区间状态及相应两个条件目标；每个32采样区间请求两份至多65类分布，不称32次标量reward。','','```json',json.dumps(decision,indent=2),'```','','## 成本及独立核验','','```json',json.dumps(cost,indent=2),json.dumps(audit,indent=2),'```','','完整signed/unknown、逐seed差值/负尾、每父实际更新数在RESULTS_PUBLIC.json。原Source448>=2pp gate仍未通过，选择半数融合教师是本轮新开发决定。当前结果不自动更改Current/Main/Paper或空间配置，也不泛化所有条件OPD。']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],spatial_config_sha256=sha(R/'methods/stvg_referdino_opd_v1/config.json'),teacher='cached Source224/448 .5 mixture, chosen development signal, not independent evidence',excluded=['media','GT coordinates','captions','parent IDs','weights','raw policy/teacher tensors/traces']))
 for f in P.glob('*.json'):f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 print(json.dumps(decision,indent=2),flush=True)

if __name__=='__main__':main()
