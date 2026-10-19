"""Post-seal joint-motion teacher capability and exact matched-context controls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,json,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_joint_motion128_v1 import D,P,PRE,FIN,DEV,FG,HR
from scripts.score_stvg_where_when_event128_v1 import avg,paired

def main():
 import torch
 torch.set_num_threads(4);p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_before_GT']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());assert read(D/'NATIVE_CONTROL.json')['status']=='pass'
 for f in bar['files']:assert sha(R/f['path'])==f['sha256']
 put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_before_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')))
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 from scripts.stvg_temporal_marginal_decode_util256_v1 import independent_cdf,probabilities
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
 from vg_tta.stvg_temporal_foreground128_v1 import project
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];oldrows={r['alias']:r for r in read(R/'results/stvg_error_state/2026-10-19/temporal_increment128/RESULTS_PUBLIC.json')['rows']};rows=[];checks=spacechecks=utilitychecks=qualitychecks=0;maxerr=0.;peaks=[]
 for r in p['records']:
  a=r['alias'];z=load(D/'predictions'/f'{a}.pt');old=load(DEV/'priors'/f'{a}.pt');inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');fg=load(FG/'predictions'/f'{a}.pt');hr=load(HR/'predictions'/f'{a}.pt');obs=z['observation'];values={};expectations={};ids=np.array(z['frame_ids']);N=len(ids);ii,jj=np.triu_indices(N,1);qh=independent_cdf(probabilities(obs['records'],obs['logits']),ids.tolist());err=float(abs(qh-obs['q']).max());assert err<1e-12;maxerr=max(maxerr,err);assert np.array_equal(qh>0,old['q']>0)and np.array_equal(old['q'],z['outputs']['Native']['q']);peaks.append(obs['peak_GPU_allocated'])
  assert np.array_equal(z['outputs']['HiRes']['q'],hr['observation']['q'])
  assert np.array_equal(z['outputs']['HiResMix']['q'],hr['outputs']['HiResMix']['q'])
  assert np.array_equal(z['outputs']['JointMix']['q'],.5*old['q']+.5*obs['q'])
  trace=z['motion_trace'];assert trace['restored']and trace['actual_encoder_calls']==1 and len(trace['calls'])==2
  assert trace['joint_feature_shape'][0]==65
  assert [c['num_frames']for c in trace['calls']]==[33,32]
  for offset,c in enumerate(trace['calls']):assert c['physical_indices']==list(range(offset,65,2))
  proj,_=project(obs['logits'],obs['actions'],obs['records'])
  assert proj['Fluct']==z['outputs']['JointFluct']and hr['outputs']['HiResFluct']==z['outputs']['HiResFluct']
  for arm,o in z['outputs'].items():
   if 'q'not in o:continue
   d=final_mbr(ids.tolist(),o['q']);assert d['interval']==o['interval'];pick=np.unique(np.r_[np.linspace(0,len(ii)-1,17).astype(int),np.where((ii==d['indices'][0])&(jj==d['indices'][1]))[0]]);exact=brute_utility(ids.tolist(),o['q'],pick);err=float(abs(exact-d['utility_grid'][ii[pick],jj[pick]]).max());assert err<1e-11;maxerr=max(maxerr,err);utilitychecks+=len(pick)
  S,E=ids[ii],ids[jj]+1;g,h=spans[r['query_ordinal']];overlap=np.maximum(0,np.minimum(E,h)-np.maximum(S,g));t=overlap/(E-S+h-g-overlap)
  for space,boxes in z['trajectories'].items():
   assert np.array_equal(boxes,inp['E']if space=='E'else sp['outputs'][f'T20_Live_{space[4:]}']);spacechecks+=1;tube=DenseTube(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True)
   for arm,o in z['outputs'].items():
    iv=o['interval'];v=tube.score(iv);off=official(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;checks+=1
    if arm=='Native':assert max(abs(v[k]-oldrows[a]['values'][space+'_L0'][k])for k in ['v','t','s'])<2e-10
    values[space+'_'+arm]=dict(**v,interval=iv,boundary_MAE=(abs(iv[0]-g)+abs(iv[1]-h))/2)
   lo=np.searchsorted(tube.fids,np.maximum(S,g));hi=np.maximum(lo,np.searchsorted(tube.fids,np.minimum(E,h)));prefix=np.r_[0.,np.cumsum(tube.iou)];v=(prefix[hi]-prefix[lo])/np.maximum(np.maximum(E,h)-np.minimum(S,g),1)
   for k in np.unique(np.linspace(0,len(ii)-1,19).astype(int)):
    actual=tube.score([S[k],E[k]]);assert abs(v[k]-actual['v'])<1e-12 and abs(t[k]-actual['t'])<1e-12;qualitychecks+=1
   for arm in ['Native','HiRes','HiResMix','Joint','JointMix']:
    q=z['outputs'][arm]['q'][ii,jj];expectations[space+'_'+arm]=dict(v=float(q@v),t=float(q@t))
  for arm in p['arms']:values['LiveMean_'+arm]={k:avg([values[f'Live{s}_{arm}'][k]for s in p['seeds']])for k in ['v','t','s','boundary_MAE']}
  for arm in ['Native','HiRes','HiResMix','Joint','JointMix']:expectations['LiveMean_'+arm]={k:avg([expectations[f'Live{s}_{arm}'][k]for s in p['seeds']])for k in ['v','t']}
  rows.append(dict(alias=a,values=values,posterior_expected_quality=expectations,motion_trace=z['motion_trace'],raw_image_height=obs['raw_RGB_shape'][1],normalized_height=obs['normalized_tensor_shape'][-2]));print('scored',a,flush=True)
 keys=['LiveMean_'+a for a in p['arms']]+['E_Native','E_Joint'];means={key:{k:avg([r['values'][key][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for key in keys};contrasts=[('LiveMean_'+a,'LiveMean_Native')for a in p['arms']if a!='Native']+[('LiveMean_JointMix','LiveMean_HiResMix'),('LiveMean_Joint','LiveMean_HiRes'),('LiveMean_JointFluct','LiveMean_HiResFluct'),('E_Joint','E_Native')];ds={}
 for a,b in contrasts:
  d={k:paired([(100 if k in ['v','t','s']else 1)*(r['values'][a][k]-r['values'][b][k])for r in rows])for k in ['v','t','s','boundary_MAE']};v=np.array([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rows]);d['tails']=dict(positive=int((v>1e-10).sum()),negative=int((v< -1e-10).sum()),flat=int((abs(v)<=1e-10).sum()),below_minus3=int((v< -3).sum()),below_minus5=int((v< -5).sum()),positive_contribution_pp=float(np.maximum(v,0).mean()),negative_contribution_pp=float(np.minimum(v,0).mean()),worst=sorted([dict(alias=r['alias'],delta_pp=float(x))for r,x in zip(rows,v)],key=lambda x:x['delta_pp'])[:8]);ds[a+'_minus_'+b]=d
 expected={a:{k:paired([100*(r['posterior_expected_quality']['LiveMean_'+a][k]-r['posterior_expected_quality']['LiveMean_Native'][k])for r in rows])for k in ['v','t']}for a in ['HiRes','HiResMix','Joint','JointMix']};cost=read(D/'COST.json');res=dict(status='completed_exposed_development_probe',parents=128,seeds=p['seeds'],means=means,paired=ds,posterior_expected_quality_delta=expected,rows=rows,cost=cost,new_OPD=False,spatial_frozen=True);put(P/'RESULTS_PUBLIC.json',res)
 audit=dict(status='pass',official_DenseTube_checks=checks,spatial_arrays_exact=spacechecks,Source_CDF_checks=128,direct_MBR_utilities=utilitychecks,vectorized_quality_checks=qualitychecks,max_error=maxerr,native_actual448_control=read(D/'NATIVE_CONTROL.json'),joint65_motion_traces=128,max_GPU_allocated_bytes=max(peaks),native_metrics_exact=True,all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']),weights_unchanged=True,GT_provenance=prov);assert audit['all128_before_GT'];put(P/'INDEPENDENT_AUDIT.json',audit);put(P/'ROOT_AUDIT.json',audit)
 selected='JointMix';gain=ds['LiveMean_JointMix_minus_LiveMean_Native']['v'];inc=ds['LiveMean_JointMix_minus_LiveMean_HiResMix']['v']
 decision=dict(primary='JointMix',primary_delta=gain,context_increment=inc,decision='freeze_for_parent_disjoint_confirmation'if gain['mean']>=2 and gain['CI95'][0]>0 and inc['CI95'][0]>0 else'no_supported_meaningful_point_readout',point_readout_failure_does_not_prove_OPD_failure=True,not_OPD=True,independent_confirmation=False);put(P/'SELECTION_PUBLIC.json',decision)
 lines=['# 同帧联合运动编码时间教师：HC128开发','','冻结同Source/448/原65实际帧；仅VideoSwin联合编码一次65，再物理选回33/32，原appearance/text/decoder长度不变。所有Source新box丢弃；既有E/Live三seed空间严格冻结。不是旧全模型dense2x，也不是完整OPD。全部128重复开发，全部输出seal后新GT。','','| 读出 | 完整vIoU% | tIoU% | v增量pp [CI95] |','|---|---:|---:|---|']
 for arm in p['arms']:
  key='LiveMean_'+arm;v=means[key];c=ds.get(key+'_minus_LiveMean_Native');d='0'if c is None else f'{c["v"]["mean"]:+.6f} [{c["v"]["CI95"][0]:+.6f},{c["v"]["CI95"][1]:+.6f}]';lines.append(f'| {arm} | {100*v["v"]:.6f} | {100*v["t"]:.6f} | {d} |')
 lines+=['','唯一主JointMix固定.5融合；Joint/JointFluct签名诊断不事后换主。既有HiRes/HiResMix/HiResFluct精确缓存复用。混合与读出增量不是蒸馏增量。原448已是开发选择，本轮匹配保持，仍有训练分布变化风险。只有同时>=2pp净收益、正CI且相对HiResMix正CI才冻候选确认。其他正趋势不追称通过。','','```json',json.dumps(decision,indent=2),'```','','## 成本与核验','','```json',json.dumps(cost,indent=2),json.dumps(audit,indent=2),'```','','新增Source/VideoSwin实际次数、单次联合65/两offset读取、历史Native448 control全部计入；既有Source448/原Source/DINO/Refer/空间fit成本不计本轮新增，不能当免费。负例/严重尾/unknown全见RESULTS_PUBLIC；未更新任何生产/论文/空间登记。']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],spatial_config_sha256=sha(R/'methods/stvg_referdino_opd_v1/config.json'),exposure='all RC128 repeatedly exposed development',excluded=['raw pixels','GT coordinates','captions','parent IDs','weights','raw Source/trajectory/posterior caches']))
 for f in P.glob('*.json'):f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in decision.items()if k!='teacher_distribution_diagnostics'},indent=2),flush=True)

if __name__=='__main__':main()
