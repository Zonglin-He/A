"""Post-seal full original-frame Source teacher, same frozen spatial predictions."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,json,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_full_context128_v1 import D,P,PRE,FIN,DEV,HR
from scripts.score_stvg_where_when_event128_v1 import avg,paired

def main():
 import torch
 torch.set_num_threads(4);p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_before_GT']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());assert read(D/'NATIVE_FULL65_CONTROL.json')['status']=='pass'
 for f in bar['files']:assert sha(R/f['path'])==f['sha256']
 put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_before_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')))
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 from scripts.stvg_temporal_marginal_decode_util256_v1 import independent_cdf,probabilities
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
 from vg_tta.stvg_temporal_foreground128_v1 import project
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];oldrows={r['alias']:r for r in read(R/'results/stvg_error_state/2026-10-19/temporal_increment128/RESULTS_PUBLIC.json')['rows']};rows=[];checks=spacechecks=utilitychecks=qualitychecks=0;maxerr=0.;peaks=[]
 for r in p['records']:
  a=r['alias'];z=load(D/'predictions'/f'{a}.pt');old=load(DEV/'priors'/f'{a}.pt');inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');hr=load(HR/'predictions'/f'{a}.pt');values={};expectations={};ids=np.array(z['frame_ids']);N=len(ids);ii,jj=np.triu_indices(N,1);obs=z['full_context_info'];peaks.append(z['peak_GPU_allocated'])
  assert len(ids)==65 and obs['frame_count']==65 and obs['normalized_shape'][0]==65 and obs['normalized_shape'][-2]==224
  assert z['raw_RGB_sha256']==hr['observation']['RGB_sha256']
  ll=z['logits'][0].double();score=ll[ii,0]+ll[jj,1];exact=__import__('torch').softmax(score,0).numpy();err=float(abs(exact-z['outputs']['Full65']['q'][ii,jj]).max());assert err<1e-12;maxerr=max(maxerr,err)
  assert np.array_equal(z['outputs']['Full65Mix']['q'],.5*old['q']+.5*z['outputs']['Full65']['q'])and np.array_equal(z['outputs']['Native']['q'],old['q'])and np.array_equal(z['outputs']['HiResMix']['q'],hr['outputs']['HiResMix']['q'])
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
   for arm in p['arms']:
    q=z['outputs'][arm]['q'][ii,jj];expectations[space+'_'+arm]=dict(v=float(q@v),t=float(q@t))
  for arm in p['arms']:values['LiveMean_'+arm]={k:avg([values[f'Live{s}_{arm}'][k]for s in p['seeds']])for k in ['v','t','s','boundary_MAE']}
  for arm in p['arms']:expectations['LiveMean_'+arm]={k:avg([expectations[f'Live{s}_{arm}'][k]for s in p['seeds']])for k in ['v','t']}
  rows.append(dict(alias=a,values=values,posterior_expected_quality=expectations,full65_shape=obs['normalized_shape']));print('scored',a,flush=True)
 keys=['LiveMean_'+a for a in p['arms']]+['E_Native','E_Full65Mix'];means={key:{k:avg([r['values'][key][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for key in keys};contrasts=[('LiveMean_'+a,'LiveMean_Native')for a in p['arms']if a!='Native']+[('LiveMean_Full65Mix','LiveMean_HiResMix'),('LiveMean_Full65','LiveMean_HiResMix'),('E_Full65Mix','E_Native')];ds={}
 for a,b in contrasts:
  d={k:paired([(100 if k in ['v','t','s']else 1)*(r['values'][a][k]-r['values'][b][k])for r in rows])for k in ['v','t','s','boundary_MAE']};v=np.array([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rows]);d['tails']=dict(positive=int((v>1e-10).sum()),negative=int((v< -1e-10).sum()),flat=int((abs(v)<=1e-10).sum()),below_minus3=int((v< -3).sum()),below_minus5=int((v< -5).sum()),positive_contribution_pp=float(np.maximum(v,0).mean()),negative_contribution_pp=float(np.minimum(v,0).mean()),worst=sorted([dict(alias=r['alias'],delta_pp=float(x))for r,x in zip(rows,v)],key=lambda x:x['delta_pp'])[:8]);ds[a+'_minus_'+b]=d
 expected={a:{k:paired([100*(r['posterior_expected_quality']['LiveMean_'+a][k]-r['posterior_expected_quality']['LiveMean_Native'][k])for r in rows])for k in ['v','t']}for a in ['Full65','Full65Mix','HiResMix']};cost=read(D/'COST.json');res=dict(status='completed_exposed_development_probe',parents=128,seeds=p['seeds'],means=means,paired=ds,posterior_expected_quality_delta=expected,rows=rows,cost=cost,new_OPD=False,spatial_frozen=True);put(P/'RESULTS_PUBLIC.json',res)
 audit=dict(status='pass',official_DenseTube_checks=checks,spatial_arrays_exact=spacechecks,single_head_softmax_checks=128,direct_MBR_utilities=utilitychecks,vectorized_quality_checks=qualitychecks,max_error=maxerr,native_full65_control=read(D/'NATIVE_FULL65_CONTROL.json'),max_GPU_allocated_bytes=max(peaks),native_metrics_exact=True,all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']),weights_unchanged=True,GT_provenance=prov);assert audit['all128_before_GT'];put(P/'INDEPENDENT_AUDIT.json',audit);put(P/'ROOT_AUDIT.json',audit)
 selected=max(p['eligible'],key=lambda a:means['LiveMean_'+a]['v']);gain=ds['LiveMean_'+selected+'_minus_LiveMean_Native']['v']
 decision=dict(primary='Full65Mix',selected_development_route=selected,selected_delta=gain,decision='freeze_for_parent_disjoint_confirmation'if gain['mean']>=2 and gain['CI95'][0]>0 else'no_supported_meaningful_point_readout',posterior_quality_diagnostic=expected,point_readout_failure_does_not_prove_OPD_failure=True,not_OPD=True,independent_confirmation=False);put(P/'SELECTION_PUBLIC.json',decision)
 lines=['# 原65帧完整Source上下文教师：HC128开发','','同Source224/同原65帧/同文本/同checkpoint；一个完整65帧Source前向与原FP32两pass replay，而非两个独立33/32再envelope。同时改变完整多模态融合上下文与decoder长度，不能将效果单独归某一层。并非dense129新增帧，也非新OPD。第一旧NH001完整65 stream FP32 logits必须exact复现历史缓存，不读它的GT；原旧Dense−9.2619是HC16而不是HC32，不能自动判单stream。当前128通常是不同clip，不能偷用旧16logits替代；本轮全128重算Source。空间E/Live三seed数组严格不动，新Source框全部丢弃。','','| 输出 | 完整vIoU% | tIoU% | v增量pp [CI95] |','|---|---:|---:|---|']
 for arm in p['arms']:
  key='LiveMean_'+arm;v=means[key];c=ds.get(key+'_minus_LiveMean_Native');d='0'if c is None else f'{c["v"]["mean"]:+.6f} [{c["v"]["CI95"][0]:+.6f},{c["v"]["CI95"][1]:+.6f}]';lines.append(f'| {arm} | {100*v["v"]:.6f} | {100*v["t"]:.6f} | {d} |')
 lines+=['','Full65Mix固定.5p0+.5q65；只两个eligible在整个128开发父均值选。原65物理决策支持、相同MBR，没有GT选流/逐视频选赢家。缓存HiResMix强比较保留。点结果/teacher分布期望GT质量分别报告，均不是学生已完成OPD。>=2pp且CI正才冻结独立确认，开发选优CI未经多重校正，仍需新父确认。','','```json',json.dumps(decision,indent=2),'```','','## 成本与核验','','```json',json.dumps(cost,indent=2),json.dumps(audit,indent=2),'```','','新增Source实际完整65调用与旧无GT数值控制计费。原DINO/Refer/空间fit和Source缓存成本不是免费；无新专家/网络训练/空间更新。预测全128 seal后新GT，所有signed/unknown/尾部保留，生产/论文/空间登记不改。']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],spatial_config_sha256=sha(R/'methods/stvg_referdino_opd_v1/config.json'),exposure='all RC128 repeatedly exposed development; old NH001 control no newGT',excluded=['raw pixels','GT coordinates','captions','parent IDs','weights','raw Source/trajectory/posterior caches']))
 for f in P.glob('*.json'):f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in decision.items()if k!='teacher_distribution_diagnostics'},indent=2),flush=True)

if __name__=='__main__':main()
