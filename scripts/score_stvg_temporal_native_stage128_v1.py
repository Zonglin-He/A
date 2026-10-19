"""All-parent post-seal evaluation of native temporal stages and auxiliary heads."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,json,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_native_stage128_v1 import D,P,MEM,PRE,FIN,DEV,HR,ARMS
from scripts.score_stvg_where_when_event128_v1 import avg,paired


def main():
 import torch
 torch.set_num_threads(4);p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_before_GT']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items())
 assert read(P/'STAGE_INDEPENDENT_AUDIT.json')['status']=='pass'
 for f in bar['files']:assert sha(R/f['path'])==f['sha256']
 put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_before_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')))
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];base={r['alias']:r for r in read(R/'results/stvg_error_state/2026-10-19/temporal_increment128/RESULTS_PUBLIC.json')['rows']};rows=[];checks=spaces=utilities=qualities=0;maxerr=0
 for r in p['records']:
  a=r['alias'];z=load(D/'predictions'/f'{a}.pt');inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');values={};expected={};ids=np.array(z['frame_ids']);ii,jj=np.triu_indices(len(ids),1)
  for arm,o in z['outputs'].items():
   dec=final_mbr(ids.tolist(),o['q']);assert dec['interval']==o['interval'];pick=np.unique(np.r_[np.linspace(0,len(ii)-1,17).astype(int),np.where((ii==dec['indices'][0])&(jj==dec['indices'][1]))[0]]);ex=brute_utility(ids.tolist(),o['q'],pick);err=float(abs(ex-dec['utility_grid'][ii[pick],jj[pick]]).max());assert err<1e-11;maxerr=max(maxerr,err);utilities+=len(pick)
  S,E=ids[ii],ids[jj]+1;g,h=spans[r['query_ordinal']];overlap=np.maximum(0,np.minimum(E,h)-np.maximum(S,g));t=overlap/(E-S+h-g-overlap)
  for space,boxes in z['trajectories'].items():
   assert np.array_equal(boxes,inp['E']if space=='E'else sp['outputs'][f'T20_Live_{space[4:]}']);spaces+=1;tube=DenseTube(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True);metric_cache={}
   for arm,o in z['outputs'].items():
    iv=o['interval'];key=tuple(iv)
    if key not in metric_cache:
     v=tube.score(iv);off=official(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;checks+=1;metric_cache[key]=v
    v=metric_cache[key]
    if arm=='Native':assert max(abs(v[k]-base[a]['values'][space+'_L0'][k])for k in ['v','t','s'])<2e-10
    values[space+'_'+arm]=dict(**v,interval=iv,boundary_MAE=(abs(iv[0]-g)+abs(iv[1]-h))/2)
   lo=np.searchsorted(tube.fids,np.maximum(S,g));hi=np.maximum(lo,np.searchsorted(tube.fids,np.minimum(E,h)));prefix=np.r_[0.,np.cumsum(tube.iou)];v=(prefix[hi]-prefix[lo])/np.maximum(np.maximum(E,h)-np.minimum(S,g),1)
   for k in np.unique(np.linspace(0,len(ii)-1,19).astype(int)):
    actual=tube.score([S[k],E[k]]);assert abs(v[k]-actual['v'])<1e-12 and abs(t[k]-actual['t'])<1e-12;qualities+=1
   for arm,o in z['outputs'].items():expected[space+'_'+arm]=dict(v=float(o['q'][ii,jj]@v),t=float(o['q'][ii,jj]@t))
  for arm in ARMS:
   values['LiveMean_'+arm]={k:avg([values[f'Live{s}_{arm}'][k]for s in p['seeds']])for k in ['v','t','s','boundary_MAE']};expected['LiveMean_'+arm]={k:avg([expected[f'Live{s}_{arm}'][k]for s in p['seeds']])for k in ['v','t']}
  traces=z['native_stage_traces'];stats=dict(changed_native_gates=sum(t['first']!=t['second']for t in traces),native_stage_teacher_TV=float(abs(z['outputs']['P1L6']['q']-z['outputs']['Native']['q']).sum()/2),query_stage_delta_mean=float(np.mean([float((t['query1']-t['query2']).norm())for t in traces])))
  rows.append(dict(alias=a,values=values,posterior_expected_quality=expected,diagnostics=stats));print('scored',a,flush=True)
 means={a:{k:avg([r['values']['LiveMean_'+a][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for a in ARMS}
 pairs=[(a,'Native')for a in ARMS if a!='Native']+[('HiResAuxMix','HiResMix'),('Pass2Mean6','NativeUniformMix'),('HiResAuxMix','HiResUniformMix'),('Pass1Mean6','Pass2Mean6'),('FirstPassMix','HiResMix')];ds={};exp={}
 for a,b in pairs:
  key=a+'_minus_'+b;d={k:paired([(100 if k in ['v','t','s']else 1)*(r['values']['LiveMean_'+a][k]-r['values']['LiveMean_'+b][k])for r in rows])for k in ['v','t','s','boundary_MAE']};v=np.array([100*(r['values']['LiveMean_'+a]['v']-r['values']['LiveMean_'+b]['v'])for r in rows]);d['tails']=dict(positive=int((v>1e-10).sum()),negative=int((v< -1e-10).sum()),flat=int((abs(v)<=1e-10).sum()),below_minus3=int((v< -3).sum()),below_minus5=int((v< -5).sum()),positive_contribution_pp=float(np.maximum(v,0).mean()),negative_contribution_pp=float(np.minimum(v,0).mean()),worst=sorted([dict(alias=r['alias'],delta_pp=float(x))for r,x in zip(rows,v)],key=lambda x:x['delta_pp'])[:8]);ds[key]=d;exp[key]={k:paired([100*(r['posterior_expected_quality']['LiveMean_'+a][k]-r['posterior_expected_quality']['LiveMean_'+b][k])for r in rows])for k in ['v','t']}
 cost=read(D/'COST.json');res=dict(status='completed_exposed_development_capability_probe',parents=128,seeds=p['seeds'],means=means,paired=ds,posterior_expected_quality_delta=exp,rows=rows,cost=cost,spatial_frozen=True,new_OPD=False);put(P/'RESULTS_PUBLIC.json',res)
 audit=dict(status='pass',official_DenseTube_checks=checks,spatial_arrays_exact=spaces,direct_MBR_utilities=utilities,vectorized_quality_checks=qualities,max_error=maxerr,all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']),numeric_independent=read(P/'STAGE_INDEPENDENT_AUDIT.json'),GT_provenance=prov);assert audit['all128_before_GT'];put(P/'ROOT_AUDIT.json',audit);put(P/'INDEPENDENT_AUDIT.json',audit)
 selected=min(p['eligible'],key=lambda a:(-means[a]['v'],a));gain=ds[selected+'_minus_Native']['v'];decision=dict(primary=p['primary'],selected=selected,selected_delta=gain,decision='candidate_requires_independent_confirmation'if gain['mean']>=2 and gain['CI95'][0]>0 else'no_supported_meaningful_readout',ranking_diagnostics_available=True,not_OPD=True,independent_confirmation=False);put(P/'SELECTION_PUBLIC.json',decision)
 lines=['# 原生时间解码阶段与六层时间头：HC128开发','','固定原Source224融合特征和E/Live三seed空间数组。严格原生temporal/spatial frame gate及attention query factory复原第一次时间解码，再由原actionness gate复原第二次query；第二次query、最后层logits和原Native必须exact。读取两次解码各六层既有时间头，不加新头/参数/像素。源码及配置为中间层提供aux loss，但无实际训练调用日志，不能宣称第一次唯一被训练或一定更强。','','| 输出 | full vIoU% | tIoU% | 相对Native v增量pp [CI95] |','|---|---:|---:|---|']
 for arm in ARMS:
  v=means[arm];c=ds.get(arm+'_minus_Native');d='0'if c is None else f'{c["v"]["mean"]:+.6f} [{c["v"]["CI95"][0]:+.6f},{c["v"]["CI95"][1]:+.6f}]';lines.append(f'| {arm} | {100*v["v"]:.6f} | {100*v["t"]:.6f} | {d} |')
 lines+=['','主HiResAuxMix=.5已有Source448 raw+.5第二次六层posterior均值，Avg6退化Native时精确回到已有HiResMix到舍入误差。FirstPassMix、Pass1/Pass2Mean6、BothMean12与主方案五eligible只在整个开发父均值选择。逐层P1/P2L1..6只是诊断，不按GT挑一层晋升；Uniform同正质量支持的两控制，检查收益是否只是变平。所有后验先逐层独立建立两个offset envelope，再等权平均，不能先平均logits后声称相同。','','```json',json.dumps(decision,indent=2),'```','','## 成本与独立核验','','```json',json.dumps(cost,indent=2),json.dumps(audit,indent=2),'```','','所有128父和所有输出signed/未知/负尾全留；全部封存后GT，三个空间seed父内平均。原生阶段改变不是新的on-policy学生更新；无新OPD成绩。若只有静态teacher读出改善，不能归OPD。没有改空间或CURRENT/MAIN/PAPER。先确认有效教师分布再有依据设计实际Direct/Fixed/Live，不以层选择或优化强度延长阴性配置。']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],spatial_config_sha256=sha(R/'methods/stvg_referdino_opd_v1/config.json'),exposure='all RC128 repeatedly exposed development; no new targetGT on TC109',excluded=['media','GT coordinates','captions','parent IDs','weights','raw caches']))
 for f in P.glob('*.json'):f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 print(json.dumps(decision,indent=2),flush=True)

if __name__=='__main__':main()
