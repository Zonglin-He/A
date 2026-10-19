"""CPU, post-global-seal semantic-ablation score and independent audit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,json,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_semantic_contrast128_v1 import D,P,PRE,FIN,DEV
from scripts.score_stvg_where_when_event128_v1 import avg,paired

def main():
 import torch
 torch.set_num_threads(4);p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_before_GT'] and bar['plan_sha256']==sha(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items())
 for f in bar['files']:assert sha(R/f['path'])==f['sha256']
 put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_before_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')))
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 from scripts.stvg_temporal_marginal_decode_util256_v1 import independent_cdf,probabilities
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];oldrows={r['alias']:r for r in read(R/'results/stvg_error_state/2026-10-19/temporal_increment128/RESULTS_PUBLIC.json')['rows']};rows=[];checks=spacechecks=cdfchecks=posteriorchecks=utilitychecks=missing=0;maxerr=0.
 for r in p['records']:
  a=r['alias'];z=load(D/'predictions'/f'{a}.pt');old=load(DEV/'priors'/f'{a}.pt');inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');values={};q0=old['q'].astype(np.longdouble);mask=q0>0;expected_square=q0*q0;expected_square/=expected_square.sum();obs=z['static_observation'];expected_contrast=q0.copy();missing+=z['static_missing']
  if obs and not obs.get('exact_same_caption_reuse',False):
   qs=independent_cdf(probabilities(obs['records'],obs['logits']),z['frame_ids']);err=float(abs(qs-obs['q']).max());assert err<1e-12;maxerr=max(maxerr,err);cdfchecks+=1;assert np.array_equal(qs>0,mask);expected_contrast=np.zeros_like(q0);expected_contrast[mask]=q0[mask]*q0[mask]/np.asarray(obs['q'],dtype=np.longdouble)[mask];expected_contrast/=expected_contrast.sum()
  for name,q in [('PriorSquare',expected_square),('Contrast',expected_contrast)]:
   err=float(abs(np.asarray(q,dtype=float)-z['outputs'][name]['q']).max());assert err<1e-12;maxerr=max(maxerr,err);posteriorchecks+=1
  assert np.array_equal(z['outputs']['Native']['q'],old['q'])
  for arm,o in z['outputs'].items():
   d=final_mbr(z['frame_ids'],o['q']);assert d['interval']==o['interval'];N=len(z['frame_ids']);ii,jj=np.triu_indices(N,1);pick=np.unique(np.r_[np.linspace(0,len(ii)-1,17).astype(int),np.where((ii==d['indices'][0])&(jj==d['indices'][1]))[0]]);exact=brute_utility(z['frame_ids'],o['q'],pick);err=float(abs(exact-d['utility_grid'][ii[pick],jj[pick]]).max());assert err<1e-11;maxerr=max(maxerr,err);utilitychecks+=len(pick)
  for space,boxes in z['trajectories'].items():
   assert np.array_equal(boxes,inp['E']if space=='E'else sp['outputs'][f'T20_Live_{space[4:]}']);spacechecks+=1;tube=DenseTube(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True)
   for arm,o in z['outputs'].items():
    iv=o['interval'];v=tube.score(iv);off=official(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;checks+=1
    if arm=='Native':assert max(abs(v[k]-oldrows[a]['values'][space+'_L0'][k])for k in ['v','t','s'])<2e-10
    values[space+'_'+arm]=dict(**v,interval=iv,boundary_MAE=(abs(iv[0]-spans[r['query_ordinal']][0])+abs(iv[1]-spans[r['query_ordinal']][1]))/2)
  for arm in p['arms']:values['LiveMean_'+arm]={k:avg([values[f'Live{s}_{arm}'][k]for s in p['seeds']])for k in ['v','t','s','boundary_MAE']}
  rows.append(dict(alias=a,static_missing=z['static_missing'],values=values));print('scored',a,flush=True)
 keys=['LiveMean_'+a for a in p['arms']]+['E_'+a for a in p['arms']];means={key:{k:avg([r['values'][key][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for key in keys};contrasts=[('LiveMean_Contrast','LiveMean_Native'),('LiveMean_PriorSquare','LiveMean_Native'),('LiveMean_Contrast','LiveMean_PriorSquare'),('E_Contrast','E_Native')];ds={}
 for a,b in contrasts:
  d={k:paired([(100 if k in ['v','t','s']else 1)*(r['values'][a][k]-r['values'][b][k])for r in rows])for k in ['v','t','s','boundary_MAE']};v=np.array([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rows]);d['tails']=dict(positive=int((v>1e-10).sum()),negative=int((v< -1e-10).sum()),flat=int((abs(v)<=1e-10).sum()),below_minus3=int((v< -3).sum()),below_minus5=int((v< -5).sum()),positive_contribution_pp=float(np.maximum(v,0).mean()),negative_contribution_pp=float(np.minimum(v,0).mean()),worst=sorted([dict(alias=r['alias'],delta_pp=float(x))for r,x in zip(rows,v)],key=lambda x:x['delta_pp'])[:8]);ds[a+'_minus_'+b]=d
 costs=[read(f)for f in D.glob('COST_*.json')];cost=dict(GPU_seconds=sum(x['GPU_seconds']for x in costs),new_Source_queries=sum(x['new_Source_queries']for x in costs),Source_offset_forwards=sum(x['Source_offset_forwards']for x in costs),new_DINO=0,new_ReferDINO=0,network_training=0,spatial_updates=0,attempts=costs);res=dict(status='completed_exposed_development_probe',parents=128,seeds=p['seeds'],means=means,paired=ds,rows=rows,cost=cost,new_OPD=False,spatial_frozen=True,GT_offline_only=True);put(P/'RESULTS_PUBLIC.json',res)
 audit=dict(status='pass',official_DenseTube_checks=checks,spatial_arrays_exact=spacechecks,independent_static_CDF=cdfchecks,longdouble_direct_posterior_checks=posteriorchecks,direct_MBR_utilities=utilitychecks,maximum_error=maxerr,missing_static_noop=missing,all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']),native_metrics_exact=True,weights_unchanged=True,GT_provenance=prov);assert audit['all128_before_GT'];put(P/'INDEPENDENT_AUDIT.json',audit);put(P/'ROOT_AUDIT.json',audit)
 gain=ds['LiveMean_Contrast_minus_LiveMean_Native']['v'];increment=ds['LiveMean_Contrast_minus_LiveMean_PriorSquare']['v'];go=gain['mean']>=2 and gain['CI95'][0]>0 and increment['CI95'][0]>0;decision=dict(primary='Contrast',delta=gain,semantic_increment_over_sharpening=increment,decision='freeze_for_parent_disjoint_confirmation'if go else'no_supported_meaningful_semantic_increment',not_OPD=True,independent_confirmation=False);put(P/'SELECTION_PUBLIC.json',decision)
 lines=['# 完整描述与静态对象描述：HC128时间对比','','同一Source、原65物理帧，两offset。静态文本仅使用旧缓存parses.old.phrase，127可用/1缺失noop，不补类别、不用GT改提示。所有新空间输出丢弃，最终空间四轨迹保持冻结。Contrast=p0²/qStatic，仅固定单位系数；PriorSquare=p0²控制单纯锐化。全部仍用同一finalMBR，不把语义消融等同纯事件因果分解或校准概率。本轮仅能力探针，非OPD。','','| 读出 | 完整vIoU% | tIoU% |','|---|---:|---:|']
 for arm in p['arms']:
  v=means['LiveMean_'+arm];lines.append(f'| {arm} | {100*v["v"]:.6f} | {100*v["t"]:.6f} |')
 lines+=['','```json',json.dumps(decision,indent=2),'```','','全部RC128反复曝光开发，未做独立确认。任何正均值不自动晋升；能力成立后仍需冻结确认/Direct/Fixed/Live才可谈OPD。完整signed/unknown、双对照与负尾在RESULTS_PUBLIC.json，不删无反馈父或挑种子。','','## 成本与审计','','```json',json.dumps(cost,indent=2),json.dumps(audit,indent=2),'```','','同RGB被新文本重新编码照常计费，不称免费缓存推理；不新增DINO/Refer/预训练权重或空间state更新。原Source checkpoint、current spatial config和登记不变。']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],spatial_config_sha256=sha(R/'methods/stvg_referdino_opd_v1/config.json'),exposure='all RC128 repeatedly exposed development',private_excluded=['captions','static phrases','parent IDs','media','GT coordinates','weights','raw posteriors/features']))
 for f in P.glob('*.json'):f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 print(json.dumps(decision,indent=2),flush=True)

if __name__=='__main__':main()
