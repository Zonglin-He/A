"""Post-seal conditional event query capability with sampled-state and fixed-teacher controls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,json,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_event_query128_v1 import D,P,PRE,FIN,DEV,HR,ST,MEM
from scripts.score_stvg_where_when_event128_v1 import avg,paired

def main():
 import torch
 torch.set_num_threads(4);p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_before_GT']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());cost_before=read(D/'COST.json');assert cost_before['conditioned_temporal_offset_decodes']==3328 and cost_before['native_temporal_checks']==256 and cost_before['spatial_attention_head_calls']==256
 for f in bar['files']:assert sha(R/f['path'])==f['sha256']
 put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_before_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')))
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 from scripts.stvg_temporal_marginal_decode_util256_v1 import independent_cdf,probabilities
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
 from vg_tta.stvg_temporal_foreground128_v1 import project
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];oldrows={r['alias']:r for r in read(R/'results/stvg_error_state/2026-10-19/temporal_increment128/RESULTS_PUBLIC.json')['rows']};rows=[];checks=spacechecks=utilitychecks=qualitychecks=0;maxerr=0.;cdfchecks=0
 for r in p['records']:
  a=r['alias'];z=load(D/'predictions'/f'{a}.pt');old=load(DEV/'priors'/f'{a}.pt');inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');hr=load(HR/'predictions'/f'{a}.pt');state=load(ST/'predictions'/f'{a}.pt');values={};expectations={};sample_before={};ids=np.array(z['frame_ids']);N=len(ids);ii,jj=np.triu_indices(N,1)
  assert np.array_equal(z['q0'],old['q'])and np.array_equal(z['q1'],state['outputs']['Live41']['q'])
  for name,ll in z['logits'].items():
   q=independent_cdf(probabilities(inp['records'],ll),ids.tolist());err=float(abs(q-z['outputs'][name]['q']).max());assert err<1e-12;maxerr=max(maxerr,err);cdfchecks+=1
  for name in ['GAll','GOcc0','RAll','ROcc0','ROcc1']:assert np.array_equal(z['outputs'][name+'Mix']['q'],.5*z['q0']+.5*z['outputs'][name]['q'])
  for state_index in [0,1]:assert np.array_equal(z['outputs'][f'Sample{state_index}Mean']['q'],sum(z['outputs'][f'Sample{state_index}_{k}']['q']for k in range(4))/4)
  for arm,o in z['outputs'].items():
   if 'q'not in o:continue
   d=final_mbr(ids.tolist(),o['q']);assert d['interval']==o['interval'];pick=np.unique(np.r_[np.linspace(0,len(ii)-1,17).astype(int),np.where((ii==d['indices'][0])&(jj==d['indices'][1]))[0]]);exact=brute_utility(ids.tolist(),o['q'],pick);err=float(abs(exact-d['utility_grid'][ii[pick],jj[pick]]).max());assert err<1e-11;maxerr=max(maxerr,err);utilitychecks+=len(pick)
  S,E=ids[ii],ids[jj]+1;g,h=spans[r['query_ordinal']];overlap=np.maximum(0,np.minimum(E,h)-np.maximum(S,g));t=overlap/(E-S+h-g-overlap)
  for space,boxes in z['trajectories'].items():
   assert np.array_equal(boxes,inp['E']if space=='E'else sp['outputs'][f'T20_Live_{space[4:]}']);spacechecks+=1;tube=DenseTube(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True)
   for arm,o in dict(**z['outputs'],HiResMix=hr['outputs']['HiResMix']).items():
    iv=o['interval'];v=tube.score(iv);off=official(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;checks+=1
    if arm=='Native':assert max(abs(v[k]-oldrows[a]['values'][space+'_L0'][k])for k in ['v','t','s'])<2e-10
    values[space+'_'+arm]=dict(**v,interval=iv,boundary_MAE=(abs(iv[0]-g)+abs(iv[1]-h))/2)
   lo=np.searchsorted(tube.fids,np.maximum(S,g));hi=np.maximum(lo,np.searchsorted(tube.fids,np.minimum(E,h)));prefix=np.r_[0.,np.cumsum(tube.iou)];v=(prefix[hi]-prefix[lo])/np.maximum(np.maximum(E,h)-np.minimum(S,g),1)
   for k in np.unique(np.linspace(0,len(ii)-1,19).astype(int)):
    actual=tube.score([S[k],E[k]]);assert abs(v[k]-actual['v'])<1e-12 and abs(t[k]-actual['t'])<1e-12;qualitychecks+=1
   vmat=np.zeros((N,N));tmat=np.zeros((N,N));vmat[ii,jj]=v;tmat[ii,jj]=t
   for state_index,pairs in enumerate(z['sampled_pairs']):sample_before[space+f'_State{state_index}']=dict(v=[float(vmat[i,j])for i,j in pairs],t=[float(tmat[i,j])for i,j in pairs])
   for arm in p['arms']:
    q=z['outputs'][arm]['q'][ii,jj];expectations[space+'_'+arm]=dict(v=float(q@v),t=float(q@t))
  for arm in p['arms']+['HiResMix']:values['LiveMean_'+arm]={k:avg([values[f'Live{s}_{arm}'][k]for s in p['seeds']])for k in ['v','t','s','boundary_MAE']}
  for arm in p['arms']:expectations['LiveMean_'+arm]={k:avg([expectations[f'Live{s}_{arm}'][k]for s in p['seeds']])for k in ['v','t']}
  rows.append(dict(alias=a,values=values,posterior_expected_quality=expectations,sampled_interval_quality=sample_before,state1_changed=z['outputs']['Student1']['interval']!=z['outputs']['Native']['interval'],state_TV=float(.5*abs(z['q1']-z['q0']).sum()),teacher_state_TV=float(.5*abs(z['outputs']['ROcc1']['q']-z['outputs']['ROcc0']['q']).sum())));print('scored',a,flush=True)
 keys=['LiveMean_'+a for a in p['arms']]+['E_Native','E_ROcc0Mix','LiveMean_HiResMix'];means={key:{k:avg([r['values'][key][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for key in keys};contrasts=[('LiveMean_'+a,'LiveMean_Native')for a in p['arms']if a!='Native']+[('LiveMean_ROcc0Mix','LiveMean_GOcc0Mix'),('LiveMean_ROcc0Mix','LiveMean_RAllMix'),('LiveMean_ROcc0Mix','LiveMean_HiResMix'),('LiveMean_ROcc1Mix','LiveMean_ROcc0Mix'),('LiveMean_Sample0Mean','LiveMean_ROcc0'),('LiveMean_Sample1Mean','LiveMean_ROcc0'),('E_ROcc0Mix','E_Native')];ds={}
 for a,b in contrasts:
  d={k:paired([(100 if k in ['v','t','s']else 1)*(r['values'][a][k]-r['values'][b][k])for r in rows])for k in ['v','t','s','boundary_MAE']};v=np.array([100*(r['values'][a]['v']-r['values'][b]['v'])for r in rows]);d['tails']=dict(positive=int((v>1e-10).sum()),negative=int((v< -1e-10).sum()),flat=int((abs(v)<=1e-10).sum()),below_minus3=int((v< -3).sum()),below_minus5=int((v< -5).sum()),positive_contribution_pp=float(np.maximum(v,0).mean()),negative_contribution_pp=float(np.minimum(v,0).mean()),worst=sorted([dict(alias=r['alias'],delta_pp=float(x))for r,x in zip(rows,v)],key=lambda x:x['delta_pp'])[:8]);ds[a+'_minus_'+b]=d
 expected={a:{k:paired([100*(r['posterior_expected_quality']['LiveMean_'+a][k]-r['posterior_expected_quality']['LiveMean_Native'][k])for r in rows])for k in ['v','t']}for a in [a for a in p['arms']if a!='Native']};cost=read(D/'COST.json');res=dict(status='completed_exposed_development_probe',parents=128,seeds=p['seeds'],means=means,paired=ds,posterior_expected_quality_delta=expected,rows=rows,cost=cost,new_OPD=False,spatial_frozen=True);put(P/'RESULTS_PUBLIC.json',res)
 audit=dict(status='pass',official_DenseTube_checks=checks,spatial_arrays_exact=spacechecks,teacher_CDF_checks=cdfchecks,direct_MBR_utilities=utilitychecks,vectorized_quality_checks=qualitychecks,max_error=maxerr,native224_temporal_checks=256,query_conditions_per_view=13,native_metrics_exact=True,all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']),weights_unchanged=True,GT_provenance=prov);assert audit['all128_before_GT'];put(P/'INDEPENDENT_AUDIT.json',audit);put(P/'ROOT_AUDIT.json',audit)
 selected=max(p['eligible'],key=lambda a:means['LiveMean_'+a]['v']);gain=ds['LiveMean_'+selected+'_minus_LiveMean_Native']['v'];sample_quality={}
 for sample in ['Sample0Mean','Sample1Mean']:
  sample_quality[sample]={base:{k:paired([100*(r['posterior_expected_quality']['LiveMean_'+sample][k]-r['posterior_expected_quality']['LiveMean_'+base][k])for r in rows])for k in ['v','t']}for base in ['Native','ROcc0']}
 sample_quality['fresh_occupancy_vs_fixed']={k:paired([100*(r['posterior_expected_quality']['LiveMean_ROcc1'][k]-r['posterior_expected_quality']['LiveMean_ROcc0'][k])for r in rows])for k in ['v','t']}
 decision=dict(primary='ROcc0Mix',selected_development_route=selected,selected_delta=gain,decision='freeze_for_parent_disjoint_confirmation'if gain['mean']>=2 and gain['CI95'][0]>0 else'no_supported_meaningful_point_readout',sample_teacher_expected_quality_vs_fixed=sample_quality,teacher_state_changes_do_not_imply_correctness=True,not_OPD=True,independent_confirmation=False);put(P/'SELECTION_PUBLIC.json',decision)
 lines=['# 目标×学生事件状态条件时间query：HC128能力检查','','仅已有Source224融合motion/text；冻结Source原始时间decoder/head，保留全局memory/time/mask。用已有Refer八框插值区域和当前时间占有率形成norm匹配query。Source空间decoder0次；最终空间仍原E/Live三数组。所有网络参数不动。G/R分别全图/Refer区域，All/Occ0分别全时/原p0的节点事件占有率，Occ1来自旧真实OPD Live41最终分布。另对S0/S1各四个实际采样区间查询教师，不用GT选候选。不是新OPD成绩。','','| 读出 | 完整vIoU% | tIoU% | v增量pp [CI95] |','|---|---:|---:|---|']
 for arm in p['arms']:
  key='LiveMean_'+arm;v=means[key];c=ds.get(key+'_minus_LiveMean_Native');d='0'if c is None else f'{c["v"]["mean"]:+.6f} [{c["v"]["CI95"][0]:+.6f},{c["v"]["CI95"][1]:+.6f}]';lines.append(f'| {arm} | {100*v["v"]:.6f} | {100*v["t"]:.6f} | {d} |')
 lines+=['','所有Mix固定.5p0+.5对应teacher；Sample均值为四份teacher posterior平均，不是学生最终适应。两个eligible仅GOcc0Mix/ROcc0Mix，按全128开发父均值选，点CI非选优校正。>=2pp且CI正才冻结确认；WHERE→WHEN因果还看相对GOcc0Mix/RAllMix，不用总收益替代。软目标期望/采样纠错与点读出分开；hard阴性不等于OPD普遍失败，target变化也不等于正确。旧ROIquery32与固定memory128负结果仍成立于各自配置。','','```json',json.dumps(decision,indent=2),'```','','## 计费与核验','','```json',json.dumps(cost,indent=2),json.dumps(audit,indent=2),'```','','新增为冻结TemporalDecoder/原spatial-attention classifier调用，无Source encoder/RGB/DINO/Refer/网络或空间训练；旧缓存计算不是免费。所有128先seal后GT；全signed/unknown/负尾及教师期望质量见完整JSON，生产/论文/空间登记不改。']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],spatial_config_sha256=sha(R/'methods/stvg_referdino_opd_v1/config.json'),exposure='all RC128 repeatedly exposed development',excluded=['raw pixels','GT coordinates','captions','parent IDs','weights','raw Source/trajectory/posterior caches']))
 for f in P.glob('*.json'):f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in decision.items()if k!='teacher_distribution_diagnostics'},indent=2),flush=True)

if __name__=='__main__':main()
