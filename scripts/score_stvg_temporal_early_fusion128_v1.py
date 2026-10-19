"""Sealed early-fusion teacher readout and distributional capability scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_early_fusion128_v1 import D,P,MEM,STRONG,PRE,FIN,DEV
from scripts.score_stvg_where_when_event128_v1 import avg,paired

def main():
 import torch
 from main_method.runtime import resource_guard
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,independent_cdf
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 torch.set_num_threads(4);beg=time.perf_counter();p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_before_GT']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json')and all(sha(R/f)==h for f,h in p['pins'].items());assert all(sha(R/f['path'])==f['sha256']for f in bar['files']+bar['early_inputs']+bar['regional_prefixes']);cdfchecks=masks=spatialchecks=0;maxcdf=0.;geometrychecks=0;geometrymax=0.;scalarcells=0;geometry_exact=0;layoutchecks=0
 from vg_tta.stvg_temporal_object_memory128_v1 import prior_mask
 for r in p['records']:
  resource_guard();assert time.perf_counter()-beg<900;a=r['alias'];z=load(D/'predictions'/f'{a}.pt');cap=load(D/'early_inputs'/f'{a}.pt');prefix=load(D/'prefixes'/f'{a}.pt');assert prefix['native_early_replay_exact'];assert cap['native_exact']and z['Source_native_exact']and z['early_native_exact']
  for name,ll in z['logits'].items():
   q=independent_cdf(probabilities(z['records'],ll),z['frame_ids']);err=float(abs(q-z['outputs'][name]['q']).max());assert err<2e-12;maxcdf=max(maxcdf,err);cdfchecks+=1
  for off,(c,v)in enumerate(zip(cap['captured_encoder_inputs'],cap['original_views'])):
   n=c['kwargs']['src_key_padding_mask'];assert n.dtype==torch.bool and torch.equal(n,v['info']['encoded_mask']);assert torch.equal(prefix['new_prefixes']['Native'][off],v['prefix']);masks+=1;h,w=v['info']['fea_map_size'];l=h*w
   for name,boxes in [('EarlyRefer',z['refer_boxes']),('EarlyShift',z['shifted_boxes'])]:
    _,meta=prior_mask(n,~n[:,-l:],torch.as_tensor(boxes[off::2],dtype=torch.float64),h,w);saved=z['prior_metadata'][name][off]['prior'];ww=meta['weights'].numpy();actual=saved['weights'].numpy();geometry_exact+=int(np.array_equal(ww,actual));summary=z['prior_metadata'][name][off];assert len(summary['actual_attention_layout_checks'])==summary['encoder_attention_calls'];
    for lay in summary['actual_attention_layout_checks']:assert lay['first_last_query_rows_same_bias']and lay['padding_folded_only_in_encoder']and lay['original_queries_values_unchanged'];layoutchecks+=1
    err=float(np.max(abs(ww-actual)/np.maximum(1,abs(actual))));assert err<=16*np.finfo(np.float32).eps and meta['empty']==saved['empty'];geometrymax=max(geometrymax,err)
    for t,b in enumerate(boxes[off::2]):
     valid=(~n[t,-l:]).numpy().reshape(h,w);vh=int(valid.any(1).sum());vw=int(valid.any(0).sum());cx,cy,bw,bh=map(float,b);coverage=np.zeros((h,w),np.float64)
     for y in range(vh):
      for x in range(vw):coverage[y,x]=max(0,min(1,min((cx+bw/2)*vw,x+1)-max((cx-bw/2)*vw,x)))*max(0,min(1,min((cy+bh/2)*vh,y+1)-max((cy-bh/2)*vh,y)));scalarcells+=1
     cc=coverage.reshape(-1).astype(np.float32)*valid.reshape(-1);mass=float(cc.sum(dtype=np.float32));N=int(valid.sum());expected=np.ones(l,np.float64)if mass<=0 else .5+.5*cc.astype(np.float64)*(N/mass);expected[~valid.reshape(-1)]=1;err=float(np.max(abs(expected-actual[t])/np.maximum(1,abs(actual[t]))));assert err<=16*np.finfo(np.float32).eps;geometrymax=max(geometrymax,err)
    geometrychecks+=1
  inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt')
  for k,boxes in z['trajectories'].items():assert np.array_equal(boxes,inp['E']if k=='E'else sp['outputs'][f'T20_Live_{k[4:]}']);spatialchecks+=1
 audit=dict(status='pass',native_Source_prefix_final_time_exact_parents=128,fullframe_exact_pilots=2,encoder_bool_masks_unchanged=masks,old_prior_geometry_replays=geometrychecks,diagnostic_bitexact_geometry_replays=geometry_exact,actual_attention_layouts_audited=layoutchecks,scalar_geometry_cells=scalarcells,geometry_max_relative_scaled_error=geometrymax,geometry_tolerance_relative_16eps32=16*np.finfo(np.float32).eps,independent_CDF_tables=cdfchecks,max_CDF_error=maxcdf,frozen_spatial_arrays_exact=spatialchecks,new_GT=False);put(D/'AUDIT_BEFORE_GT.json',audit)
 put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_before_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')))
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];oldlate={r['alias']:r for r in read(R/'results/stvg_error_state/2026-10-19/temporal_object_memory128/RESULTS_PUBLIC.json')['rows']};oldstrong={r['alias']:r for r in read(R/'results/stvg_error_state/2026-10-20/temporal_native_stage128/RESULTS_PUBLIC.json')['rows']};ii,jj=np.triu_indices(65,1);rows=[];officialchecks=scalarcalls=0
 for r in p['records']:
  resource_guard();assert time.perf_counter()-beg<900;a=r['alias'];z=load(D/'predictions'/f'{a}.pt');ids=np.array(z['frame_ids']);values={};curves={k:[]for k in ['v','t']}
  for space,boxes in z['trajectories'].items():
   tube=DenseTube(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True)
   for name,o in z['outputs'].items():
    assert final_mbr(ids,o['q'])['interval']==o['interval'];iv=o['interval'];v=tube.score(iv);off=official(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;officialchecks+=1;values[space+'_'+name]=dict(**v,boundary_MAE=(abs(iv[0]-spans[r['query_ordinal']][0])+abs(iv[1]-spans[r['query_ordinal']][1]))/2)
    if name in ['Native','HiResAuxMix']:assert all(abs(v[k]-oldstrong[a]['values'][space+'_'+name][k])<2e-10 for k in ['v','t','s'])
    if name=='LateRefer':assert all(abs(v[k]-oldlate[a]['values'][space+'_ReferROI'][k])<2e-10 for k in ['v','t','s'])
   if space!='E':
    vals=[tube.score([int(ids[i]),int(ids[j])+1])for i,j in zip(ii,jj)];scalarcalls+=len(vals)
    for k in curves:curves[k].append(np.array([v[k]for v in vals]))
  for name in p['arms']:values['Mean_'+name]={k:avg([values[f'Live{s}_{name}'][k]for s in p['seeds']])for k in ['v','t','s','boundary_MAE']}
  quality={k:np.mean(v,0)for k,v in curves.items()};expected={name:{k:float(o['q'][ii,jj]@curve)for k,curve in quality.items()}for name,o in z['outputs'].items()};rows.append(dict(alias=a,values=values,expected_candidate_quality=expected));print('score_early',a,flush=True)
 means={key:{k:avg([r['values'][key][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for key in ['Mean_'+a for a in p['arms']]};contrasts={};expected={}
 for baseline in ['Native','LateRefer','EarlyShift','HiResAuxMix']:
  ds={}
  for metric in ['v','t','s','boundary_MAE']:
   delta=np.array([(100 if metric in ['v','t','s']else 1)*(r['values']['Mean_EarlyRefer'][metric]-r['values']['Mean_'+baseline][metric])for r in rows]);ds[metric]=paired(delta)
   if metric=='v':ds['tails']=dict(positive=int((delta>1e-10).sum()),negative=int((delta< -1e-10).sum()),flat=int((abs(delta)<=1e-10).sum()),below_minus3=int((delta< -3).sum()),below_minus5=int((delta< -5).sum()),leave_one_out_mean_range_pp=[float(((delta.sum()-delta)/(len(delta)-1)).min()),float(((delta.sum()-delta)/(len(delta)-1)).max())],worst=sorted([dict(alias=r['alias'],delta_pp=float(v))for r,v in zip(rows,delta)],key=lambda x:x['delta_pp'])[:8])
  contrasts['EarlyRefer_minus_'+baseline]=ds;expected[baseline]={k:paired([100*(r['expected_candidate_quality']['EarlyRefer'][k]-r['expected_candidate_quality'][baseline][k])for r in rows])for k in ['v','t']}
 cost=read(D/'COST.json');put(P/'RESULTS_PUBLIC.json',dict(status='completed_exposed_teacher_capability_not_OPD',parents=128,rows=rows,means=means,paired=contrasts,expected_quality=expected,cost=cost,spatial_frozen=True));dist=expected['Native']['v'];hard=contrasts['EarlyRefer_minus_Native']['v'];go_dist=dist['mean']>=2 and dist['CI95'][0]>0 and all(expected[k]['v']['CI95'][0]>0 for k in ['LateRefer','EarlyShift','HiResAuxMix']);go_hard=hard['mean']>=2 and hard['CI95'][0]>0 and all(contrasts['EarlyRefer_minus_'+k]['v']['CI95'][0]>0 for k in ['LateRefer','EarlyShift','HiResAuxMix']);decision=dict(primary_hard=hard,primary_expected=dist,decision='qualifies_one_actual_controlled_transfer'if go_dist else'eligible_static_component_confirmation_only'if go_hard else'close_this_early_motion_prior_configuration',no_OPD_result=True,not_independent_confirmation=True,no_strength_mixture_layer_search=True,goal_achieved=False);put(P/'DECISION_PUBLIC.json',decision)
 audit.update(new_GT=True,official_Dense_output_checks=officialchecks,scalar_candidate_quality_calls=scalarcalls,GT_provenance=prov,all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']+bar['early_inputs']+bar['regional_prefixes']),Native_LateRefer_HiResAuxMix_exact_previous=True);assert audit['all128_before_GT'];put(P/'ROOT_AUDIT.json',audit);put(P/'INDEPENDENT_AUDIT.json',audit);put(D/'SCORING_COST.json',dict(CPU_seconds=time.perf_counter()-beg,GPU_seconds=0))
 lines=['# 早期跨模态motion区域先验：HC128教师能力','','同一旧late motion prior仅前移到逐帧encoder的motion keys；appearance/text prior1不代表attention mass不变。原生查询/门控从新memory重建，不能把收益唯一归memory。没有新跨帧encoder层；全空间数组冻结、无新专家/训练/OPD。','','| 对照 | EarlyRefer完整v增量pp [CI] | 候选期望GTv增量pp [CI] |','|---|---|---|']
 for name,x in contrasts.items():b=name[len('EarlyRefer_minus_'):];lines.append(f'| {b} | {x["v"]["mean"]:+.6f} {x["v"]["CI95"]} | {expected[b]["v"]["mean"]:+.6f} {expected[b]["v"]["CI95"]} |')
 lines+=['','HC128均已曝光，3空间seed父内平均，不按GT选头/样本。早期输入原缓存未保存，本轮实际补同Source捕获并完整计费；新Source空间框只控数值，不进入结果。旧LateRefer/强HiResAuxMix逐值复用。Raw Action/SA边界负结果不改。','','```json',json.dumps(decision,indent=2),json.dumps(cost,indent=2),json.dumps(audit,indent=2),'```','','单一EarlyRefer候选，无强度/层/模态/混合参数搜索，hard失败不是OPD失败；teacher期望质量只是能力，非已兑现方法增益。保留全128父、负尾、unknown；CURRENT/MAIN/PAPER及空间不动。']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],exposure='RC128 repeatedly exposed development, TC109 not used',excluded=['captions','parentIDs','weights','media','raw early inputs','annotations']))
 for f in P.glob('*.json'):f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 print(json.dumps(decision,indent=2),flush=True)
if __name__=='__main__':main()
