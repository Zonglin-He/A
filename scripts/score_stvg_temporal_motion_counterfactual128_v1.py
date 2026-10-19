"""All-parent sealed motion-input intervention quality, no online GT selection."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,hashlib,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_motion_counterfactual128_v1 import D,P,EARLY,PRE,FIN,DEV,STRONG
from scripts.score_stvg_where_when_event128_v1 import avg,paired

def main():
 import torch
 from main_method.runtime import resource_guard
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,independent_cdf
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 from vg_tta.stvg_temporal_motion_counterfactual128_v1 import temporal_mean
 torch.set_num_threads(4);beg=time.perf_counter();p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_before_GT']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json')and all(sha(R/f)==h for f,h in p['pins'].items());assert all(sha(R/f['path'])==f['sha256']for f in bar['files']+bar['prefixes']);cdfchecks=spaces=meanchecks=algebra=0;maxcdf=maxmean=maxscaled=0.;changed=0;normratios=[]
 for r in p['records']:
  resource_guard();assert time.perf_counter()-beg<900;a=r['alias'];assert sha(EARLY/'early_inputs'/f'{a}.pt')==r['early_input_sha256'];cap=load(EARLY/'early_inputs'/f'{a}.pt');z=load(D/'predictions'/f'{a}.pt');pref=load(D/'prefixes'/f'{a}.pt');assert z['native_exact']and pref['native_exact']
  for off,(c,v,s)in enumerate(zip(cap['captured_encoder_inputs'],cap['original_views'],z['motion_statistics'])):
   assert torch.equal(pref['prefixes']['Native'][off],v['prefix']);cc,ss=temporal_mean(c,v);assert hashlib.sha256(cc['src'].contiguous().numpy().tobytes()).hexdigest()==s['actual_input_sha256'];h,w=v['info']['fea_map_size'];l=h*w;x=c['src'][-l:].numpy().astype(np.float64);actual=s['actual_motion_template'].numpy().astype(np.float64);valid=(~c['kwargs']['src_key_padding_mask'][0,-l:]).numpy();assert torch.equal(cc['src'][-l:,0],s['actual_motion_template'])and torch.equal(cc['src'][:-l],c['src'][:-l]);expected=x.mean(1);err=np.abs(expected[valid]-actual[valid]);bound=(x.shape[1]+2)*np.finfo(np.float32).eps*np.abs(x[valid]).mean(1)+np.finfo(np.float32).eps*np.abs(expected[valid]);assert np.all(err<=bound);maxmean=max(maxmean,float(err.max(initial=0)));maxscaled=max(maxscaled,float(np.max(err/np.maximum(bound,1e-30),initial=0)));meanchecks+=int(valid.sum())*256;changed+=s['changed_scalar_values'];normratios.append(s['motion_norm_after']/s['motion_norm_before']if s['motion_norm_before']else 1.)
  for name,ll in z['logits'].items():
   q=independent_cdf(probabilities(z['records'],ll),z['frame_ids']);err=float(abs(q-z['outputs'][name]['q']).max());assert err<2e-12;maxcdf=max(maxcdf,err);cdfchecks+=1
  q=z['outputs']['Native']['q'];m=z['outputs']['MeanMotion']['q'];mask=q>0;assert np.array_equal(mask,m>0)
  for name,lv in [('PriorSquare',2*np.log(q[mask])),('Contrast',2*np.log(q[mask])-np.log(m[mask]))]:
   ex=np.zeros_like(q);xx=np.exp(lv-lv.max());ex[mask]=xx/xx.sum();assert np.max(abs(ex-z['outputs'][name]['q']))<2e-14;algebra+=int(mask.sum())
  inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt')
  for k,b in z['trajectories'].items():assert np.array_equal(b,inp['E']if k=='E'else sp['outputs'][f'T20_Live_{k[4:]}']);spaces+=1
 audit=dict(status='pass',native_encoder_prefix_final_time_exact_parents=128,independent_FP64_mean_values=meanchecks,mean_absolute_max_error=maxmean,mean_error_over_prelocked_bound_max=maxscaled,actual_encoder_input_hashes=256,nonmotion_padding_pos_exact=True,changed_motion_scalar_values=changed,motion_norm_ratio_mean=float(np.mean(normratios)),motion_norm_ratio_min=float(min(normratios)),quotient_scalar_checks=algebra,independent_CDFs=cdfchecks,max_CDF_error=maxcdf,frozen_spatial_arrays_exact=spaces,new_GT=False);put(D/'AUDIT_BEFORE_GT.json',audit)
 put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_before_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')))
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];old={r['alias']:r for r in read(R/'results/stvg_error_state/2026-10-20/temporal_native_stage128/RESULTS_PUBLIC.json')['rows']};ii,jj=np.triu_indices(65,1);rows=[];officialchecks=scalarcalls=0
 for r in p['records']:
  resource_guard();assert time.perf_counter()-beg<900;a=r['alias'];z=load(D/'predictions'/f'{a}.pt');ids=np.array(z['frame_ids']);values={};curves={k:[]for k in ['v','t']}
  for space,boxes in z['trajectories'].items():
   tube=DenseTube(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True)
   for name,o in z['outputs'].items():
    assert final_mbr(ids,o['q'])['interval']==o['interval'];iv=o['interval'];v=tube.score(iv);off=official(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;officialchecks+=1;values[space+'_'+name]=dict(**v,boundary_MAE=(abs(iv[0]-spans[r['query_ordinal']][0])+abs(iv[1]-spans[r['query_ordinal']][1]))/2)
    if name in ['Native','HiResAuxMix']:assert all(abs(v[k]-old[a]['values'][space+'_'+name][k])<2e-10 for k in ['v','t','s'])
   if space!='E':
    vals=[tube.score([int(ids[i]),int(ids[j])+1])for i,j in zip(ii,jj)];scalarcalls+=len(vals)
    for k in curves:curves[k].append(np.array([v[k]for v in vals]))
  for name in p['arms']:values['Mean_'+name]={k:avg([values[f'Live{s}_{name}'][k]for s in p['seeds']])for k in ['v','t','s','boundary_MAE']}
  quality={k:np.mean(v,0)for k,v in curves.items()};expected={name:{k:float(o['q'][ii,jj]@curve)for k,curve in quality.items()}for name,o in z['outputs'].items()};rows.append(dict(alias=a,values=values,expected_candidate_quality=expected));print('score_motion_mean',a,flush=True)
 means={key:{k:avg([r['values'][key][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for key in ['Mean_'+a for a in p['arms']]};ds={};expected={}
 for a,b in [('Contrast','Native'),('Contrast','PriorSquare'),('Contrast','HiResAuxMix'),('MeanMotion','Native')]:
  key=a+'_minus_'+b;delta=np.array([100*(r['values']['Mean_'+a]['v']-r['values']['Mean_'+b]['v'])for r in rows]);ds[key]={k:paired([(100 if k in ['v','t','s']else 1)*(r['values']['Mean_'+a][k]-r['values']['Mean_'+b][k])for r in rows])for k in ['v','t','s','boundary_MAE']};ds[key]['tails']=dict(positive=int((delta>1e-10).sum()),negative=int((delta< -1e-10).sum()),flat=int((abs(delta)<=1e-10).sum()),below_minus3=int((delta< -3).sum()),below_minus5=int((delta< -5).sum()),leave_one_out_mean_range_pp=[float(((delta.sum()-delta)/127).min()),float(((delta.sum()-delta)/127).max())],worst=sorted([dict(alias=r['alias'],delta_pp=float(v))for r,v in zip(rows,delta)],key=lambda x:x['delta_pp'])[:8]);expected[key]={k:paired([100*(r['expected_candidate_quality'][a][k]-r['expected_candidate_quality'][b][k])for r in rows])for k in ['v','t']}
 cost=read(D/'COST.json');put(P/'RESULTS_PUBLIC.json',dict(status='completed_exposed_counterfactual_capability_not_OPD',parents=128,rows=rows,means=means,paired=ds,expected_quality=expected,cost=cost,spatial_frozen=True));dist=expected['Contrast_minus_Native']['v'];hard=ds['Contrast_minus_Native']['v'];go_dist=dist['mean']>=2 and dist['CI95'][0]>0 and all(expected['Contrast_minus_'+k]['v']['CI95'][0]>0 for k in ['PriorSquare','HiResAuxMix']);go_hard=hard['mean']>=2 and hard['CI95'][0]>0 and all(ds['Contrast_minus_'+k]['v']['CI95'][0]>0 for k in ['PriorSquare','HiResAuxMix']);decision=dict(primary_hard=hard,primary_expected=dist,decision='qualifies_one_actual_controlled_transfer'if go_dist else'eligible_static_component_confirmation_only'if go_hard else'close_this_mean_motion_quotient_configuration',not_OPD=True,not_independent_confirmation=True,no_coefficient_norm_mean_search=True,goal_achieved=False);put(P/'DECISION_PUBLIC.json',decision)
 audit.update(new_GT=True,official_Dense_output_checks=officialchecks,scalar_candidate_quality_calls=scalarcalls,GT_provenance=prov,all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']+bar['prefixes']),Native_HiResAuxMix_exact_previous=True);assert audit['all128_before_GT'];put(P/'ROOT_AUDIT.json',audit);put(P/'INDEPENDENT_AUDIT.json',audit);scorecost=dict(CPU_audit_and_scoring_seconds=time.perf_counter()-beg,GPU_seconds=0,publication_and_archive_cost_excluded=True);put(D/'SCORING_COST.json',scorecost);put(P/'SCORING_COST_PUBLIC.json',scorecost)
 lines=['# motion时间均值反事实：HC128冻结能力检查','','每offset、每motion cell/channel沿T均值，仅改变有效motion输入；appearance/text/pos不改。同一Source原生融合、query/gate两pass重新运行。不是静态视频或纯先验；动作特征均值可能缩范数、混合前景背景并分布外。所有空间坐标冻结。','','旧noun-caption quotient阴性保留；此处只换视觉输入干预，不复现官方VCD的图像噪声/文本token流程。Contrast=normalize(p0²/qMeanMotion)，PriorSquare控制泛化锐化，单位系数固定；没有新专家/训练/学生或OPD。','','| 对照 | 完整v增量pp [CI] | 候选期望GTv增量pp [CI] |','|---|---|---|']
 for name,x in ds.items():lines.append(f'| {name} | {x["v"]["mean"]:+.6f} {x["v"]["CI95"]} | {expected[name]["v"]["mean"]:+.6f} {expected[name]["v"]["CI95"]} |')
 lines+=['','全部RC128旧曝光父query，空间三seed父内平均，TC109未选参。预测及prefix全封存后GT；MeanMotion原始输出只诊断，不事后晋升。新fusion encoder计算如实计费，未新RGB/专家观测。','','```json',json.dumps(decision,indent=2),json.dumps(cost,indent=2),json.dumps(audit,indent=2),'```','','hard读出和完整分布质量分别判定，不以hard阴性否定所有OPD。无单位系数/范数/均值范围/混合搜索，失败只关闭本配置。CURRENT/MAIN/PAPER及空间不动。']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],exposure='RC128 repeatedly exposed development, TC109 not used',primary_source='https://arxiv.org/abs/2311.16922',transfer_not_official_VCD=True,excluded=['captions','parentIDs','weights','media','raw input/prefix caches','annotations']))
 for f in P.glob('*.json'):f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 print(json.dumps(decision,indent=2),flush=True)
if __name__=='__main__':main()
