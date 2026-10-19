"""All-parent independent registration audit before new GT task scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,fcntl,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_registered_motion128_v1 import D,P,MOV,EARLY,PRE,FIN,DEV,STAGE,ARMS
from scripts.score_stvg_where_when_event128_v1 import avg,paired


def sample(a,x,y):
 h,w=a.shape[-2:];px=np.clip(x*w-.5,0,w-1);py=np.clip(y*h-.5,0,h-1);ix,iy=int(px),int(py);jx,jy=min(ix+1,w-1),min(iy+1,h-1);sx,sy=px-ix,py-iy
 return (1-sy)*((1-sx)*a[...,iy,ix]+sx*a[...,iy,jx])+sy*((1-sx)*a[...,jy,ix]+sx*a[...,jy,jx])


def sample_grid(a,gx,gy):
 a=np.asarray(a,dtype=np.float64);h,w=a.shape[-2:];f=np.float32
 # Match the installed CPU grid_sampler_unnormalize floating operation order.
 px=float(f(float(f(f(gx)+f(1)))*(w/2)-.5));py=float(f(float(f(f(gy)+f(1)))*(h/2)-.5));px=np.clip(px,0,w-1);py=np.clip(py,0,h-1);ix,iy=int(px),int(py);jx,jy=min(ix+1,w-1),min(iy+1,h-1);sx,sy=px-ix,py-iy
 return (1-sy)*((1-sx)*a[...,iy,ix]+sx*a[...,iy,jx])+sy*((1-sx)*a[...,jy,ix]+sx*a[...,jy,jx])


def main():
 import torch
 from main_method.runtime import resource_guard
 from vg_tta.stvg_temporal_registered_motion128_v1 import geometry,independent_geometry,feature_crop,register,tensor_sha
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table,independent_cdf
 lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);torch.set_num_threads(4);beg=time.perf_counter();phase={'GT':False}
 def guard(event,args):
  if not phase['GT']and event=='open'and args and isinstance(args[0],(str,bytes,Path))and str(args[0]).endswith('/data/hcstvg2_official_metadata/val_v2.json'):raise PermissionError('registered pre-audit forbids actual labels')
 sys.addaudithook(guard);p=read(D/'SCORING_PLAN_LOCK001.json');assert p['source_measurement_plan_sha256']==sha(D/'PLAN_LOCK.json')and p['source_prediction_barrier_sha256']==sha(D/'GLOBAL_PREDICTION_BARRIER.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_before_GT']and bar['weights_and_flags_unchanged']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in bar['files']);assert sha(R/bar['source_control_file']['path'])==bar['source_control_file']['sha256']and sha(D/'SOURCE_CONTROLS.json')==bar['source_control_summary_sha256'];raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];spaces=oldexact=cdfs=utils=geometry_cells=independent_scalars=0;maxcdf=maxscalar=maxlocal=maxregistered=0.;ii,jj=np.triu_indices(65,1);diagnostics={}
 def check():resource_guard();assert time.perf_counter()-beg<900
 for nr,r in enumerate(p['records']):
  check();a=r['alias']
  for path,h in [(EARLY/'early_inputs'/f'{a}.pt',r['early_sha256']),(MOV/'predictions'/f'{a}.pt',r['moving_sha256']),(PRE/'inputs'/f'{a}.pt',r['input_sha256']),(FIN/'Live/predictions'/f'{a}.pt',r['spatial_sha256']),(DEV/'priors'/f'{a}.pt',r['prior_sha256']),(STAGE/'predictions'/f'{a}.pt',r['native_stage_sha256'])]:assert sha(path)==h
  z=load(D/'predictions'/f'{a}.pt');old=load(MOV/'predictions'/f'{a}.pt');e=load(EARLY/'early_inputs'/f'{a}.pt');row=raw[r['query_ordinal']]['input'];rect=r['rectangles']['MovingLive'];assert z['RGB_sha256']==old['observations']['MovingLive']['raw_RGB_sha256'];diag=[]
  for off in [0,1]:
   cap=e['captured_encoder_inputs'][off];ov=e['original_views'][off];hh,ww=ov['info']['fea_map_size'];l=hh*ww;src=cap['src'];native=src[-l:].permute(1,2,0).reshape(len(z['frame_ids'][off::2]),256,hh,ww);valid=(~cap['kwargs']['src_key_padding_mask'][:,-l:]).reshape(-1,hh,ww);alpha,grid,cg=geometry(rect[off::2],row['width'],row['height'],valid);aa,gg=independent_geometry(rect[off::2],row['width'],row['height'],valid);md=z['metadata'][str(off)];assert np.array_equal(aa,alpha.numpy())and np.array_equal(gg,grid.numpy())and torch.equal(alpha,md['alpha'])and torch.equal(grid,md['grid'])and torch.equal(valid,md['valid']);geometry_cells+=alpha.numel();assert tensor_sha(native)==md['native_motion_sha256'];local=z['crop_motion'][off];assert tensor_sha(local)==md['new_motion_sha256'];fl=feature_crop(native,cg,tuple(local.shape[-2:])).to(local.dtype);assert str(fl.dtype)==md['FeatureWarp_local_dtype']==md['local_dtype']and str(native.dtype)==md['native_dtype'];rgb=register(native,local,alpha,grid);fw=register(native,fl,alpha,grid)
   assert tensor_sha(fl)==md['FeatureWarp_local_sha256']and tensor_sha(rgb)==md['RGBRegistered_sha256']and tensor_sha(fw)==md['FeatureWarp_sha256']
   for arm,motion in [('Native',native),('RGBRegistered',rgb),('FeatureWarp',fw)]:
    x=src.clone();x[-l:]=motion.flatten(2).permute(2,0,1);meta=md['conditions'][arm];assert tensor_sha(x)==meta['actual_input_sha256']and meta['appearance_text_input_exact']and meta['position_padding_exact']and meta['actual_encoder_entry_verified'];assert torch.equal(x[:-l],src[:-l]);assert all(z['metadata'][str(off)]['conditions'][k]['position_padding_exact']for k in ['Native','RGBRegistered','FeatureWarp'])
   # Fixed frame/cell/channel indices, independent scalar border bilinear.
   frames=np.unique(np.linspace(0,len(native)-1,5).astype(int));cells=np.unique(np.linspace(0,l-1,13).astype(int));channels=[0,17,63,127,255];nn=native.float().numpy();loc=local.float().numpy();ff=fl.float().numpy();rr=rgb.float().numpy();wwarp=fw.float().numpy()
   for t in frames:
    left,top,right,bottom,vh,vw=cg[t]
    for cy,cx in [(0,0),(0,3),(1,5),(3,3),(5,1),(6,6)]:
     want=sample(nn[t,channels,:vh,:vw],left+(cx+.5)/7*(right-left),top+(cy+.5)/7*(bottom-top));err=float(np.max(abs(want-ff[t,channels,cy,cx])/np.maximum(1,abs(want))));assert err<2*np.finfo('float16').eps;maxlocal=max(maxlocal,err);maxscalar=max(maxscalar,err);independent_scalars+=len(channels)
    for cell in cells:
     y,x=divmod(int(cell),nn.shape[-1]);u,v=(gg[t,y,x]+1)/2;A=float(aa[t,y,x])
     for source,actual in [(loc,rr),(ff,wwarp)]:
      want=(1-A)*nn[t,channels,y,x].astype(np.float64)+A*sample_grid(source[t,channels],gg[t,y,x,0],gg[t,y,x,1]);err=float(np.max(abs(want-actual[t,channels,y,x])/np.maximum(1,abs(want))));assert err<(16 if native.dtype==torch.float32 else 2)*torch.finfo(native.dtype).eps;maxregistered=max(maxregistered,err);maxscalar=max(maxscalar,err);independent_scalars+=len(channels)
   diag.append(dict(offset=off,changed_motion_entries=int((rgb!=native).sum()),FeatureWarp_changed_entries=int((fw!=native).sum()),valid_cells=int(valid.sum()),crop_coverage_alpha_mean=float(alpha[valid].mean()),RGB_FeatureWarp_abs_difference_mean=float((rgb.float()-fw.float()).abs().mean())))
  for arm in ['Native','HiResAuxMix','MovingLive']:assert np.array_equal(z['outputs'][arm]['q'],old['outputs'][arm]['q'])and z['outputs'][arm]['interval']==old['outputs'][arm]['interval'];oldexact+=1
  for arm in ['Native','RGBRegistered','FeatureWarp']:
   gs=probabilities(z['records'],z['logits'][arm]);ids,q=mass_table(gs);ex=independent_cdf(gs,ids);err=float(abs(q-ex).max());assert ids==z['frame_ids']and err<2e-13 and np.array_equal(q,z['outputs'][arm]['q']);maxcdf=max(maxcdf,err);cdfs+=1
   if arm!='Native':assert np.array_equal(.5*old['outputs']['Native']['q']+.5*q,z['outputs'][arm+'Mix']['q'])
  for arm,o in z['outputs'].items():
   d=final_mbr(z['frame_ids'],o['q']);assert d['interval']==o['interval'];sel=np.unique(np.r_[np.linspace(0,len(ii)-1,13).astype(int),np.flatnonzero((ii==d['indices'][0])&(jj==d['indices'][1]))]);u=brute_utility(z['frame_ids'],o['q'],sel);assert abs(u-d['utility_grid'][ii[sel],jj[sel]]).max()<1e-11;utils+=len(sel)
  for name,bx in old['trajectories'].items():assert np.array_equal(bx,z['trajectories'][name]);spaces+=1
  diagnostics[a]=diag
 check();assert spaces==512 and oldexact==384 and cdfs==384;control=read(D/'SOURCE_CONTROLS.json');cc=load(D/'SOURCE_CONTROLS.pt');first=p['records'][0];firstz=load(D/'predictions'/f"{first['alias']}.pt");firstold=load(MOV/'predictions'/f"{first['alias']}.pt");firstinp=load(PRE/'inputs'/f"{first['alias']}.pt");firstprior=load(DEV/'priors'/f"{first['alias']}.pt");firstearly=load(EARLY/'early_inputs'/f"{first['alias']}.pt");assert torch.equal(cc['Native']['values']['boxes'],firstinp['Source'])and all(torch.equal(x,y)for x,y in zip(cc['Native']['values']['logits'],firstprior['logits']))and all(torch.equal(x,y)for x,y in zip(cc['Crop']['values']['logits'],firstold['observations']['MovingLive']['logits']));assert np.array_equal(cc['Native']['q'],firstold['outputs']['Native']['q'])and np.array_equal(cc['Crop']['q'],firstold['outputs']['MovingLive']['q']);assert all(torch.equal(x,y)for x,y in zip(firstz['crop_motion'],cc['Crop']['projected_motion']))
 for off,pr in enumerate(cc['Native']['projected_motion']):
  hh,ww=firstearly['original_views'][off]['info']['fea_map_size'];want=firstearly['captured_encoder_inputs'][off]['src'][-hh*ww:].permute(1,2,0).reshape(pr.shape);assert torch.equal(pr,want)
 assert control['status']=='pass'and control['Native_boxes_logits_q_raw_motion_exact']and control['Crop_logits_q_exact_R28'];audit=dict(status='pass',frozen_spatial_arrays_exact=spaces,old_posteriors_intervals_exact=oldexact,independent_CDF_checks=cdfs,max_CDF_error=maxcdf,all_geometry_cells_independent=geometry_cells,independent_bilinear_channel_scalars=independent_scalars,max_bilinear_relative_error=maxscalar,max_FP16_local_relative_error=maxlocal,max_FP32_registered_relative_error=maxregistered,numerical_oracle='CPU vectorized contracted FP32 coordinates plus independent FP64 interpolation',actual_fusion_inputs_SHA_reconstructed=True,independent_MBR_sample_utilities=utils,Source_controls=control,Source_control_raw_receipt_and_tensor_values_exact=True,new_GT=False);put(D/'AUDIT_BEFORE_GT.json',audit);assert all(sha(R/f)==h for f,h in p['pins'].items());put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_before_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')));phase['GT']=True
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));rows=[];officialchecks=scalarcalls=oldmetric=0;oldpublic={r['alias']:r for r in read(R/'results/stvg_error_state/2026-10-20/temporal_moving_rgb128/RESULTS_PUBLIC.json')['rows']}
 for r in p['records']:
  check();a=r['alias'];z=load(D/'predictions'/f'{a}.pt');ids=np.array(z['frame_ids']);values={};curves={k:[]for k in ['v','t']};actual={}
  for space,boxes in z['trajectories'].items():
   tube=DenseTube(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True);sel=np.isin(tube.fids,np.array(sorted(set(ids)-set(r['physical_frame_ids8']))));actual[space]=dict(nodes=int(sel.sum()),sIoU=float(tube.iou[sel].mean())if sel.any()else None,all_temporal_arms_identical=True)
   for name,o in z['outputs'].items():
    iv=o['interval'];v=tube.score(iv);off=official(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;officialchecks+=1;values[space+'_'+name]=dict(**v,boundary_MAE=(abs(iv[0]-spans[r['query_ordinal']][0])+abs(iv[1]-spans[r['query_ordinal']][1]))/2)
    if name in ['Native','HiResAuxMix','MovingLive']:assert all(abs(values[space+'_'+name][k]-oldpublic[a]['values'][space+'_'+name][k])<2e-10 for k in ['v','t','s','boundary_MAE']);oldmetric+=1
   assert len({values[space+'_'+name]['s']for name in ARMS})==1
   if space!='E':
    vals=[tube.score([int(ids[i]),int(ids[j])+1])for i,j in zip(ii,jj)];scalarcalls+=len(vals)
    for k in curves:curves[k].append(np.array([v[k]for v in vals]))
  for name in ARMS:values['Mean_'+name]={k:avg([values[f'Live{s}_{name}'][k]for s in p['seeds']])for k in ['v','t','s','boundary_MAE']}
  quality={k:np.mean(v,0)for k,v in curves.items()};expected={name:{k:float(o['q'][ii,jj]@curve)for k,curve in quality.items()}for name,o in z['outputs'].items()};rows.append(dict(alias=a,values=values,expected_candidate_quality=expected,diagnostics=diagnostics[a],fixed_unobserved_actual_sIoU=actual));print('score_registered_motion',a,flush=True)
 means={key:{k:avg([r['values'][key][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for key in ['Mean_'+a for a in ARMS]};ds={};expected={};pairs=[('RGBRegisteredMix',k)for k in ['Native','HiResAuxMix','MovingLive','FeatureWarpMix','RGBRegistered']]+[('RGBRegistered',k)for k in ['Native','HiResAuxMix','FeatureWarp','MovingLive']]+[(k,'Native')for k in ['FeatureWarpMix','FeatureWarp','HiResAuxMix']]
 for a,b in pairs:
  key=a+'_minus_'+b;delta=np.array([100*(r['values']['Mean_'+a]['v']-r['values']['Mean_'+b]['v'])for r in rows]);ds[key]={k:paired([(100 if k in ['v','t','s']else 1)*(r['values']['Mean_'+a][k]-r['values']['Mean_'+b][k])for r in rows])for k in ['v','t','s','boundary_MAE']};ds[key]['tails']=dict(positive=int((delta>1e-10).sum()),negative=int((delta< -1e-10).sum()),flat=int((abs(delta)<=1e-10).sum()),below_minus3=int((delta< -3).sum()),below_minus5=int((delta< -5).sum()),leave_one_out_mean_range_pp=[float(((delta.sum()-delta)/127).min()),float(((delta.sum()-delta)/127).max())],worst=sorted([dict(alias=r['alias'],delta_pp=float(v))for r,v in zip(rows,delta)],key=lambda x:x['delta_pp'])[:8]);expected[key]={k:paired([100*(r['expected_candidate_quality'][a][k]-r['expected_candidate_quality'][b][k])for r in rows])for k in ['v','t']}
 check();assert all(sha(R/f)==h for f,h in p['pins'].items());cost=read(D/'COST.json');put(P/'RESULTS_PUBLIC.json',dict(status='completed_exposed_registered_motion_capability_not_OPD',parents=128,rows=rows,means=means,paired=ds,expected_quality=expected,cost=cost,spatial_frozen=True));dist=expected['RGBRegistered_minus_Native']['v'];hard=ds['RGBRegisteredMix_minus_Native']['v'];fresh=expected['RGBRegistered_minus_FeatureWarp']['v'];go_dist=dist['mean']>=2 and dist['CI95'][0]>0 and fresh['CI95'][0]>0;go_hard=hard['mean']>=2 and hard['CI95'][0]>0;decision=dict(primary_hard=hard,pure_teacher_expected=dist,new_RGB_versus_FeatureWarp_expected=fresh,decision='qualifies_one_actual_controlled_transfer'if go_dist else'eligible_static_component_confirmation_only'if go_hard else'close_this_registered_motion_configuration',versus_Strong=ds['RGBRegisteredMix_minus_HiResAuxMix']['v'],not_OPD=True,not_independent_confirmation=True,no_crop_alpha_mixture_temperature_sweep=True,goal_achieved=False);put(P/'DECISION_PUBLIC.json',decision)
 audit.update(new_GT=True,official_Dense_output_checks=officialchecks,scalar_candidate_quality_calls=scalarcalls,old_metrics_exact=oldmetric,GT_provenance=prov,all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']),all_temporal_arms_sIoU_identical=True);assert audit['all128_before_GT'];put(P/'ROOT_AUDIT.json',audit);put(P/'INDEPENDENT_AUDIT.json',audit);sc=dict(CPU_audit_and_scoring_seconds=time.perf_counter()-beg,GPU_seconds=0,publication_archive_cost_excluded=True,additional_pre_GT_failed_and_diagnostic_CPU_wall='not separately instrumented; all retained; zero GPU/model/GT');put(D/'SCORING_COST.json',sc);put(P/'SCORING_COST_PUBLIC.json',sc)
 lines=['# 保留全图上下文的注册RGB-motion：HC2 RC128','','实际评分入口：`scripts/score_stvg_temporal_registered_motion128_audit001.py`。原scorer预GT浮点坐标审计失败保留；修订001匹配本机CPU vectorized contracted FP32坐标，再独立FP64插值，原16eps32门槛、93测量pins/PLAN/barrier和预测均不变，无模型重跑。见NUMERICAL_AUDIT_REVISION001_PUBLIC.json及同名audit001协议。额外失败/诊断CPU时间未分别精确计时，不能将下面成功评分时间称全部研究CPU成本。','','原appearance/text融合输入逐值保留；实际局部RGB先经同Swin/project2，再按均匀有效cell/面积覆盖率注册，仅替换原motion。FeatureWarp在原motion上裁/重采样后走同注册，区分新RGB与重采样。位置/布尔padding/65物理帧及两offset/MBR均不动，新空间输出全丢弃，无学生或网络更新。本轮不是OPD。','','| 对比 | 完整v增量pp [CI] | 候选期望GTv增量pp [CI] |','|---|---|---|']
 for name,x in ds.items():lines.append(f'| {name} | {x["v"]["mean"]:+.6f} {x["v"]["CI95"]} | {expected[name]["v"]["mean"]:+.6f} {expected[name]["v"]["CI95"]} |')
 tail=ds['RGBRegisteredMix_minus_Native']['tails'];lines+=['',f'主比较正/负/零 {tail["positive"]}/{tail["negative"]}/{tail["flat"]}；<-3 {tail["below_minus3"]}，<-5 {tail["below_minus5"]}；最坏{tail["worst"]}。','RC128反复曝光、三冻结空间seed父内平均、全部128先封存后GT，不是独立确认。均匀feature cell注册不是精确CNN感受野；不能逆转Swin在移动crop中已混合的运动参照，也可能丢上下文或回粗格丢细节。原appearance/text只在fusion输入固定，融合后输出可变。未超过FeatureWarp不归因新RGB；未超过Strong不称方法领先。','','```json',json.dumps(decision,indent=2),json.dumps(cost,indent=2),json.dumps(audit,indent=2),'```','','固定GT支持和未观察actual sIoU不变；3584最终输出official独立核，798720候选质量仅Dense逐项计算，MBR是预定抽点核，不夸大全候选双实现。旧缓存不免费，新Swin/fusion/两pass成本已计。关闭精确配置不否定全部空间条件时间/OPD；无crop/alpha/混合救援，不晋升空间或CURRENT/MAIN/PAPER。']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),scoring_plan_sha256=sha(D/'SCORING_PLAN_LOCK001.json'),audit_revision='001',pins=p['pins'],exposure='HC2 RC128 repeatedly exposed; no independent confirmation',excluded=['captions','parentIDs','media','GT','weights','raw caches']))
 for f in P.glob('*.json'):
  if str(f.relative_to(R))not in p['pins']:f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 check();assert all(sha(R/f)==h for f,h in p['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in bar['files']);print(json.dumps(decision,indent=2),flush=True);lock.close()
if __name__=='__main__':main()
