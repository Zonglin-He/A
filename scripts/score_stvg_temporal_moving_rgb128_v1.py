"""Independent real pixel/geometry/CDF audit, then all-parent post-seal scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,fcntl,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_moving_rgb128_v1 import D,P,PRE,FIN,STAGE,DEV,UNION,ARMS
from scripts.score_stvg_where_when_event128_v1 import avg,paired

def _main():
 import torch,cv2
 from main_method.runtime import resource_guard
 from vg_tta.stvg_temporal_moving_rgb128_v1 import independent_rectangles,digest
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table,independent_cdf
 from methods.stvg_referdino_opd_v1.runtime import decode
 torch.set_num_threads(4);cv2.setNumThreads(2);beg=time.perf_counter();phase={'GT':False}
 def reject_early_GT(event,args):
  if not phase['GT']and event=='open'and args and isinstance(args[0],(str,bytes,Path))and str(args[0]).endswith('/data/hcstvg2_official_metadata/val_v2.json'):raise PermissionError('pre-GT audit forbids labels')
 sys.addaudithook(reject_early_GT);p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_before_GT']and bar['weights_and_flags_unchanged']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json')and all(sha(R/f)==h for f,h in p['pins'].items());assert all(sha(R/f['path'])==f['sha256']for f in bar['files']);raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];spaces=oldexact=cdfs=utilities=rectchecks=pixelchecks=0;maxerr=0.;ii,jj=np.triu_indices(65,1);diagnostics={}
 def check():resource_guard();assert time.perf_counter()-beg<900
 for nr,r in enumerate(p['records']):
  check();a=r['alias']
  for path,h in [(STAGE/'predictions'/f'{a}.pt',r['native_stage_sha256']),(PRE/'inputs'/f'{a}.pt',r['input_sha256']),(FIN/'Live/predictions'/f'{a}.pt',r['spatial_sha256']),(DEV/'priors'/f'{a}.pt',r['prior_sha256']),(UNION/'predictions'/f'{a}.pt',r['union_sha256'])]:assert sha(path)==h
  z=load(D/'predictions'/f'{a}.pt');old=load(STAGE/'predictions'/f'{a}.pt');inp=load(PRE/'inputs'/f'{a}.pt');prior=load(DEV/'priors'/f'{a}.pt');union=load(UNION/'predictions'/f'{a}.pt');row=raw[r['query_ordinal']];b=dict(MovingLive=z['trajectories']['Live41'],MovingSource=inp['Source'].numpy());dg={}
  if nr<2:
   frames,ids=decode(row['input']);assert ids==z['frame_ids']and digest(frames)==z['original_RGB_sha256']
  for arm in ['MovingLive','MovingSource']:
   ob=z['observations'][arm];rect=independent_rectangles(b[arm],row['input']['width'],row['input']['height']);assert rect==r['rectangles'][arm]==ob['rectangles'];rectchecks+=len(rect);assert ob['shape']==[65,224,224,3]and ob['normalized_shape']==[65,3,224,224]
   if nr<2:
    pixels=np.stack([cv2.resize(frames[i][v[1]:v[3],v[0]:v[2]],(224,224),interpolation=cv2.INTER_LINEAR)for i,v in enumerate(rect)]);assert digest(pixels)==ob['raw_RGB_sha256'];pixelchecks+=1;del pixels
   gs=probabilities(ob['records'],ob['logits']);ids,q=mass_table(gs);ex=independent_cdf(gs,ids);err=float(abs(q-ex).max());assert err<2e-13;maxerr=max(maxerr,err);cdfs+=1;assert ids==z['frame_ids']and np.array_equal(q,ob['q'])and np.array_equal(q,z['outputs'][arm]['q']);mix=.5*prior['q']+.5*q;assert np.array_equal(mix,z['outputs'][arm+'Mix']['q']);ar=np.asarray(rect,float);size=ar[:,2:]-ar[:,:2];center=(ar[:,2:]+ar[:,:2])/2/np.array([row['input']['width'],row['input']['height']]);dg[arm]=dict(crop_area_fraction_mean=float((size.prod(1)/(row['input']['width']*row['input']['height'])).mean()),normalized_crop_center_jump_mean=float(np.linalg.norm(np.diff(center,axis=0),axis=1).mean()),log_crop_size_jump_mean=float(abs(np.diff(np.log(size),axis=0)).mean()),reused=ob['reused'])
  if nr<2:del frames
  assert z['original_RGB_sha256']==union['original_RGB_sha256']and z['original_RGB_shape']==union['original_RGB_shape']
  for name,ref in [('Native',old['outputs']['Native']),('HiResAuxMix',old['outputs']['HiResAuxMix']),('UnionCrop0Mix',union['outputs']['Crop0Mix'])]:assert np.array_equal(z['outputs'][name]['q'],ref['q'])and z['outputs'][name]['interval']==ref['interval'];oldexact+=1
  for name,bx in old['trajectories'].items():assert np.array_equal(bx,z['trajectories'][name]);spaces+=1
  for name,o in z['outputs'].items():
   dec=final_mbr(z['frame_ids'],o['q']);assert dec['interval']==o['interval'];pick=np.unique(np.r_[np.linspace(0,len(ii)-1,13).astype(int),np.flatnonzero((ii==dec['indices'][0])&(jj==dec['indices'][1]))]);u=brute_utility(z['frame_ids'],o['q'],pick);assert abs(u-dec['utility_grid'][ii[pick],jj[pick]]).max()<1e-11;utilities+=len(pick)
  diagnostics[a]=dict(**dg,Live_Source_view_pixels_same=z['observations']['MovingLive']['raw_RGB_sha256']==z['observations']['MovingSource']['raw_RGB_sha256'],geometry_changed_nodes=sum(v!=w for v,w in zip(r['rectangles']['MovingLive'],r['rectangles']['MovingSource'])),TV_Live_Source=float(abs(z['outputs']['MovingLive']['q']-z['outputs']['MovingSource']['q']).sum()/2))
 assert cdfs==256 and spaces==512 and oldexact==384 and rectchecks==16640 and pixelchecks==4
 audit=dict(status='pass',independent_scalar_rectangle_checks=rectchecks,independent_real_RGB_redecode_view_checks=pixelchecks,original_all_RGB_SHA_matches_R13=True,independent_CDF_checks=cdfs,old_posteriors_intervals_exact=oldexact,frozen_spatial_arrays_exact=spaces,independent_MBR_original_action_utilities=utilities,max_CDF_error=maxerr,Native_full_Source_boxes_logits_q_exact=read(D/'NATIVE_CONTROL.json')['Source_boxes_logits_q_exact'],new_GT=False);put(D/'AUDIT_BEFORE_GT.json',audit);check();assert all(sha(R/f)==h for f,h in p['pins'].items());put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_before_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')));phase['GT']=True
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));rows=[];officialchecks=scalarcalls=0
 for r in p['records']:
  check();a=r['alias'];z=load(D/'predictions'/f'{a}.pt');ids=np.array(z['frame_ids']);values={};curves={k:[]for k in ['v','t']};actual={}
  for space,boxes in z['trajectories'].items():
   tube=DenseTube(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True);sel=np.isin(tube.fids,np.array(sorted(set(ids)-set(r['physical_frame_ids8']))));actual[space]=dict(nodes=int(sel.sum()),sIoU=float(tube.iou[sel].mean())if sel.any()else None,all_temporal_arms_identical=True)
   for name,o in z['outputs'].items():
    iv=o['interval'];v=tube.score(iv);off=official(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;officialchecks+=1;values[space+'_'+name]=dict(**v,boundary_MAE=(abs(iv[0]-spans[r['query_ordinal']][0])+abs(iv[1]-spans[r['query_ordinal']][1]))/2)
   assert len({values[space+'_'+name]['s']for name in ARMS})==1
   if space!='E':
    vals=[tube.score([int(ids[i]),int(ids[j])+1])for i,j in zip(ii,jj)];scalarcalls+=len(vals)
    for k in curves:curves[k].append(np.array([v[k]for v in vals]))
  for name in ARMS:values['Mean_'+name]={k:avg([values[f'Live{s}_{name}'][k]for s in p['seeds']])for k in ['v','t','s','boundary_MAE']}
  quality={k:np.mean(v,0)for k,v in curves.items()};expected={name:{k:float(o['q'][ii,jj]@curve)for k,curve in quality.items()}for name,o in z['outputs'].items()};rows.append(dict(alias=a,values=values,expected_candidate_quality=expected,diagnostics=diagnostics[a],fixed_unobserved_actual_sIoU=actual));print('score_moving_RGB',a,flush=True)
 means={key:{k:avg([r['values'][key][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for key in ['Mean_'+a for a in ARMS]};ds={};expected={};pairs=[('MovingLiveMix',k)for k in ['Native','HiResAuxMix','UnionCrop0Mix','MovingSourceMix','MovingLive']]+[('MovingLive',k)for k in ['Native','HiResAuxMix','MovingSource']]+[(k,'Native')for k in ['MovingSourceMix','MovingSource','UnionCrop0Mix','HiResAuxMix']]
 for a,b in pairs:
  key=a+'_minus_'+b;delta=np.array([100*(r['values']['Mean_'+a]['v']-r['values']['Mean_'+b]['v'])for r in rows]);ds[key]={k:paired([(100 if k in ['v','t','s']else 1)*(r['values']['Mean_'+a][k]-r['values']['Mean_'+b][k])for r in rows])for k in ['v','t','s','boundary_MAE']};ds[key]['tails']=dict(positive=int((delta>1e-10).sum()),negative=int((delta< -1e-10).sum()),flat=int((abs(delta)<=1e-10).sum()),below_minus3=int((delta< -3).sum()),below_minus5=int((delta< -5).sum()),leave_one_out_mean_range_pp=[float(((delta.sum()-delta)/127).min()),float(((delta.sum()-delta)/127).max())],worst=sorted([dict(alias=r['alias'],delta_pp=float(v))for r,v in zip(rows,delta)],key=lambda x:x['delta_pp'])[:8]);expected[key]={k:paired([100*(r['expected_candidate_quality'][a][k]-r['expected_candidate_quality'][b][k])for r in rows])for k in ['v','t']}
 check();assert all(sha(R/f)==h for f,h in p['pins'].items());cost=read(D/'COST.json');put(P/'RESULTS_PUBLIC.json',dict(status='completed_exposed_moving_RGB_temporal_capability_not_OPD',parents=128,rows=rows,means=means,paired=ds,expected_quality=expected,cost=cost,spatial_frozen=True));dist=expected['MovingLive_minus_Native']['v'];hard=ds['MovingLiveMix_minus_Native']['v'];go_dist=dist['mean']>=2 and dist['CI95'][0]>0;go_hard=hard['mean']>=2 and hard['CI95'][0]>0;decision=dict(primary_hard=hard,pure_teacher_expected=dist,decision='qualifies_one_actual_controlled_transfer'if go_dist else'eligible_static_component_confirmation_only'if go_hard else'close_this_moving_RGB_configuration',versus_Strong=ds['MovingLiveMix_minus_HiResAuxMix']['v'],versus_SourceGuidance=ds['MovingLiveMix_minus_MovingSourceMix']['v'],not_OPD=True,not_independent_confirmation=True,no_crop_seed_mixture_temperature_sweep=True,goal_achieved=False);put(P/'DECISION_PUBLIC.json',decision)
 audit.update(new_GT=True,official_Dense_output_checks=officialchecks,scalar_candidate_quality_calls=scalarcalls,GT_provenance=prov,all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']),all_temporal_arms_sIoU_identical=True);assert audit['all128_before_GT'];put(P/'ROOT_AUDIT.json',audit);put(P/'INDEPENDENT_AUDIT.json',audit);sc=dict(CPU_audit_and_scoring_seconds=time.perf_counter()-beg,GPU_seconds=0,extra_original_RGB_decodes_for_audit=2,publication_archive_cost_excluded=True);put(D/'SCORING_COST.json',sc);put(P/'SCORING_COST_PUBLIC.json',sc)
 lines=['# 冻结轨迹引导的逐帧RGB时间观察：HC2 RC128','','沿用旧cycle的每帧2倍框、图内最小16像素、224方形INTER_LINEAR；真实RGB重新进入同Source Vid。MovingLive由预锁冻结Live41指导，MovingSource为原Source轨迹指导；新框全部丢弃，原时间全跨度65帧/两流envelope/MBR不变。旧HC1八＋Vid八不是本轮确认，其已有移动RGB前向不重新命名新颖性。原空间和网络不更新，本轮不是OPD。','','| 对比 | 完整v增量pp [CI] | 候选期望GTv增量pp [CI] |','|---|---|---|']
 for name,x in ds.items():lines.append(f'| {name} | {x["v"]["mean"]:+.6f} {x["v"]["CI95"]} | {expected[name]["v"]["mean"]:+.6f} {expected[name]["v"]["CI95"]} |')
 tail=ds['MovingLiveMix_minus_Native']['tails'];lines+=['',f'主比较正/负/零 {tail["positive"]}/{tail["negative"]}/{tail["flat"]}；<-3pp {tail["below_minus3"]}，<-5pp {tail["below_minus5"]}；最坏 {tail["worst"]}。','RC128反复曝光、三冻结空间seed父内平均、全128先封存后GT。直接/混合读出不等于OPD贡献。旧缓存不免费；本轮完整Source实际成本包含真实RGB解码/重编码、原生spatial/temporal解码和FP32 replay。裁剪同时改变尺度/身份/上下文/运动参照与形变，可能放大抖动或消除真实移动，不能单独解释为像素细节优势。未超过Source指导/Strong时不声称WHERE→WHEN或方法领先；硬失败不单独否定分布转移资格。','','```json',json.dumps(decision,indent=2),json.dumps(cost,indent=2),json.dumps(audit,indent=2),'```','','固定GT支持及未观察实际节点sIoU不变；不扫crop/seed/混合/温度，不晋升空间/CURRENT/MAIN/PAPER。']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],exposure='HC2 RC128 repeatedly exposed; old HC1/Vid legacy metadata audit is not current evaluation or independent confirmation',legacy_qualification='LEGACY_QUALIFICATION_PUBLIC.json',excluded=['captions','parentIDs','media','GT','weights','raw caches']))
 for f in P.glob('*.json'):
  if str(f.relative_to(R))not in p['pins']:f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 check();assert all(sha(R/f)==h for f,h in p['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in bar['files']);print(json.dumps(decision,indent=2),flush=True)

def main():
 lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:_main()
 finally:lock.close()
if __name__=='__main__':main()
