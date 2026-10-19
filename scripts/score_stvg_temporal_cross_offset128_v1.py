"""Independent actual attention and two-stream probability audit before sealed GT."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,fcntl,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_cross_offset128_v1 import D,P,MEM,PRE,FIN,STAGE,DEV,ARMS
from scripts.score_stvg_where_when_event128_v1 import avg,paired

def _main():
 import torch
 from main_method.runtime import resource_guard
 from vg_tta.stvg_temporal_cross_offset128_v1 import audit_attention
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr,brute_utility
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,mass_table,independent_cdf
 torch.set_num_threads(4);beg=time.perf_counter();p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_before_GT']and bar['weights_and_flags_unchanged']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json')and all(sha(R/f)==h for f,h in p['pins'].items());assert all(sha(R/f['path'])==f['sha256']for f in bar['files']);spaces=oldexact=cdfs=utilities=attentions=0;maxerr=attnerr=0.;ii,jj=np.triu_indices(65,1);diagnostics={}
 def check():resource_guard();assert time.perf_counter()-beg<900
 for nr,r in enumerate(p['records']):
  check();a=r['alias']
  for path,h in [(STAGE/'predictions'/f'{a}.pt',r['native_stage_sha256']),(MEM/'temporal_inputs'/f'{a}.pt',r['memory_sha256']),(PRE/'inputs'/f'{a}.pt',r['input_sha256']),(FIN/'Live/predictions'/f'{a}.pt',r['spatial_sha256']),(DEV/'priors'/f'{a}.pt',r['prior_sha256'])]:assert sha(path)==h
  old=load(STAGE/'predictions'/f'{a}.pt');prior=load(DEV/'priors'/f'{a}.pt');z=load(D/'predictions'/f'{a}.pt');dg={}
  for arm in ['CrossOffset','WrongBinding']:
   gs=probabilities(z['records'],[v[-1]for v in z['logits'][arm]]);ids,q=mass_table(gs);ind=independent_cdf(gs,ids);err=float(abs(q-ind).max());assert err<2e-13;maxerr=max(maxerr,err);cdfs+=1;actual=z['outputs'][arm]['q'];assert ids==z['frame_ids']and np.array_equal(actual,q)and np.array_equal(q>0,z['outputs']['Native']['q']>0);dec=final_mbr(ids,q);assert dec['interval']==z['outputs'][arm]['interval'];pick=np.unique(np.r_[np.linspace(0,len(ii)-1,19).astype(int),np.where((ii==dec['indices'][0])&(jj==dec['indices'][1]))[0]]);u=brute_utility(ids,q,pick);err=float(abs(u-dec['utility_grid'][ii[pick],jj[pick]]).max());assert err<1e-11;maxerr=max(maxerr,err);utilities+=len(pick)
   assert len(z['traces'][arm])==12
   for item in z['traces'][arm]:
    assert item['wrong_binding']==(arm=='WrongBinding')and item['layer']in range(6)and item['offset']in [0,1]
    if nr<2:
     er=audit_attention(item);attnerr=max(attnerr,max(er.values()));attentions+=1
   dg[arm]=dict(cross_mass_by_layer=[avg([t['cross_mass_mean']for t in z['traces'][arm]if t['layer']==i])for i in range(6)],other_temporal_variation_by_layer=[max(t['other_content_temporal_variation']for t in z['traces'][arm]if t['layer']==i)for i in range(6)])
  for name in ['Native','HiResAuxMix']:assert np.array_equal(z['outputs'][name]['q'],old['outputs'][name]['q'])and z['outputs'][name]['interval']==old['outputs'][name]['interval'];oldexact+=1
  for off,nl in enumerate(z['native_logits']):assert torch.equal(nl[-1],prior['logits'][off])
  for name,b in old['trajectories'].items():assert np.array_equal(b,z['trajectories'][name]);spaces+=1
  diagnostics[a]=dict(**dg,TV_Cross_vs_Native=float(abs(z['outputs']['CrossOffset']['q']-z['outputs']['Native']['q']).sum()/2),TV_Cross_vs_Wrong=float(abs(z['outputs']['CrossOffset']['q']-z['outputs']['WrongBinding']['q']).sum()/2))
 assert attentions==48 and cdfs==256 and spaces==512 and oldexact==256
 audit=dict(status='pass',independent_CDF_checks=cdfs,independent_actual_SA_checks=attentions,max_attention_scaled_error=attnerr,old_posteriors_intervals_exact=oldexact,frozen_spatial_arrays_exact=spaces,independent_MBR_original_action_utilities=utilities,max_error=maxerr,native_logits_exact=True,new_GT=False);put(D/'AUDIT_BEFORE_GT.json',audit);check();assert all(sha(R/f)==h for f,h in p['pins'].items());put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_before_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')))
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];rows=[];officialchecks=scalarcalls=0
 for r in p['records']:
  check();a=r['alias'];z=load(D/'predictions'/f'{a}.pt');ids=np.array(z['frame_ids']);values={};curves={k:[]for k in ['v','t']};actual={}
  for space,boxes in z['trajectories'].items():
   tube=DenseTube(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True);sel=np.isin(tube.fids,np.array(sorted(set(ids)-set(r['physical_frame_ids8']))));actual[space]=dict(nodes=int(sel.sum()),sIoU=float(tube.iou[sel].mean())if sel.any()else None,all_temporal_arms_identical=True)
   for name,o in z['outputs'].items():
    assert final_mbr(ids,o['q'])['interval']==o['interval'];iv=o['interval'];v=tube.score(iv);off=official(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;officialchecks+=1;values[space+'_'+name]=dict(**v,boundary_MAE=(abs(iv[0]-spans[r['query_ordinal']][0])+abs(iv[1]-spans[r['query_ordinal']][1]))/2)
   assert len({values[space+'_'+name]['s']for name in ARMS})==1
   if space!='E':
    vals=[tube.score([int(ids[i]),int(ids[j])+1])for i,j in zip(ii,jj)];scalarcalls+=len(vals)
    for k in curves:curves[k].append(np.array([v[k]for v in vals]))
  for name in ARMS:values['Mean_'+name]={k:avg([values[f'Live{s}_{name}'][k]for s in p['seeds']])for k in ['v','t','s','boundary_MAE']}
  quality={k:np.mean(v,0)for k,v in curves.items()};expected={name:{k:float(o['q'][ii,jj]@curve)for k,curve in quality.items()}for name,o in z['outputs'].items()};rows.append(dict(alias=a,values=values,expected_candidate_quality=expected,diagnostics=diagnostics[a],fixed_unobserved_actual_sIoU=actual));print('score_cross_offset',a,flush=True)
 means={key:{k:avg([r['values'][key][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for key in ['Mean_'+a for a in ARMS]};ds={};expected={};pairs=[('CrossOffset',k)for k in ['Native','WrongBinding','HiResAuxMix']]+[('WrongBinding','Native')]
 for a,b in pairs:
  key=a+'_minus_'+b;delta=np.array([100*(r['values']['Mean_'+a]['v']-r['values']['Mean_'+b]['v'])for r in rows]);ds[key]={k:paired([(100 if k in ['v','t','s']else 1)*(r['values']['Mean_'+a][k]-r['values']['Mean_'+b][k])for r in rows])for k in ['v','t','s','boundary_MAE']};ds[key]['tails']=dict(positive=int((delta>1e-10).sum()),negative=int((delta< -1e-10).sum()),flat=int((abs(delta)<=1e-10).sum()),below_minus3=int((delta< -3).sum()),below_minus5=int((delta< -5).sum()),leave_one_out_mean_range_pp=[float(((delta.sum()-delta)/127).min()),float(((delta.sum()-delta)/127).max())],worst=sorted([dict(alias=r['alias'],delta_pp=float(v))for r,v in zip(rows,delta)],key=lambda x:x['delta_pp'])[:8]);expected[key]={k:paired([100*(r['expected_candidate_quality'][a][k]-r['expected_candidate_quality'][b][k])for r in rows])for k in ['v','t']}
 check();assert all(sha(R/f)==h for f,h in p['pins'].items());cost=read(D/'COST.json');put(P/'RESULTS_PUBLIC.json',dict(status='completed_exposed_cross_offset_SA_capability_not_OPD',parents=128,rows=rows,means=means,paired=ds,expected_quality=expected,cost=cost,spatial_frozen=True));dist=expected['CrossOffset_minus_Native']['v'];hard=ds['CrossOffset_minus_Native']['v'];controls=['WrongBinding','HiResAuxMix'];go_dist=dist['mean']>=2 and dist['CI95'][0]>0 and all(expected['CrossOffset_minus_'+k]['v']['CI95'][0]>0 for k in controls);go_hard=hard['mean']>=2 and hard['CI95'][0]>0 and all(ds['CrossOffset_minus_'+k]['v']['CI95'][0]>0 for k in controls);decision=dict(primary_hard=hard,primary_expected=dist,decision='qualifies_one_actual_controlled_transfer'if go_dist else'eligible_static_component_confirmation_only'if go_hard else'close_this_cross_offset_SA_configuration',not_OPD=True,not_independent_confirmation=True,no_layer_strength_position_sweep=True,goal_achieved=False);put(P/'DECISION_PUBLIC.json',decision)
 audit.update(new_GT=True,official_Dense_output_checks=officialchecks,scalar_candidate_quality_calls=scalarcalls,GT_provenance=prov,all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']),all_temporal_arms_sIoU_identical=True);assert audit['all128_before_GT'];put(P/'ROOT_AUDIT.json',audit);put(P/'INDEPENDENT_AUDIT.json',audit);sc=dict(CPU_audit_and_scoring_seconds=time.perf_counter()-beg,GPU_seconds=0,publication_archive_cost_excluded=True);put(D/'SCORING_COST.json',sc);put(P/'SCORING_COST_PUBLIC.json',sc)
 lines=['# 仅开放TimeDecoder跨offset自注意力：HC128能力检查','','原第二pass query/位置与逐帧CA memory/FFN/head/两流envelope均保留；只有SA允许另一流状态。WrongBinding固定移动另一流content和V、不移动位置。不是原65完整Source重编码，也不更新空间或网络。原位置存在两相位局部索引重合边界，不能宣称完整物理时间对齐。','','| 对比 | 完整v增量pp [CI] | 候选期望GTv增量pp [CI] |','|---|---|---|']
 for name,x in ds.items():lines.append(f'| {name} | {x["v"]["mean"]:+.6f} {x["v"]["CI95"]} | {expected[name]["v"]["mean"]:+.6f} {expected[name]["v"]["CI95"]} |')
 tail=ds['CrossOffset_minus_Native']['tails'];lines+=['',f'主比较正/负/零 {tail["positive"]}/{tail["negative"]}/{tail["flat"]}；低于−3pp {tail["below_minus3"]}，低于−5pp {tail["below_minus5"]}，最坏 {tail["worst"]}。','RC128反复曝光、三个冻结空间seed父内平均、全128先封存后GT。静态/期望质量不等于OPD价值；本轮没有学生更新，不以硬读出失败否定全部蒸馏，不按控制晋升。固定GT空间与未观察实际节点sIoU不变。','','```json',json.dumps(decision,indent=2),json.dumps(cost,indent=2),json.dumps(audit,indent=2),'```','','Source Vid检查点固定，0新RGB/encoder/专家/网络训练；旧缓存创建成本不免费。不扫层/强度/位置码，空间/CURRENT/MAIN/PAPER保持原样。']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],exposure='RC128 repeatedly exposed; TC109 not selected',source_evidence='actual frozen TimeDecoder source and old R8/R11 protocols, not a novel literature claim',excluded=['captions','parentIDs','media','GT','weights','raw caches']))
 for f in P.glob('*.json'):f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 check();assert all(sha(R/f)==h for f,h in p['pins'].items())and all(sha(R/f['path'])==f['sha256']for f in bar['files']);print(json.dumps(decision,indent=2),flush=True)

def main():
 lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:_main()
 finally:lock.close()
if __name__=='__main__':main()
