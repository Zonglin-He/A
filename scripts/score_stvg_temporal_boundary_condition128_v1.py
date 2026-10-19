"""Post-seal physical-mask/label audit and conditional information scoring."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_boundary_condition128_v1 import D,P,MEM,EVENT,DEV,PRE,FIN
from scripts.score_stvg_where_when_event128_v1 import paired,avg

def main():
 import torch
 from main_method.runtime import resource_guard
 from scripts.stvg_temporal_marginal_decode_util256_v1 import probabilities,independent_cdf
 torch.set_num_threads(4);beg=time.perf_counter();p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_before_GT']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json')and all(sha(R/f)==h for f,h in p['pins'].items());assert all(sha(R/f['path'])==f['sha256']for f in bar['files']);masks=labels=cdfchecks=spatialchecks=0;maxerr=0.
 for r in p['records']:
  resource_guard();assert time.perf_counter()-beg<900;a=r['alias'];z=load(D/'predictions'/f'{a}.pt');event=load(EVENT/'predictions'/f'{a}.pt');assert np.array_equal(z['paired_uniforms'],event['paired_uniforms'])and all(np.array_equal(x,y)for x,y in zip(z['sampled_pairs'],event['sampled_pairs']));ids=np.array(z['frame_ids'])
  for name,o in z['queries'].items():
   direction=o['direction'];variant=o['variant'];pair=z['sampled_pairs'][o['state']][o['sample']];assert np.array_equal(pair,o['pair']);i,j=map(int,pair);cut=int(ids[j if direction=='start'else i]);assert cut==o['physical_cut']
   for rec,mask in zip(z['records'],o['masks']):
    actual=np.array(rec['frame_ids']);keep=np.array([t<=cut if direction=='start'else t>=cut for t in actual]);n=int(keep.sum());assert n>0
    if variant=='Wrong':keep=np.array([k>=len(actual)-n if direction=='start'else k<n for k in range(len(actual))])
    assert np.array_equal(mask,~keep);masks+=1
   q=independent_cdf(probabilities(z['records'],o['logits']),z['frame_ids']);err=float(abs(q-o['joint']).max());assert err<2e-12;maxerr=max(maxerr,err);cdfchecks+=1
   v=o['joint'][:,j].copy()if direction=='start'else o['joint'][i,:].copy();support=z['q0'][:,j]>0 if direction=='start'else z['q0'][i,:]>0;v[~support]=0;v/=v.sum();assert np.array_equal(v,o['label'])and abs(v.sum()-1)<1e-12 and np.array_equal(v>0,support);base=z['q0'][:,j].copy()if direction=='start'else z['q0'][i,:].copy();base/=base.sum();assert np.array_equal(base,o['native_label']);labels+=1
  oldsp=load(FIN/'Live/predictions'/f'{a}.pt');oldinp=load(PRE/'inputs'/f'{a}.pt')
  for key,boxes in z['trajectories'].items():assert np.array_equal(boxes,oldinp['E']if key=='E'else oldsp['outputs'][f'T20_Live_{key[4:]}']);spatialchecks+=1
 audit=dict(status='pass',physical_masks_exact=masks,conditional_labels_exact=labels,independent_CDF_tables=cdfchecks,maxerror=maxerr,frozen_spatial_arrays_exact=spatialchecks,all_inputs_outputs_before_GT=True,new_GPU=0);put(D/'AUDIT_BEFORE_GT.json',audit)
 put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_before_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')))
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];ii,jj=np.triu_indices(65,1);rows=[];officialchecks=scalarcalls=0
 for r in p['records']:
  resource_guard();assert time.perf_counter()-beg<900;a=r['alias'];z=load(D/'predictions'/f'{a}.pt');ids=np.array(z['frame_ids']);curves={k:[]for k in ['v','t']}
  for space,boxes in z['trajectories'].items():
   if space=='E':continue
   tube=DenseTube(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True);vals=[tube.score([int(ids[i]),int(ids[j])+1])for i,j in zip(ii,jj)];scalarcalls+=len(vals)
   for k in curves:
    M=np.zeros((65,65));M[ii,jj]=[v[k]for v in vals];curves[k].append(M)
   for pos in np.linspace(0,len(ii)-1,9).astype(int):
    off=official(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],[int(ids[ii[pos]]),int(ids[jj[pos]])+1],'hc2');assert max(abs(off[k]-vals[pos][k])for k in ['v','t','s'])<2e-10;officialchecks+=1
  quality={k:np.mean(v,0)for k,v in curves.items()};cases=[]
  for name,o in z['queries'].items():
   i,j=map(int,o['pair']);v={}
   for key,M in quality.items():
    curve=M[:,j]if o['direction']=='start'else M[i,:];v[key]=float(o['label']@curve);v['Native_'+key]=float(o['native_label']@curve);v['Sample_'+key]=float(M[i,j])
   cases.append(dict(name=name,state=o['state'],sample=o['sample'],direction=o['direction'],variant=o['variant'],values=v,label_TV_from_Native=float(abs(o['label']-o['native_label']).sum()/2)))
  values={}
  for state in [0,1]:
   for variant in ['Correct','Wrong']:
    c=[x for x in cases if x['state']==state and x['variant']==variant];assert len(c)==8;values[f'S{state}_{variant}']={key:avg([x['values'][key]for x in c])for key in c[0]['values']};values[f'S{state}_{variant}']['label_TV_from_Native']=avg([x['label_TV_from_Native']for x in c])
  rows.append(dict(alias=a,values=values,cases=cases));print('score_condition',a,flush=True)
 contrasts={}
 for state in [0,1]:
  for baseline in ['Native','Wrong']:
   ds={}
   for metric in ['v','t']:
    diff=[100*(r['values'][f'S{state}_Correct'][metric]-(r['values'][f'S{state}_Correct']['Native_'+metric]if baseline=='Native'else r['values'][f'S{state}_Wrong'][metric]))for r in rows];ds[metric]=paired(diff)
    if metric=='v':
     a=np.array(diff);ds['tails']=dict(positive=int((a>1e-10).sum()),negative=int((a< -1e-10).sum()),flat=int((abs(a)<=1e-10).sum()),below_minus3=int((a< -3).sum()),below_minus5=int((a< -5).sum()),leave_one_out_mean_range_pp=[float(((a.sum()-a)/(len(a)-1)).min()),float(((a.sum()-a)/(len(a)-1)).max())],worst=sorted([dict(alias=r['alias'],delta_pp=float(d))for r,d in zip(rows,a)],key=lambda x:x['delta_pp'])[:8])
   contrasts[f'S{state}_Correct_minus_{baseline}']=ds
 means={name:{key:avg([r['values'][name][key]for r in rows])for key in rows[0]['values'][name]}for name in rows[0]['values']};cost=read(D/'COST.json');result=dict(status='completed_exposed_conditional_capability_not_system_gain_or_OPD',parents=128,rows=rows,means=means,paired=contrasts,cost=cost,no_new_final_intervals=True,no_new_student_updates=True,spatial_frozen=True);put(P/'RESULTS_PUBLIC.json',result)
 a=contrasts['S0_Correct_minus_Native']['v'];b=contrasts['S0_Correct_minus_Wrong']['v'];go=a['mean']>=2 and a['CI95'][0]>0 and b['CI95'][0]>0;decision=dict(primary=a,matched_wrong_control=b,decision='qualifies_one_actual_controlled_transfer'if go else'close_this_boundary_SA_mask_configuration',conditional_not_system_gain=True,not_OPD=True,no_mask_layer_strength_sweep=True,goal_achieved=False);put(P/'DECISION_PUBLIC.json',decision)
 audit.update(official_Dense_checks=officialchecks,scalar_candidate_quality_calls=scalarcalls,GT_provenance=prov,all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']),attention_layer_calls_audited=cost['attention_layer_calls_audited'],max_blocked_attention=cost['max_blocked_attention']);assert audit['all128_before_GT'];put(P/'ROOT_AUDIT.json',audit);put(P/'INDEPENDENT_AUDIT.json',audit);put(D/'SCORING_COST.json',dict(CPU_seconds=time.perf_counter()-beg,GPU_seconds=0))
 lines=['# 原生时间decoder边界条件SA：HC128能力资格','','重用旧event_query的S0与真实R7 Live41 S1及各四个样本，不是本轮新学生更新。仅六层SA keys按当前另一边界截prefix/suffix；Wrong等keys数量保留相反侧。原native query/full CA逐帧memory仍包含全局上下文，不称严格因果隔离、原生校准条件概率或一致联合teacher。固定另外一侧边界的条件期望GT fullv/t是能力统计，不是新增最终系统vIoU。','','| 配对 | 条件期望fullv增量pp [CI] | t增量pp [CI] |','|---|---|---|']
 for name,x in contrasts.items():lines.append(f'| {name} | {x["v"]["mean"]:+.6f} {x["v"]["CI95"]} | {x["t"]["mean"]:+.6f} {x["t"]["CI95"]} |')
 lines+=['','全部128父、两方向/四样本/三个冻结空间seed先父内平均，保留null与负尾，不以多个查询/种子虚增n。S1只是响应检查，不把旧S1收益归本轮刷新；没有最终全局MBR或OPD成绩。','','```json',json.dumps(decision,indent=2),json.dumps(cost,indent=2),json.dumps(audit,indent=2),'```','','旧RawAction teacher净负与相对减损都保留；本轮没有重调其温度/先验/步长。0新encoder/RGB/专家/网络训练/空间/学生update，GPU只是冻结时间decoder重解码，缓存成本不免费。CURRENT/MAIN/PAPER不变，全部RC128重复开发曝光，仍需独立确认才方法结论。']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],old_states_and_samples='exact old event_query128 saved real R7 states and paired uniforms',exposure='RC128 exposed development; TC109 not used',excluded=['media','annotations','weights','raw caches','text','parentIDs']))
 for f in P.glob('*.json'):f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 print(json.dumps(decision,indent=2),flush=True)
if __name__=='__main__':main()
