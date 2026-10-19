"""Post-seal scoring and independent audit. All condition/readout arms retained."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,json,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_roi_sampler128_v1 import D,P,OLD,PRE,FIN,DEV,CONDITIONS
from scripts.score_stvg_where_when_event128_v1 import avg,paired

def main():
 import torch
 torch.set_num_threads(4);p=read(D/'PLAN_LOCK.json');b=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert b['count']==128 and b['all_before_GT'] and b['plan_sha256']==sha(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items())
 for f in b['files']:assert sha(R/f['path'])==f['sha256']
 put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json'),all128_before_GT=True))
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 from vg_tta.stvg_temporal_foreground128_v1 import project
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];base={r['alias']:r for r in read(R/'results/stvg_error_state/2026-10-19/temporal_increment128/RESULTS_PUBLIC.json')['rows']};rows=[];metric_checks=space_checks=projection_checks=0;pool_checks=native_pool_checks=0
 for r in p['records']:
  a=r['alias'];z=load(D/'predictions'/f'{a}.pt');old=load(OLD/'predictions'/f'{a}.pt');inp=load(PRE/'inputs'/f'{a}.pt');values={}
  assert all(sha(R/f)==h for f,h in r['cache_pins'].items())
  for c in CONDITIONS:
   assert all(torch.equal(x,y)for x,y in zip(z['endpoint_logits'][c],z['endpoint_logits']['NativeGlobal']))
   v,au=project(z['endpoint_logits'][c],z['actions'][c],inp['records']);assert all(v[n]==z['outputs'][c+n]for n in ['Full','Fluct']);projection_checks+=sum(x[n]['candidates']for x in au for n in ['Full','Fluct'])
  assert z['native_equivalent_checks']==2;native_pool_checks+=2
  for meta in z['pooling_audit'].values():
   w=meta['weights'];assert torch.isfinite(w).all() and (w>=0).all() and torch.allclose(w.sum(-1),torch.ones(len(w)),atol=2e-6);pool_checks+=len(w)
  for space,boxes in z['trajectories'].items():
   assert np.array_equal(boxes,old['trajectories'][space]);space_checks+=1
   tube=DenseTube(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True)
   for arm,o in z['outputs'].items():
    iv=o['interval'];v=tube.score(iv);off=official(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;metric_checks+=1
    if arm=='NativeMBR':assert max(abs(v[k]-base[a]['values'][space+'_L0'][k])for k in ['v','t','s'])<2e-10
    values[space+'_'+arm]=dict(**v,interval=iv,boundary_MAE=(abs(iv[0]-spans[r['query_ordinal']][0])+abs(iv[1]-spans[r['query_ordinal']][1]))/2)
  for arm in p['arms']:
   values['LiveMean_'+arm]={k:avg([values[f'Live{s}_{arm}'][k]for s in p['seeds']])for k in ['v','t','s','boundary_MAE']}
  for n in ['Full','Fluct']:values['LiveOwn_'+n]={k:avg([values[f'Live{s}_Live{s}_ROI{n}'][k]for s in p['seeds']])for k in ['v','t','s','boundary_MAE']}
  rows.append(dict(alias=a,values=values));print('scored',a,flush=True)
 keys=['LiveMean_'+a for a in p['arms']]+['LiveOwn_Full','LiveOwn_Fluct'];means={key:{k:avg([r['values'][key][k]for r in rows])for k in ['v','t','s','boundary_MAE']}for key in keys};contrasts=[(key,'LiveMean_NativeMBR')for key in keys if key!='LiveMean_NativeMBR']+[(f'LiveMean_ReferROI{n}',f'LiveMean_NativeGlobal{n}')for n in ['Full','Fluct']]+[(f'LiveMean_ReferROI{n}',f'LiveMean_ShiftedReferROI{n}')for n in ['Full','Fluct']]+[(f'LiveOwn_{n}',f'LiveMean_E_ROI{n}')for n in ['Full','Fluct']]+[(f'LiveMean_ReferROI{n}',f'LiveMean_ValidGlobal{n}')for n in ['Full','Fluct']];paireds={}
 for a,bkey in contrasts:
  differences={k:[(100 if k in ['v','t','s']else 1)*(r['values'][a][k]-r['values'][bkey][k])for r in rows]for k in ['v','t','s','boundary_MAE']};v=np.array(differences['v']);paireds[a+'_minus_'+bkey]=dict(**{k:paired(x)for k,x in differences.items()},tails=dict(positive=int((v>1e-10).sum()),negative=int((v< -1e-10).sum()),flat=int((abs(v)<=1e-10).sum()),below_minus3=int((v< -3).sum()),below_minus5=int((v< -5).sum()),positive_contribution_pp=float(np.maximum(v,0).mean()),negative_contribution_pp=float(np.minimum(v,0).mean()),worst=sorted([dict(alias=r['alias'],delta_pp=float(d))for r,d in zip(rows,v)],key=lambda r:r['delta_pp'])[:8]))
 costs=[read(f)for f in D.glob('COST_*.json')];cost=dict(GPU_seconds=sum(x['GPU_seconds']for x in costs),temporal_offset_decodes=0,temporal_classifier_forwards=sum(x['temporal_classifier_forwards']for x in costs),equivalent_native_classifier_checks=sum(x['equivalent_native_classifier_checks']for x in costs),new_RGB=0,new_Source_encoder=0,new_experts=0,network_training=0,spatial_updates=0,attempts=costs)
 res=dict(status='completed_exposed_development_probe',parents=128,seeds=p['seeds'],means=means,paired=paireds,rows=rows,cost=cost,primary=p['primary'],new_OPD=False,spatial_frozen=True,GT_offline_only=True);put(P/'RESULTS_PUBLIC.json',res)
 audit=dict(status='pass',official_DenseTube_checks=metric_checks,frozen_spatial_arrays=space_checks,normalized_pool_checks=pool_checks,native_prepool_exact_checks=native_pool_checks,independent_slice_projection_values=projection_checks,native_old_metrics_exact=True,native_endpoints_same_prior_all_conditions=True,all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in read(D/'GLOBAL_PREDICTION_BARRIER.json')['files']),weights_unchanged=True,GT_provenance=prov);assert audit['all128_before_GT'];put(P/'INDEPENDENT_AUDIT.json',audit);put(P/'ROOT_AUDIT.json',audit)
 selected=min(p['eligible'],key=lambda x:(-means['LiveMean_'+x]['v'],x));gain=paireds['LiveMean_'+selected+'_minus_LiveMean_NativeMBR']['v'];decision=dict(selected=selected,development_selected_delta=gain,primary_delta=paireds['LiveMean_'+p['primary']+'_minus_LiveMean_NativeMBR']['v'],decision='freeze_for_parent_disjoint_confirmation'if gain['mean']>=2 and gain['CI95'][0]>0 else'no_supported_meaningful_candidate',not_OPD=True,independent_confirmation=False);put(P/'SELECTION_PUBLIC.json',decision)
 lines=['# 目标区域运动池化：HC128能力检查','','复用原第二pass融合运动与文本，区域池化后读取已有source监督的TemporalSampling。全局原算子/valid全局/对象区域/横移错误区域同分类器，原endpoint prior完全相同；不训练、不读新RGB/专家、不改空间。RC128均为曝光开发父。Full/Fluct为旧报告既有规则，本轮检验当前基线与目标条件记忆，不能包装成新算法或OPD。','','| 条件与读出 | full vIoU% | tIoU% | 相对原MBR v增量pp [CI95] |','|---|---:|---:|---|']
 for arm in p['arms']:
  key='LiveMean_'+arm;v=means[key];d=paireds.get(key+'_minus_LiveMean_NativeMBR');delta='0'if d is None else f'{d["v"]["mean"]:+.6f} [{d["v"]["CI95"][0]:+.6f},{d["v"]["CI95"][1]:+.6f}]';lines.append(f'| {arm} | {100*v["v"]:.6f} | {100*v["t"]:.6f} | {delta} |')
 lines+=['','全部条件和全部父保留，ReferROIFluct为先定主假设；四个eligible开发选优有选择偏差。空间s按构造固定。旧64为不同空间基线，不宣称本轮独立确认；池化变化不是新模型或新增视觉观测。与错位/E/Own对照完整在全结果。','','```json',json.dumps(decision,ensure_ascii=False,indent=2),'```','','## 计费与核验','','```json',json.dumps(cost,indent=2),json.dumps(audit,indent=2),'```','','全逐父×空间×条件/读出signed和unknown保留，负尾完整在RESULTS_PUBLIC.json。生产/论文/研究配置不自动晋升。若能力成立还需确认和Direct/Fixed/Live，不能把projection有效称OPD有效。']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],spatial_config_sha256=sha(R/'methods/stvg_referdino_opd_v1/config.json'),exposure='all128 exposed development; existing historical projection rules',excluded=['media','GT coordinates','captions','parent IDs','weights','raw caches']))
 for f in P.glob('*.json'):f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 print(json.dumps(decision,indent=2),flush=True)

if __name__=='__main__':main()
