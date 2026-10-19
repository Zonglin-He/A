"""Explicit post-main wrong-video control; raw diagnostics remain immutable."""
from pathlib import Path
import sys,os,time,numpy as np,json,shutil,copy
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R));os.environ['CUDA_VISIBLE_DEVICES']=''
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
from scripts.run_stvg_temporal_foreground_info128_v1 import D,P,PRE,FIN,classification,rankdiag
A=D/'revision001'


def repair_and_lock():
 source=P/'RESULTS_PUBLIC.json';backup=D/'recovery/invalid_aux_rho';backup.mkdir(parents=True,exist_ok=True)
 for name in ['RESULTS_PUBLIC.json','REPORT.md']:shutil.copyfile(P/name,backup/name)
 old=read(source);z=copy.deepcopy(old)
 for r in z['rows']:
  for k in r['ranking']:r['ranking'][k]['within_length_rank_correlation']=None
 for k in z['fields']:z['fields'][k]['ranking']['within_length_rank_correlation']=dict(mean=None,defined_parents=0,valid=False,reason='FP64 pseudo-ties contaminate rankdata auxiliary statistic')
 for k in z['paired']:z['paired'][k]['ranking']['within_length_rank_correlation']=dict(mean=None,CI95=None,parents=0,valid=False)
 def strip(v):
  if isinstance(v,dict):return {k:strip(x)for k,x in v.items()if k!='within_length_rank_correlation'}
  if isinstance(v,list):return [strip(x)for x in v]
  return v
 assert strip(z)==strip(old);z['auxiliary_measurement_validity']=dict(within_length_rank_correlation='INVALID_EXCLUDED; original bytes preserved',all_main_statistics='unchanged exact')
 source.write_text(json.dumps(z,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n');(P/'REPORT.md').write_text((P/'REPORT.md').read_text()+'\n\n修订001：辅助within_length_rank_correlation由于Flat的FP64伪tie排序无效，统一置null，不用于解释。原结果/报告按原字节存档；主AUROC/Brier/带容差pair accuracy/有益AUC及区间逐项exact。此后附加一个post-main固定错视频对照，见ADDENDUM_001.md，不追认为prelock。\n')
 put(P/'CORRECTION_001_PUBLIC.json',dict(status='valid_main_with_invalid_aux_excluded',original_results_sha256=sha(backup/'RESULTS_PUBLIC.json'),corrected_results_sha256=sha(source),all_non_rho_statistics_exact=True,original_fields_energies_predictions_unchanged=True,no_new_GT_calculation_for_correction=True,original_bytes_preserved=True,invalid_aux_reason='rankdata exact ties admit prefix-subtraction FP64 noise in theoretically constant Flat'))
 plan=read(D/'PLAN_LOCK.json');records=sorted(plan['records'],key=lambda r:r['alias']);put(A/'LOCK.json',dict(records=records,donors={r['alias']:records[(k+1)%128]['alias']for k,r in enumerate(records)},pins={str(f.relative_to(R)):sha(f)for f in [Path(__file__),P/'ADDENDUM_001.md']},original_plan_sha256=sha(D/'PLAN_LOCK.json'),original_diagnostic_barrier_sha256=sha(D/'GLOBAL_DIAGNOSTIC_BARRIER.json'),main_results_sha256=sha(source),post_main_GT_review=True,created_epoch=time.time(),CPU_seconds_cap_total=120));print('revision locked after main score',flush=True)


def export():
 from methods.stvg_referdino_opd_v1.runtime import setup
 from main_method.runtime import resource_guard
 setup();p=read(A/'LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());beg=time.perf_counter();files=[];byalias={r['alias']:r for r in p['records']};ii,jj=np.triu_indices(65,1)
 for r in p['records']:
  resource_guard();alias=r['alias'];donor=p['donors'][alias];assert alias!=donor and r['parent']!=byalias[donor]['parent'];z=load(D/'diagnostics'/f'{alias}.pt');other=load(D/'diagnostics'/f'{donor}.pt');x=np.array(z['ids'],float);dx=np.array(other['ids'],float);v=z['fields']['Action'];w=other['fields']['Action'];x=(x-x[0])/(x[-1]-x[0]);dx=(dx-dx[0])/(dx[-1]-dx[0]);interp=np.interp(x,dx,w);mu=float(v.mean());sd=float(v.std());ds=float(interp.std());degenerate=sd<=1e-12 or ds<=1e-12;field=np.full(65,mu)if degenerate else mu+(interp-interp.mean())*(sd/ds)
  assert abs(field.mean()-mu)<1e-12
  if not degenerate:assert abs(field.std()-sd)<1e-12
  pref=np.r_[0.,np.cumsum(field)];energy=pref[jj+1]-pref[ii];files.append(save(A/'controls'/f'{alias}.pt',dict(alias=alias,donor=donor,field=field,energy=energy,source_std=sd,interpolated_donor_std=ds,degenerate=degenerate,GT_read=False)));print('wrong',alias,flush=True)
 put(A/'GLOBAL_CONTROL_BARRIER.json',dict(count=128,files=files,lock_sha256=sha(A/'LOCK.json'),all_controls_before_GT=True,post_main_review_disclosed=True));put(A/'EXPORT_COST.json',dict(CPU_seconds=time.perf_counter()-beg,GPU_seconds=0))


def score():
 import torch
 from scripts.score_stvg_where_when_event128_v1 import avg,paired
 torch.set_num_threads(4);p=read(A/'LOCK.json');bar=read(A/'GLOBAL_CONTROL_BARRIER.json');assert bar['count']==128 and bar['lock_sha256']==sha(A/'LOCK.json')and all(sha(R/f)==h for f,h in p['pins'].items());assert sha(P/'RESULTS_PUBLIC.json')==p['main_results_sha256']
 for f in bar['files']:assert sha(R/f['path'])==f['sha256']
 put(A/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_control_fields_before_GT=True,post_main_GT_revision=True))
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube
 gt,spans,_=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];old={r['alias']:r for r in read(P/'RESULTS_PUBLIC.json')['rows']};ii,jj=np.triu_indices(65,1);rows=[];beg=time.perf_counter();correction_main=copy.deepcopy(read(P/'RESULTS_PUBLIC.json'));used=read(D/'EXPORT_COST.json')['CPU_seconds']+read(D/'COST.json')['CPU_seconds_score']+read(A/'EXPORT_COST.json')['CPU_seconds']
 for r in p['records']:
  assert used+time.perf_counter()-beg<120;alias=r['alias'];z=load(D/'diagnostics'/f'{alias}.pt');c=load(A/'controls'/f'{alias}.pt');ids=np.array(z['ids']);g,h=spans[r['query_ordinal']];labels=(ids>=g)&(ids<h);sp=load(FIN/'Live/predictions'/f'{alias}.pt');qualities=[];baselines=[]
  for seed in [41,42,43]:
   tube=DenseTube(np.asarray(sp['outputs'][f'T20_Live_{seed}']),raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True);S,E=ids[ii],ids[jj]+1;lo=np.searchsorted(tube.fids,np.maximum(S,g));hi=np.maximum(lo,np.searchsorted(tube.fids,np.minimum(E,h)));prefix=np.r_[0.,np.cumsum(tube.iou)];qualities.append((prefix[hi]-prefix[lo])/np.maximum(np.maximum(E,h)-np.minimum(S,g),1));baselines.append(tube.score(z['native_interval'])['v'])
  quality=np.mean(qualities,axis=0);base=float(np.mean(baselines));actual_rank=rankdiag(z['mask_energies']['Action'],quality,ii,jj,base);original=old[alias]['ranking']['Action'];assert all(actual_rank[k]==original[k]for k in original if k!='within_length_rank_correlation');assert classification(z['fields']['Action'],labels)==old[alias]['classification']['Action'];wrong_rank=rankdiag(c['energy'],quality,ii,jj,base);wrong_rank['within_length_rank_correlation']=None;rows.append(dict(alias=alias,donor=c['donor'],degenerate=c['degenerate'],classification=classification(c['field'],labels),ranking=wrong_rank));print('wrong_info',alias,flush=True)
 means={g:{m:dict(mean=avg([r[g][m]for r in rows]),defined_parents=sum(r[g][m]is not None for r in rows))for m in rows[0][g]}for g in ['classification','ranking']};contrast={g:{m:paired([old[r['alias']][g]['Action'][m]-r[g][m]for r in rows if r[g][m]is not None and old[r['alias']][g]['Action'][m]is not None])for m in metrics}for g,metrics in [('classification',['AUROC','Brier','logloss','recall']),('ranking',['equal_node_length_pair_accuracy','beneficial_vs_harmful_AUC'])]}
 cost=dict(CPU_seconds_export=read(A/'EXPORT_COST.json')['CPU_seconds'],CPU_seconds_score=time.perf_counter()-beg,total_main_and_addendum_CPU_seconds=used+time.perf_counter()-beg,GPU_seconds=0,new_model_expert_or_updates=0);assert cost['total_main_and_addendum_CPU_seconds']<120
 result=dict(status='completed_post_main_fixed_wrong_video_control_not_confirmation',parents=128,means=means,Action_minus_Wrong=contrast,rows=rows,cost=cost,original_main_recomputed_exact=True,all128_controls_before_GT=True,post_main_review_disclosed=True,new_method_outputs=False,confidence_distribution_not_preserved_only_mean_std=True);put(P/'WRONG_VIDEO_CONTROL_PUBLIC.json',result);put(A/'COST.json',cost);assert read(P/'RESULTS_PUBLIC.json')==correction_main
 (P/'REPORT.md').write_text((P/'REPORT.md').read_text()+'\n\n## Post-main固定错父视频控制\n\n按预定alias+1及归一化physicalclock，插值后只匹配接收者均值/std；不声称完整分布相同。128控制全部先seal后新GT，主原统计重算exact。此修订在主GT后，非原prelock或独立确认。\n\n```json\n'+json.dumps(dict(means=means,Action_minus_Wrong=contrast,cost=cost),indent=2)+'\n```\n')
 print(json.dumps(contrast,indent=2),flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
