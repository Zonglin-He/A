"""One bounded cached information qualification; no new temporal readout method."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import sys,time,fcntl,json,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,save,sha
D=R/'artifacts/stvg_temporal_foreground_info128_v1';P=R/'results/stvg_error_state/2026-10-20/temporal_foreground_info128';MEM=R/'artifacts/stvg_temporal_object_memory128_v1';FG=R/'artifacts/stvg_temporal_foreground128_v1';ST=R/'artifacts/stvg_temporal_native_stage128_v1';DEV=R/'artifacts/stvg_temporal_increment128_v1';PRE=R/'artifacts/stvg_referdino_condition_hc128_v1';FIN=R/'artifacts/stvg_referdino_finalize_v1'
FIELDS=['Action','ActionEOS','ActionFlat','ActionShift','Motion','MotionFlat','MotionShift','Appearance','AppearanceFlat','AppearanceShift']


def auc(x,y):
 x=np.asarray(x);y=np.asarray(y,bool);a=x[y];b=x[~y]
 if not len(a)or not len(b):return None
 d=a[:,None]-b;return float(np.mean(np.where(abs(d)<=1e-12,.5,d>0)))


def classification(x,y):
 prob=np.exp(-np.logaddexp(0.,-x));pred=x>=0;y=np.asarray(y,bool);TP=int((pred&y).sum());return dict(AUROC=auc(x,y),Brier=float(np.mean((prob-y)**2)),logloss=float(np.mean(np.logaddexp(0,x)-x*y)),predicted_positive_fraction=float(pred.mean()),GT_positive_fraction=float(y.mean()),recall=TP/int(y.sum())if y.any()else None,precision=TP/int(pred.sum())if pred.any()else None)


def rankdiag(score,quality,ii,jj,base):
 from scipy.stats import rankdata
 valid=(quality>=base+.02)|(quality<=base-.02);good=quality[valid]>=base+.02
 A=auc(score[valid],good)if valid.any()else None;correct=total=0;rx=[];ry=[]
 for length in range(1,65):
  loc=np.where(jj-ii==length)[0];x=score[loc];y=quality[loc];a,b=np.triu_indices(len(loc),1);dx=x[a]-x[b];dy=y[a]-y[b];v=abs(dy)>1e-12
  correct+=float(np.where(abs(dx[v])<=1e-10,.5,dx[v]*dy[v]>0).sum());total+=int(v.sum())
  if len(loc)>1:
   xx=rankdata(x);yy=rankdata(y);rx.extend((xx-xx.mean())/(len(loc)-1));ry.extend((yy-yy.mean())/(len(loc)-1))
 rx=np.array(rx);ry=np.array(ry);rho=float(rx@ry/np.sqrt((rx@rx)*(ry@ry)))if rx@rx>0 and ry@ry>0 else None
 return dict(equal_node_length_pair_accuracy=correct/total if total else None,equal_length_quality_pairs=total,within_length_rank_correlation=rho,beneficial_vs_harmful_AUC=A,beneficial_candidates=int((quality>=base+.02).sum()),harmful_candidates=int((quality<=base-.02).sum()),candidate_oracle_extra_pp=100*float(quality.max()-base))


def prepare():
 from scipy.stats import rankdata
 records=[]
 for r in read(MEM/'PLAN_LOCK.json')['records']:
  files=[FG/'predictions'/f"{r['alias']}.pt",ST/'predictions'/f"{r['alias']}.pt",DEV/'priors'/f"{r['alias']}.pt",PRE/'inputs'/f"{r['alias']}.pt",FIN/'Live/predictions'/f"{r['alias']}.pt"]
  records.append(dict(r,cache_pins={str(f.relative_to(R)):sha(f)for f in files}))
 files=['scripts/run_stvg_temporal_foreground_info128_v1.py','scripts/publish_stvg_temporal_foreground_info128_v1.py','protocols/stvg_temporal_foreground_info128_v1.md','results/stvg_error_state/2026-10-20/temporal_foreground_info128/independent_info_audit.py','external/TA-STVG/models/criterion.py','external/TA-STVG/experiments/vidstg.yaml','vg_tta/structured_temporal_v1.py','methods/stvg_referdino_opd_v1/config.json']
 p=dict(records=records,count=128,fields=FIELDS,primary='Action',EOS_background_weight=.3,cyclic_shift=32,benefit_harm_margin=.02,seeds=[41,42,43],CPU_seconds_cap=120,pins={f:sha(R/f)for f in files},created_epoch=time.time(),new_final_intervals=False,new_OPD=False,new_models=0,new_RGB=0,new_experts=0,spatial_frozen=True)
 put(D/'PLAN_LOCK.json',p);put(P/'PROTOCOL_PUBLIC.json',{k:v for k,v in p.items()if k not in ['records','created_epoch']});print('locked diagnostic128',flush=True)


def qualify():
 import torch
 from methods.stvg_referdino_opd_v1.runtime import setup
 from main_method.runtime import resource_guard
 setup();p=read(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());beg=time.perf_counter();files=[];ii,jj=np.triu_indices(65,1)
 for r in p['records']:
  resource_guard();assert time.perf_counter()-beg<120;assert all(sha(R/f)==h for f,h in r['cache_pins'].items());a=r['alias'];f=load(FG/'predictions'/f'{a}.pt');s=load(ST/'predictions'/f'{a}.pt');old=load(DEV/'priors'/f'{a}.pt');base={k:np.empty(65)for k in ['Action','Motion','Appearance']}
  for off in [0,1]:
   base['Action'][off::2]=f['actions']['Native'][off].numpy().reshape(-1);trace=s['native_stage_traces'][off];base['Motion'][off::2]=trace['initial_motion_logits'].numpy().reshape(-1);base['Appearance'][off::2]=trace['initial_appearance_logits'].numpy().reshape(-1)
  fields={}
  for k,x in base.items():fields[k]=x;fields[k+'Flat']=np.full(65,x.mean());fields[k+'Shift']=np.roll(x,32)
  fields['ActionEOS']=base['Action']+np.log(.3);assert set(fields)==set(FIELDS);energies={}
  for k,x in fields.items():
   prefix=np.r_[0.,np.cumsum(x)];energies[k]=prefix[jj+1]-prefix[ii];assert np.isfinite(energies[k]).all()
  assert np.max(abs(energies['ActionEOS']-energies['Action']-np.log(.3)*(jj-ii+1)))<1e-10
  files.append(save(D/'diagnostics'/f'{a}.pt',dict(alias=a,ids=old['ids'],fields=fields,mask_energies=energies,native_interval=old['interval'],GT_read=False,no_new_final_output=True)))
 put(D/'GLOBAL_DIAGNOSTIC_BARRIER.json',dict(count=128,files=files,plan_sha256=sha(D/'PLAN_LOCK.json'),all_before_GT=True));put(D/'EXPORT_COST.json',dict(CPU_seconds=time.perf_counter()-beg,GPU_seconds=0));print('sealed128 diagnostic fields only',flush=True)


def score():
 import torch
 from main_method.runtime import resource_guard
 from scripts.score_stvg_where_when_event128_v1 import avg,paired
 torch.set_num_threads(4);beg=time.perf_counter();p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_DIAGNOSTIC_BARRIER.json');assert bar['count']==128 and bar['plan_sha256']==sha(D/'PLAN_LOCK.json')and all(sha(R/f)==h for f,h in p['pins'].items());assert read(P/'INFORMATION_INDEPENDENT_AUDIT.json')['status']=='pass'
 for f in bar['files']:assert sha(R/f['path'])==f['sha256']
 put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_fields_before_GT=True,barrier_sha256=sha(D/'GLOBAL_DIAGNOSTIC_BARRIER.json')))
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];ii,jj=np.triu_indices(65,1);rows=[];checks=spaces=0
 for r in p['records']:
  resource_guard();assert time.perf_counter()-beg+read(D/'EXPORT_COST.json')['CPU_seconds']<120;a=r['alias'];z=load(D/'diagnostics'/f'{a}.pt');inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');ids=np.array(z['ids']);g,h=spans[r['query_ordinal']];labels=(ids>=g)&(ids<h);classification_stats={k:classification(x,labels)for k,x in z['fields'].items()};quality=[];baselines=[]
  for space in ['E','Live41','Live42','Live43']:
   boxes=np.asarray(inp['E']if space=='E'else sp['outputs'][f'T20_Live_{space[4:]}']);tube=DenseTube(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True);S,E=ids[ii],ids[jj]+1;lo=np.searchsorted(tube.fids,np.maximum(S,g));hi=np.maximum(lo,np.searchsorted(tube.fids,np.minimum(E,h)));prefix=np.r_[0.,np.cumsum(tube.iou)];v=(prefix[hi]-prefix[lo])/np.maximum(np.maximum(E,h)-np.minimum(S,g),1)
   for k in np.linspace(0,2079,9).astype(int):
    off=official(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],[S[k],E[k]],'hc2');assert abs(v[k]-off['v'])<2e-10;checks+=1
   spaces+=1
   if space!='E':quality.append(v);baselines.append(tube.score(z['native_interval'])['v'])
  quality=np.mean(quality,axis=0);baseline=float(np.mean(baselines));ranking={k:rankdiag(s,quality,ii,jj,baseline)for k,s in z['mask_energies'].items()};assert classification_stats['Action']['AUROC']==classification_stats['ActionEOS']['AUROC'];x=ranking['Action']['equal_node_length_pair_accuracy'];y=ranking['ActionEOS']['equal_node_length_pair_accuracy'];assert x==y or abs(x-y)<1e-12
  rows.append(dict(alias=a,classification=classification_stats,ranking=ranking,actual_GT_positive_nodes=int(labels.sum()),native_fixed_spatial_mean_v=baseline));print('info',a,flush=True)
 fields={};contrasts={}
 for k in FIELDS:
  fields[k]=dict(classification={m:dict(mean=avg([r['classification'][k][m]for r in rows]),defined_parents=sum(r['classification'][k][m]is not None for r in rows))for m in classification_stats[k]},ranking={m:dict(mean=avg([r['ranking'][k][m]for r in rows]),defined_parents=sum(r['ranking'][k][m]is not None for r in rows))for m in ranking[k]})
 for a,b in [('Action','ActionFlat'),('Action','ActionShift'),('Motion','MotionFlat'),('Motion','MotionShift'),('Appearance','AppearanceFlat'),('Appearance','AppearanceShift'),('ActionEOS','Action')]:
  contrasts[a+'_minus_'+b]={}
  for group,metrics in [('classification',['AUROC','Brier','logloss','recall']),('ranking',['equal_node_length_pair_accuracy','within_length_rank_correlation','beneficial_vs_harmful_AUC'])]:
   contrasts[a+'_minus_'+b][group]={m:paired([r[group][a][m]-r[group][b][m]for r in rows if r[group][a][m]is not None and r[group][b][m]is not None])for m in metrics}
 cost=dict(CPU_seconds_export=read(D/'EXPORT_COST.json')['CPU_seconds'],CPU_seconds_score=time.perf_counter()-beg,GPU_seconds=0,new_models=0,new_RGB=0,new_experts=0,new_spatial_or_student_updates=0,upstream_cost_not_free=True);result=dict(status='completed_exposed_diagnostic_not_method_accuracy',parents=128,fields=fields,paired=contrasts,rows=rows,cost=cost,new_final_intervals=False,new_OPD=False);put(P/'RESULTS_PUBLIC.json',result);put(D/'COST.json',cost)
 audit=dict(status='pass',official_candidate_quality_checks=checks,frozen_spatial_cases=spaces,all128_fields_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']),EOS_AUROC_and_equal_length_preference_identity_all_parents=True,numeric_independent=read(P/'INFORMATION_INDEPENDENT_AUDIT.json'),GT_provenance=prov);assert audit['all128_fields_before_GT'];put(P/'ROOT_AUDIT.json',audit);put(P/'INDEPENDENT_AUDIT.json',audit)
 lines=['# 原生前景信息资格：HC128纯缓存诊断','','没有新最终区间、MBR赢家或OPD成绩。Source既有Action/两前置motion/appearance BCE头；原65节点、全部128父及3冻结空间seed，GT只在所有字段/能量封存后读取。ActionEOS=z+log(.3)仅理想class-weight反校准假设，官方配置不代表真实训练日志或目标域校准。旧structured_temporal已有Bernoulli结构，不声称新方法。','','| field | AUROC (定义父数) | Brier | 同长度pair正确率 | 有益/有害候选AUC (定义父数) |','|---|---|---:|---:|---|']
 for k in FIELDS:
  c=fields[k]['classification'];rr=fields[k]['ranking'];lines.append(f'| {k} | {c["AUROC"]} | {c["Brier"]["mean"]} | {rr["equal_node_length_pair_accuracy"]["mean"]} | {rr["beneficial_vs_harmful_AUC"]} |')
 lines+=['','所有区间在原65格点2080支持；同长度指相同节点数j−i（物理时长可差1帧），纠错候选标记相对同父原Native完整v至少±2pp，GT仅诊断不作在线门控。所有可区分GT质量的同长度pair都保留，教师tie算.5；条件未定义记null不填0。总体排序可混有时长偏置，必须与同长度和Shift控制分开。三个seed只父内平均，独立n仍128。','','EOS修正的AUROC及同长度排序逐父严格相同；Brier/固定.5工作点改变不是新定位语义。若只有校准/长度项改善，不进入新读出。即使有时间特异信息，也不自动证明2–3pp或OPD优势；需要对后续能力机会单独裁决。','','```json',json.dumps(cost,indent=2),json.dumps(audit,indent=2),'```']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],exposure='all RC128 previously exposed development; TC109 not used',raw_caches_labels_coordinates_media_captions_parentIDs_excluded=True))
 for f in P.glob('*.json'):f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 print('diagnostic completed no method outputs',flush=True)

if __name__=='__main__':
 D.mkdir(parents=True,exist_ok=True);lock=(D/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 try:globals()[sys.argv[1]]()
 finally:lock.close()
