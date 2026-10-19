"""Post-seal proper-joint-decision probe with spatial/time tradeoffs explicit."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import sys,time,json,numpy as np
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
from scripts.run_stvg_joint_event_critic_v1 import read,put,load,sha
from scripts.run_stvg_temporal_joint_risk128_v1 import D,P,PRE,FIN,DEV,PRIORS,ARMS
from scripts.score_stvg_where_when_event128_v1 import avg,paired


def main():
 import torch
 from scipy.stats import spearmanr
 torch.set_num_threads(4);p=read(D/'PLAN_LOCK.json');bar=read(D/'GLOBAL_PREDICTION_BARRIER.json');assert bar['count']==128 and bar['all_before_GT']and bar['plan_sha256']==sha(D/'PLAN_LOCK.json');assert all(sha(R/f)==h for f,h in p['pins'].items());assert read(P/'RISK_INDEPENDENT_AUDIT.json')['status']=='pass'
 for f in bar['files']:assert sha(R/f['path'])==f['sha256']
 put(D/'GT_EXPOSURE.json',dict(epoch=time.time(),all128_before_GT=True,barrier_sha256=sha(D/'GLOBAL_PREDICTION_BARRIER.json')))
 from scripts.score_stvg_opd_p1_v1 import truths
 from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official
 gt,spans,prov=truths('hc2',dict(parents=[r['query_ordinal']for r in p['records']]));raw=read(R/'artifacts/decota_paper_experiments_v1/hc2/PLAN.json')['rows'];old={r['alias']:r for r in read(R/'results/stvg_error_state/2026-10-19/temporal_increment128/RESULTS_PUBLIC.json')['rows']};rows=[];checks=spaces=qualitychecks=0
 for r in p['records']:
  a=r['alias'];z=load(D/'predictions'/f'{a}.pt');inp=load(PRE/'inputs'/f'{a}.pt');sp=load(FIN/'Live/predictions'/f'{a}.pt');ids=np.array(z['frame_ids']);ii,jj=np.triu_indices(len(ids),1);S,E=ids[ii],ids[jj]+1;g,h=spans[r['query_ordinal']];overlap=np.maximum(0,np.minimum(E,h)-np.maximum(S,g));t=overlap/(E-S+h-g-overlap);values={};expected={};diag={}
  for space,boxes in z['trajectories'].items():
   assert np.array_equal(boxes,inp['E']if space=='E'else sp['outputs'][f'T20_Live_{space[4:]}']);spaces+=1;tube=DenseTube(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],clip=True);metric_cache={}
   for arm,o in z['outputs'][space].items():
    iv=o['interval'];key=tuple(iv)
    if key not in metric_cache:
     v=tube.score(iv);off=official(boxes,raw[r['query_ordinal']],gt[r['query_ordinal']],spans[r['query_ordinal']],iv,'hc2');assert max(abs(v[k]-off[k])for k in ['v','t','s'])<2e-10;checks+=1;metric_cache[key]=v
    v=metric_cache[key];inter=max(0,min(iv[1],h)-max(iv[0],g));assert abs(v['s']-tube.s)<1e-15
    if arm=='N_Time':assert max(abs(v[k]-old[a]['values'][space+'_L0'][k])for k in ['v','t','s'])<2e-10
    values[space+'_'+arm]=dict(**v,interval=iv,boundary_MAE=(abs(iv[0]-g)+abs(iv[1]-h))/2,precision=inter/(iv[1]-iv[0]),recall=inter/(h-g),duration=iv[1]-iv[0])
   lo=np.searchsorted(tube.fids,np.maximum(S,g));hi=np.maximum(lo,np.searchsorted(tube.fids,np.minimum(E,h)));prefix=np.r_[0.,np.cumsum(tube.iou)];v=(prefix[hi]-prefix[lo])/np.maximum(np.maximum(E,h)-np.minimum(S,g),1)
   for k in np.unique(np.linspace(0,len(ii)-1,19).astype(int)):
    actual=tube.score([S[k],E[k]]);assert abs(v[k]-actual['v'])<1e-12 and abs(t[k]-actual['t'])<1e-12;qualitychecks+=1
   curves=z['quality_curves'][space];diag[space]=dict(proxy_mean=curves['meta'].get('mean',1.),known_frames=curves['meta']['known_frames'],flat=bool(np.ptp(curves['curve'])<=1e-12),GT_candidate_capacity_pp=100*(float(v.max())-values[space+'_N_Time']['v']),candidate_risk_spearman={})
   for name in PRIORS:
    risk=z['risk'][space][name];q=z['priors'][name];expected[space+'_'+name]={}
    for key,target in [('Prior',q),('Target',risk['target']),('FlatTarget',risk['flat_target'])]:expected[space+'_'+name][key]=dict(v=float(target[ii,jj]@v),t=float(target[ii,jj]@t))
    u=risk['real_utility'][ii,jj];diag[space]['candidate_risk_spearman'][name]=None if np.std(u)<1e-12 or np.std(v)<1e-12 else float(spearmanr(u,v).statistic)
  fields=['v','t','s','boundary_MAE','precision','recall','duration']
  for arm in ARMS:values['LiveMean_'+arm]={k:avg([values[f'Live{s}_{arm}'][k]for s in p['seeds']])for k in fields}
  for name in PRIORS:expected['LiveMean_'+name]={kind:{k:avg([expected[f'Live{s}_{name}'][kind][k]for s in p['seeds']])for k in ['v','t']}for kind in ['Prior','Target','FlatTarget']}
  rows.append(dict(alias=a,values=values,expected_candidate_quality=expected,diagnostics=diag));print('scored',a,flush=True)
 fields=['v','t','s','boundary_MAE','precision','recall','duration'];means={arm:{k:avg([r['values']['LiveMean_'+arm][k]for r in rows])for k in fields}for arm in ARMS};pairs=[(arm,'N_Time')for arm in ARMS if arm!='N_Time']+[(n+'_Risk',n+'_Time')for n in PRIORS]+[(n+'_Risk',n+'_Shift')for n in PRIORS]+[(n+'_Target',n+'_FlatTarget')for n in PRIORS];ds={}
 for a,b in pairs:
  key=a+'_minus_'+b;d={k:paired([(100 if k in ['v','t','s','precision','recall']else 1)*(r['values']['LiveMean_'+a][k]-r['values']['LiveMean_'+b][k])for r in rows])for k in fields};delta=np.array([100*(r['values']['LiveMean_'+a]['v']-r['values']['LiveMean_'+b]['v'])for r in rows]);d['tails']=dict(positive=int((delta>1e-10).sum()),negative=int((delta< -1e-10).sum()),flat=int((abs(delta)<=1e-10).sum()),below_minus3=int((delta< -3).sum()),below_minus5=int((delta< -5).sum()),positive_contribution_pp=float(np.maximum(delta,0).mean()),negative_contribution_pp=float(np.minimum(delta,0).mean()),worst=sorted([dict(alias=r['alias'],delta_pp=float(v))for r,v in zip(rows,delta)],key=lambda v:v['delta_pp'])[:8]);ds[key]=d
 expected={n:{kind:{k:paired([100*(r['expected_candidate_quality']['LiveMean_'+n]['Target'][k]-r['expected_candidate_quality']['LiveMean_'+n][kind][k])for r in rows])for k in ['v','t']}for kind in ['Prior','FlatTarget']}for n in PRIORS};interaction={n:paired([100*((r['values']['LiveMean_'+n+'_Risk']['v']-r['values']['LiveMean_'+n+'_Time']['v'])-(r['values']['E_'+n+'_Risk']['v']-r['values']['E_'+n+'_Time']['v']))for r in rows])for n in PRIORS};cost=read(D/'COST.json');put(P/'RESULTS_PUBLIC.json',dict(status='completed_exposed_joint_decision_probe',parents=128,seeds=p['seeds'],means=means,paired=ds,soft_teacher_expected_quality=expected,spatial_state_interaction_pp=interaction,rows=rows,cost=cost,spatial_frozen=True,new_OPD=False))
 audit=dict(status='pass',official_DenseTube_unique_interval_checks=checks,spatial_arrays_exact=spaces,all_outputs_have_same_fixed_spatial_sIoU=True,native_metrics_exact=True,vectorized_GT_quality_checks=qualitychecks,all128_before_GT=all((R/f['path']).stat().st_mtime<read(D/'GT_EXPOSURE.json')['epoch']for f in bar['files']),risk_independent=read(P/'RISK_INDEPENDENT_AUDIT.json'),GT_provenance=prov);assert audit['all128_before_GT'];put(P/'ROOT_AUDIT.json',audit);put(P/'INDEPENDENT_AUDIT.json',audit)
 selected=min(p['eligible'],key=lambda x:(-means[x]['v'],x));gain=ds[selected+'_minus_N_Time']['v'];decision=dict(primary=p['primary'],selected=selected,selected_delta=gain,selected_temporal_delta=ds[selected+'_minus_N_Time']['t'],decision='freeze_for_independent_parent_confirmation'if gain['mean']>=2 and gain['CI95'][0]>0 else'no_supported_2pp_joint_decision',point_readout_not_OPD=True,spatial_proxy_not_event_probability=True,temporal_tradeoff_must_be_reported=True,independent_confirmation=False);put(P/'SELECTION_PUBLIC.json',decision)
 lines=['# 固定空间下的时空联合决策效用：HC128开发','','空间预测/专家/观测位置/模型权重全部不动。现有Temporal MBR只评价区间交并比；此探针将同一个事件区间后验与学生-ReferDINO空间IoU代理组合为expected pseudo-tube utility。空间IoU不是事件概率，也不是校准的GT质量。八帧teacher只在其支持内插值，不能称逐帧独立专家观察。无新模型/训练/专家/OPD。','','| 输出 | full vIoU% | tIoU% | event recall% | v增量pp [CI95] |','|---|---:|---:|---:|---|']
 for arm in ARMS:
  v=means[arm];c=ds.get(arm+'_minus_N_Time');d='0'if c is None else f'{c["v"]["mean"]:+.6f} [{c["v"]["CI95"][0]:+.6f},{c["v"]["CI95"][1]:+.6f}]';lines.append(f'| {arm} | {100*v["v"]:.6f} | {100*v["t"]:.6f} | {100*v["recall"]:.6f} | {d} |')
 lines+=['','N原Native，H旧HiResMix，A旧HiResAuxMix；Time原timeMBR、Risk最大expected pseudo-tube utility、Shift时间错位空间质量、Flat相同均值常曲线必须回到Time。Target为prior×joint utility归一化后的静态诊断，FlatTarget控制仅self-risk重加权；二者不eligible挑方法。主A_Risk，三个Risk全128共同选择，不按父/seed挑赢家。与各自Time的差才是新增联合读出。','','GT只封存后解释，全部父/signed/unknown/尾部保留。若fullv上升但t/recall下降，结论是联合指标取舍，不能称事件边界理解更好；固定空间真实节点指标全部不变。teacher软目标期望与风险排序诊断也保留，不用单点阴性宣布所有OPD失败。','','```json',json.dumps(decision,indent=2),'```','','## 成本与审计','','```json',json.dumps(cost,indent=2),json.dumps(audit,indent=2),'```','','所有成本为缓存后新增CPU推理，原Source/ReferDINO/空间适应成本不是免费。模型/空间/CURRENT/MAIN/PAPER未变。>=2pp及正区间只支持冻结父独立确认；如果确认成立再设计实际Direct/Fixed/Live，静态联合读出不算OPD贡献。']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');put(P/'PROVENANCE_PUBLIC.json',dict(plan_sha256=sha(D/'PLAN_LOCK.json'),pins=p['pins'],spatial_config_sha256=sha(R/'methods/stvg_referdino_opd_v1/config.json'),exposure='all128 exposed development, no confirmation GT used',excluded=['media','GT coordinates','captions','parent IDs','weights','raw caches']))
 for f in P.glob('*.json'):f.write_text(json.dumps(read(f),ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
 print(json.dumps(decision,indent=2),flush=True)

if __name__=='__main__':main()
