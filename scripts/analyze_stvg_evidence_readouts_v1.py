"""Fixed-trajectory temporal64 metrics and original online32 rollout diagnosis."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,collections,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_stvg_evidence_strata_v1 import D,P,OLD,PAPER,read,put,load,sha,mean,summarize
import numpy as np
import torch
torch.set_num_threads(2)
from scripts.score_stvg_opd_p1_v1 import truths
from vg_tta.tastvg_oracle_event5_v1 import DenseTube,official,box_iou

def temporal():
 priv=read(D/'FEATURES_PRIVATE.json');fby={ds:{f['query_ordinal']:f for f in fs}for ds,fs in priv['rows'].items()};base=ROOT/'artifacts/stvg_interval_decision_v1';plan=read(base/'PLAN_LOCK.json');bar=read(base/'GLOBAL_READOUT_BARRIER.json');rs={r['path']:r for r in bar['files']};zs=[];checks=0
 for r in plan['files']:
  assert sha(ROOT/r['path'])==r['sha256'];p=base/'readouts'/(r['alias']+'.pt');assert sha(p)==rs[str(p.relative_to(ROOT))]['sha256'];z=load(p);assert z['input_source_sha256']==r['sha256'];old=load(ROOT/r['path']);zs.append((r,z,old))
 targets={ds:truths(ds,dict(parents=[z['query_ordinal']for r,z,old in zs if z['dataset']==ds]))for ds in fby};rows=[]
 for r,z,old in zs:
  ds=z['dataset'];q=z['query_ordinal'];f=fby[ds][q];row=read(PAPER/ds/'PLAN.json')['rows'][q];gt=targets[ds][0][q];span=targets[ds][1][q]
  assert z['fused_intervals']['Native']==old['predictions']['Native']['physical_interval']
  for geometry,boxes in [('Frozen',z['boxes']),('Before',old['original_spatial_before']['boxes']),('After',old['predictions']['SpatialOPD']['boxes'])]:
   tube=DenseTube(boxes.numpy(),row,gt,span,clip=ds=='hc2')
   for support in ['offset0','offset1','fused']:
    ivs=z['fused_intervals']if support=='fused'else{a:v[int(support[-1])]for a,v in z['offset_intervals'].items()};ni=ivs['Native'];native=(tube.fids>=ni[0])&(tube.fids<ni[1]);vals={}
    for arm in ['Native','projection','native_MBR']:
     iv=ivs[arm];m=official(boxes.numpy(),row,gt,span,iv,ds);assert all(abs(m[k]-tube.score(iv)[k])<2e-10 for k in ['v','t','s']);checks+=3
     now=(tube.fids>=iv[0])&(tube.fids<iv[1]);add=now&~native;lost=native&~now
     # New temporal support outside GT has unknown boxes, never filled with zeros.
     added_frames=max(0,iv[1]-iv[0]-max(0,min(iv[1],ni[1])-max(iv[0],ni[0])))
     vals[arm]=dict(**m,R03=float(m['v']>.3),R05=float(m['v']>.5),length_input_fraction=(iv[1]-iv[0])/(row['frame_ids'][-1]+1-row['frame_ids'][0]),added_annotated_frames=int(add.sum()),added_known_s=mean(tube.iou[add]),lost_annotated_frames=int(lost.sum()),lost_known_s=mean(tube.iou[lost]),added_frames_without_GT=int(added_frames-add.sum()),delta_v_vs_Native=m['v']-tube.score(ni)['v'],delta_t_vs_Native=m['t']-tube.score(ni)['t'])
    rows.append(dict(dataset=ds,query_alias=f['query_alias'],parent_alias=f['parent_alias'],geometry=geometry,support=support,event_length_bin=f['bins']['event_fraction'],event_start_bin=f['bins']['event_start'],values=vals))
  print('TEMPORAL',r['alias'],flush=True)
 def stats(rr):
  out={}
  for a in ['Native','projection','native_MBR']:
   parents=collections.defaultdict(list)
   for r in rr:parents[r['parent_alias']].append(r['values'][a])
   keys=list(rr[0]['values'][a]);out[a]=dict(queries=len(rr),parents=len(parents),query_macro={k:mean([r['values'][a][k]for r in rr])for k in keys},parent_macro={k:mean([mean([v[k]for v in g])for g in parents.values()])for k in keys})
  return out
 groups={ds:{geo:{sup:dict(overall=stats([r for r in rows if(r['dataset'],r['geometry'],r['support'])==(ds,geo,sup)]),strata={key:{v:stats([r for r in rows if(r['dataset'],r['geometry'],r['support'])==(ds,geo,sup)and r[key]==v])for v in sorted({r[key]for r in rows if r['dataset']==ds})}for key in ['event_length_bin','event_start_bin']})for sup in ['offset0','offset1','fused']}for geo in ['Frozen','Before','After']}for ds in fby}
 put(P/'TEMPORAL64.json',dict(status='complete',count=64,new_models=0,groups=groups,rows=rows,exact_fixed_spatial_geometries=['Frozen','Before','After'],independent_official_metric_checks=checks,historical_online_state_conditional_not_new_stream=True))
 put(D/'TEMPORAL_ROOT_READBACK.json',dict(status='pass',count=64,checks=checks,no_new_model_calls=True))

def spatial():
 priv=read(D/'FEATURES_PRIVATE.json');feat={ds:{f['query_ordinal']:f for f in fs}for ds,fs in priv['rows'].items()};base=ROOT/'artifacts/stvg_dino_rollback32_v1';records=read(base/'PLAN_LOCK.json')['records'];targets={ds:truths(ds,dict(parents=[r['query_ordinal']for r in records if r['dataset']==ds]))for ds in feat};plans={ds:read(PAPER/ds/'PLAN.json')for ds in feat};rows=[];checks=0
 def xy(b):return np.concatenate((b[...,:2]-b[...,2:]/2,b[...,:2]+b[...,2:]/2),-1)
 for rec in records:
  ds=rec['dataset'];q=rec['query_ordinal'];p=base/'predictions'/(rec['alias']+'.pt');receipt=read(p.with_suffix('.receipt.json'));assert sha(p)==receipt['sha256'];z=load(p);fit=z['fit'];f=feat[ds][q];gt=targets[ds][0][q];span=targets[ds][1][q];row=plans[ds]['rows'][q];pos=fit['positions'];ids=np.array(row['frame_ids']);rounds=[]
  for k,rd in enumerate(fit['rounds']):
   if 'rollout'not in rd:continue
   rr=rd['rollout'];mu=rr['mean'].double().numpy();samples=rr['samples'].double().numpy();b=1/(1+np.exp(-mu));a=1/(1+np.exp(-samples));delta=(a[...,:2]-b[:,None,:2])/b[:,None,2:];iou=box_iou(xy(a),xy(b[:,None,:]));rewards=rr['rewards'].double().numpy();weights=rr['weights'].double().numpy()
   # Independently recover the actual Gaussian coordinates and antithetic seed structure.
   eps=(samples-mu[:,None,:])/fit['config']['sigma'];assert np.max(abs(eps[:,:16]+eps[:,16:]))<1e-3;assert np.max(abs(weights.sum(-1)-1))<1e-5;checks+=2
   bef=DenseTube(fit['path'][k]['boxes'].numpy(),row,gt,span,clip=ds=='hc2');aft=DenseTube(fit['path'][k+1]['boxes'].numpy(),row,gt,span,clip=ds=='hc2');native=np.isin(bef.fids,ids);observed=np.isin(bef.fids,ids[pos]);unobs=native&~observed;reward_delta=float((rd['central_reward_after']-rd['central_reward_before']).mean())if'central_reward_after'in rd else None
   rounds.append(dict(relative_center_RMS=float(np.sqrt(np.mean(delta**2))),IoU_perturbation=float(np.mean(1-iou)),log_size_RMS=float(np.sqrt(np.mean(np.log(a[...,2:]/b[:,None,2:])**2))),reward_variance=float(np.var(rewards,axis=-1).mean()),flat_reward_anchors=int(np.sum(np.all(rewards==rewards[:,:1],axis=1))),anchors=len(pos),ESS=float(np.mean(1/np.sum(weights**2,-1))),reward_delta=reward_delta,unobserved_actual_decoder_iou_delta=mean((aft.iou-bef.iou)[unobs]),observed_actual_decoder_iou_delta=mean((aft.iou-bef.iou)[observed])))
  before=DenseTube(fit['before'].numpy(),row,gt,span,clip=ds=='hc2');after=DenseTube(fit['final'].numpy(),row,gt,span,clip=ds=='hc2');iv=z['arms']['OPD_native_readout']['interval'];assert np.array_equal(xy(fit['final'].double().numpy()),np.asarray(z['arms']['OPD_native_readout']['boxes']))
  rows.append(dict(dataset=ds,query_alias=f['query_alias'],parent_alias=f['parent_alias'],bins=f['bins'],rounds=rounds,relative_center_RMS=mean([r['relative_center_RMS']for r in rounds]),IoU_perturbation=mean([r['IoU_perturbation']for r in rounds]),current_v=after.score(iv)['v']-before.score(iv)['v'],reward_up_unobserved_actual_down_rounds=sum(r['reward_delta']is not None and r['reward_delta']>0 and r['unobserved_actual_decoder_iou_delta']is not None and r['unobserved_actual_decoder_iou_delta']<0 for r in rounds),config=fit['config']))
  print('SPATIAL',rec['alias'],len(rounds),flush=True)
 vals=[r['relative_center_RMS']for r in rows if r['relative_center_RMS']is not None];edges=np.quantile(vals,[.25,.75]);lo=[v for v in vals if v<=edges[0]];hi=[v for v in vals if v>=edges[1]];ratio=mean(hi)/mean(lo)
 fullrows=read(D/'FULL_PRIVATE_ROWS.json');gate={};associations={}
 for ds,rr in fullrows.items():
  primary=[r for r in rr if r['primary']];low=[r for r in primary if r['bins']['exploration_proxy']=='Q1'];high=[r for r in primary if r['bins']['exploration_proxy']=='Q4'];sl=summarize(low,True)if low else None;sh=summarize(high,True)if high else None;orders={o:mean([r['delta_current_v']for r in high if r['order']==o])-mean([r['delta_current_v']for r in low if r['order']==o])for o in ['order1','order2','order3']}if low and high else{}
  conditions=dict(actual_RMS_contrast_at_least2=ratio>=2,full_groups_at_least20parents=bool(sl and sh and min(sl['parents'],sh['parents'])>=20),high_current_mean_negative=bool(sh and sh['query_macro']['delta_current_v']<0),high_unobserved_mean_negative=bool(sh and sh['query_macro']['unobserved_iou_delta']is not None and sh['query_macro']['unobserved_iou_delta']<0),high_minus_low_negative_all_orders=bool(orders and all(v<0 for v in orders.values())))
  gate[ds]=dict(conditions=conditions,passed=all(conditions.values()),order_high_minus_low_current_v=orders,low=sl,high=sh)
  # Within-domain Spearman avoids merely rediscovering different domain sigmas.
  from scipy.stats import spearmanr
  qgroup=collections.defaultdict(list)
  for r in primary:qgroup[r['query_ordinal']].append(r)
  ff=feat[ds];x=[ff[q]['exploration_proxy']for q in qgroup];y=[mean([r['delta_current_v']for r in g])for g in qgroup.values()];assoc=spearmanr(x,y)
  associations[ds]=dict(query_Spearman_scale_vs_current=float(assoc.statistic),query_count=len(x),p_value_not_independent_due_parent_clustering=float(assoc.pvalue))
 passed=any(x['passed']for x in gate.values());put(P/'SPATIAL_SCALE_GATE.json',dict(status='complete',actual_online_sources=32,actual_RMS_upper_lower_quartile_ratio=ratio,actual_RMS_quartiles=edges.tolist(),groups=gate,within_domain_associations=associations,passed=passed,decision='one_matched_online32_relative_coordinate_variant_authorized'if passed else'stop_no_new_spatial_model_or_parameter_fits',full_Source_scale_is_proxy_not_actual_inherited_state=True,full_raw_fits_retired_not_new_failure=True,rows=rows,checks=checks,confounding_not_eliminated=True))
 put(D/'SPATIAL_ROOT_READBACK.json',dict(status='pass',count=32,checks=checks,conditional_variant_triggered=passed))
 print('GATE',passed,{d:g['conditions']for d,g in gate.items()},flush=True)
if __name__=='__main__':globals()[sys.argv[1]]()
