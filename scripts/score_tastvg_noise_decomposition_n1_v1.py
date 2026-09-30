"""Independent audit and matched outcomes for an explicitly GT-assisted oracle."""
import sys,time,hashlib,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,load,sha
from scripts.run_tastvg_noise_decomposition_n1_v1 import OUT,J01,S0,TEMP,NATIVE,verify,ARMS,MATCHED
from scripts.score_tastvg_paper48_raw_v1 import loss_dist,aggregate,describe
from scripts.score_tastvg_schedule_j01_v1 import independent_scores
from scripts.analyze_spatial10_components_v1 import checked_score
from methods.decota_final_simplified_v1.tensors import state_hash

def rectangle_iou(a,b):
 a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
 lo=np.maximum(a[:,:2]-a[:,2:]/2,b[:,:2]-b[:,2:]/2);hi=np.minimum(a[:,:2]+a[:,2:]/2,b[:,:2]+b[:,2:]/2)
 inter=np.maximum(hi-lo,0).prod(1);return inter/(a[:,2:].prod(1)+b[:,2:].prod(1)-inter)

def independent_pool(arm,e,g):
 result=[]
 for k,(a,b) in enumerate(zip(e,g),1):
  use=(a!=0 if arm=='all' else a*b==1 if arm in ['useful','useful_matched'] else a*b==-1 if arm in ['noisy','noisy_matched'] else a==b==1 if arm=='useful_positive' else a==b==-1)
  if use:result.append(k)
 return result

def run():
 import ijson
 torch.set_num_threads(4);p=verify();bar=read(OUT/'PREDICTION_BARRIER.json');assert len(bar['files'])==3360 and bar['GT_for_signal_filter'] and not bar['deployable']
 for f,h in {**bar['files'],**bar['final_states']}.items():assert sha(OUT/f)==h
 keys={r['key'] for r in p['rows']}
 with (ROOT/p['GT_container']).open('rb') as f:gt={k:v for k,v in ijson.kvitems(f,'',use_float=True) if k in keys}
 assert set(gt)==keys
 write(OUT/'SCORING_EXPOSURE.json',dict(time=time.time(),queries=16,sources=16,old_exposed_only=True,GT_used_during_filtering=True,post_seal_evaluation=True,deployable=False,prediction_barrier_sha256=sha(OUT/'PREDICTION_BARRIER.json')))
 base=read(J01/'ROWS.json');ref={(r['order'],r['condition'],r['parent']):r for r in base};initial=load(NATIVE/'PARAMETER_SUPPORT.pt')['center']
 rows=[];diagnostics=[];count=coords=losschecks=teacherchecks=rewardchecks=qualitychecks=0;maxparam=maxloss=maxdistance=0.;matched={}
 for arm in ARMS:
  for order in p['orders']:
   for cond in p['conditions']:
    previous=initial
    for arrival,parent in enumerate(p['orders'][order]):
     r=next(v for v in p['rows'] if v['ordinal']==parent);name=f'{parent:03}.pt';x=load(OUT/'online'/arm/order/cond/name);z=ref[(order,cond,parent)]
     scheduled=arrival in p['expert_indices'];assert x['expert_scheduled']==scheduled and x['arrival']==arrival and x['arm']==arm
     assert state_hash(x['pre_state'])==x['pre_state_sha256'] and state_hash(x['post_state'])==x['post_state_sha256'];assert all(torch.equal(v,previous[n]) for n,v in x['pre_state'].items());previous=x['post_state'];count+=1
     assert torch.equal(x['output_prediction']['boxes'],x['prediction']['boxes'])
     diag=dict(arm=arm,order=order,condition=cond,parent=parent,arrival=arrival,scheduled=scheduled,updated=x['updated'],step_norm=x['parameter_displacement'],inherited_norm=x['displacement_from_source'])
     if scheduled:
      assert x['expert_reads']==['temporal','spatial'] and x['GT_read'];teacher=load(TEMP/'c2'/cond/name)
      for field in ['temporal','fast_control']:
       d=x[field];scores=independent_scores(d['candidates'],teacher);np.testing.assert_allclose(scores,d['scores'],atol=1e-14,rtol=0);assert int(np.argmax(scores))==d['selected'];teacherchecks+=len(scores)
      expert=load(S0/'expert'/cond/name);valid=np.asarray(expert['valid'],bool);candidate=np.stack([c['prediction']['boxes'].numpy() for c in x['candidates']]);assert len(candidate)==9
      if valid.any():
       re=np.array([rectangle_iou(c[valid],expert['boxes'][valid]).mean() for c in candidate]);np.testing.assert_allclose(re,x['rewards'],atol=1e-12,rtol=0);rewardchecks+=9
       e=np.array([int(v>1e-12)-int(v< -1e-12) for v in re[1:]-re[0]])
      else:assert x['rewards'] is None;e=np.zeros(8,dtype=int);re=None
      truth=gt[r['key']];v=np.asarray(truth['valid'],bool);g=np.array([rectangle_iou(c[v],np.asarray(truth['boxes'])[v]).mean() for c in candidate]);np.testing.assert_allclose(g,x['gt_sIoU'],atol=1e-12,rtol=0);qualitychecks+=9
      yg=np.array([int(z>1e-12)-int(z< -1e-12) for z in g[1:]-g[0]])
      np.testing.assert_array_equal(e,x['teacher_labels']);np.testing.assert_array_equal(yg,x['gt_labels']);pool=independent_pool(arm,e,yg);assert pool==x['eligible_indices']
      if arm in MATCHED:
       key=f'{order}|{parent}';assert key==x['hash_key'];n=x['match_count'];chosen=sorted(sorted(pool,key=lambda k:hashlib.sha256(f'N1-count-v1|{key}|{k}'.encode()).hexdigest())[:n]);assert n<=len(pool)
       matched.setdefault((order,cond,parent),{})[arm]=dict(available=len(pool),count=n,updated=x['updated'])
      else:chosen=pool
      assert chosen==x['selected_indices'] and len(chosen)==x['selected_count'];assert x['support_center_sha256']==x['pre_state_sha256']
      classes=collections.Counter({k:0 for k in ['correct_positive','correct_negative','noisy_positive','noisy_negative','tie']})
      for a,b in zip(e,yg):classes[{(1,1):'correct_positive',(-1,-1):'correct_negative',(1,-1):'noisy_positive',(-1,1):'noisy_negative'}.get((int(a),int(b)),'tie')]+=1
      diag.update(rewards=None if re is None else re.tolist(),gt_sIoU=g.tolist(),teacher_labels=e.tolist(),gt_labels=yg.tolist(),classes=dict(classes),eligible_indices=pool,selected_indices=chosen,selected_count=len(chosen),valid_expert_frames=int(valid.sum()),match_count=x['match_count'],hash_key=x['hash_key'])
     else:assert x['expert_reads']==[] and not x['GT_read'] and 'update' not in x
     if 'update' in x:
      u=x['update'];assert scheduled and chosen and u['target_detached'] and u['lr']==.005
      d=loss_dist(x['prediction']['boxes'].numpy()[None],candidate,u['coefficients']);err=float(np.max(np.abs(d-np.array(u['distances']))));assert err<2e-6;maxdistance=max(maxdistance,err)
      d=np.asarray(u['distances'],dtype=float);ids=np.array(chosen);terms=np.logaddexp(0,e[ids-1]*(d[ids]-d[0]));np.testing.assert_allclose(terms,u['terms'],atol=2e-6,rtol=0);err=abs(float(terms.mean())-u['loss_before']);assert err<2e-6;maxloss=max(maxloss,err)
      after_d=loss_dist(x['post_prediction']['boxes'].numpy()[None],candidate,u['coefficients']);after=float(np.logaddexp(0,e[ids-1]*(after_d[ids]-after_d[0])).mean());assert abs(after-u['loss_after'])<2e-6;losschecks+=1
      gn=float(np.sqrt(sum(float(g.double().square().sum()) for g in u['gradients'].values())));assert abs(gn-u['global_gradient_norm'])<1e-12;assert u['update_scale']==(.005 if gn>0 else 0.) and x['updated']==(gn>0)
      for n,b in x['pre_state'].items():
       grad=u['gradients'][n];expected=(b.double().numpy()-u['update_scale']*grad.double().numpy()).astype(np.float32);err=float(np.max(np.abs(expected-x['post_state'][n].numpy())));assert err<=1.5e-7;maxparam=max(maxparam,err);coords+=grad.numel()
      diag.update(loss_before=u['loss_before'],loss_after=u['loss_after'],gradient_norm=gn,distances=d.tolist(),terms=terms.tolist())
     else:assert not x['updated'] and all(torch.equal(v,x['pre_state'][n]) for n,v in x['post_state'].items())
     pred=x['output_prediction'];m,_=checked_score(pred['boxes'],gt[r['key']],r['frame_ids'],pred['indices']);rec=dict(arm=arm,order=order,condition=cond,parent=parent,arrival=arrival,expert_scheduled=scheduled)
     for code,key in [('s','sIoU'),('t','tIoU'),('v','vIoU_corrected')]:rec.update({code:m[key],'delta_'+code:m[key]-z['frozen_'+code],'minus_final_'+code:m[key]-z['final_'+code]})
     rows.append(rec);diagnostics.append(diag)
    assert all(torch.equal(v,previous[n]) for n,v in load(OUT/'final_states'/arm/order/f'{cond}.pt').items())
 for key,pair in matched.items():
  a,b=[pair[k] for k in MATCHED];assert a['count']==b['count']==min(a['available'],b['available']),(key,pair)
 assert len(matched)==120
 for x in base:
  for arm,prefix in [('Frozen','frozen'),('Fast-only','fast'),('Slow-only','slow'),('RKL-Final','final')]:
   rec={k:x[k] for k in ['order','condition','parent','arrival','expert_scheduled']};rec['arm']=arm
   for code in ['s','t','v']:rec.update({code:x[prefix+'_'+code],'delta_'+code:x[prefix+'_'+code]-x['frozen_'+code],'minus_final_'+code:x[prefix+'_'+code]-x['final_'+code]})
   rows.append(rec)
 summary,across,effects=aggregate(rows)
 for name,value in [('ROWS.json',rows),('DIAGNOSTICS.json',diagnostics),('SUMMARY.json',summary),('ACROSS_ORDERS.json',across),('SOURCE_EFFECTS.json',effects)]:write(OUT/name,value)
 signals={}
 for arm in ARMS:
  signals[arm]={}
  for group in ['corruption','clean']:
   dd=[d for d in diagnostics if d['arm']==arm and d['scheduled'] and ((d['condition']!='clean') if group=='corruption' else d['condition']=='clean')]
   cc={k:sum(d['classes'][k] for d in dd if d['valid_expert_frames']) for k in ['correct_positive','correct_negative','noisy_positive','noisy_negative','tie']};dd_updates=[d for d in dd if d['updated']]
   signals[arm][group]=dict(scheduled=len(dd),valid_specialist_cells=sum(d['valid_expert_frames']>0 for d in dd),**cc,selected_signals=sum(d['selected_count'] for d in dd),updates=len(dd_updates),empty_selection=sum(d['selected_count']==0 for d in dd),mean_gradient_norm=float(np.mean([d['gradient_norm'] for d in dd_updates])) if dd_updates else None,mean_step_norm=float(np.mean([d['step_norm'] for d in dd_updates])) if dd_updates else None)
 write(OUT/'SIGNAL_SUMMARY.json',signals)
 contrasts={}
 for a,b in [('useful','all'),('noisy','all'),('useful_negative','useful_positive'),('useful_matched','noisy_matched')]:
  contrasts[a+' minus '+b]={g:{sub:{m:describe([summary[a][o][g][sub][m]-summary[b][o][g][sub][m] for o in p['orders']]) for m in ['delta_s','delta_v']} for sub in ['all','nonexpert']} for g in ['corruption','clean']}
 write(OUT/'PAIRED_CONTRASTS.json',contrasts)
 write(OUT/'AUDIT.json',dict(status='pass',new_cells=count,reused_arm_cells=1920,state_resets=210,state_links=count,SGD_coordinates=coords,pairwise_updates=losschecks,independent_reward_values=rewardchecks,independent_GT_quality_values=qualitychecks,independent_temporal_scores=teacherchecks,dual_metric_calls=count,matched_scheduled_pairs=120,all_matched_signal_counts_exact=True,maximum_parameter_error=maxparam,maximum_loss_error=maxloss,maximum_distance_error=maxdistance,GT_used_during_filtering=True,GT_read_before_updates=True,deployable=False,frozen_method_unchanged=True))
 print('N1 audit pass',count,'arrivals',losschecks,'updates')
if __name__=='__main__':run()
