"""CPU-only A/R scoring with independent Rank-RKL and SGD reconstruction."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.tastvg_routed_common_v1 import *
def run(ds,arm):
 import torch
 torch.set_num_threads(2)
 from scripts.score_tastvg_extended_v3 import labels
 from scripts.score_tastvg_best_quick_v1 import source_summary
 from scripts.tastvg_event_support_metrics_v1 import geometry as geom,weights_from_candidates,weighted_rewards,frame_geometry
 from scripts.diagnose_tastvg_pipeline_cpu_v1 import evaluator
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.tastvg_spatial_rank_s11_v1 import average_ranks
 from vg_tta.tastvg_spatial_critic_s06_v1 import rewards
 from scripts.score_tastvg_schedule_j01_v1 import independent_scores
 from vg_tta.tastvg_paper48_metrics_v1 import DenseMetric,xyxy
 from vg_tta.tastvg_paper48_hc2_metrics_v1 import HC2DenseMetric
 from vg_tta.tastvg_paper_readouts_v1 import dense_official_metrics
 p=verify(ds)
 out=BASE/ds/arm;req=read(out/'REQUEST.json');cfg=p['params']
 globalbar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json');assert globalbar['cells']==1536 and globalbar['GT_read'] is False
 for rel,h in read(BASE/'SCORING_RUNTIME_LOCK.json')['pins'].items():assert sha(ROOT/rel)==h
 for rel,h in globalbar['files'].items():assert sha(BASE/rel)==h
 bar=read(out/'PREDICTION_BARRIER.json');assert bar['cells']==384 and len(bar['files'])==384 and bar['GT_read'] is False
 assert len(globalbar['files'])==4 and globalbar['time']>=bar['time']
 for f,h in bar['files'].items():assert sha(out/f)==h and sha((out/f).with_suffix('.pt'))==read(out/f)['sha256']
 # These identical development labels were already exposed; they are never
 # opened by a model worker and are not used for evidence-strength calibration.
 assert time.time()>=globalbar['time']
 gt=labels(ds,'search',p)
 write(out/'GT_EXPOSURE.json',dict(scope='development comparison, after four-stream global seal',globally_fresh=False,barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),time=time.time()))
 assert sha(BASE/'BASIS.pt')==read(BASE/'BASIS_LOCK.json')['sha256'];basis=load(BASE/'BASIS.pt').numpy()
 initial=read(out/'SUPPORT.json')['center_sha256'];rows=[];steps=[];links=0;coords=0;maxerr=0.;paramerr=0.;cost=collections.Counter();source_ids={s:i for i,s in enumerate(sorted({p['rows'][i]['source'] for i in p['splits']['search']['orders']['order1']}))}
 metric=DenseMetric() if ds=='vidstg' else HC2DenseMetric()
 for cond in p['conditions']:
  for order,seq in p['splits']['search']['orders'].items():
   previous=initial;has_prior_write=False
   for at,parent in enumerate(seq):
    x=load(out/'online'/cond/order/f'{at:05}.pt');row=p['rows'][parent];g=gt[str(parent)];truth={int(k):v for k,v in g['truth'].items()};span=g['span'];clip=ds=='hc2'
    assert x['GT_read'] is False and x['parent']==parent and x['arrival']==at and x['expert_scheduled']==(at%4==0)
    assert previous==x['pre_sha']==state_hash(x['pre_state']) and x['post_sha']==state_hash(x['post_state']);previous=x['post_sha'];links+=1
    fs=evaluator(x['source_native']['boxes'],row,truth,span,clip);ss=evaluator(x['slow']['boxes'],row,truth,span,clip);ps=evaluator(x['post_prediction']['boxes'],row,truth,span,clip)
    f=fs(x['source_native']['indices']);b=ss(x['source_native']['indices']);slow=ss(x['slow']['indices']);final=ss(x['final_indices']);post=ps(x['slow']['indices'])
    r=dict(arm=arm,parent=parent,source_id=source_ids[row['source']],condition=cond,order=order,arrival=at,expert_scheduled=x['expert_scheduled'],updated=x['updated'],has_prior_write=has_prior_write)
    for name,m in [('frozen',f),('boxes_only',b),('slow',slow),('final',final),('post_fixed_time',post)]:r.update({name+'_'+k:float(v) for k,v in m.items()})
    for name,pred,idx in [('Frozen',x['source_native'],x['source_native']['indices']),('Ours',x['slow'],x['final_indices'])]:
     pix=xyxy(pred['boxes'],row['input']['width'],row['input']['height']);pix=np.maximum(pix,0) if clip else pix;interval=[row['frame_ids'][idx[0]],row['frame_ids'][idx[1]]+1]
     m=metric(pix,row['frame_ids'],interval,truth,span);ind=dense_official_metrics(pix,row['frame_ids'],interval,truth,span)
     for k in ['m_tIoU','m_vIoU','vIoU@0.3','vIoU@0.5']:err=abs(m[k]-ind[k]);assert err<1e-10;maxerr=max(maxerr,err)
     r.update({name+'_'+k:float(v) for k,v in m.items()})
    for k in ['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']:r['delta_'+k]=r['Ours_'+k]-r['Frozen_'+k]
    r.update(delta_inherited_boxes=b['v']-f['v'],delta_inherited_interval=slow['v']-b['v'],delta_temporal_rerank=final['v']-slow['v'],delta_post_fixed_time=post['v']-slow['v'])
    evidence={}
    if x['expert_scheduled']:
     for stage in ['spatial','temporal']:
      er=x['routed_expert'] if stage=='spatial' and arm=='R' else read(POOL/ds/'experts'/stage/cond/f'{parent:05}.json')
      cf=ROOT/er['cache'] if stage=='spatial' and arm=='R' else POOL/ds/'experts'/er['cache']
      assert sha(cf)==er['cache_sha256'] and er['pixel_sha256']==x['pixel_sha256'];evidence[stage]=load(cf)
     td=x['temporal'];scores=independent_scores(td['candidates'],evidence['temporal']);np.testing.assert_allclose(scores,td['scores'],atol=1e-14,rtol=0);assert int(np.argmax(scores))==td['selected']
     r.update(temporal_candidate_v=[ss(c['indices'])['v'] for c in td['candidates']],temporal_scores=td['scores'],temporal_selected=td['selected'])
    prev=x['pre_state'];valid=0;extra=0
    for j,s in enumerate(x['update_steps']):
     assert state_hash(prev)==s['pre_state_sha256']==state_hash(s['pre_state']);assert state_hash(s['post_state'])==s['post_state_sha256'];prev=s['post_state'];u=s['update'];cs=[c['prediction']['boxes'] for c in s['candidates']];assert torch.equal(cs[0],s['prediction']['boxes'])
     ew=None;support=s.get('event_support');er=evidence['spatial'];ev=np.asarray(er['valid'],bool)
     assert support is None
     rew=rewards([c.numpy() for c in cs],er['boxes'],ev)
     if rew is None:assert u is None and s['rewards'] is None
     else:np.testing.assert_allclose(rew,s['rewards'],atol=1e-14,rtol=0)
     if u is not None:
      valid+=1;rank=average_ranks(rew);np.testing.assert_array_equal(rank,u['rank'])
      d=geom(s['prediction']['boxes'],np.stack(cs),u['coefficients'],ew);np.testing.assert_allclose(d,u['distances'],atol=3e-6,rtol=2e-5)
      lp=-d/cfg['student_temperature'];lp-=np.logaddexp.reduce(lp);lq=-rank/cfg['teacher_temperature'];lq-=np.logaddexp.reduce(lq);pi=np.exp(lp);qr=np.exp(lq)
      np.testing.assert_allclose(pi,u['p'],atol=2e-6,rtol=2e-5);np.testing.assert_allclose(qr,u['q_rank'],atol=2e-7,rtol=1e-6)
      spread=float(np.ptp(rew));lam=1. if req['method']['target_mode']=='rank' else req['method'].get('fixed_lambda',spread/(spread+req['method']['s_ref']))
      assert abs(lam-u['strength'])<1e-14 and abs(spread-u['reward_spread'])<1e-14
      tq=(1-lam)*pi+lam*qr;np.testing.assert_allclose(tq,u['q'],atol=2e-6,rtol=2e-5)
      np.testing.assert_allclose(np.sum(pi*(lp-np.log(tq))),u['loss_before'],atol=2e-5,rtol=2e-5)
      # Independent full-space selected-target and FP32 actuation reconstruction.
      mode=req['method']['actuation'];selected=int(np.argmax(rew));flat=bool(np.ptp(rew)==0.)
      assert selected==u['selected_index'] and flat==u['selected_flat_noop'] and u['selected_target_detached']
      gn=float(torch.sqrt(sum(g.double().square().sum() for g in u['gradients'].values())));desired=cfg['lr']*gn
      assert abs(desired-u['counterfactual_rkl_step_norm'])<1e-10
      np.testing.assert_allclose(d[selected],u['selected_loss_before'],atol=3e-6,rtol=2e-5)
      need_srd=mode=='selected_rollout' and selected!=0 and not flat
      assert u['srd_backward_calls']==int(need_srd) and u['rkl_backward_calls']==1;extra+=int(need_srd)
      assert mode=='rkl' and req['method']['support_mode']=='full'
      if mode=='rkl':
       proposed={n:v.clone().add_(u['gradients'][n],alpha=-u['update_scale']) for n,v in s['pre_state'].items()}
      else:
       gg=np.concatenate([g.double().numpy().reshape(-1) for g in u['selected_gradients'].values()]);sn=float(np.linalg.norm(gg))
       assert abs(sn-u['selected_gradient_norm'])<1e-10
       reason='flat_rewards' if flat else 'central_selected' if selected==0 else 'zero_selected_gradient' if sn==0. else 'zero_rkl_magnitude' if desired==0. else None
       assert u['no_op_reason']==reason and u['magnitude_match_applicable']==(reason is None) and u['actuated']==(reason is None)
       movement=-desired*gg/(sn+1e-12) if reason is None else np.zeros_like(gg)
       proposed={};offset=0
       for n,v in s['pre_state'].items():
        piece=torch.from_numpy(movement[offset:offset+v.numel()]).reshape(v.shape).to(v);proposed[n]=v.clone().add_(piece);offset+=v.numel()
       assert offset==1792
       if not need_srd:assert not gg.any()
       ga=np.concatenate([g.double().numpy().reshape(-1) for g in u['gradients'].values()]);co=float(np.dot(ga,gg)/(gn*sn)) if gn>0 and sn>0 else None
       if co is None:assert u['rkl_selected_gradient_cosine'] is None
       else:assert abs(co-u['rkl_selected_gradient_cosine'])<1e-10
      for n,v in proposed.items():
       err=float((v-s['post_state'][n]).abs().max());paramerr=max(paramerr,err)
       np.testing.assert_allclose(v,s['post_state'][n],atol=2e-7,rtol=1e-7);coords+=v.numel()
      actualstep=float(torch.sqrt(sum((s['post_state'][n]-s['pre_state'][n]).double().square().sum() for n in proposed)))
      assert abs(actualstep-u['actual_step_norm'])<1e-9
      if mode!='rkl' and u['no_op_reason'] is not None:assert s['pre_state_sha256']==s['post_state_sha256'] and actualstep==0
      else:assert abs(actualstep-desired)<2e-6+1e-5*desired
      assert u['arrival_radius'] is None and u['projection_factor']==1.
      if u['flat_noop']:assert u['loss_before']==0 and all(torch.count_nonzero(g)==0 for g in u['gradients'].values()) and s['pre_state_sha256']==s['post_state_sha256']
      dp=geom(s['post_prediction']['boxes'],np.stack(cs),u['coefficients'],ew)
      np.testing.assert_allclose(dp[selected],u['selected_loss_after'],atol=3e-6,rtol=2e-5)
      dd=(s['post_prediction']['boxes'].double()-s['prediction']['boxes'].double()).numpy().reshape(-1)
      ww=(cs[selected].double()-s['prediction']['boxes'].double()).numpy().reshape(-1);dn=float(np.linalg.norm(dd));wn=float(np.linalg.norm(ww))
      assert abs(np.mean(np.abs(dd))-u['box_movement_mean_abs'])<1e-12
      assert abs(dn-u['box_movement_l2'])<1e-12 and abs(wn-u['selected_target_movement_l2'])<1e-12
      output_cosine=float(np.dot(dd,ww)/(dn*wn)) if dn>0 and wn>0 else None
      if output_cosine is None:assert u['output_space_cosine'] is None
      else:assert abs(output_cosine-u['output_space_cosine'])<1e-12
      lpp=-dp/cfg['student_temperature'];lpp-=np.logaddexp.reduce(lpp);frozen_logq=np.array(u['frozen_log_target']);np.testing.assert_allclose(np.sum(np.exp(lpp)*(lpp-frozen_logq)),u['loss_after'],atol=2e-5,rtol=2e-5)
     else:assert all(torch.equal(v,s['post_state'][n]) for n,v in s['pre_state'].items())
     vals=[evaluator(c,row,truth,span,clip)(x['slow']['indices'])['v'] for c in cs];after=evaluator(s['post_prediction']['boxes'],row,truth,span,clip)(x['slow']['indices'])['v'];top=np.flatnonzero(np.asarray(rew)>=max(rew)-1e-12).tolist() if rew is not None else []
     sr=dict(arm=arm,parent=parent,source_id=r['source_id'],condition=cond,order=order,arrival=at,step=j,pre_v=vals[0],post_v=after,delta_update=after-vals[0],candidate_v=vals,rewards=rew.tolist() if rew is not None else None,teacher_top_indices=top,unique_useful_top_harm=bool(len(top)==1 and vals[top[0]]>vals[0]+1e-12 and after<vals[0]-1e-12),valid_frames=s['valid_expert_frames'],updated=s['updated'],loss_decreased=bool(u and u['loss_after']<u['loss_before']-1e-12))
     if u:
      sr.update({k:float(u[k]) for k in ['strength','reward_spread','loss_before','loss_after','global_gradient_norm','proposed_arrival_norm','projection_factor','actual_arrival_norm','counterfactual_rkl_step_norm','actual_step_norm','selected_loss_before','selected_loss_after','box_movement_mean_abs','box_movement_l2','selected_target_movement_l2']})
      sr.update({k:u[k] for k in ['actuation','actuated','no_op_reason','magnitude_match_applicable','selected_index','selected_flat_noop','srd_backward_calls','rkl_backward_calls','objective_direction','output_space_cosine','output_space_cosine_defined']})
      sr['selected_gradient_norm']=u.get('selected_gradient_norm')
      sr['rkl_selected_gradient_cosine']=u.get('rkl_selected_gradient_cosine')
      sr['teacher_ranks']=rank.tolist();sr['flat_noop']=u['flat_noop'];sr['executed_to_probe_radius']=u['actual_step_norm']/read(out/'SUPPORT.json')['spec']['radius']
      sr['selected_target_GT_delta']=float(vals[selected]-vals[0]);sr['selected_target_GT_v']=float(vals[selected]);sr['candidate_oracle_v']=float(max(vals))
      sr['selected_useful_harm']=bool(vals[selected]>vals[0]+1e-12 and after<vals[0]-1e-12)
      sr['selected_distance_decreased']=bool(u['selected_loss_after']<u['selected_loss_before']-1e-12)
      sr['output_direction_positive']=bool(u['output_space_cosine'] is not None and u['output_space_cosine']>0)

     # Support diagnostics are posthoc only; never feed GT back to the worker.
     cand_support=support['candidates'] if support else (x['temporal']['candidates'] if j==0 else None)
     ids=np.asarray(row['frame_ids']);event=(ids>=span[0])&(ids<=span[1]);scored=np.array([int(i) in truth for i in ids])
     ww=weights_from_candidates(cand_support,row['frame_ids']) if cand_support is not None else None
     sr['support_mode']=req['method']['support_mode'];sr['support_observed_this_step']=cand_support is not None
     sr['raw_valid_reference_frames']=int(ev.sum());sr['raw_reference_GT_frames']=int((ev&scored).sum())
     sr['GT_sample_frames']=int(scored.sum());sr['update_absent_reason']=support['no_update_reason'] if support else ('empty_expert' if rew is None else None)
     sr['support_candidate_indices']=[c['indices'] for c in cand_support] if cand_support is not None else None
     sr['support_weights']=ww.tolist() if ww is not None else None
     if ww is not None:
      sr.update(support_candidate_count=len(cand_support),support_weight_sum=float(ww.sum()),
       reference_weight_mass=float(ww[ev].sum()),positive_reference_frames=int((ev&(ww>0)).sum()),
       reference_GT_weight_mass=float(ww[ev&scored].sum()),
       support_GT_mass_fraction=float(ww[scored].sum()/ww.sum()),
       support_GT_frame_coverage=float((ww[scored]>0).mean()) if scored.any() else None,
       support_frame_fraction=float((ww>0).mean()))
     if u is not None:
      df=geom(s['prediction']['boxes'],np.stack(cs),u['coefficients']);dfp=geom(s['post_prediction']['boxes'],np.stack(cs),u['coefficients'])
      sr.update(selected_full_D_before=float(df[selected]),selected_full_D_after=float(dfp[selected]),
       selected_full_D_decreases=bool(dfp[selected]<df[selected]-1e-12))
      if scored.any():
       dg=geom(s['prediction']['boxes'],np.stack(cs),u['coefficients'],scored.astype(float))
       dgp=geom(s['post_prediction']['boxes'],np.stack(cs),u['coefficients'],scored.astype(float))
       sr.update(selected_GT_support_D_before=float(dg[selected]),selected_GT_support_D_after=float(dgp[selected]),
        full_D_down_GT_support_D_up=bool(dfp[selected]<df[selected]-1e-12 and dgp[selected]>dg[selected]+1e-12))
      else:sr.update(selected_GT_support_D_before=None,selected_GT_support_D_after=None,full_D_down_GT_support_D_up=None)
     steps.append(sr)
    assert state_hash(prev)==x['post_sha'] and x['updated']==any(z['updated'] for z in x['update_steps'])
    cnt=x['compute'];ns=len(x['update_steps']);assert cnt['inner_steps']==ns and cnt['backward_calls']==valid+extra and cnt['rkl_backward_calls']==valid and cnt['srd_backward_calls']==extra
    assert cnt['spatial_candidate_replays']==ns*9 and cnt['native_replays']==1+max(0,ns-1)+ns*9+2*valid
    assert cnt['spatial_provider_calls']==cnt['temporal_provider_calls']==int(x['expert_scheduled'])
    assert (1<=ns<=cfg['steps'] if x['expert_scheduled'] else ns==0) and x['reinsertion'] is None
    if ns and ns<cfg['steps']:assert x['update_steps'][-1]['update'] is None
    if arm=='R' and x['expert_scheduled']:
     from vg_tta.tastvg_reference_selection_v1 import student_frames
     route=student_frames(x['temporal']['candidates'],row['frame_ids']);assert route==x['routing'] and route['positions']==x['routed_expert']['positions']
     r.update(route_actual_unique=route['actual_unique'],route_raw_unique=route['raw_unique'])
    has_prior_write=has_prior_write or x['updated'];cost.update(cnt);rows.append(r)
 assert links==384 and not torch.cuda.is_initialized()
 fields=[a+'_'+k for a in ['Frozen','Ours','delta'] for k in ['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']]+['delta_inherited_boxes','delta_inherited_interval','delta_temporal_rerank','delta_post_fixed_time']
 summary={g:{sub:source_summary([r for r in rows if (r['condition']=='clean')==(g=='clean') and (sub=='all' or r['expert_scheduled']==(sub=='expert'))],fields) for sub in ['all','expert','nonexpert']} for g in ['clean','corruption']}
 write(out/'ROWS.json',rows);write(out/'SPATIAL_STEP_ROWS.json',steps);write(out/'SUMMARY.json',summary)
 write(out/'AUDIT.json',dict(status='pass',state_links=links,actuation_coordinates=coords,max_dense_metric_error=maxerr,max_parameter_reconstruction_error=paramerr,compute=dict(cost),worker_wall_seconds=read(out/'STATUS.json')['seconds'],GPU_initialized=False,GT_after_global_barrier=True,time=time.time()))
 write(out/'COMPLETION.json',dict(status='scored_pending_root_public_audit',rows_sha256=sha(out/'ROWS.json'),audit_sha256=sha(out/'AUDIT.json'),time=time.time()))
 print('SCORED',ds,arm,summary['corruption']['nonexpert']['metrics']['delta_m_vIoU']['mean']*100,flush=True)
if __name__=='__main__':run(*sys.argv[1:])
