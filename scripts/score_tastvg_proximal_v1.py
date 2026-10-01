"""CPU-only scoring and numerical audit after the complete four-arm seal."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.tastvg_proximal_common_v1 import *
def run(ds,arm):
 import torch
 torch.set_num_threads(2)
 from scripts.score_tastvg_extended_v3 import labels
 from scripts.score_tastvg_best_quick_v1 import source_summary,geom
 from scripts.diagnose_tastvg_pipeline_cpu_v1 import evaluator
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.tastvg_spatial_rank_s11_v1 import average_ranks
 from vg_tta.tastvg_spatial_critic_s06_v1 import rewards
 from scripts.score_tastvg_schedule_j01_v1 import independent_scores
 from vg_tta.tastvg_paper48_metrics_v1 import DenseMetric,xyxy
 from vg_tta.tastvg_paper48_hc2_metrics_v1 import HC2DenseMetric
 from vg_tta.tastvg_paper_readouts_v1 import dense_official_metrics
 p=verify(ds)
 if arm=='T':
  tb=read(BASE/'TEMPORAL_PREDICTION_BARRIER.json')
  for rel,h in tb['files'].items():assert sha(BASE/rel)==h
 out=BASE/ds/arm;req=read(out/'REQUEST.json');cfg=p['params']
 globalbar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
 for rel,h in globalbar['files'].items():assert sha(BASE/rel)==h
 bar=read(out/'PREDICTION_BARRIER.json');assert bar['cells']==384
 for f,h in bar['files'].items():assert sha(out/f)==h and sha((out/f).with_suffix('.pt'))==read(out/f)['sha256']
 # These identical development labels were already exposed; they are never
 # opened by a model worker and are not used for evidence-strength calibration.
 gt=labels(ds,'search',p)
 write(out/'GT_EXPOSURE.json',dict(scope='development comparison, after four-arm global seal',globally_fresh=False,barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),time=time.time()))
 initial=read(out/'SUPPORT.json')['center_sha256'];rows=[];steps=[];links=0;coords=0;maxerr=0.;cost=collections.Counter();source_ids={s:i for i,s in enumerate(sorted({p['rows'][i]['source'] for i in p['splits']['search']['orders']['order1']}))}
 metric=DenseMetric() if ds=='vidstg' else HC2DenseMetric()
 for cond in p['conditions']:
  for order,seq in p['splits']['search']['orders'].items():
   previous=initial
   for at,parent in enumerate(seq):
    x=load(out/'online'/cond/order/f'{at:05}.pt');row=p['rows'][parent];g=gt[str(parent)];truth={int(k):v for k,v in g['truth'].items()};span=g['span'];clip=ds=='hc2'
    assert x['GT_read'] is False and x['parent']==parent and x['arrival']==at and x['expert_scheduled']==(at%4==0)
    assert previous==x['pre_sha']==state_hash(x['pre_state']) and x['post_sha']==state_hash(x['post_state']);previous=x['post_sha'];links+=1
    fs=evaluator(x['source_native']['boxes'],row,truth,span,clip);ss=evaluator(x['slow']['boxes'],row,truth,span,clip);ps=evaluator(x['post_prediction']['boxes'],row,truth,span,clip)
    f=fs(x['source_native']['indices']);b=ss(x['source_native']['indices']);slow=ss(x['slow']['indices']);final=ss(x['final_indices']);post=ps(x['slow']['indices'])
    r=dict(arm=arm,parent=parent,source_id=source_ids[row['source']],condition=cond,order=order,arrival=at,expert_scheduled=x['expert_scheduled'],updated=x['updated'])
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
      er=read(POOL/ds/'experts'/stage/cond/f'{parent:05}.json');cf=POOL/ds/'experts'/er['cache'];assert sha(cf)==er['cache_sha256'] and er['pixel_sha256']==x['pixel_sha256'];evidence[stage]=load(cf)
     td=x['temporal'];scores=independent_scores(td['candidates'],evidence['temporal']);np.testing.assert_allclose(scores,td['scores'],atol=1e-14,rtol=0);assert int(np.argmax(scores))==td['selected']
     r.update(temporal_candidate_v=[ss(c['indices'])['v'] for c in td['candidates']],temporal_scores=td['scores'],temporal_selected=td['selected'])
    prev=x['pre_state'];valid=0
    for j,s in enumerate(x['update_steps']):
     assert state_hash(prev)==s['pre_state_sha256']==state_hash(s['pre_state']);assert state_hash(s['post_state'])==s['post_state_sha256'];prev=s['post_state'];u=s['update'];cs=[c['prediction']['boxes'] for c in s['candidates']];assert torch.equal(cs[0],s['prediction']['boxes'])
     rew=rewards([c.numpy() for c in cs],evidence['spatial']['boxes'],evidence['spatial']['valid'])
     if rew is None:assert u is None and s['rewards'] is None
     else:np.testing.assert_allclose(rew,s['rewards'],atol=1e-14,rtol=0)
     if u is not None:
      valid+=1;rank=average_ranks(rew);np.testing.assert_array_equal(rank,u['rank'])
      d=geom(s['prediction']['boxes'],np.stack(cs),u['coefficients']);np.testing.assert_allclose(d,u['distances'],atol=3e-6,rtol=2e-5)
      lp=-d/cfg['student_temperature'];lp-=np.logaddexp.reduce(lp);lq=-rank/cfg['teacher_temperature'];lq-=np.logaddexp.reduce(lq);pi=np.exp(lp);qr=np.exp(lq)
      np.testing.assert_allclose(pi,u['p'],atol=2e-6,rtol=2e-5);np.testing.assert_allclose(qr,u['q_rank'],atol=2e-7,rtol=1e-6)
      spread=float(np.ptp(rew));lam=1. if req['method']['target_mode']=='rank' else req['method'].get('fixed_lambda',spread/(spread+req['method']['s_ref']))
      assert abs(lam-u['strength'])<1e-14 and abs(spread-u['reward_spread'])<1e-14
      tq=(1-lam)*pi+lam*qr;np.testing.assert_allclose(tq,u['q'],atol=2e-6,rtol=2e-5)
      np.testing.assert_allclose(np.sum(pi*(lp-np.log(tq))),u['loss_before'],atol=2e-5,rtol=2e-5)
      # Reconstruct FP32 SGD then shared-arrival projection from saved gradients.
      proposed={n:v.clone().add_(u['gradients'][n],alpha=-u['update_scale']) for n,v in s['pre_state'].items()}
      delta={n:proposed[n]-x['pre_state'][n] for n in proposed};pn=float(torch.sqrt(sum(v.double().square().sum() for v in delta.values())));rad=req['method']['arrival_radius'];factor=min(1.,rad/(pn+1e-30)) if rad is not None else 1.
      assert abs(pn-u['proposed_arrival_norm'])<1e-9 and abs(factor-u['projection_factor'])<1e-9 and u['arrival_center_sha256']==x['pre_sha']
      for n,v in proposed.items():
       expected=x['pre_state'][n]+factor*delta[n] if factor<1 else v;np.testing.assert_array_equal(expected,s['post_state'][n]);coords+=v.numel()
      actual=float(torch.sqrt(sum((s['post_state'][n]-x['pre_state'][n]).double().square().sum() for n in proposed)))
      if rad is not None:assert actual<=rad+2e-6
      if u['flat_noop']:assert u['loss_before']==0 and all(torch.count_nonzero(g)==0 for g in u['gradients'].values()) and s['pre_state_sha256']==s['post_state_sha256']
      dp=geom(s['post_prediction']['boxes'],np.stack(cs),u['coefficients']);lpp=-dp/cfg['student_temperature'];lpp-=np.logaddexp.reduce(lpp);frozen_logq=np.array(u['frozen_log_target']);np.testing.assert_allclose(np.sum(np.exp(lpp)*(lpp-frozen_logq)),u['loss_after'],atol=2e-5,rtol=2e-5)
     else:assert all(torch.equal(v,s['post_state'][n]) for n,v in s['pre_state'].items())
     vals=[evaluator(c,row,truth,span,clip)(x['slow']['indices'])['v'] for c in cs];after=evaluator(s['post_prediction']['boxes'],row,truth,span,clip)(x['slow']['indices'])['v'];top=np.flatnonzero(np.asarray(rew)>=max(rew)-1e-12).tolist() if rew is not None else []
     sr=dict(arm=arm,parent=parent,source_id=r['source_id'],condition=cond,order=order,arrival=at,step=j,pre_v=vals[0],post_v=after,delta_update=after-vals[0],candidate_v=vals,rewards=rew.tolist() if rew is not None else None,teacher_top_indices=top,unique_useful_top_harm=bool(len(top)==1 and vals[top[0]]>vals[0]+1e-12 and after<vals[0]-1e-12),valid_frames=s['valid_expert_frames'],updated=s['updated'],loss_decreased=bool(u and u['loss_after']<u['loss_before']-1e-12))
     if u:sr.update({k:float(u[k]) for k in ['strength','reward_spread','loss_before','loss_after','global_gradient_norm','proposed_arrival_norm','projection_factor','actual_arrival_norm']});sr['flat_noop']=u['flat_noop']
     steps.append(sr)
    assert state_hash(prev)==x['post_sha'];cost.update(x['compute']);rows.append(r)
 assert links==384 and not torch.cuda.is_initialized()
 fields=[a+'_'+k for a in ['Frozen','Ours','delta'] for k in ['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']]+['delta_inherited_boxes','delta_inherited_interval','delta_temporal_rerank','delta_post_fixed_time']
 summary={g:{sub:source_summary([r for r in rows if (r['condition']=='clean')==(g=='clean') and (sub=='all' or r['expert_scheduled']==(sub=='expert'))],fields) for sub in ['all','expert','nonexpert']} for g in ['clean','corruption']}
 write(out/'ROWS.json',rows);write(out/'SPATIAL_STEP_ROWS.json',steps);write(out/'SUMMARY.json',summary)
 write(out/'AUDIT.json',dict(status='pass',state_links=links,SGD_projection_coordinates=coords,max_dense_metric_error=maxerr,compute=dict(cost),worker_wall_seconds=read(out/'STATUS.json')['seconds'],GPU_initialized=False,GT_after_global_barrier=True,time=time.time()))
 write(out/'COMPLETION.json',dict(status='scored_pending_root_public_audit',rows_sha256=sha(out/'ROWS.json'),audit_sha256=sha(out/'AUDIT.json'),time=time.time()))
 print('SCORED',ds,arm,summary['corruption']['nonexpert']['metrics']['delta_m_vIoU']['mean']*100,flush=True)
if __name__=='__main__':run(*sys.argv[1:])
