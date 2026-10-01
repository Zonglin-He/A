"""CPU-only full evaluation plus saved-tube pipeline diagnosis, after global seal."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,collections,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.tastvg_best_full_common_v1 import *
def source_summary(rows,fields):
 # Multiple queries share a source: query mean per source/order/condition first.
 if not rows:return dict(sources=0,cells=0,metrics={})
 groups=collections.defaultdict(list)
 for r in rows:groups[r['source_id'],r['order'],r['condition']].append([r[k] for k in fields])
 so=collections.defaultdict(list)
 for (s,o,c),v in groups.items():so[s,o].append(np.mean(v,axis=0))
 sources=collections.defaultdict(list);orders=collections.defaultdict(list)
 for (s,o),v in so.items():z=np.mean(v,axis=0);sources[s].append(z);orders[o].append(z)
 matrix=np.array([np.mean(sources[s],axis=0) for s in sorted(sources)]);rng=np.random.default_rng(20261001);boots=[]
 for i in range(0,10000,100):boots.append(matrix[rng.integers(0,len(matrix),(100,len(matrix)))].mean(1))
 ci=np.quantile(np.concatenate(boots),[.025,.975],axis=0);ov=np.array([np.mean(orders[o],axis=0) for o in sorted(orders)])
 return dict(sources=len(matrix),cells=len(rows),metrics={k:dict(mean=float(matrix[:,j].mean()),ci95=ci[:,j].tolist(),query_macro=float(np.mean([r[k] for r in rows])),order_values=ov[:,j].tolist(),order_sample_SD=float(ov[:,j].std(ddof=1)) if len(ov)>1 else None,harm_gt5pp_sources=int((matrix[:,j]<-.05).sum()) if k.startswith('delta_') else None) for j,k in enumerate(fields)})
def truth(dataset,p):
 if dataset=='vidstg':
  from scripts.diagnose_tastvg_pipeline_cpu_v1 import truth as oldtruth
  return oldtruth('P1',p)
 ap=ROOT/'data/hcstvg2_official_metadata/val_v2.json';a=read(ap);assert sha(ap)==read(BASE/'DESIGN_LOCK.json')['hc_annotation_metadata_sha256'];dense={};spans={}
 for i,r in enumerate(p['rows']):
  v=a[r['annotation_key']];assert v['English'].lower()==r['input']['caption'];start=int(v['st_frame'])-1;end=start+len(v['bbox'])-1;spans[i]=[start,end]
  dense[i]={start+j:[x,y,min(x+w,r['input']['width']),min(y+h,r['input']['height'])] for j,(x,y,w,h) in enumerate(v['bbox'])}
 return dense,spans,{str(ap.relative_to(ROOT)):sha(ap)}
def geom(p,q,coeff):
 p=np.asarray(p,float)[None];q=np.asarray(q,float);a,b=p[...,:2]-p[...,2:]/2,p[...,:2]+p[...,2:]/2;c,d=q[...,:2]-q[...,2:]/2,q[...,:2]+q[...,2:]/2
 inter=np.maximum(np.minimum(b,d)-np.maximum(a,c),0).prod(-1);union=p[...,2:].prod(-1)+q[...,2:].prod(-1)-inter;enc=(np.maximum(b,d)-np.minimum(a,c)).prod(-1)
 return coeff[0]*abs(p-q).sum(-1).mean(-1)+coeff[1]*(1-inter/union+(enc-union)/enc).mean(-1)
def iou(a,b):
 inter=max(0,min(a[1],b[1])-max(a[0],b[0]));return inter/max(max(a[1],b[1])-min(a[0],b[0]),1e-12)
def run(dataset):
 import torch
 torch.set_num_threads(2)
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.tastvg_paper48_metrics_v1 import DenseMetric,xyxy
 from vg_tta.tastvg_paper48_hc2_metrics_v1 import HC2DenseMetric
 from vg_tta.tastvg_paper_readouts_v1 import dense_official_metrics
 from scripts.diagnose_tastvg_pipeline_cpu_v1 import evaluator,transitions
 from scripts.score_tastvg_schedule_j01_v1 import independent_scores
 from vg_tta.tastvg_spatial_critic_s06_v1 import rewards
 from vg_tta.tastvg_spatial_rank_s11_v1 import average_ranks
 p=verify(dataset);out=BASE/dataset;cfg=p['params'];globalbar=read(BASE/'GLOBAL_PREDICTION_BARRIER.json')
 for d in DATASETS:
  f=BASE/d/'PREDICTION_BARRIER.json';assert sha(f)==globalbar['datasets'][d] and read(f)['cells']==read(BASE/d/'PLAN.json')['total']
 bar=read(out/'PREDICTION_BARRIER.json')
 for f,h in bar['files'].items():
  rf=out/f;assert sha(rf)==h and sha(rf.with_suffix('.gz'))==read(rf)['sha256']
 dense,spans,provenance=truth(dataset,p)
 if not (out/'GT_EXPOSURE.json').exists():write(out/'GT_EXPOSURE.json',dict(scope='Full fixed evaluation and posthoc pipeline diagnosis; no reselection',global_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),annotation_sources=provenance,time=time.time()))
 metric=DenseMetric() if dataset=='vidstg' else HC2DenseMetric();sourceids={s:i for i,s in enumerate(sorted({r['source'] for r in p['rows']}))};rows=[];steps=[];maxerr=0.;links=sgd=teachers=reinserts=0;initial=read(out/'SUPPORT.json')['center_sha256'];compute=collections.Counter()
 for cond in p['conditions']:
  for order,seq in p['orders'].items():
   previous=initial
   for at,parent in enumerate(seq):
    x=loadz(out/'online'/cond/order/f'{at:05}.pt.gz');row=p['rows'][parent];scheduled=at%4==0
    assert x['parent']==parent and x['condition']==cond and x['order']==order and x['arrival']==at and x['expert_scheduled']==scheduled and x['GT_read'] is False
    assert previous==x['pre_sha']==state_hash(x['pre_state']) and x['post_sha']==state_hash(x['post_state']);previous=x['post_sha'];links+=1
    ids=row['frame_ids'];gt=dense[parent];span=spans[parent];clip=dataset=='hc2'
    fs=evaluator(x['source_native']['boxes'],row,gt,span,clip);ss=evaluator(x['slow']['boxes'],row,gt,span,clip);ps=evaluator(x['post_prediction']['boxes'],row,gt,span,clip)
    f=fs(x['source_native']['indices']);b=ss(x['source_native']['indices']);s=ss(x['slow']['indices']);z=ss(x['final_indices']);post=ps(x['slow']['indices'])
    r=dict(parent=parent,source_id=sourceids[row['source']],order=order,condition=cond,arrival=at,expert_scheduled=scheduled,updated=x['updated'],tuning_source_exposed=row['tuning_source_exposed'])
    for name,m in [('frozen',f),('boxes_only',b),('slow',s),('final',z),('post_fixed_time',post)]:r.update({name+'_'+k:float(v) for k,v in m.items()})
    for arm,pred,idx in [('Frozen',x['source_native'],x['source_native']['indices']),('Ours',x['slow'],x['final_indices'])]:
     pix=xyxy(pred['boxes'],row['input']['width'],row['input']['height']);pix=np.maximum(pix,0) if clip else pix;interval=[ids[idx[0]],ids[idx[1]]+1];m=metric(pix,ids,interval,gt,span);ind=dense_official_metrics(pix,ids,interval,gt,span)
     for k in ['m_tIoU','m_vIoU','vIoU@0.3','vIoU@0.5']:
      err=abs(m[k]-ind[k]);assert err<1e-10;maxerr=max(maxerr,err)
     ref=f if arm=='Frozen' else z
     assert abs(ref['v']-m['m_vIoU'])<1e-10 and abs(ref['t']-m['m_tIoU'])<1e-10
     r.update({arm+'_'+k:float(v) for k,v in m.items()})
    r.update(delta_inherited_boxes=b['v']-f['v'],delta_inherited_interval=s['v']-b['v'],delta_inherited_total=s['v']-f['v'],delta_temporal_rerank=z['v']-s['v'],delta_total=z['v']-f['v'],delta_post_fixed_time=post['v']-s['v'])
    for k in ['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']:r['delta_'+k]=r['Ours_'+k]-r['Frozen_'+k]
    evidence={}
    if scheduled:
     for stage in ['spatial','temporal']:
      er=read(out/'experts'/stage/cond/f'{parent:05}.json');assert er['pixel_sha256']==x['pixel_sha256'];cf=out/'experts'/er['cache'];assert sha(cf)==er['cache_sha256'];evidence[stage]=load(cf)
     td=x['temporal'];e=evidence['temporal'];sc=independent_scores(td['candidates'],e);np.testing.assert_allclose(sc,td['scores'],atol=1e-14,rtol=0);assert int(np.argmax(sc))==td['selected'] and x['final_indices']==td['candidates'][td['selected']]['indices'];teachers+=1
     cm=[ss(c['indices']) for c in td['candidates']];oracle=int(np.argmax([m['v'] for m in cm]));proposals=e['proposals'];selected=td['selected'];contributors=[iou(td['candidates'][selected]['physical_interval'],q)*w for q,w in zip(proposals,e['proposal_confidence'])];wi=int(np.argmax(contributors)) if contributors else None
     r.update(temporal_candidate_v=[m['v'] for m in cm],temporal_candidate_t=[m['t'] for m in cm],temporal_scores=td['scores'],temporal_selected=selected,temporal_best_v=max(m['v'] for m in cm),temporal_best_t=max(m['t'] for m in cm),temporal_oracle_top_tie=bool(abs(sc[oracle]-sc[selected])<=1e-12),best_teacher_t=max([iou(q,span) for q in proposals],default=0),winning_teacher_t=iou(proposals[wi],span) if wi is not None else 0.)
    trace=x['update_steps'];assert bool(trace)==scheduled and len(trace)<=cfg['steps'];prev=x['pre_state'];valid=0
    for j,step in enumerate(trace):
     assert state_hash(prev)==step['pre_state_sha256']==state_hash(step['pre_state']);assert state_hash(step['post_state'])==step['post_state_sha256'];prev=step['post_state'];cs=[c['prediction']['boxes'] for c in step['candidates']];assert len(cs)==9 and torch.equal(cs[0],step['prediction']['boxes'])
     rew=rewards([c.numpy() for c in cs],evidence['spatial']['boxes'],evidence['spatial']['valid']);u=step['update']
     assert step['valid_expert_frames']==int(evidence['spatial']['valid'].sum())
     if rew is None:assert u is None and step['rewards'] is None
     else:np.testing.assert_allclose(rew,step['rewards'],atol=1e-14,rtol=0)
     if u is not None:
      assert u['lr']==cfg['lr'] and u['teacher_temperature']==cfg['teacher_temperature'] and u['student_temperature']==cfg['student_temperature'] and u['candidate_targets_detached'] and u['reward_detached'];rank=average_ranks(rew);np.testing.assert_allclose(rank,u['rank'],atol=0,rtol=0)
      d=geom(step['prediction']['boxes'],np.stack(cs),u['coefficients']);np.testing.assert_allclose(d,u['distances'],atol=3e-6,rtol=2e-5)
      lp=-d/cfg['student_temperature'];lp-=np.logaddexp.reduce(lp);lq=-rank/cfg['teacher_temperature'];lq-=np.logaddexp.reduce(lq);np.testing.assert_allclose(np.exp(lp),u['p'],atol=2e-6,rtol=2e-5);np.testing.assert_allclose(np.exp(lq),u['q'],atol=2e-7,rtol=1e-6);np.testing.assert_allclose(np.sum(np.exp(lp)*(lp-lq)),u['loss_before'],atol=2e-5,rtol=2e-5)
      for n,before in step['pre_state'].items():np.testing.assert_allclose((before.double().numpy()-u['update_scale']*u['gradients'][n].double().numpy()).astype(np.float32),step['post_state'][n],atol=1e-6,rtol=2e-6)
      assert u['update_scale'] in [0.,cfg['lr']];valid+=1;sgd+=1
     else:assert all(torch.equal(v,step['post_state'][n]) for n,v in step['pre_state'].items())
     # Box-only candidate and update utility: all use the original pre-update time.
     scores=[evaluator(c,row,gt,span,clip)(x['slow']['indices'])['v'] for c in cs];before=scores[0];after=evaluator(step['post_prediction']['boxes'],row,gt,span,clip)(x['slow']['indices'])['v'];top=np.flatnonzero(np.asarray(rew)>=max(rew)-1e-12).tolist() if rew is not None else []
     steps.append(dict(parent=parent,source_id=r['source_id'],condition=cond,order=order,arrival=at,step=j,tuning_source_exposed=r['tuning_source_exposed'],pre_v=before,post_v=after,delta_update=after-before,candidate_v=scores,rewards=None if rew is None else rew.tolist(),oracle_v=max(scores),teacher_top_indices=top,teacher_top_best_v=max([scores[k] for k in top],default=before),teacher_top_first_v=scores[top[0]] if top else before,valid_frames=step['valid_expert_frames'],updated=step['updated'],flat_rewards=bool(rew is not None and np.ptp(rew)<=1e-12),loss_decreased=bool(u is not None and u['loss_after']<u['loss_before']-1e-12)))
    assert state_hash(prev)==x['post_sha'];c=x['compute'];assert c['inner_steps']==len(trace) and c['backward_calls']==valid and c['spatial_candidate_replays']==len(trace)*9 and c['native_replays']==1+max(0,len(trace)-1)+9*len(trace)+2*valid and c['spatial_provider_calls']==c['temporal_provider_calls']==int(scheduled);compute.update(c)
    if x['reinsertion']:assert x['reinsertion']['full_pipeline_exact'];reinserts+=1
    rows.append(r)
    if len(rows)%250==0:status(out/'SCORE_STATUS.json',dict(status='running_cpu',done=len(rows),total=p['total']));print('SCORE',dataset,len(rows),p['total'],flush=True)
 assert links==p['total'] and reinserts==24 and not torch.cuda.is_initialized()
 # Anonymous scalar records suffice to reproduce all aggregate claims.
 write(out/'ROWS.json',rows);write(out/'SPATIAL_STEP_ROWS.json',steps)
 fields=[a+'_'+k for a in ['Frozen','Ours','delta'] for k in ['m_vIoU','m_tIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']]+['delta_inherited_boxes','delta_inherited_interval','delta_inherited_total','delta_temporal_rerank','delta_post_fixed_time']
 summary={};diagnosis={}
 for cohort in ['full','outside_tuning_sources','tuning_sources']:
  rr=[r for r in rows if cohort=='full' or r['tuning_source_exposed']==(cohort=='tuning_sources')];summary[cohort]={};diagnosis[cohort]={}
  for group in ['clean','corruption',*p['conditions'][1:]]:
   seq=[r for r in rr if (r['condition']!='clean' if group=='corruption' else r['condition']==group)];summary[cohort][group]={}
   for sub in ['all','expert','nonexpert']:
    selected=[r for r in seq if sub=='all' or r['expert_scheduled']==(sub=='expert')];summary[cohort][group][sub]=source_summary(selected,fields)
   if group not in ['clean','corruption']:continue
   z={name:transitions(seq,a,b) for name,a,b in [('inherited_boxes','frozen_v','boxes_only_v'),('inherited_interval','boxes_only_v','slow_v'),('temporal_rerank','slow_v','final_v'),('total','frozen_v','final_v'),('post_update_fixed_time','slow_v','post_fixed_time_v')]}
   for name,a,b in [('inherited_boxes','frozen_v','boxes_only_v'),('temporal_rerank','slow_v','final_v')]:
    z[name]['damaged_source_count']=len({r['source_id'] for r in seq if r[b]<r[a]-1e-12})
   ex=[r for r in seq if r['expert_scheduled']];st=[r for r in steps if (cohort=='full' or r['tuning_source_exposed']==(cohort=='tuning_sources')) and ((r['condition']!='clean')==(group=='corruption'))]
   z['temporal']={str(t):dict(scheduled=len(ex),correct_candidate_exists=sum(r['temporal_best_v']>t for r in ex),missed_correct_candidate=sum(r['temporal_best_v']>t and r['final_v']<=t for r in ex),no_correct_candidate=sum(r['temporal_best_v']<=t for r in ex),correct_destroyed=sum(r['slow_v']>t and r['final_v']<=t for r in ex),correct_rescued=sum(r['slow_v']<=t and r['final_v']>t for r in ex),good_teacher_bad_contributor=sum(r['slow_v']>t and r['final_v']<=t and r['best_teacher_t']>.5 and r['winning_teacher_t']<=.5 for r in ex)) for t in [.3,.5]}
   z['spatial_by_step']={}
   for j in sorted({r['step'] for r in st}):
    ss=[r for r in st if r['step']==j];z['spatial_by_step'][str(j)]=dict(cells=len(ss),empty_evidence=sum(r['valid_frames']==0 for r in ss),flat_rewards=sum(r['flat_rewards'] for r in ss),loss_down_but_GT_harm=sum(r['loss_decreased'] and r['delta_update']< -1e-12 for r in ss),transitions=transitions(ss,'pre_v','post_v'),thresholds={str(t):dict(support_correct=sum(r['oracle_v']>t for r in ss),expert_top_has_no_correct=sum(r['oracle_v']>t and r['teacher_top_best_v']<=t and bool(r['teacher_top_indices']) for r in ss),empty_evidence_with_correct_support=sum(r['oracle_v']>t and r['valid_frames']==0 for r in ss),useful_top_but_update_wrong=sum(r['teacher_top_best_v']>t and r['post_v']<=t for r in ss)) for t in [.3,.5]})
   diagnosis[cohort][group]=z
 write(out/'SUMMARY.json',summary);write(out/'PIPELINE_DIAGNOSIS.json',diagnosis)
 write(out/'CASES.json',{field:{'negative':sorted(rows,key=lambda r:r[field])[:12],'positive':sorted(rows,key=lambda r:-r[field])[:12]} for field in ['delta_total','delta_temporal_rerank','delta_post_fixed_time']})
 write(out/'AUDIT.json',dict(status='pass',cells=links,SGD_steps=sgd,teacher_checks=teachers,reinserts=reinserts,compute=dict(compute),max_dense_error=maxerr,GT_after_global_barrier=True,GPU_initialized=False,diagnostic_steps=len(steps),time=time.time()))
 write(out/'COMPLETION.json',dict(status='scored_diagnosed_pending_root_audit_publication',audit_sha256=sha(out/'AUDIT.json'),rows_sha256=sha(out/'ROWS.json'),time=time.time()));status(out/'SCORE_STATUS.json',dict(status='completed',done=links,total=links))
if __name__=='__main__':run(sys.argv[1])
