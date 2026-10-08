"""Postseal tuned P1 full-query evaluation and independent Gaussian/Adam/state/dense audit.
Adapted from the original locked OPD scorer, with target-specific final rounds and alpha.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,time,json,gzip,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_common_v1 import *
import numpy as np

def truths(ds,stage):
 p=read(PAPER/ds/'PLAN.json');p={**p,'orders':{'selected':stage['parents']}}
 if ds=='vidstg':
  from scripts.diagnose_tastvg_pipeline_cpu_v1 import truth
  return truth('P1',p)
 ap=ROOT/'data/hcstvg2_official_metadata/val_v2.json';a=read(ap)
 assert sha(ap)==read(PAPER/'DESIGN_LOCK.json')['hc_annotation_metadata_sha256']
 dense={};spans={}
 for parent in stage['parents']:
  r=p['rows'][parent];v=a[r['annotation_key']];assert v['English'].lower()==r['input']['caption']
  start=int(v['st_frame'])-1;end=start+len(v['bbox'])-1;spans[parent]=[start,end]
  dense[parent]={start+j:[x,y,min(x+w,r['input']['width']),min(y+h,r['input']['height'])] for j,(x,y,w,h) in enumerate(v['bbox'])}
 return dense,spans,{str(ap.relative_to(ROOT)):sha(ap)}

def source_stats(rows,fields):
 sources=sorted({r['source_id'] for r in rows});orders=sorted({r['order'] for r in rows})
 groups=collections.defaultdict(list)
 for r in rows:groups[r['source_id']].append(r)
 matrix=np.array([np.array([[r[f] for f in fields] for r in groups[s]]).mean(0) for s in sources])
 rng=np.random.default_rng(20261006);boots=[]
 for _ in range(100):boots.append(matrix[rng.integers(0,len(matrix),(100,len(matrix)))].mean(1))
 ci=np.quantile(np.concatenate(boots),[.025,.975],axis=0)
 return dict(sources=len(sources),cells=len(rows),metrics={f:dict(mean=float(matrix[:,j].mean()),ci95=ci[:,j].tolist(),
  query_macro=float(np.mean([r[f] for r in rows])),order_values=[float(np.mean([r[f] for r in rows if r['order']==o])) for o in orders],
  gross_gain_pp=float(np.maximum(matrix[:,j],0).mean()*100) if f.startswith('delta_') else None,
  gross_loss_pp=float(-np.minimum(matrix[:,j],0).mean()*100) if f.startswith('delta_') else None,
  harm_gt5pp_sources=int((matrix[:,j]<-.05).sum()) if f.startswith('delta_') else None,
  harm_gt20pp_sources=int((matrix[:,j]<-.20).sum()) if f.startswith('delta_') else None,
  harm_gt5pp_cells=sum(r[f]<-.05 for r in rows) if f.startswith('delta_') else None,
  harm_gt20pp_cells=sum(r[f]<-.20 for r in rows) if f.startswith('delta_') else None) for j,f in enumerate(fields)})

def run(stage_name):
 import torch
 torch.set_num_threads(2);verify();global_barrier=read(BASE/'P1_PREDICTION_BARRIER.json');assert global_barrier['status']=='sealed' and global_barrier['all_deployment_OPD_directions'];stage=read(BASE/'DESIGN_LOCK.json')['stages'][stage_name];ds=stage['dataset'];dest=BASE/'stages'/stage_name
 barrier=read(dest/'PREDICTION_BARRIER.json');assert barrier['status']=='sealed' and not barrier['GT_read']
 assert sha(dest/'PREDICTION_BARRIER.json')==global_barrier['barriers'][str((dest/'PREDICTION_BARRIER.json').relative_to(BASE))]
 for rel,h in global_barrier['barriers'].items():
  b=read(BASE/rel);assert sha(BASE/rel)==h and not b['GT_read']
  for group in ['files','inputs']:
   for f,digest in b[group].items():
    path=BASE/f;rc=read(path.with_suffix('.json'));assert sha(path)==digest==rc['sha256'] and not rc['GT_read']
    assert rc['time']<=b['time']<=global_barrier['time']
 cfg=read(BASE/'DESIGN_LOCK.json')['datasets'][ds]['config'];arms=stage['arms'];assert arms==['on_policy']
 for f,h in barrier['files'].items():assert sha(BASE/f)==h and read((BASE/f).with_suffix('.json'))['sha256']==h
 out=PUB/stage_name
 if (dest/'CPU_COMPLETION.json').exists():return
 out.mkdir(parents=True,exist_ok=True)
 write(dest/'GT_EXPOSURE.json',dict(complete_both_direction_OPD_barrier_sha256=sha(BASE/'P1_PREDICTION_BARRIER.json'),GT_after_all_stage_predictions=True,time=time.time()))
 dense,spans,provenance=truths(ds,stage);paperplan=read(PAPER/ds/'PLAN.json');sourceids={s:i for i,s in enumerate(sorted({r['source'] for r in paperplan['rows']}))}
 from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube,box_iou
 from vg_tta.tastvg_paper48_metrics_v1 import xyxy
 from scripts.run_decota_paper_main_v1 import unpack_expert
 from vg_tta.decota_spatial_opd_tunable_audit_v1 import audit
 allrows=[];sample_records=[];statechecks=0;densechecks=0;mathchecks=0;start=time.time()
 for condition in stage['conditions']:
  for order,seq in stage['orders'].items():
   previous={a:None for a in arms};prevhash={a:None for a in arms}
   for at,parent in enumerate(seq):
    row=paperplan['rows'][parent];gt=dense[parent];span=spans[parent];cell={}
    for arm in arms:
     f=dest/condition/order/arm/f'{at:05}.pt';z=load(f);fit=z['fit'];inp=load(BASE/z['input']['path']);ex=unpack_expert(inp['expert'])
     assert sha(BASE/z['input']['path'])==z['input']['sha256']
     assert z['parent']==parent and z['previous_payload_sha256']==prevhash[arm] and not z['GT_read']
     if previous[arm] is not None:
      assert all(torch.equal(v,torch.zeros_like(v) if n=='spatial.query_residual' else previous[arm][n]) for n,v in fit['initial'].items())
     assert fit['selected_step']==cfg['steps'] and fit['config']==cfg and z['config']==cfg
     expected=committed(fit['initial'],fit['state'],cfg['writeback']);assert all(torch.equal(expected[n],z['committed'][n]) for n in expected)
     check=audit(fit,ex);assert check==z['math_audit'];statechecks+=1792*2;mathchecks+=1
     previous[arm]=z['committed'];prevhash[arm]=sha(f)
     values={};tubes={}
     for name,box in [('Frozen',inp['native_boxes']),('Before',fit['before']),('After',fit['final'])]:
      values[name]=official(box.numpy(),row,gt,span,z['interval'],ds)
      tubes[name]=DenseTube(box.numpy(),row,gt,span,clip=ds=='hc2')
      ev=tubes[name].score(z['interval']);assert all(abs(ev[f]-values[name][f])<2e-10 for f in ['v','t','s'])
      densechecks+=1
     r=dict(source_id=sourceids[row['source']],parent_ordinal_in_locked_roster=stage['parents'].index(parent),arrival=at,arm=arm,order=order,condition=condition,
      values=values,delta_total_v=values['After']['v']-values['Frozen']['v'],delta_inherited_v=values['Before']['v']-values['Frozen']['v'],
      delta_current_v=values['After']['v']-values['Before']['v'],delta_total_s=values['After']['s']-values['Frozen']['s'],compute=z['compute'],
      observed_frames=len(fit['positions']),gradient_calls=fit['gradient_calls'],empty=fit['empty'],roundtrip_error=z['chart_roundtrip_max_error'],query_ordinal=parent,
      event_duration_fraction=len(gt)/row['input']['frame_count'],query_type=row['query_type'])
     for name in values:
      for met,val in values[name].items():r[name+'_'+met]=val
      for threshold in [.3,.5]:r[name+'_R'+str(int(threshold*100))]=float(values[name]['v']>threshold)
     assert values['Frozen']['t']==values['Before']['t']==values['After']['t']
     for threshold in [.3,.5]:
      pre=values['Frozen']['v']>threshold;post=values['After']['v']>threshold
      r['correct_to_wrong_'+str(threshold)]=float(pre and not post);r['wrong_to_correct_'+str(threshold)]=float(not pre and post)
     validids=np.array(sorted(gt));mask=np.isin(validids,[row['frame_ids'][p] for p in fit['positions']]);diff=tubes['After'].iou-tubes['Before'].iou
     motion=np.array([gt[f] for f in sorted(gt)],float);centers=(motion[:,:2]+motion[:,2:])/2
     r['normalized_target_motion']=float(np.linalg.norm(np.diff(centers,axis=0),axis=1).mean()/np.hypot(row['input']['width'],row['input']['height'])) if len(centers)>1 else 0.
     central_reward_changes=[float((rd['central_reward_after']-rd['central_reward_before']).mean()) for rd in fit['rounds'] if 'central_reward_after' in rd]
     r['central_expert_reward_round_mean_delta']=float(np.mean(central_reward_changes)) if central_reward_changes else None
     qualities=[]
     for pos,e in __import__('vg_tta.decota_fixed_full_audit_v1',fromlist=['top1_support']).top1_support(ex):
      fid=row['frame_ids'][pos]
      if fid in gt:qualities.append(float(box_iou(xyxy(np.array(e),row['input']['width'],row['input']['height']),gt[fid])))
     r['admitted_expert_GT_IoU']=float(np.mean(qualities)) if qualities else None
     r.update(observed_GT_frames=int(mask.sum()),unobserved_GT_frames=int((~mask).sum()),observed_iou_delta=float(diff[mask].mean()) if mask.any() else None,
      unobserved_iou_delta=float(diff[~mask].mean()) if (~mask).any() else None,
      sample_diagnosis_unknown_frames=sum(row['frame_ids'][p] not in gt for p in fit['positions']))
     teacher_correct_harmed=0;details=[]
     for k,rd in enumerate(fit['rounds']):
      if not fit['positions']:continue
      rr=rd['rollout'];a=rr['samples'].numpy();actions=1/(1+np.exp(-a));weights=rr['weights'].numpy()
      for j,pos in enumerate(fit['positions']):
       fid=row['frame_ids'][pos]
       if fid not in gt:continue
       quality=box_iou(xyxy(actions[j],row['input']['width'],row['input']['height']),gt[fid])
       before=float(box_iou(xyxy(fit['path'][k]['boxes'][pos].numpy(),row['input']['width'],row['input']['height']),gt[fid]))
       after=float(box_iou(xyxy(fit['path'][k+1]['boxes'][pos].numpy(),row['input']['width'],row['input']['height']),gt[fid]))
       details.append(dict(round=k,position=pos,sample_best_gt_iou=float(quality.max()),sample_mean_gt_iou=float(quality.mean()),
        teacher_weighted_gt_iou=float(weights[j]@quality),teacher_vs_uniform_gt_iou=float(weights[j]@quality-quality.mean()),
        central_before_gt_iou=before,central_after_gt_iou=after,central_gt_delta=after-before,
        reward_variance=float(rd['reward_variance'][j]),ESS=float(rd['ESS'][j]),weight_max=float(rd['weight_max'][j])))
       teacher_correct_harmed+=int(before>.5 and after<=.5)
     r['observed_correct_to_wrong_rounds']=teacher_correct_harmed
     if details:
      for key in ['sample_best_gt_iou','teacher_vs_uniform_gt_iou','central_gt_delta','reward_variance','ESS','weight_max']:r[key]=float(np.mean([v[key] for v in details]))
     else:
      for key in ['sample_best_gt_iou','teacher_vs_uniform_gt_iou','central_gt_delta','reward_variance','ESS','weight_max']:r[key]=None
     cell[arm]=r;allrows.append(r)
    if at%10==0:print('OPD_CPU_SCORE',stage_name,condition,order,at,len(seq),flush=True)
 fields=['Frozen_v','Before_v','After_v','Frozen_s','Before_s','After_s','Frozen_t','Before_t','After_t','Frozen_R30','After_R30','Frozen_R50','After_R50','delta_total_v','delta_inherited_v','delta_current_v','delta_total_s','correct_to_wrong_0.3','correct_to_wrong_0.5','wrong_to_correct_0.3','wrong_to_correct_0.5']
 summaries={a:source_stats([r for r in allrows if r['arm']==a],fields) for a in arms}
 devsources={paperplan['rows'][q]['source'] for q in read(BASE/'DESIGN_LOCK.json')['datasets'][ds]['development_query_ordinals']}
 dev_ids={sourceids[s] for s in devsources};assert len(dev_ids)==32
 excluding=[r for r in allrows if r['source_id'] not in dev_ids]
 exclusion=source_stats(excluding,fields)
 summary=dict(stage=stage_name,dataset=ds,setting='same-domain corruption' if stage['split']=='mechanism' else 'clean cross-domain',
  split=stage['split'],historically_exposed=True,config=cfg,arms=summaries,
  diagnosis={a:{k:float(np.mean([r[k] for r in allrows if r['arm']==a and r[k] is not None])) if any(r['arm']==a and r[k] is not None for r in allrows) else None for k in ['observed_iou_delta','unobserved_iou_delta','sample_best_gt_iou','teacher_vs_uniform_gt_iou','central_gt_delta','reward_variance','ESS','weight_max','admitted_expert_GT_IoU','central_expert_reward_round_mean_delta']} for a in arms},
  audit=dict(status='pass',state_coordinate_checks=statechecks,independent_math_arrivals=mathchecks,independent_dense_metrics=densechecks),
  excluding_32_tuning_parent_sources=exclusion,excluded_development_anonymous_source_ids=sorted(dev_ids),
  GT_provenance=provenance,paired_parent_bootstrap=10000,time=time.time(),CPU_seconds=time.time()-start,automatic_promotion=False)
 write(out/'SUMMARY.json',summary)
 for name,rr in [('ROWS.jsonl.gz',allrows)]:
  with gzip.open(out/name,'wt',encoding='utf-8') as f:
   for r in rr:f.write(json.dumps(r,allow_nan=False)+'\n')
 write(dest/'CPU_COMPLETION.json',dict(status='postseal_score_and_independent_math_state_dense_audit_complete',output=str(out),rows=len(allrows),GT_after_all_stage_predictions=True,
  files={str(f.relative_to(ROOT)):sha(f) for f in out.iterdir() if f.is_file()},time=time.time()))

if __name__=='__main__':
 try:run(sys.argv[1])
 except BaseException:
  import traceback
  fd=BASE/'CPU_failures'/str(time.time_ns());fd.mkdir(parents=True,exist_ok=True);(fd/'traceback.txt').write_text(traceback.format_exc())
  status(BASE/'CPU_STAGE.json',dict(status='failed_preserved',evidence=str(fd),time=time.time()))
  raise
