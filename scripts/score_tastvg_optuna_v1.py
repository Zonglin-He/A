"""Isolated CPU scoring; labels never enter the online process."""
import sys,time,collections,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_optuna_common_v1 import *
def labels(dataset,split,p):
 import numpy as np
 parents=sorted(next(iter(p['splits'][split]['orders'].values())));f=BASE/dataset/f'GT_LABELS_{split}.json'
 if f.exists():return read(f)
 if split=='confirm':assert (BASE/dataset/'SELECTION.json').exists()
 result={}
 if dataset=='vidstg':
  import ijson
  from scripts.prepare_vidstg_wrong_domain_support import load_vidor_annotations,trajectory_for_target
  keys={p['rows'][i]['key'] for i in parents};lp=ROOT/'artifacts/stvg_fullscale_diagnostics_v1/labels_diagnostic_only.json'
  with lp.open('rb') as h:gts={k:v for k,v in ijson.kvitems(h,'',use_float=True) if k in keys}
  ap=ROOT/'external/VidSTG-Dataset/annotations/test_annotations.json';vp=ROOT/'downloads/vidor/validation-annotation.zip';anns=read(ap);raw=load_vidor_annotations(vp,None)
  for i in parents:
   row=p['rows'][i];a=gts[row['key']]['official_annotation'];v=anns[a['annotation_index']];assert v['vid']==row['source'] and v[a['field']][a['query_index']]['description']==row['input']['caption'];span=[v['temporal_gt']['begin_fid'],v['temporal_gt']['end_fid']];truth={}
   for fid,rec in trajectory_for_target(raw[row['source']],a['target_id']).items():
    if span[0]<=int(fid)<span[1]:
     x,y,w,h=rec['bbox'];truth[int(fid)]=[x,y,x+w,y+h]
   assert truth;result[str(i)]=dict(span=span,truth=truth)
  hashes=dict(labels=sha(lp),annotations=sha(ap),vidor=sha(vp))
 else:
  lock=read(ROOT/'artifacts/tastvg_paper48_v1/P5/LOCK.json');ap=Path(lock['metadata_intake_path']);assert sha(ap)==lock['metadata_intake_sha256'];anns=read(ap)
  for i in parents:
   row=p['rows'][i];v=anns[row['annotation_index']];assert v['original_video_id']==row['input']['original_video_id'] and v['caption'].lower()==row['input']['caption'];start=int(v['tube_start_frame'])-1;end=start+len(v['trajectory'])-1;truth={}
   for j,(x,y,w,h) in enumerate(v['trajectory']):truth[start+j]=[x,y,min(x+w,row['input']['width']),min(y+h,row['input']['height'])]
   result[str(i)]=dict(span=[start,end],truth=truth)
  hashes=dict(annotations=sha(ap))
 write(f,result);write(BASE/dataset/f'GT_EXPOSURE_{split}.json',dict(scope='supervised hyperparameter development' if split=='search' else 'confirmation after selected config sealed; not used for selection',sources=len(parents),globally_fresh=False,hashes=hashes,labels_sha256=sha(f),time=time.time()));return result

def run(dataset,request):
 import torch,numpy as np
 from methods.decota_final_simplified_v1.tensors import state_hash
 from vg_tta.tastvg_native_spatial_rollout_s05_v1 import central_state
 from vg_tta.tastvg_paper48_metrics_v1 import DenseMetric,xyxy,source_summary
 from vg_tta.tastvg_paper48_hc2_metrics_v1 import HC2DenseMetric
 from vg_tta.tastvg_paper_readouts_v1 import dense_official_metrics
 from scripts.score_tastvg_schedule_j01_v1 import independent_scores
 torch.set_num_threads(4);p=verify(dataset);out=Path(request).parent;req=read(Path(request));split=req['split'];bar=read(out/'PREDICTION_BARRIER.json');assert bar['cells']==p['splits'][split]['total'];cfg=req['params']
 for f,h in bar['files'].items():assert sha(out/f)==h and sha((out/f).with_suffix('.pt'))==read(out/f)['sha256']
 gts=labels(dataset,split,p);official=DenseMetric() if dataset=='vidstg' else HC2DenseMetric();rows=[];links=updates=teachers=reinserts=0;maxerr=0.;initial_sha=read(out/'SUPPORT.json')['center_sha256']
 for cond in p['conditions']:
  for order,seq in p['splits'][split]['orders'].items():
   previous=initial_sha
   for at,parent in enumerate(seq):
    x=load(out/'online'/cond/order/f'{at:05}.pt');assert x['parent']==parent and x['arrival']==at and x['expert_scheduled']==(at%4==0) and x['GT_read'] is False
    assert previous==x['pre_sha']==state_hash(x['pre_state']) and state_hash(x['post_state'])==x['post_sha'];previous=x['post_sha'];links+=1
    if x['updated']:
     u=x['update'];assert u['lr']==cfg['lr'] and u['teacher_temperature']==cfg['teacher_temperature'];d=np.asarray(u['distances']);pp=np.exp(-d+d.min());pp/=pp.sum();rank=np.asarray(u['rank']);qq=np.exp(-rank/cfg['teacher_temperature']);qq/=qq.sum()
     np.testing.assert_allclose(pp,u['p'],atol=2e-7,rtol=1e-6);np.testing.assert_allclose(qq,u['q'],atol=2e-7,rtol=1e-6);np.testing.assert_allclose(np.sum(pp*np.log(pp/qq)),u['loss_before'],atol=2e-5,rtol=2e-5)
     for n,before in x['pre_state'].items():np.testing.assert_allclose((before.double().numpy()-cfg['lr']*u['gradients'][n].double().numpy()).astype(np.float32),x['post_state'][n],atol=1e-6,rtol=2e-6)
     updates+=1
    else:assert all(torch.equal(v,x['pre_state'][n]) for n,v in x['post_state'].items())
    if x['expert_scheduled']:
     rr=read(BASE/dataset/'experts/temporal'/cond/f'{parent:05}.json');assert rr['pixel_sha256']==x['pixel_sha256'];e=load(BASE/dataset/'experts'/rr['cache']);sc=independent_scores(x['temporal']['candidates'],e);np.testing.assert_allclose(sc,x['temporal']['scores'],atol=1e-14,rtol=0);assert int(np.argmax(sc))==x['temporal']['selected'];teachers+=1
    if x['reinsertion']:assert x['reinsertion']['full_pipeline_exact'];reinserts+=1
    row=p['rows'][parent];ids=row['frame_ids'];gt=gts[str(parent)];gt['truth']={int(k):v for k,v in gt['truth'].items()};result=dict(parent=parent,order=order,condition=cond,arrival=at,expert_scheduled=x['expert_scheduled'],updated=x['updated'])
    for arm,boxes,idx in [('Frozen',x['source_native']['boxes'],x['source_native']['indices']),('Ours',x['slow']['boxes'],x['final_indices'])]:
     pix=xyxy(boxes,row['input']['width'],row['input']['height']);pix=np.maximum(pix,0) if dataset=='hc2' else pix;interval=[ids[idx[0]],ids[idx[1]]+1];z=official(pix,ids,interval,gt['truth'],gt['span']);ind=dense_official_metrics(pix,ids,interval,gt['truth'],gt['span'])
     for k in ['m_tIoU','m_vIoU','vIoU@0.3','vIoU@0.5']:
      err=abs(float(z[k])-float(ind[k]));assert err<1e-10;maxerr=max(err,maxerr)
     for k,v in z.items():assert np.isfinite(v);result[arm+'_'+k]=float(v)
    for k in ['m_tIoU','m_vIoU','sIoU_dense_GT','vIoU@0.3','vIoU@0.5']:result['delta_'+k]=result['Ours_'+k]-result['Frozen_'+k]
    if not x['expert_scheduled']:assert result['delta_m_tIoU']==0
    rows.append(result)
 fields=[a+'_'+k for a in ['Frozen','Ours','delta'] for k in ['m_vIoU','m_tIoU','sIoU_dense_GT']];summary={g:{sub:source_summary([r for r in rows if (r['condition']=='clean')==(g=='clean') and (sub=='all' or not r['expert_scheduled'])],fields) for sub in ['all','nonexpert']} for g in ['clean','corruption']}
 objective=summary['corruption']['nonexpert']['metrics']['delta_m_vIoU']['mean'];assert np.isfinite(objective)
 if cfg==DEFAULT or split=='confirm':assert reinserts==24
 write(out/'ROWS.json',rows);write(out/'SUMMARY.json',summary);write(out/'AUDIT.json',dict(status='pass',state_links=links,SGD_updates=updates,teacher_checks=teachers,full_reinsertions=reinserts,independent_metric_checks=links*2,max_error=maxerr,all_predictions_before_scoring=True))
 write(out/'SCORE.json',dict(status='completed',objective=objective,objective_pp=objective*100,params=cfg,split=split,cells=len(rows),audit_sha256=sha(out/'AUDIT.json'),prediction_barrier_sha256=sha(out/'PREDICTION_BARRIER.json'),time=time.time()));print('SCORED',dataset,req['tag'],objective*100,flush=True)
if __name__=='__main__':run(*sys.argv[1:])
