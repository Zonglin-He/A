"""Only post-global-barrier: full-query metrics, source/order aggregation and harm tails."""
import sys,json,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.decota_matrix_common_v1 import read,write,load,sha,status
from scripts.run_tastvg_full_b1_experts_v1 import OUT,verify
from scripts.analyze_spatial10_components_v1 import checked_score
from scripts.score_tastvg_spatial_online_opd_s1_v1 import write_exposure
from methods.decota_final_simplified_v1.tensors import state_hash
from scripts.score_tastvg_schedule_j01_v1 import independent_scores
ARMS=['Frozen','Fast-only','Slow-only','Final'];METRICS=['sIoU','tIoU','vIoU_corrected'];SUBSETS=['all','nonexpert','first_source_nonexpert']

def stats(a):
 a=np.asarray(a,float);a=a[np.isfinite(a)];assert len(a)
 rng=np.random.default_rng(20260929);means=[]
 for j in range(0,10000,1000):means.extend(a[rng.integers(0,len(a),(1000,len(a)))].mean(1).tolist())
 return dict(sources=len(a),mean=float(a.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),positive=int((a>0).sum()),negative=int((a<0).sum()),source_harm_gt5pp=int((a<-.05).sum()),min=float(a.min()),max=float(a.max()))

def run():
 import ijson
 torch.set_num_threads(4);tick=time.monotonic();p=verify();bar=read(OUT/'PREDICTION_BARRIER.json');assert bar['cells']==451728 and len(bar['files'])==451728
 # No labels are loaded until the complete prediction receipt inventory is verified.
 for f,h in bar['files'].items():assert sha(OUT/f)==h
 keys={r['key'] for r in p['rows']};lp=ROOT/'artifacts/stvg_fullscale_diagnostics_v1/labels_diagnostic_only.json'
 with lp.open('rb') as f:gt={k:v for k,v in ijson.kvitems(f,'',use_float=True) if k in keys}
 assert set(gt)==keys
 write_exposure(OUT/'GT_EXPOSURE.json',dict(time=time.time(),queries=len(keys),sources=670,all_predictions_before_GT=True,globally_fresh=False,prediction_barrier_sha256=sha(OUT/'PREDICTION_BARRIER.json'),GT_container_sha256=sha(lp)))
 sources=sorted({r['source'] for r in p['rows']});source_index={s:i for i,s in enumerate(sources)};orders=list(p['orders']);conds=p['conditions'];metric_calls=sgd=kl=links=0;maxparam=maxkl=0.;support=load(ROOT/'artifacts/tastvg_native_spatial_rollout_s05_v1/PARAMETER_SUPPORT.pt');initial=support['center']
 # [source,condition,order,subset,arm,metric]. Mean queries before all other averaging.
 sums=np.zeros((670,16,3,3,4,3));counts=np.zeros((670,16,3,3),dtype=np.int64);cellharm=np.zeros((16,3,3,4,3),dtype=np.int64)
 for ci,cond in enumerate(conds):
  for oi,order in enumerate(orders):
   previous=initial;dst=OUT/'metrics'/cond/f'{order}.jsonl';dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_suffix('.working')
   # A completed scored stream can be re-read without relaunching any model.
   if dst.exists():
    streamaudit=read(dst.with_suffix('.audit.json'));assert sha(dst)==streamaudit['metrics_sha256']
    with dst.open() as f:
     for line in f:
      z=json.loads(line);si=source_index[p['rows'][z['parent']]['source']];values=np.array(z['metrics']);delta=values-values[0]
      for sub,keep in enumerate([True,not z['expert_scheduled'],z['first_query_of_source'] and not z['expert_scheduled']]):
       if keep:sums[si,ci,oi,sub]+=values;counts[si,ci,oi,sub]+=1;cellharm[ci,oi,sub]+=(delta<-.05)
    metric_calls+=streamaudit['metric_calls'];sgd+=streamaudit['SGD_coordinates'];kl+=streamaudit['KL_updates'];links+=streamaudit['links'];maxparam=max(maxparam,streamaudit['max_parameter_error']);maxkl=max(maxkl,streamaudit['max_KL_error']);continue
   localcoords=localkl=localmetric=0;localparam=localloss=0.
   with tmp.open('w') as target:
    for arrival,parent in enumerate(p['orders'][order]):
     row=p['rows'][parent];f=OUT/'online'/cond/order/f'{arrival:05}.pt';receipt=read(f.with_suffix('.json'));assert sha(f)==receipt['sha256'];x=load(f)
     assert x['parent']==parent and x['arrival']==arrival and x['expert_scheduled']==(arrival%4==0)
     assert state_hash(previous)==x['pre_sha'] and state_hash(x['pre_state'])==x['pre_sha'] and state_hash(x['post_state'])==x['post_sha'];previous=x['post_state']
     if x['updated']:
      u=x['update'];d=np.asarray(u['distances'],dtype=float);pp=np.exp(-d+d.min());pp/=pp.sum();rank=np.asarray(u['rank']);qq=np.exp(-rank);qq/=qq.sum();err=abs(float(np.sum(pp*np.log(pp/qq)))-u['loss_before']);assert err<2e-6;localloss=max(localloss,err);localkl+=1
      np.testing.assert_allclose(pp,u['p'],atol=1e-7,rtol=0);np.testing.assert_allclose(qq,u['q'],atol=1e-7,rtol=0)
      for name,before in x['pre_state'].items():
       expected=(before.double().numpy()-.005*u['gradients'][name].double().numpy()).astype(np.float32);err=float(np.max(np.abs(expected-x['post_state'][name].numpy())));assert err<1.5e-7;localparam=max(localparam,err);localcoords+=before.numel()
     else:assert all(torch.equal(v,x['pre_state'][k]) for k,v in x['post_state'].items())
     if x['expert_scheduled']:
      er=read(OUT/'temporal'/cond/f'{parent:05}.json');e=load(OUT/er['cache']);assert sha(OUT/er['cache'])==er['cache_sha256']
      for field,indices in [('temporal',x['final_indices']),('fast_temporal',x['fast_indices'])]:
       z=x[field];sc=independent_scores(z['candidates'],e);np.testing.assert_allclose(sc,z['scores'],atol=1e-14,rtol=0);j=int(np.argmax(sc));assert j==z['selected'] and indices==z['candidates'][j]['indices']
     fz=x['source_native'];sl=x['slow'];preds=[(fz['boxes'],fz['indices']),(fz['boxes'],x['fast_indices']),(sl['boxes'],sl['indices']),(sl['boxes'],x['final_indices'])];values=[]
     for boxes,indices in preds:
      m,_=checked_score(boxes,gt[row['key']],row['frame_ids'],indices);values.append([m[k] for k in METRICS]);localmetric+=1
     values=np.array(values);si=source_index[row['source']];delta=values-values[0]
     if not x['expert_scheduled']:assert np.array_equal(values[1],values[0]) and np.array_equal(values[3],values[2])
     for sub,keep in enumerate([True,not x['expert_scheduled'],x['first_query_of_source'] and not x['expert_scheduled']]):
      if keep:sums[si,ci,oi,sub]+=values;counts[si,ci,oi,sub]+=1;cellharm[ci,oi,sub]+=(delta<-.05)
     target.write(json.dumps(dict(parent=parent,arrival=arrival,expert_scheduled=x['expert_scheduled'],first_query_of_source=x['first_query_of_source'],metrics=values.tolist()),separators=(',',':'))+'\n')
   tmp.rename(dst);write(dst.with_suffix('.audit.json'),dict(metrics_sha256=sha(dst),metric_calls=localmetric,links=9411,SGD_coordinates=localcoords,KL_updates=localkl,max_parameter_error=localparam,max_KL_error=localloss));metric_calls+=localmetric;sgd+=localcoords;kl+=localkl;links+=9411;maxparam=max(maxparam,localparam);maxkl=max(maxkl,localloss)
   status(OUT/'SCORE_STATUS.json',dict(status='running',streams_done=ci*3+oi+1,total=48));print('SCORED',ci*3+oi+1,48,flush=True)
 means=np.divide(sums,counts[...,None,None],out=np.full_like(sums,np.nan),where=counts[...,None,None]>0);assert (counts[:,:,:,0]>0).all()
 np.savez_compressed(OUT/'SOURCE_METRICS.npz',means=means,counts=counts,cell_harm=cellharm)
 groups={'corruption':list(range(1,16)),'clean':[0],**{c:[i] for i,c in enumerate(conds) if i}}
 for family in ['frame_drop','frame_freeze','motion_blur','occlusion','exposure']:groups[family]=[i for i,c in enumerate(conds) if c.startswith(family+'_')]
 summary={};source_rows=[]
 for group,idx in groups.items():
  summary[group]={}
  for sub,subset in enumerate(SUBSETS):
   perorder=np.nanmean(means[:,:,:,sub][:,idx],axis=1);primary=np.nanmean(perorder,axis=1);summary[group][subset]={}
   fields={}
   for ai,arm in enumerate(ARMS):
    for mi,metric in enumerate(['s','t','v']):fields[f'{arm}_{metric}']=primary[:,ai,mi];fields[f'{arm}_minus_Frozen_{metric}']=primary[:,ai,mi]-primary[:,0,mi]
   for mi,metric in enumerate(['s','t','v']):fields['Final_minus_Fast_'+metric]=primary[:,3,mi]-primary[:,1,mi]
   for key,value in fields.items():summary[group][subset][key]=stats(value)
   summary[group][subset]['across_order']={f'{arm}_{metric}':dict(values=np.nanmean(perorder[:,:,ai,mi],axis=0).tolist(),sample_std=float(np.nanmean(perorder[:,:,ai,mi],axis=0).std(ddof=1))) for ai,arm in enumerate(ARMS) for mi,metric in enumerate(['s','t','v'])}
   summary[group][subset]['cell_harm']=cellharm[idx,:,sub].sum((0,1)).tolist();summary[group][subset]['query_arrivals']=int(counts[:,idx,:,sub].sum())
   if group in ['clean','corruption']:
    for si in range(670):source_rows.append(dict(source=f'S{si+1:04}',group=group,subset=subset,metrics={k:float(v[si]) if np.isfinite(v[si]) else None for k,v in fields.items()}))
 excess={}
 for sub,subset in enumerate(SUBSETS):
  c=np.nanmean(np.nanmean(means[:,1:,:,sub],axis=1),axis=1);b=np.nanmean(means[:,0,:,sub],axis=1)
  excess[subset]={metric:stats((c[:,3,mi]-c[:,0,mi])-(b[:,3,mi]-b[:,0,mi])) for mi,metric in enumerate(['s','t','v'])}
 write(OUT/'SUMMARY.json',summary);write(OUT/'SOURCE_ROWS.json',source_rows);write(OUT/'CORRUPTION_EXCESS.json',excess)
 write(OUT/'AUDIT.json',dict(status='pass',links=links,streams=48,dual_metric_calls=metric_calls,SGD_coordinates=sgd,KL_updates=kl,max_parameter_error=maxparam,max_KL_error=maxkl,queries=9411,sources=670,all_predictions_sealed_before_GT=True,seconds=time.monotonic()-tick))
 status(OUT/'SCORE_STATUS.json',dict(status='completed',streams_done=48,time=time.time()));print('FULL SCORE COMPLETE',flush=True)
if __name__=='__main__':run()
