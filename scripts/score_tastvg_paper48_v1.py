"""Post-panel-barrier CPU scoring using official dense geometry plus historical metrics."""
import sys,time,json,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from scripts.tastvg_paper48_common_v1 import BASE,read,write,load,sha,status,verify
from methods.decota_final_simplified_v1.tensors import state_hash
from scripts.analyze_spatial10_components_v1 import checked_score
from scripts.score_tastvg_schedule_j01_v1 import independent_scores
from scripts.score_tastvg_spatial_online_opd_s1_v1 import write_exposure
from vg_tta.tastvg_paper48_metrics_v1 import DenseMetric,xyxy,source_summary
from vg_tta.tastvg_paper_readouts_v1 import dense_official_metrics

def run(panel):
 import ijson
 from scripts.prepare_vidstg_wrong_domain_support import load_vidor_annotations,trajectory_for_target
 torch.set_num_threads(4);out=BASE/panel;p=verify(panel);bar=read(out/'PREDICTION_BARRIER.json');assert bar['cells']==p['total']==len(bar['files'])
 for f,h in bar['files'].items():assert sha(out/f)==h,f
 parents=sorted(set(next(iter(p['orders'].values()))));keys={p['rows'][i]['key'] for i in parents};lp=ROOT/'artifacts/stvg_fullscale_diagnostics_v1/labels_diagnostic_only.json'
 with lp.open('rb') as f:gt={k:v for k,v in ijson.kvitems(f,'',use_float=True) if k in keys}
 assert set(gt)==keys
 ap=ROOT/'external/VidSTG-Dataset/annotations/test_annotations.json';vp=ROOT/'downloads/vidor/validation-annotation.zip'
 write_exposure(out/'GT_EXPOSURE.json',dict(time=time.time(),queries=len(keys),sources=len(parents),globally_fresh=False,all_panel_predictions_before_GT=True,prediction_barrier_sha256=sha(out/'PREDICTION_BARRIER.json'),labels_sha256=sha(lp),official_annotation_sha256=sha(ap),vidor_sha256=sha(vp)))
 annotations=read(ap);raw=load_vidor_annotations(vp,None);dense={};spans={}
 for i in parents:
  row=p['rows'][i];g=gt[row['key']];a=g['official_annotation'];v=annotations[a['annotation_index']];assert v['vid']==row['source'] and v[a['field']][a['query_index']]['description']==row['input']['caption'];assert v[a['field']][a['query_index']]['target_id']==a['target_id']
  spans[i]=[v['temporal_gt']['begin_fid'],v['temporal_gt']['end_fid']];track=trajectory_for_target(raw[row['source']],a['target_id']);truth={}
  for fid,rec in track.items():
   if spans[i][0]<=int(fid)<spans[i][1]:
    x,y,w,h=rec['bbox'];truth[int(fid)]=[x,y,x+w,y+h]
  assert truth;dense[i]=truth
  for j,fid in enumerate(row['frame_ids']):
   if g['valid'][j]:np.testing.assert_allclose(xyxy([g['boxes'][j]],row['input']['width'],row['input']['height'])[0],truth[fid],atol=1e-7,rtol=0)
 del raw,annotations
 initial=load(ROOT/'artifacts/tastvg_native_spatial_rollout_s05_v1/PARAMETER_SUPPORT.pt')['center'];official=DenseMetric();results=[];links=updates=teacherchecks=0;max_geometry=0.
 for cond in p['conditions']:
  for order,seq in p['orders'].items():
   previous=initial
   for arrival,parent in enumerate(seq):
    row=p['rows'][parent];f=out/'online'/cond/order/f'{arrival:05}.pt';receipt=read(f.with_suffix('.json'));assert sha(f)==receipt['sha256'];x=load(f)
    expected=p['availability']==100 or (p['availability']==25 and arrival%4==0)
    assert x['parent']==parent and x['arrival']==arrival and x['expert_scheduled']==expected and x['first_query_of_source']
    assert state_hash(previous)==x['pre_sha']==state_hash(x['pre_state']) and state_hash(x['post_state'])==x['post_sha'];previous=x['post_state'];links+=1
    if x['updated']:
     u=x['update'];d=np.asarray(u['distances'],float);pp=np.exp(-d+d.min());pp/=pp.sum();q=np.exp(-np.asarray(u['rank']));q/=q.sum();assert abs(float(np.sum(pp*np.log(pp/q)))-u['loss_before'])<2e-6
     np.testing.assert_allclose(pp,u['p'],atol=1e-7,rtol=0);np.testing.assert_allclose(q,u['q'],atol=1e-7,rtol=0)
     for name,before in x['pre_state'].items():np.testing.assert_allclose((before.double().numpy()-.005*u['gradients'][name].double().numpy()).astype(np.float32),x['post_state'][name],atol=1.5e-7,rtol=0)
     updates+=1
    else:assert all(torch.equal(v,x['pre_state'][k]) for k,v in x['post_state'].items())
    if expected:
     rr=read(BASE/'experts/temporal'/cond/f'{parent:05}.json');cf=BASE/'experts'/rr['cache'];assert sha(cf)==rr['cache_sha256'];e=load(cf);td=x['temporal'];sc=independent_scores(td['candidates'],e);np.testing.assert_allclose(sc,td['scores'],atol=1e-14,rtol=0);assert int(np.argmax(sc))==td['selected'] and x['final_indices']==td['candidates'][td['selected']]['indices'];teacherchecks+=1
    result=dict(parent=parent,order=order,condition=cond,arrival=arrival,expert_scheduled=expected,quartile=min(3,4*arrival//len(seq)),query_type=row['query_type'],updated=x['updated'],drift=float(torch.sqrt(sum((v-initial[n]).square().sum() for n,v in previous.items()))))
    for arm,boxes,idx in [('Frozen',x['source_native']['boxes'],x['source_native']['indices']),('Ours',x['slow']['boxes'],x['final_indices'])]:
     m,_=checked_score(boxes,gt[row['key']],row['frame_ids'],idx);ids=row['frame_ids'];interval=[ids[idx[0]],ids[idx[1]]+1];pix=xyxy(boxes,row['input']['width'],row['input']['height']);dm=official(pix,ids,interval,dense[parent],spans[parent]);ind=dense_official_metrics(pix,ids,interval,dense[parent],spans[parent])
     for key in ['m_tIoU','m_vIoU','vIoU@0.3','vIoU@0.5']:
      err=abs(float(dm[key])-float(ind[key]));assert err<1e-10,(panel,key,err);max_geometry=max(max_geometry,err)
     for key,value in {**dm,'sIoU_sampled':m['sIoU'],'vIoU_sampled':m['vIoU_corrected']}.items():result[arm+'_'+key]=float(value)
    for key in ['m_tIoU','m_vIoU','vIoU@0.3','vIoU@0.5','sIoU_dense_GT','sIoU_sampled','vIoU_sampled']:result['delta_'+key]=result['Ours_'+key]-result['Frozen_'+key]
    if p['availability']==0:assert all(result[k]==0 for k in result if k.startswith('delta_'))
    results.append(result)
   status(out/'SCORE_STATUS.json',dict(status='running',scored=len(results),total=p['total']))
 assert links==p['total'];write(out/'ROWS.json',results)
 fields=[a+'_'+k for a in ['Frozen','Ours','delta'] for k in ['m_tIoU','m_vIoU','vIoU@0.3','vIoU@0.5','sIoU_dense_GT','sIoU_sampled']]
 groups={'clean':[r for r in results if r['condition']=='clean'],'corruption':[r for r in results if r['condition']!='clean']}
 groups.update({c:[r for r in results if r['condition']==c] for c in p['conditions'] if c!='clean'})
 summary={g:{sub:source_summary([r for r in rr if sub=='all' or not r['expert_scheduled']],fields) for sub in ['all','nonexpert']} for g,rr in groups.items()}
 paired=[]
 for parent in parents:
  for order in p['orders']:
   rr=[r for r in results if r['parent']==parent and r['order']==order];c=next(r for r in rr if r['condition']=='clean');bad=[r for r in rr if r['condition']!='clean'];paired.append(dict(parent=parent,order=order,delta_excess=float(np.mean([r['delta_m_vIoU'] for r in bad])-c['delta_m_vIoU'])))
 write(out/'SUMMARY.json',summary);write(out/'CORRUPTION_EXCESS.json',source_summary(paired,['delta_excess']))
 write(out/'QUARTILES.json',{str(k):source_summary([r for r in groups['corruption'] if r['quartile']==k],['delta_m_vIoU','drift']) for k in range(4)})
 write(out/'AUDIT.json',dict(status='pass',state_links=links,SGD_updates=updates,teacher_checks=teacherchecks,independent_dense_metric_checks=links*2,maximum_dense_discrepancy=max_geometry,GT_after_panel_barrier=True,official_functions_executed=True,frozen_method_unchanged=True))
 lines=[f'# Paper48 {panel}: Frozen vs Ours','',f"{p['sources']} hash-selected sources, one query/source, two orders; availability {p['availability']}%. Historical project exposure disclosed. Official dense metrics and sampled continuity metrics kept separate.",'','| Group | Arm | m_tIoU % | m_vIoU % | vIoU@.3 % | vIoU@.5 % |','|---|---|---:|---:|---:|---:|']
 for group in ['clean','corruption']:
  for arm in ['Frozen','Ours']:
   vals=[summary[group]['all']['metrics'][arm+'_'+k] for k in ['m_tIoU','m_vIoU','vIoU@0.3','vIoU@0.5']];lines.append('| '+group+' | '+arm+' | '+' | '.join(f"{100*v['mean']:.4f} ± {100*v['order_sample_SD']:.4f}" for v in vals)+' |')
 lines+=['','Future nonexpert, source bootstrap, both-order variation, clean/corruption excess, harm tails and quartile drift are in companion JSON. Orders/subsets reuse sources; not independent cohorts. No tuning or method promotion.','']
 (out/'REPORT.md').write_text('\n'.join(lines));write(out/'COMPLETION.json',dict(status='completed',publication='pending',audit_sha256=sha(out/'AUDIT.json'),time=time.time()));status(out/'SCORE_STATUS.json',dict(status='completed',scored=links));print('COMPLETED',panel,links,flush=True)
if __name__=='__main__':run(sys.argv[1])
