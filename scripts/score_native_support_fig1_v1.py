"""CPU oracle readout. Global three-backbone prediction seal precedes label I/O."""
import sys,time,collections,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.native_support_fig1_common_v1 import *
from vg_tta.native_support_ratios_v1 import paired_summary,recoverable

def scalar_ci(a):
 a=np.asarray(a,np.float64);assert a.ndim==1
 if not len(a):return dict(mean=None,ci95=None,sources=0)
 rng=np.random.default_rng(20260930);boot=np.concatenate([a[rng.integers(0,len(a),(500,len(a)))].mean(1) for _ in range(20)])
 return dict(mean=float(a.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),sources=len(a))

def read_labels(roster):
 # Called only AFTER global barrier, all local hashes and pixel IDs verified.
 import ijson
 from scripts.prepare_vidstg_wrong_domain_support import load_vidor_annotations,trajectory_for_target
 lp=ROOT/'artifacts/stvg_fullscale_diagnostics_v1/labels_diagnostic_only.json';ap=ROOT/'external/VidSTG-Dataset/annotations/test_annotations.json';vp=ROOT/'downloads/vidor/validation-annotation.zip'
 keys={r['key'] for r in roster['rows']}
 with lp.open('rb') as f:old={k:v for k,v in ijson.kvitems(f,'',use_float=True) if k in keys}
 anns=read(ap);raw=load_vidor_annotations(vp,None);result={}
 for row in roster['rows']:
  a=old[row['key']]['official_annotation'];v=anns[a['annotation_index']];assert v['vid']==row['source'] and v[a['field']][a['query_index']]['description']==row['input']['caption']
  span=[v['temporal_gt']['begin_fid'],v['temporal_gt']['end_fid']];truth={}
  for fid,rec in trajectory_for_target(raw[row['source']],a['target_id']).items():
   if span[0]<=int(fid)<span[1]:
    x,y,w,h=rec['bbox'];truth[int(fid)]=[x,y,x+w,y+h]
  assert truth;result[row['ordinal']]=dict(span=span,truth=truth)
 write(OUT/'GT_EXPOSURE.json',dict(status='read_after_global_prediction_seal',sources=128,historical_exposure=True,hashes={str(f.relative_to(ROOT)):sha(f) for f in [lp,ap,vp]},global_barrier_sha256=sha(OUT/'GLOBAL_PREDICTION_BARRIER.json'),time=time.time()))
 return result

def component_scores(pred,truth,span,width,height,official):
 # Raw spatial support is independent of each temporal candidate. Missing
 # support stays zero in the full fixed-GT-frame denominator.
 cls=official['VidSTGiouEvaluator'];ev=cls.__new__(cls);ev.vid2steds={0:span};ev.vid2box={0:{k:[v] for k,v in truth.items()}};ev.vid2names={0:'anonymous'};ev.vid2sents={0:''};ev.iou_thresholds=[.3,.5]
 ids=np.asarray(pred['box_frame_ids'],int);dense_ids=np.array(sorted(truth));gt=np.array([truth[int(i)] for i in dense_ids],float);out_t=[];out_s=[];coverage=[];invalid=[];maxerr=0.
 for k in range(6):
  interval=pred['intervals'][k] if pred['temporal_valid'][k] else [0,0];interval=list(map(int,interval))
  inter=max(0,min(interval[1],span[1])-max(interval[0],span[0]));den=interval[1]-interval[0]+span[1]-span[0]-inter;ti=inter/den if den>0 else 0.
  tube={};ind=np.zeros(len(gt));covered=np.zeros(len(gt),bool);bad=0
  if len(ids) and pred['spatial_valid'][k]:
   boxes=np.asarray(pred['boxes'][k],float)*np.array([width,height,width,height]);assert boxes.shape==(len(ids),4) and np.isfinite(boxes).all()
   # Official interpolation first; geometry is judged per interpolated frame.
   full=official['linear_interp']({int(i):[b.tolist()] for i,b in zip(ids,boxes)})
   for fid,bb in full.items():
    b=np.asarray(bb[0]);valid=bool(np.isfinite(b).all() and (b[2:]>b[:2]).all())
    if valid:tube[fid]=bb
    else:bad+=1
   covered=(dense_ids>=ids[0])&(dense_ids<=ids[-1]);interp=np.stack([np.interp(dense_ids,ids,boxes[:,j]) for j in range(4)],1)
   geom=(interp[:,2:]>interp[:,:2]).all(1)&covered
   area=np.maximum(interp[:,2:]-interp[:,:2],0).prod(1);ga=np.maximum(gt[:,2:]-gt[:,:2],0).prod(1);ia=np.maximum(np.minimum(interp[:,2:],gt[:,2:])-np.maximum(interp[:,:2],gt[:,:2]),0).prod(1);un=area+ga-ia
   ind[geom]=np.divide(ia[geom],un[geom],out=np.zeros(int(geom.sum())),where=un[geom]>0)
  m=ev.evaluate({0:tube},{0:dict(sted=interval,qtype='declarative')},{},{})[0][0]
  error=max(abs(float(m['tiou'])-ti),abs(float(m['gt_viou'])-float(ind.mean())));assert error<1e-10,error;maxerr=max(maxerr,error)
  out_t.append(ti);out_s.append(float(ind.mean()));coverage.append(float(covered.mean()));invalid.append(bad)
 return dict(temporal=out_t,spatial=out_s,coverage=coverage,invalid_interpolated_box_frames=invalid,max_kernel_error=maxerr)

def run():
 verify()
 lock=read(OUT/'SCORING_IMPLEMENTATION_LOCK.json')
 for rel,h in lock['code_sha256'].items():assert sha(ROOT/rel)==h,rel
 roster=read(OUT/'ROSTER.json');g=read(OUT/'GLOBAL_PREDICTION_BARRIER.json');assert g['cells']==2304 and g['GT_read'] is False
 assert not (OUT/'GT_EXPOSURE.json').exists(),'Preserve prior scoring attempt; root must scope any rerun'
 payloads={};hash_count=0
 for model in MODELS:
  bar=read(OUT/model/'PREDICTION_BARRIER.json');assert sha(OUT/model/'PREDICTION_BARRIER.json')==g['barriers'][model] and bar['cells']==768 and bar['time']<=g['time']
  for rel,h in bar['files'].items():
   f=OUT/model/rel;assert sha(f)==h;z=read(f);assert z['GT_read'] is False and z['parameter_updates']==0 and z['native_parity'];r=roster['rows'][z['ordinal']];assert z['frame_ids']==r['frame_ids']
   shared=read(OUT/'common_pixels'/z['condition']/f"{z['ordinal']:05}.json");assert z['pixel_sha256']==shared['pixel_sha256'] and z['frame_ids']==shared['frame_ids'];payloads[model,z['ordinal'],z['condition']]=z;hash_count+=1
 assert hash_count==2304
 labels=read_labels(roster)
 from vg_tta.tastvg_paper48_metrics_v1 import official_functions
 official=official_functions();allrows=[];summaries={};maxerr=0.
 for model in MODELS:
  rr=[]
  for row in roster['rows']:
   i=row['ordinal'];truth=labels[i]
   for cond in roster['conditions']:
    z=payloads[model,i,cond];v=component_scores(z,truth['truth'],truth['span'],row['input']['width'],row['input']['height'],official);maxerr=max(maxerr,v.pop('max_kernel_error'))
    rec=dict(backbone=model,parent=i,condition=cond,**v,unique_temporal=z['unique_temporal'],unique_spatial=z['unique_spatial'],temporal_valid=z['temporal_valid'],spatial_valid=z['spatial_valid'],format_valid=z['format_valid'],observed_burst_fraction=0. if z['corruption'] is None else z['corruption']['actual_observed_fraction']);rr.append(rec)
  summaries[model]={}
  for group,conditions in [('clean',['clean']),('corruption',[x for x in roster['conditions'] if x!='clean'])]:
   cells=[[next(x for x in rr if x['parent']==i and x['condition']==c) for c in conditions] for i in range(128)]
   ts=np.array([[x['temporal'] for x in src] for src in cells]);ss=np.array([[x['spatial'] for x in src] for src in cells]);result=paired_summary(ts[:,:,0],ts,ss[:,:,0],ss)
   result['baseline_temporal']=scalar_ci(ts[:,:,0].mean(1));result['baseline_spatial']=scalar_ci(ss[:,:,0].mean(1));result['marginal']={}
   for name,a in [('temporal',ts),('spatial',ss)]:
    delta,ratio,eligible=recoverable(a[:,:,0],a);vals=np.array([ratio[j,eligible[j]].mean() for j in range(128) if eligible[j].any()]);result['marginal'][name]={**scalar_ci(vals),'eligible_cells':int(eligible.sum())};result['raw_'+name+'_gain_pp_with_ci']=scalar_ci(100*delta.mean(1))
   flat=[x for src in cells for x in src];result['coverage_mean']=float(np.mean([x['coverage'][0] for x in flat]));result['format_invalid_cells']=sum(not x['format_valid'] for x in flat);result['unique_temporal_mean']=float(np.mean([x['unique_temporal'] for x in flat]));result['unique_spatial_mean']=float(np.mean([x['unique_spatial'] for x in flat]));result['observed_burst_fraction_mean']=float(np.mean([x['observed_burst_fraction'] for x in flat]));summaries[model][group]=result
  allrows+=rr
 write(OUT/'SCALAR_ROWS.json',allrows);write(OUT/'SUMMARY.json',summaries)
 write(OUT/'SCORING_AUDIT.json',dict(status='pass',prediction_payloads=hash_count,component_candidate_scores=hash_count*12,max_independent_kernel_error=maxerr,GT_after_global_seal=True,source_bootstrap=10000,whole_tube_oracle=True,time=time.time()))
 set_status(dict(status='scored_pending_root_audit_figure_publication',model_predictions=2304,GT_read=True,time=time.time()));print('Scored2304; root audit/figure/publication pending')
if __name__=='__main__':run()
