"""Sealed CPU temporal trials on a fixed official parent-disjoint panel."""
import argparse,json,hashlib,os,sys,time
from pathlib import Path
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));PROVIDER=ROOT.parent/'stvg-opd';OUT=ROOT/'outputs/temporal_hc_v7';PUBLIC=ROOT/'results/temporal_hc_v7'
import numpy as np,torch
from stvg.temporal_late_guard import BASE,validate,tables,combine,decode
from stvg.temporal import final_mbr
from stvg.spatial import SpatialGeometry,evidence_dp
from stvg.metrics import official,DenseTube,box_iou
from stvg.metrics_base import xyxy
from stvg.evaluation import add_reporting_metrics,finite_mean,main_table

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');os.replace(t,p)
def inputs(panel):
 protocol=read(OUT/'protocol.json');manifest=OUT/'panel.jsonl';assert sha(manifest)==protocol['panel_manifest_sha256'];rows=[json.loads(s)for s in manifest.read_text().splitlines()];indices=range(128)if panel=='tune'else range(128,163);seal=read(ROOT/'outputs/temporal_hc_v4/captures/sealed.json');assert seal['queries']==163 and sha(ROOT/'outputs/temporal_hc_v4/captures/sealed.json')==protocol['capture_source_seal_sha256']
 return protocol,[(f'P{i+1:03d}',rows[i],seal['files'][f'P{i+1:03d}'])for i in indices]
def trial(name,params,panel):
 torch.set_num_threads(1);validate(params);protocol,rows=inputs(panel);dest=OUT/('trials'if panel=='tune'else'validation')/name;assert not(dest/'lock.json').exists(),'No overwrite'
 if panel=='tune':assert len(list((OUT/'trials').glob('*/sealed.json')))<3
 else:
  assert (OUT/'selected.json').exists()and read(OUT/'selected.json')['params']==params;assert not list((OUT/'validation').glob('*/sealed.json'))
  dest=OUT/'validation'/name
 code={str(p.relative_to(ROOT)):sha(p)for p in [Path(__file__),ROOT/'stvg/temporal_late_guard.py',ROOT/'stvg/temporal.py',ROOT/'stvg/spatial.py',ROOT/'stvg/_spatial_math.py',ROOT/'stvg/temporal_readout.py',ROOT/'stvg/temporal_source_calibration.py']}
 atomic(dest/'lock.json',dict(params=params,panel=panel,protocol_sha256=sha(OUT/'protocol.json'),code=code,GT_read=False));files={};begin=time.perf_counter()
 for key,r,item in rows:
  path=ROOT/'outputs/temporal_hc_v4/captures'/item['path'];assert sha(path)==item['sha256'];cap=torch.load(path,map_location='cpu',weights_only=False);assert cap['GT_read']is False
  td=OUT/'tables'/key/(hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()[:16]+'.npz');td.parent.mkdir(parents=True,exist_ok=True)
  if td.exists():
   with np.load(td,allow_pickle=False)as z:assert str(z['capture_sha256'])==item['sha256']and str(z['readout_sha256'])==sha(ROOT/'stvg/temporal_late_guard.py');t=z['tables'].copy()
  else:
   t=tables(cap['records'],cap['logits'],cap['frame_ids'],params);np.savez(td,tables=t,capture_sha256=item['sha256'],readout_sha256=sha(ROOT/'stvg/temporal_late_guard.py'))
  q=combine(t,cap['frame_ids'],params);span=decode(cap['frame_ids'],q,params)['interval'];q=q[-3:].mean(0)if params['mode']=='last3_mean'else q.mean(0);g=SpatialGeometry.__new__(SpatialGeometry);g.__dict__.update(cap['geometry']);s=evidence_dp(g.for_interval(span));b=s['final'].numpy()
  if params==BASE:assert np.array_equal(b,cap['reference'])and np.array_equal(span,cap['reference_interval'])
  p=dest/'predictions'/(key+'.npz');p.parent.mkdir(parents=True,exist_ok=True);np.savez(p,Candidate=b,Candidate_interval=span,Baseline=cap['reference'],Baseline_interval=cap['reference_interval'],Source=cap['Source'],Source_interval=cap['native_interval'],frame_ids=cap['frame_ids'],positions8=cap['positions8'])
  ids=np.asarray(cap['frame_ids']);L=(ids[None,:]+1-ids[:,None])/cap['fps'];positive=q[q>0]
  files[key]=dict(path=str(p.relative_to(dest)),sha256=sha(p),entropy=float(-(positive*np.log(positive)).sum()),expected_length_seconds=float((q*L).sum()),selected_length_seconds=(span[1]-span[0])/cap['fps'],interval_changed=not np.array_equal(span,cap['reference_interval']))
 assert not torch.cuda.is_initialized();atomic(dest/'sealed.json',dict(queries=len(rows),files=files,params=params,panel=panel,GT_read=False,seconds=time.perf_counter()-begin,lock_sha256=sha(dest/'lock.json')));print(json.dumps(dict(status='sealed',name=name,panel=panel,queries=len(rows),seconds=time.perf_counter()-begin)))

def metric(boxes,r,g,iv,pos):
 meta=r['metadata'];ids=meta['frame_ids'];truth={int(k):v for k,v in g['boxes'].items()};row=dict(input=meta,frame_ids=ids);span=g['interval'];v=official(boxes,row,truth,span,iv,'hc2');dense=DenseTube(boxes,row,truth,span,clip=True).score(iv);err=max(abs(v[k]-dense[k])for k in ['v','t','s']);assert err<2e-10
 pixels=np.maximum(xyxy(boxes,meta['width'],meta['height']),0);fixed=[f for f in ids if f in truth];lookup={f:i for i,f in enumerate(ids)};obs={ids[int(p)]for p in pos}
 for name,group in [('fixed_actual_sIoU',fixed),('observed_actual_sIoU',[f for f in fixed if f in obs]),('unobserved_actual_sIoU',[f for f in fixed if f not in obs])]:v[name]=float(np.mean([box_iou(pixels[lookup[f]],truth[f])for f in group]))if group else None
 fps=meta['fps'];v.update(boundary_MAE=(abs(iv[0]-span[0])+abs(iv[1]-span[1]))/2,start_signed_seconds=(iv[0]-span[0])/fps,end_signed_seconds=(iv[1]-span[1])/fps,start_error_seconds=abs(iv[0]-span[0])/fps,end_error_seconds=abs(iv[1]-span[1])/fps,interval_length_seconds=(iv[1]-iv[0])/fps,GT_length_seconds=(span[1]-span[0])/fps,length_error_seconds=abs(iv[1]-iv[0]-span[1]+span[0])/fps)
 return add_reporting_metrics(v,meta),err

def score(name,panel):
 torch.set_num_threads(1);protocol,rows=inputs(panel);dest=OUT/('trials'if panel=='tune'else'validation')/name;seal=read(dest/'sealed.json');assert not (dest/'score_receipt.json').exists()
 lock=read(dest/'lock.json');assert sha(dest/'lock.json')==seal['lock_sha256'];assert all(sha(ROOT/f)==h for f,h in lock['code'].items());assert seal['queries']==len(rows)
 for e in seal['files'].values():assert sha(dest/e['path'])==e['sha256']
 old=read(PROVIDER/'outputs/boxfixed_full_v1/plan.json');labels=PROVIDER/old['jobs']['hc2']['labels'];assert sha(labels)==protocol['labels_sha256'];wanted={r['id']for _,r,_ in rows};truth={g['id']:g for g in [json.loads(s)for s in labels.read_text().splitlines()]if g['dataset']=='hc2'and g['id']in wanted};assert len(truth)==len(rows)
 result=[];error=0.;checks=0
 for key,r,item in rows:
  with np.load(dest/seal['files'][key]['path'],allow_pickle=False)as z:
   values={}
   for arm in ['Source','Baseline','Candidate']:
    values[arm],e=metric(z[arm],r,truth[r['id']],z[arm+'_interval'].tolist(),z['positions8']);error=max(error,e);checks+=1
   values['Candidate_fixed_space'],e=metric(z['Baseline'],r,truth[r['id']],z['Candidate_interval'].tolist(),z['positions8']);error=max(error,e);checks+=1
  result.append(dict(query=key,parent=str(truth[r['id']]['parent']),values=values,process=seal['files'][key]))
 assert len({r['parent']for r in result})==len(result)
 means={a:{f:finite_mean([r['values'][a][f]for r in result])for f in result[0]['values'][a]}for a in result[0]['values']};paired={}
 for b in ['Baseline','Source']:
  paired['Candidate-'+b]={}
  for f in ['v','t','unobserved_actual_sIoU','observed_actual_sIoU','boundary_MAE_seconds']:
   scale=1 if f.endswith('seconds')else 100;d=np.asarray([scale*(r['values']['Candidate'][f]-r['values'][b][f])for r in result if r['values']['Candidate'][f]is not None and r['values'][b][f]is not None]);ix=np.random.default_rng(20261022).integers(0,len(d),(10000,len(d)));paired['Candidate-'+b][f]=dict(mean=float(d.mean()),CI95=np.quantile(d[ix].mean(1),[.025,.975]).tolist(),improved=int((d>1e-10).sum()),harmed=int((d < -1e-10).sum()),unchanged=int((abs(d)<=1e-10).sum()),declined_over5pp=int((d < -5).sum())if scale==100 else None)
 severe=lambda a:sum(100*(r['values'][a]['v']-r['values']['Source']['v']) < -5 for r in result)
 p=paired['Candidate-Baseline'];ps=paired['Candidate-Source'];short=[z for z in result if z['values']['Source']['GT_length_seconds']<=5];long=[z for z in result if z['values']['Source']['GT_length_seconds']>5];groups=[100*float(np.mean([z['values']['Candidate']['t']-z['values']['Source']['t']for z in g]))for g in [short,long]]
 gates=dict(t_above_Source=ps['t']['mean']>=(.5 if panel=='tune'else 1e-10),t_not_below_Global6=p['t']['mean']>=-1e-10,v_above_Source=ps['v']['mean']>=-1e-10,v_protection=p['v']['mean']>=-.1,unobserved_protection=p['unobserved_actual_sIoU']['mean']>=-.1,severe_protection=severe('Candidate')<=severe('Baseline')+(4 if panel=='tune'else 2))
 gates['short_and_long_strictly_above_Source']=all(v>0 for v in groups)
 main=main_table(means)
 for a in means:
  for f in ['start_signed_seconds','end_signed_seconds','start_error_seconds','end_error_seconds','interval_length_seconds','length_error_seconds','GT_length_seconds']:main[a][f]=means[a][f]
 process={k:float(np.mean([r['process'][k]for r in result]))for k in ['entropy','expected_length_seconds','selected_length_seconds','interval_changed']}
 baseline_strata={}
 # Predefined GT-duration and baseline success slices are diagnosis only, never online rules.
 for label,keep in [('short_event',lambda x:x['values']['Source']['GT_length_seconds']<=5),('long_event',lambda x:x['values']['Source']['GT_length_seconds']>5),('baseline_time_good',lambda x:x['values']['Baseline']['t']>=.5),('baseline_time_bad',lambda x:x['values']['Baseline']['t']<.5)]:
  rr=[r for r in result if keep(r)];baseline_strata[label]=dict(queries=len(rr),t_delta_pp=100*finite_mean([x['values']['Candidate']['t']-x['values']['Baseline']['t']for x in rr])if rr else None)
 anonymous=[dict(query=x['query'],values=x['values'])for x in result]
 report=dict(status='scored',short_long_Source_t_delta_pp=groups,efficiency=dict(CPU_trial_seconds=seal['seconds'],CPU_trial_seconds_per_query=seal['seconds']/len(result),scope='Output-only cached posterior/MBR/spatial-DP and file-write wall; excludes original Source/expert/RGB cost and is not deployment acceleration.'),panel=panel,name=name,params=seal['params'],queries=len(result),parents=len(result),means=main,paired=paired,severe_v_declines_vs_Source=dict(Baseline=severe('Baseline'),Candidate=severe('Candidate')),process=process,diagnostic_strata=baseline_strata,gates=gates,eligible=all(gates.values()),anonymous_queries=anonymous,proof=dict(predictions_sealed_before_GT=True,labels_sha256=sha(labels),seal_sha256=sha(dest/'sealed.json'),official_dense_checks=checks,max_error=error,CUDA_initialized=False),exposure=protocol['exposure'])
 pub=PUBLIC/panel/(name+'.json');atomic(pub,report);atomic(dest/'score_receipt.json',dict(status='scored',public_report_sha256=sha(pub),seal_sha256=sha(dest/'sealed.json')));print(json.dumps(dict(name=name,panel=panel,params=seal['params'],means={a:{k:v[k]for k in ['m_vIoU','m_tIoU','m_sIoU_unobserved_actual','interval_length_seconds','start_signed_seconds','end_signed_seconds']}for a,v in main.items()},delta={b:{f:v['mean']for f,v in d.items()}for b,d in paired.items()},process=process,gates=gates,eligible=report['eligible'],strata=baseline_strata)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['trial','score']);p.add_argument('--name',required=True);p.add_argument('--panel',choices=['tune','validation'],default='tune');p.add_argument('--params');a=p.parse_args();trial(a.name,json.loads(a.params),a.panel)if a.command=='trial'else score(a.name,a.panel)
