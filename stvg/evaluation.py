"""Official metric bridge and parent-level paired aggregation, separate from inference."""
import json,hashlib
from pathlib import Path
import numpy as np
from .metrics import DenseTube,official,box_iou
from .metrics_base import xyxy

def read_rows(path):return [json.loads(s)for s in Path(path).read_text().splitlines()if s.strip()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fps_value(metadata):
 """Only an explicit, finite positive input FPS can convert physical frames to seconds."""
 value=metadata.get('fps')
 if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:return None
 return float(value)

def add_reporting_metrics(values,metadata):
 """Keep compatibility fields and add per-query official threshold outcomes."""
 result=dict(values);fps=fps_value(metadata)
 result.update(vIoU_at_03=float(values['v']>.3),vIoU_at_05=float(values['v']>.5),
               vIoU_ge_03=float(values['v']>=.3),vIoU_ge_05=float(values['v']>=.5),
               tIoU_ge_05=float(values['t']>=.5),tIoU_eq_0=float(values['t']==0),
               boundary_MAE_seconds=values['boundary_MAE']/fps if fps is not None else None)
 return result

def finite_mean(values):
 valid=[float(x)for x in values if x is not None and np.isfinite(x)]
 return float(np.mean(valid))if valid else None

def main_table(means):
 names={'m_vIoU':'v','vIoU@0.3':'vIoU_at_03','vIoU@0.5':'vIoU_at_05',
        'm_tIoU':'t','m_sIoU':'s',
        'm_sIoU_fixed_actual':'fixed_actual_sIoU',
        'vIoU_ge_03':'vIoU_ge_03','vIoU_ge_05':'vIoU_ge_05',
        'm_sIoU_unobserved_actual':'unobserved_actual_sIoU',
        'm_sIoU_observed_actual':'observed_actual_sIoU',
        'tIoU_ge_05':'tIoU_ge_05','tIoU_eq_0':'tIoU_eq_0',
        'boundary_MAE_seconds':'boundary_MAE_seconds'}
 return {arm:{name:fields.get(field)for name,field in names.items()}for arm,fields in means.items()}

def summarize_rows(rows):
 """Query-weighted main table; separately retain equal-parent historical summaries."""
 summaries={}
 for ds in sorted({r['dataset']for r in rows}):
  rr=[r for r in rows if r['dataset']==ds];parents=list(dict.fromkeys(r['parent']for r in rr))
  groups={p:[r for r in rr if r['parent']==p]for p in parents};arms=list(rr[0]['values']);fields=list(rr[0]['values'][arms[0]])
  query_means={a:{k:finite_mean([r['values'][a].get(k)for r in rr])for k in fields}for a in arms}
  valid_queries={a:{k:sum(r['values'][a].get(k)is not None and np.isfinite(r['values'][a][k])for r in rr)for k in fields}for a in arms}
  matrix={a:{k:np.asarray([finite_mean([r['values'][a].get(k)for r in groups[p]])for p in parents],dtype=float)for k in fields}for a in arms}
  means={a:{k:finite_mean(x)for k,x in m.items()}for a,m in matrix.items()};paired={};query_paired={}
  rng=np.random.default_rng(20261020);indices=rng.integers(0,len(parents),(10000,len(parents)))
  comparisons=[('Full','Source'),('Full','WithoutOPD')]if 'Full'in arms else [('BoxFixed','Source'),('BoxFixed','Initialization')]
  for a,b in comparisons:
   d=100*(matrix[a]['v']-matrix[b]['v']);paired[a+'-'+b]=dict(mean_pp=float(d.mean()),CI95_pp=np.quantile(d[indices].mean(1),[.025,.975]).tolist(),positive=int((d>1e-10).sum()),negative=int((d< -1e-10).sum()),unchanged=int((abs(d)<=1e-10).sum()),below_minus5=int((d< -5).sum()))
   query_d=100*np.asarray([r['values'][a]['v']-r['values'][b]['v']for r in rr])
   cluster_sums=np.asarray([sum(100*(r['values'][a]['v']-r['values'][b]['v'])for r in groups[p])for p in parents]);cluster_counts=np.asarray([len(groups[p])for p in parents])
   sampled_means=cluster_sums[indices].sum(1)/cluster_counts[indices].sum(1)
   query_paired[a+'-'+b]=dict(mean_pp=float(query_d.mean()),CI95_pp=np.quantile(sampled_means,[.025,.975]).tolist(),positive=int((query_d>1e-10).sum()),negative=int((query_d< -1e-10).sum()),unchanged=int((abs(query_d)<=1e-10).sum()),below_minus5=int((query_d< -5).sum()),bootstrap='parent-cluster resampling; equal-query estimate')
  strata={}
  for flag in ['current_development_parent','not_current_development_parent','unknown']:
   subset=[r for r in rr if r.get('development_exposure','unknown')==flag]
   if subset:
    sm={a:{k:finite_mean([r['values'][a].get(k)for r in subset])for k in fields}for a in arms}
    strata[flag]=dict(queries=len(subset),parents=len({r['parent']for r in subset}),query_means=sm,query_main_table=main_table(sm))
  summaries[ds]=dict(parents=len(parents),queries=len(rr),means=means,paired=paired,
                    query_means=query_means,query_main_table=main_table(query_means),query_paired=query_paired,
                    query_valid_counts=valid_queries,parent_main_table=main_table(means),
                    development_strata=strata)
 return summaries

def cohort_provenance(records,manifest,development=None):
 """Exposure is determined by input parent IDs, never by performance or GT categories."""
 development=Path(development)if development is not None else Path(__file__).resolve().parents[1]/'data/evaluation/development.jsonl'
 known={};receipt=dict(manifest_sha256=sha(manifest),queries=len(records),
  development_manifest_available=development.is_file(),exposure_scope='current 256-query development manifest only; remaining parents may have earlier exposure')
 if development.is_file():
  dev=read_rows(development);receipt['development_manifest_sha256']=sha(development)
  for r in dev:known.setdefault(r['dataset'],set()).add(str(r['metadata']['source']))
 datasets={}
 for ds in sorted({r['dataset']for r in records}):
  rr=[r for r in records if r['dataset']==ds];parents={str(r['metadata']['source'])for r in rr}
  overlap=parents&known.get(ds,set())
  datasets[ds]=dict(queries=len(rr),parents=len(parents),current_development_parents=len(overlap)if development.is_file()else None,
    current_development_queries=sum(str(r['metadata']['source'])in overlap for r in rr)if development.is_file()else None,
    missing_or_invalid_fps_queries=sum(fps_value(r['metadata'])is None for r in rr))
 receipt['datasets']=datasets
 return receipt,known

def score(manifest,labels,predictions,output):
 records=read_rows(manifest);root=Path(predictions);seal=json.loads((root/'sealed.json').read_text())
 if seal['manifest_sha256']!=sha(manifest) or len(records)!=seal['queries'] or {r['id']for r in records}!=set(seal['files']):raise ValueError('Prediction seal does not match the evaluation manifest')
 # Verify every prediction before opening any labels, including later queries.
 for entry in seal['files'].values():
  candidate=(root/entry['path']).resolve()
  if not candidate.is_relative_to(root.resolve()) or sha(candidate)!=entry['sha256']:raise ValueError('Prediction content does not match its seal')
 provenance,development_parents=cohort_provenance(records,manifest)
 truth={r['id']:r for r in read_rows(labels)}
 rows=[];checks=0
 for r in records:
  key=r['id'];assert key in truth;g=truth[key];assert g['dataset']==r['dataset'];meta=r['metadata'];dataset=r['dataset'];path=root/seal['files'][key]['path'];assert sha(path)==seal['files'][key]['sha256'];gt={int(k):v for k,v in g['boxes'].items()};span=g['interval'];values={}
  with np.load(path,allow_pickle=False)as z:
   ids=z['frame_ids'].tolist();assert ids==meta['frame_ids'];positions=z['positions8'].tolist();fixed=[f for f in ids if f in gt];observed={ids[p]for p in positions};obs=[f for f in fixed if f in observed];unobs=[f for f in fixed if f not in observed];lookup={f:i for i,f in enumerate(ids)};row=dict(input=meta,frame_ids=ids)
   if 'BoxFixed'in z:
    combinations=[('Source',z['Source'],'Native'),('Initialization',z['E_Ref'],'Selected'),('BoxFixed',z['BoxFixed'],'Selected')]
   else:combinations=[('Source',z['Source'],'Native'),('Spatial',z['Live'],'Native'),('Temporal',z['Source'],'Strong'),('Full',z['Live'],'Strong'),('WithoutOPD',z['E_Ref'],'Strong')]
   for arm,b,kind in combinations:
    iv=z[kind+'_interval'].tolist();v=official(b,row,gt,span,iv,dataset);dense=DenseTube(b,row,gt,span,clip=dataset=='hc2').score(iv);assert max(abs(v[k]-dense[k])for k in ['v','t','s'])<2e-10;checks+=1;pix=xyxy(b,meta['width'],meta['height']);pix=np.maximum(pix,0)if dataset=='hc2'else pix;node={f:float(box_iou(pix[lookup[f]],gt[f]))for f in fixed}
    for name,group in [('fixed_actual_sIoU',fixed),('observed_actual_sIoU',obs),('unobserved_actual_sIoU',unobs)]:v[name]=float(np.mean([node[f]for f in group]))if group else None
    v['boundary_MAE']=(abs(iv[0]-span[0])+abs(iv[1]-span[1]))/2;values[arm]=add_reporting_metrics(v,meta)
  exposure=('current_development_parent'if str(meta['source'])in development_parents.get(dataset,set())else'not_current_development_parent')if provenance['development_manifest_available']else'unknown'
  rows.append(dict(id=key,parent=g['parent'],dataset=dataset,fps=fps_value(meta),development_exposure=exposure,values=values))
 summaries=summarize_rows(rows)
 out=dict(status='scored_sealed_predictions',aggregation='queries within parent, then equal parents; all zero-overlap and no-update parents retained',main_table_aggregation='equal queries; official vIoU@ thresholds use strict >; ge fields are supplemental inclusive >=; m_sIoU is dense fixed-GT-frame spatial IoU; all IoU/recall values are fractions, boundary seconds separate',provenance=provenance,official_dense_checks=checks,datasets=summaries,rows=rows);dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,indent=2)+'\n');return out
