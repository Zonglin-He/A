"""Official metric bridge and parent-level paired aggregation, separate from inference."""
import json,hashlib
from pathlib import Path
import numpy as np
from .metrics import DenseTube,official,box_iou
from .metrics_base import xyxy

def read_rows(path):return [json.loads(s)for s in Path(path).read_text().splitlines()if s.strip()]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def score(manifest,labels,predictions,output):
 records=read_rows(manifest);root=Path(predictions);seal=json.loads((root/'sealed.json').read_text())
 if seal['manifest_sha256']!=sha(manifest) or len(records)!=seal['queries'] or {r['id']for r in records}!=set(seal['files']):raise ValueError('Prediction seal does not match the evaluation manifest')
 # Verify every prediction before opening any labels, including later queries.
 for entry in seal['files'].values():
  candidate=(root/entry['path']).resolve()
  if not candidate.is_relative_to(root.resolve()) or sha(candidate)!=entry['sha256']:raise ValueError('Prediction content does not match its seal')
 truth={r['id']:r for r in read_rows(labels)}
 rows=[];checks=0
 for r in records:
  key=r['id'];assert key in truth;g=truth[key];assert g['dataset']==r['dataset'];meta=r['metadata'];dataset=r['dataset'];path=root/seal['files'][key]['path'];assert sha(path)==seal['files'][key]['sha256'];gt={int(k):v for k,v in g['boxes'].items()};span=g['interval'];values={}
  with np.load(path,allow_pickle=False)as z:
   ids=z['frame_ids'].tolist();assert ids==meta['frame_ids'];positions=z['positions8'].tolist();fixed=[f for f in ids if f in gt];observed={ids[p]for p in positions};obs=[f for f in fixed if f in observed];unobs=[f for f in fixed if f not in observed];lookup={f:i for i,f in enumerate(ids)};row=dict(input=meta,frame_ids=ids)
   for arm,b,kind in [('Source',z['Source'],'Native'),('Spatial',z['Live'],'Native'),('Temporal',z['Source'],'Strong'),('Full',z['Live'],'Strong'),('WithoutOPD',z['E_Ref'],'Strong')]:
    iv=z[kind+'_interval'].tolist();v=official(b,row,gt,span,iv,dataset);dense=DenseTube(b,row,gt,span,clip=dataset=='hc2').score(iv);assert max(abs(v[k]-dense[k])for k in ['v','t','s'])<2e-10;checks+=1;pix=xyxy(b,meta['width'],meta['height']);pix=np.maximum(pix,0)if dataset=='hc2'else pix;node={f:float(box_iou(pix[lookup[f]],gt[f]))for f in fixed}
    for name,group in [('fixed_actual_sIoU',fixed),('observed_actual_sIoU',obs),('unobserved_actual_sIoU',unobs)]:v[name]=float(np.mean([node[f]for f in group]))if group else None
    v['boundary_MAE']=(abs(iv[0]-span[0])+abs(iv[1]-span[1]))/2;values[arm]=v
  rows.append(dict(id=key,parent=g['parent'],dataset=dataset,values=values))
 summaries={}
 for ds in sorted({r['dataset']for r in rows}):
  rr=[r for r in rows if r['dataset']==ds];parents=list(dict.fromkeys(r['parent']for r in rr));arms=list(rr[0]['values']);fields=list(rr[0]['values'][arms[0]]);matrix={a:{k:np.asarray([np.mean([r['values'][a][k]for r in rr if r['parent']==p and r['values'][a][k]is not None])if any(r['parent']==p and r['values'][a][k]is not None for r in rr)else np.nan for p in parents])for k in fields}for a in arms};means={a:{k:float(np.nanmean(x))for k,x in m.items()}for a,m in matrix.items()};paired={};rng=np.random.default_rng(20261020);indices=rng.integers(0,len(parents),(10000,len(parents)))
  for a,b in [('Full','Source'),('Full','WithoutOPD')]:
   d=100*(matrix[a]['v']-matrix[b]['v']);paired[a+'-'+b]=dict(mean_pp=float(d.mean()),CI95_pp=np.quantile(d[indices].mean(1),[.025,.975]).tolist(),positive=int((d>1e-10).sum()),negative=int((d< -1e-10).sum()),unchanged=int((abs(d)<=1e-10).sum()),below_minus5=int((d< -5).sum()))
  summaries[ds]=dict(parents=len(parents),queries=len(rr),means=means,paired=paired)
 out=dict(status='scored_sealed_predictions',aggregation='queries within parent, then equal parents; all zero-overlap and no-update parents retained',official_dense_checks=checks,datasets=summaries,rows=rows);dest=Path(output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,indent=2)+'\n');return out
