"""CPU-only score after ALL 256 trial predictions have been sealed."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;TRIAL=HERE.parent
ROOT=Path(__import__('os').environ['STVG_PROJECT_ROOT']).resolve();sys.path.insert(0,str(HERE))
from stvg.metrics import DenseTube,box_iou,official
from stvg.metrics_base import xyxy
from stvg.evaluation import add_reporting_metrics,summarize_rows

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 seal=read(TRIAL/'scope_hc2/sealed.json');run=read(TRIAL/'run.json');manifest=ROOT/'data/evaluation/development.jsonl'
 assert run['status']=='completed_hc2_sealed_by_user_scope' and seal['GT_read'] is False and seal['queries']==128 and seal['manifest_sha256']==sha(manifest)
 assert seal['fingerprint']==run['fingerprint']
 records=[json.loads(s)for s in manifest.read_text().splitlines()if s.strip() and json.loads(s)['dataset']=='hc2'];assert len(records)==128
 for r in records:
  key=r['id'];entry=seal['files'][key];p=TRIAL/entry['path'];assert sha(p)==entry['sha256'];receipt=read(p.with_suffix('.json'));assert receipt['fingerprint']==seal['fingerprint'] and receipt['GT_read'] is False
 labels_path=ROOT/'data/evaluation/development_labels.jsonl';labels={r['id']:r for r in [json.loads(s)for s in labels_path.read_text().splitlines()if s.strip()]};rows=[];maxerr=0
 for r in records:
  key=r['id'];g=labels[key];meta=r['metadata'];truth={int(k):v for k,v in g['boxes'].items()};span=g['interval'];values={}
  with np.load(TRIAL/seal['files'][key]['path'],allow_pickle=False)as z:
   ids=z['frame_ids'].tolist();pos=z['positions8'].tolist();iv=z['Strong_interval'].tolist();row=dict(input=meta,frame_ids=ids);observed={ids[p]for p in pos};fixed=[f for f in ids if f in truth];obs=[f for f in fixed if f in observed];unobs=[f for f in fixed if f not in observed];lookup={f:i for i,f in enumerate(ids)}
   for arm,name in [('Source','Source'),('WithoutOPD','E_Ref'),('GeometryOPD','GeometryOPD'),('Full','LayoutOPD'),('LayoutDirect','LayoutDirect')]:
    b=z[name];d=DenseTube(b,row,truth,span,clip=r['dataset']=='hc2');v=d.score(iv);check=official(b,row,truth,span,iv,r['dataset']);maxerr=max(maxerr,max(abs(v[k]-check[k])for k in ['v','t','s']))
    pix=xyxy(b,meta['width'],meta['height']);pix=np.maximum(pix,0)if r['dataset']=='hc2'else pix
    for field,frames in [('fixed_actual_sIoU',fixed),('observed_actual_sIoU',obs),('unobserved_actual_sIoU',unobs)]:v[field]=float(np.mean([box_iou(pix[lookup[f]],truth[f])for f in frames]))if frames else None
    v['boundary_MAE']=(abs(iv[0]-span[0])+abs(iv[1]-span[1]))/2;values[arm]=add_reporting_metrics(v,meta)
  rows.append(dict(id=key,parent=g['parent'],dataset=r['dataset'],values=values,development_exposure='exposed_development'))
 assert maxerr<2e-10
 # summarize_rows uses numpy scalar counts: serialize them explicitly, not by relaxing metrics.
 result=dict(status='completed_scored_sealed_development',comparison='Full means LayoutOPD; same Strong interval for all arms',queries=128,datasets=summarize_rows(rows),rows=rows,seal_sha256=sha(TRIAL/'scope_hc2/sealed.json'),labels_sha256=sha(labels_path),official_dense_max_absolute_error=maxerr,config=read(TRIAL/'config.json'),paired={})
 rng=np.random.default_rng(20261021)
 for ds in ['hc2']:
  rr=[r for r in rows if r['dataset']==ds];assert len(rr)==len({r['parent']for r in rr})==128;idx=rng.integers(0,128,(10000,128));out={}
  for a,b in [('Full','GeometryOPD'),('Full','WithoutOPD'),('Full','LayoutDirect'),('LayoutDirect','WithoutOPD')]:
   stat={}
   for k in ['v','s','unobserved_actual_sIoU']:
    d=np.array([r['values'][a][k]-r['values'][b][k]for r in rr])*100;stat[k]=dict(mean_pp=float(d.mean()),CI95_pp=np.quantile(d[idx].mean(1),[.025,.975]).tolist(),positive=int((d>1e-8).sum()),negative=int((d < -1e-8).sum()),below_minus5=int((d < -5).sum()))
   out[a+'-'+b]=stat
  result['paired'][ds]=out
 def default(x):
  if isinstance(x,np.generic):return x.item()
  raise TypeError(type(x).__name__)
 (TRIAL/'scope_hc2/metrics.json').write_text(json.dumps(result,indent=2,default=default)+'\n')
 for ds,g in result['datasets'].items():print(ds,g['query_main_table']);print('paired',result['paired'][ds])
 print('Scored HC128 only after complete seal; no model calls')
if __name__=='__main__':main()
