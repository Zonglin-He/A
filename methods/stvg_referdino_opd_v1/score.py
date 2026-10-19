"""Independent offline scorer for a sealed prediction bundle.

python -m methods.stvg_referdino_opd_v1.score --barrier manifest.json \
  --labels offline_labels.json --output scores.json

The manifest contains immutable .pt paths and SHA256, not model requests.
Labels are explicit offline pixel xyxy frame boxes, metadata(width/height), and temporal span.
No labels or scoring module are imported by the predictor or run entry.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import torch

def iou(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);a0=a[:,:2]-a[:,2:]/2;a1=a[:,:2]+a[:,2:]/2;b0=b[:,:2]-b[:,2:]/2;b1=b[:,:2]+b[:,2:]/2;inter=np.maximum(0,np.minimum(a1,b1)-np.maximum(a0,b0)).prod(-1);return inter/(a[:,2:].prod(-1)+b[:,2:].prod(-1)-inter)

def score_one(z,labels):
 from vg_tta.tastvg_oracle_event5_v1 import official,DenseTube,box_iou
 from vg_tta.tastvg_paper48_metrics_v1 import xyxy
 ids=np.asarray(z['frame_ids']);truth={int(k):v for k,v in labels['boxes_xyxy_pixels'].items()};row=dict(frame_ids=z['frame_ids'],input=labels['metadata']);span=labels['interval'];known=np.array([int(f)in truth for f in ids]);observed=np.isin(np.arange(len(ids)),z['planned_positions']);out={}
 for arm,boxes in z['outputs'].items():
  b=np.asarray(boxes,float);v=official(b,row,truth,span,z['interval'],'hc2');tube=DenseTube(b,row,truth,span,clip=True);assert max(abs(v[k]-tube.score(z['interval'])[k])for k in ['v','t','s'])<2e-10;pred=np.maximum(xyxy(b,row['input']['width'],row['input']['height']),0);si=np.array([float(box_iou(pred[j:j+1],np.asarray(truth[int(f)],float)[None])[0])if known[j]else np.nan for j,f in enumerate(ids)])
  def avg(mask):return float(si[mask].mean())if mask.any()else None
  out[arm]=dict(**v,observed_actual_s=avg(known&observed),unobserved_actual_s=avg(known&~observed),benchmark_interpolated_s=float(tube.iou[~np.isin(tube.fids,ids)].mean())if (~np.isin(tube.fids,ids)).any()else None)
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--barrier',required=True);p.add_argument('--labels',required=True);p.add_argument('--output',required=True);a=p.parse_args();bar=json.loads(Path(a.barrier).read_text());assert bar['all_before_new_GT'];dst=Path(a.output);assert not dst.exists()
 for f in bar['files']:assert hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()==f['sha256']
 labels=json.loads(Path(a.labels).read_text());rows=[]
 for f in bar['files']:
  z=torch.load(f['path'],map_location='cpu',weights_only=False);rows.append(dict(alias=z['alias'],values=score_one(z,labels[z['alias']])))
 dst.write_text(json.dumps(dict(rows=rows,offline_GT_only=True,normalization='all GT temporal frames; temporal IoU times intersection spatial support'),indent=2))
if __name__=='__main__':main()
