"""All16 readback of normalization sensitivity and non-aggregate policy changes."""
import sys,json,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from vg_tta.desta3d_v3_oracle_io import *

def run(name):
 d=OUT/name;verify_seal(d);r=read(d/'independent_readback_v1/REPORT.json');old=OUT/'oracle001';verify_seal(old);rr=read(old/'independent_readback_v1/REPORT.json')
 cases=[]
 for i,row in enumerate(read(d/'INPUTS.json')):
  case=dict(key=row['key'],arms={})
  for b,oldarm in [('event','wrong_temporal'),('spatial','wrong_spatial')]:
   arm=b+'_late_wrong';p=torch.load(d/'episodes'/f'{i:02}'/(arm+'.pt'),map_location='cpu',weights_only=False);q=torch.load(old/'episodes'/f'{i:02}'/(oldarm+'.pt'),map_location='cpu',weights_only=False)
   n=next(z for z in r['raw_normalization'] if z['key']==row['key'] and z['arm']==arm)
   def endpoint(p):
    x=p['time_distribution']['endpoint_logits']
    if x is None:return None
    a=x.double().numpy();order=np.argsort(-a,axis=-1,kind='stable')[:,:2]
    return dict(top2_positions=order.tolist(),top2_values=np.take_along_axis(a,order,axis=-1).tolist(),top1_minus_top2=(np.take_along_axis(a,order,axis=-1)[:,0]-np.take_along_axis(a,order,axis=-1)[:,1]).tolist())
   before=rr['arms'][oldarm]['rows'][i];after=r['arms'][arm]['rows'][i];assert before['key']==after['key']==row['key']
   case['arms'][b]=dict(normalization_scale=n['scale'],old_interval=q['interval'],matched_interval=p['interval'],
    interval_changed=q['interval']!=p['interval'],metric_delta_pp={m:100*(after['metrics'][m]-before['metrics'][m]) for m in ['vIoU','sIoU','tIoU']},old_endpoint=endpoint(q),matched_endpoint=endpoint(p))
  cases.append(case)
 scales={}
 for b in ['event','spatial']:
  for loc in ['late','early']:
   for w in ['correct','wrong']:
    arm=f'{b}_{loc}_{w}';v=[x['scale'] for x in r['raw_normalization'] if x['arm']==arm and x['target_norm']>0]
    scales[arm]=dict(nonzero_support=len(v),min=min(v),median=float(np.median(v)),max=max(v))
 out=dict(status='completed_saved_evidence_only',cases=cases,scales=scales,
  interval_changes={b:sum(x['arms'][b]['interval_changed'] for x in cases) for b in ['event','spatial']},
  interpretation='Norm scaling changes the intervention; matched wrong is not an exact reuse of old wrong. A contrast driven by wrong-control degradation is not improvement over original. No attribution of every discrete change to cast alone.',target_read=False,new_GPU=False)
 write(d/'ROOT_NORM_SENSITIVITY_READBACK.json',out);print(json.dumps({k:v for k,v in out.items() if k!='cases'}))
 for c in cases:
  if c['arms']['event']['interval_changed']:print(c)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(p.parse_args().name)
