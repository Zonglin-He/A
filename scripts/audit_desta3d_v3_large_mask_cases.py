"""Root CPU readback: all cases, stock identity, old scale, native endpoint scope."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from vg_tta.desta3d_v3_oracle_io import OUT,PANEL,read,write,sha,tensor_sha
from scripts.desta3d_v3_large_mask import ARMS,SMALL
from scripts.desta3d_v3_privileged_ptd_qualification import equal

def run(name):
 d=OUT/name;o=d/'independent_readback_v1';r=read(o/'REPORT.json');assert read(o/'ROOT_SUMMARY_CROSSCHECK.json')['status']=='passed'
 old=read(SMALL/'independent_readback_v1/REPORT.json');labels=read(PANEL/'SOURCE_RECORDS.json');rows=read(d/'INPUTS.json')
 torch.set_num_threads(4);cases=[];stockchecks=0;neutral=[];fixedspatial=0
 def load(p):return torch.load(p,weights_only=False,map_location='cpu')
 for i,(row,lab) in enumerate(zip(rows,labels)):
  ep=d/'episodes'/f'{i:02}';ident=read(ep/'INPUT.json');e=load(ep/'FULL_ENDPOINT_BASES.pt');base=load(ep/'original.pt')
  assert row['key']==lab['key'] and tensor_sha(e['stock'])==ident['support']['visual_grid'];stockchecks+=1
  comp=read(ep/'COMPLETE.json');ap={a:comp['norms'][a]['applicable'] for a in ARMS[1:]}
  assert ap['event_large_correct']==ap['event_large_wrong'] and ap['spatial_large_correct'] and ap['spatial_large_wrong']
  if not ap['event_large_correct']:neutral.append(row['key'].split(':')[-1])
  case={}
  for arm in ARMS:
   p=load(ep/(arm+'.pt'));v=r['arms'][arm]['rows'][i];assert p['key']==v['key']==row['key']
   ids=p['frame_ids'];iv=p['interval'];direct_t=None
   if isinstance(iv,(tuple,list)) and len(iv)==2 and 0<=iv[0]<=iv[1]<len(ids):
    lo,hi=ids[iv[0]],ids[iv[1]]+1;gl,gh=lab['event_interval']['begin_fid'],lab['event_interval']['end_fid']
    direct_t=max(0,min(hi,gh)-max(lo,gl))/max(1,max(hi,gh)-min(lo,gl))
    if p['format_ok']:assert abs(direct_t-v['metrics']['tIoU'])<1e-12
   if arm.startswith('spatial'):
    assert p['interval']==base['interval'] and equal(p['time_distribution'],base['time_distribution']) and p['event_completion']==base['event_completion'];fixedspatial+=1
   smallarm=arm.replace('_large_','_late_')
   oldrow=old['arms'][smallarm]['rows'][i];assert oldrow['key']==row['key']
   diff={m:100*(v['metrics'][m]-oldrow['metrics'][m]) for m in ['tIoU','sIoU','vIoU']}
   if arm=='original':assert max(abs(x) for x in diff.values())<1e-10
   case[arm]=dict(metrics=v['metrics'],delta_pp_from_small_late=diff,format_ok=p['format_ok'],interval=iv,
       original_interval=base['interval'],direct_endpoint_tIoU=direct_t,
       direct_endpoint_note='Descriptive temporal interval readback only; official whole-tube failure scoring is unchanged',
       event_exactly_base=equal(p['time_distribution'],base['time_distribution']) and p['event_completion']==base['event_completion'],
       parsed_positions=p['positions'],positions_equal_base=p['positions']==base['positions'],
       reference_equal_base=p['readout']['spatial_reference_token_ids']==base['readout']['spatial_reference_token_ids'],
       coordinates_changed=int((p['boxes_cxcywh']!=base['boxes_cxcywh']).sum()) if p['positions']==base['positions'] and p['boxes_cxcywh'].shape==base['boxes_cxcywh'].shape else None)
  cases.append(dict(key=row['key'],source=row['source'],index=i,arms=case,norms=comp['norms']))
 assert sorted(neutral)==['15560','28649','4591']
 summary={}
 for arm in ARMS:
  vv=[c['arms'][arm] for c in cases]
  summary[arm]=dict(format_valid=sum(x['format_ok'] for x in vv),interval_changes=sum(x['interval']!=x['original_interval'] for x in vv),
   coordinate_changed_same_parsed_support=sum((x['coordinates_changed'] or 0)>0 for x in vv),
   reference_changes=sum(not x['reference_equal_base'] for x in vv),
   mean_delta_pp_from_small_late={m:float(np.mean([x['delta_pp_from_small_late'][m] for x in vv])) for m in ['tIoU','sIoU','vIoU']},
   direct_endpoint_tIoU_mean=float(np.mean([x['direct_endpoint_tIoU'] if x['direct_endpoint_tIoU'] is not None else 0 for x in vv])))
  if arm!='original':
   nn=[c['norms'][arm] for c in cases if c['norms'][arm]['applicable']]
   summary[arm]['requested_ratio']=nn[0]['requested_ratio'];summary[arm]['applicable']=len(nn)
   for k in ['realized_ratio','scale','raw_norm','stock_norm']:
    summary[arm][k+'_range']=[min(x[k] for x in nn),max(x[k] for x in nn)]
   inj=[r['diagnostics'][i]['arms'][arm]['injection'].get(arm.split('_')[0]) for i in range(16)]
   summary[arm]['BF16_changed_fraction_range']=[min(x['fraction'] for x in inj if x),max(x['fraction'] for x in inj if x)]
   summary[arm]['BF16_relative_delta_range']=[min(x['delta_norm']/cases[i]['norms'][arm]['stock_norm'] for i,x in enumerate(inj) if x),max(x['delta_norm']/cases[i]['norms'][arm]['stock_norm'] for i,x in enumerate(inj) if x)]
 failed_replay={}
 for arm in ['original','event_large_correct']:
  first=load(OUT/'large_mask001/episodes/00'/(arm+'.pt'));replay=load(d/'episodes/00'/(arm+'.pt'))
  failed_replay[arm]=equal(first,replay);assert failed_replay[arm]
 result=dict(status='passed',failed_partial_native_exact_replay=failed_replay,report_sha=sha(o/'REPORT.json'),stock_identity_checks=stockchecks,fixed_spatial_event_checks=fixedspatial,
  neutral_time_cases=neutral,summary=summary,cases=cases,new_GPU=False,new_target=False,
  limitations='All16 exposed source cases; old/new magnitude comparison is no new inference. Box/endpoint changes remain jointly conditioned for event interventions. Whole-format failures score zero by unchanged contract.')
 write(d/'ROOT_CASE_AND_MAGNITUDE_READBACK.json',result)
 print({k:v for k,v in result.items() if k not in ['cases']})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(p.parse_args().name)
