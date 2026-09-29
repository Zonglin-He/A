"""All-case root readback, including positive cases, malformed/zero boxes and baseline equality."""
import sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.desta3d_v3_external_policy_native import D,TEACHER,OUT,GAP,read,write,sha,load,equal
from vg_tta.external_qualification_io import verify_seal

def main():
 import torch,numpy as np
 start=time.monotonic();run=OUT/'extgate_policy001';verify_seal(run);report=read(D/'evaluation/REPORT.json');assert read(D/'evaluation/ROOT_SCORE_CROSSCHECK.json')['status']=='passed';rows=read(run/'INPUTS.json');cases=[];checks=0
 for i,row in enumerate(rows):
  ep=run/'episodes'/f'{i:02}';base=load(ep/'B1.pt');old=load(GAP/'episodes'/f"{row['gap_index']:04}"/'B1.pt');tr=load(GAP/'episodes'/f"{row['gap_index']:04}"/'BASE_TRACE.pt')
  for k in ['key','source','video_sha256','frame_ids','positions','boxes_cxcywh','geometry_valid','interval','format_ok','preprocess','adapter_sha','event_completion','spatial_completion','event_logits','readout','support']:assert equal(base[k],old[k]);checks+=1
  assert torch.equal(base['native_time_logits'],tr['branches'][0]['logits']['time']);checks+=1
  out=dict(query=f'Q{i+1:02}',arms={});inp=read(ep/'INPUT.json');ev=read(TEACHER/'episodes'/f'{i:02}'/'EVIDENCE.json')
  out['provider']=dict(temporal_usable=ev['temporal_usable'],spatial_usable=ev['spatial_usable'],format_valid=ev['format_valid'],errors=ev['errors'],possibly_truncated=read(TEACHER/'episodes'/f'{i:02}'/'RAW_PREDICTION.json')['possibly_truncated'])
  for a in ['B1','T','S','TS']:
   p=load(ep/(a+'.pt'));assert p['frame_ids']==base['frame_ids'] and p['physical_support']==base['physical_support'] and p['adapter_sha']==base['adapter_sha'];checks+=3
   assert not p['GT_read'] and not p['decoder_GT_prefix'] and p['optimizer_steps']==0 and not p['target_read'];checks+=4
   rs=report['arms'][a]['rows'][i];bs=report['arms']['B1']['rows'][i];assert rs['key']==bs['key']==row['key']
   out['arms'][a]=dict(metrics_percent={k:100*rs['metrics'][k] for k in ['tIoU','sIoU','vIoU']},delta_pp={k:100*(rs['metrics'][k]-bs['metrics'][k]) for k in ['tIoU','sIoU','vIoU']},format_ok=p['format_ok'],boxes=len(p['positions']),invalid_geometry=int((~p['geometry_valid']).sum()),interval_changed=not equal(p['interval'],base['interval']),semantic_reference_changed=not equal(p['readout']['spatial_reference_token_ids'],base['readout']['spatial_reference_token_ids']),input_pixels_changed=inp['view_pixel_sha'][a]!=inp['view_pixel_sha']['B1'])
  cases.append(out)
 summary={a:dict(format_failures=sum(not x['arms'][a]['format_ok'] for x in cases),invalid_boxes=sum(x['arms'][a]['invalid_geometry'] for x in cases),interval_changed=sum(x['arms'][a]['interval_changed'] for x in cases),reference_changed=sum(x['arms'][a]['semantic_reference_changed'] for x in cases),input_changed=sum(x['arms'][a]['input_pixels_changed'] for x in cases)) for a in ['B1','T','S','TS']}
 write(D/'ROOT_CASE_READBACK.json',dict(status='passed',checks=checks,cases=cases,summary=summary,CPU_seconds=time.monotonic()-start,all16_retained=True,metrics_source='already dual-scored exact16; no new label pool'))
 print('CASES',checks,summary)
if __name__=='__main__':main()
