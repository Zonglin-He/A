"""Post-seal protocol correction: official7D geometry, preserve invalid9D history."""
from pathlib import Path
import json,hashlib,sys,numpy as np
R=Path(__file__).resolve().parent
for run_name in ['pilot_v1','pilot_v2_scale']:
 run=R/'runs'/run_name;seal=json.loads((run/'sealed.json').read_text())
 for name,h in seal['files'].items():assert hashlib.sha256((run/name).read_bytes()).hexdigest()==h
 original=json.loads((run/'scored.json').read_text());rows=[]
 for row in original['rows']:
  r={k:v for k,v in row.items() if k not in ['iou_full','top10_best_iou_full']};r['iou']=r['iou_yaw_only'];r['independent_vs_native_abs_error']=abs(r['iou']-r['iou_native']);rows.append(r)
 summaries=[]
 for readout in ['text_only','target_class_assisted']:
  for platform in ['all','waymo','drone','quad']:
   for arm in sorted(set(r['arm'] for r in rows)):
    rr=[r for r in rows if r['readout']==readout and r['arm']==arm and (platform=='all' or r['platform']==platform)]
    if not rr:continue
    summaries.append({'readout':readout,'platform':platform,'arm':arm,'n':len(rr),'mean_iou':float(np.mean([r['iou'] for r in rr])),'acc25':float(np.mean([r['iou']>.25 for r in rr])),'acc50':float(np.mean([r['iou']>.5 for r in rr]))})
 base={r['id']:r for r in rows if r['arm']=='baseline' and r['readout']=='text_only'};paired=[]
 for arm in sorted(set(r['arm'] for r in rows)):
  if arm=='baseline':continue
  pairs=[]
  for r in rows:
   if r['arm']!=arm or r['readout']!='text_only':continue
   b=base[r['id']];pairs.append({'id':r['id'],'platform':r['platform'],'baseline_iou':b['iou'],'adapted_iou':r['iou'],'delta_iou':r['iou']-b['iou'],'rescue25':r['iou']>.25 and b['iou']<=.25,'harm25':r['iou']<=.25 and b['iou']>.25,'rescue50':r['iou']>.5 and b['iou']<=.5,'harm50':r['iou']<=.5 and b['iou']>.5})
  paired.append({'arm':arm,'mean_delta_iou':float(np.mean([p['delta_iou'] for p in pairs])),'min_delta_iou':min(p['delta_iou'] for p in pairs),'max_delta_iou':max(p['delta_iou'] for p in pairs),'rescued25':sum(p['rescue25'] for p in pairs),'harmed25':sum(p['harm25'] for p in pairs),'rescued50':sum(p['rescue50'] for p in pairs),'harmed50':sum(p['harm50'] for p in pairs),'pairs':pairs})
 traces=json.loads((run/'trace.json').read_text());gate=[{'id':r['id'],'agreement':r['diagnostics']['gated']['agreement'],'accept':r['diagnostics']['gated']['gate_accept'],'baseline_iou':base[r['id']]['iou'],'baseline_correct25':base[r['id']]['iou']>.25} for r in traces]
 corrected={'measurement':'valid_official_7d_protocol_fixed_pilot','claim_scope':'exploratory9samples_only','supersedes_invalid_full9d_report_sha256':hashlib.sha256((run/'scored.json').read_bytes()).hexdigest(),'reason':'Official loader geometry is bbox[:7]; metadata tail8/9 likely score,class and never validated as pitch,roll. Full9D rotation results invalid and retained only for error lineage. No predictions rerun or changed.','prediction_seal_sha256':hashlib.sha256((run/'sealed.json').read_bytes()).hexdigest(),'independent_native_max_abs_error':max(r['independent_vs_native_abs_error'] for r in rows),'rows':rows,'summary':summaries,'paired':paired,'gate':gate}
 (run/'scored_native.json').write_text(json.dumps(corrected,indent=2))
 (run/'measurement_alert.json').write_text(json.dumps({'measurement_full9d':'invalid','reason':'Misinterpreted9D schema tail as pitch/roll','replacement':'scored_native.json','measurement_7d':'valid_official_protocol','predictions':'unchanged_and_sealed'},indent=2))
 print(run_name,json.dumps([r for r in summaries if r['platform']=='all' and r['readout']=='text_only']))
