"""Own code/config and all anonymous outcomes/process diagnostics only."""
from common132 import *
import csv,io
O132=H132/'publication/public_export_attempt001'
def table(p,rows):
 if not rows:return
 p.parent.mkdir(parents=True,exist_ok=True);out=io.StringIO();w=csv.DictWriter(out,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows);p.write_text(out.getvalue())
def main():
 assert rd(H132/'SCALAR_AUDIT.json')['n']==1798;O132.mkdir(exist_ok=False,parents=True);Q=H132/'scores_attempt001';allrows=[r for p in sorted(Q.glob('*_ALL_ROWS_PRIVATE.json')) for r in rd(p)];frames=sorted({r['frame_PRIVATE'] for r in allrows});contents=sorted({r['content_group_PRIVATE'] for r in allrows});scenes={t:sorted({r['scene_PRIVATE'] for r in allrows if r['target']==t}) for t in ['waymo','drone','quad']};mapping={}
 def anonymous(r):return {'frame_cluster':f'FRAME{frames.index(r["frame_PRIVATE"])+1:04d}','content_cluster':f'CONTENT{contents.index(r["content_group_PRIVATE"])+1:04d}','scene_cluster':f'CLUSTER{scenes[r["target"]].index(r["scene_PRIVATE"])+1:03d}'}
 for p in sorted(Q.glob('*_ALL_ROWS_PRIVATE.json')):
  rows=[]
  for i,r in enumerate(rd(p)):
   d={k:r[k] for k in ['source','target','sampling_weight','old_skip_reason','old_skip_detail','seconds_new']};d.update(row=f'ROW{i+1:04d}',**anonymous(r));mapping[(r['source'],r['target'],r['id_PRIVATE'])]={k:d[k] for k in ['row','frame_cluster','content_cluster','scene_cluster']};d.update({k+'__IoU':v for k,v in r['IoUs'].items()})
   for kind,diag in r['diagnostics'].items():d.update({kind+'__'+k:v for k,v in diag.items()})
   rows.append(d)
  table(O132/'tasks'/p.name.replace('_ALL_ROWS_PRIVATE.json','.csv'),rows)
 for n in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS']:table(O132/'results'/(n+'.csv'),rd(Q/(n+'.json')))
 scene=[]
 for r in rd(Q/'ALL_SCENE_ROWS_PRIVATE.json'):
  d=dict(r);d['scene_cluster']=f'CLUSTER{scenes[r["target"]].index(d.pop("scene_PRIVATE"))+1:03d}';scene.append(d)
 table(O132/'results/ALL_SCENE_ROWS.csv',scene)
 for n in ['PROCESS_SUMMARY','SUBSET_COMPARISONS']:table(O132/'diagnostics'/(n+'.csv'),rd(H132/(n+'.json')))
 for name,rows in [('CHANGED_CASES',rd(H132/'CHANGED_CASES_PRIVATE.json')),('R131_REASON_AND_DECISION_ROWS',rd(H132/'DIAGNOSIS_ROWS_PRIVATE.json'))]:
  pub=[]
  for r in rows:d={k:v for k,v in r.items() if not k.endswith('_PRIVATE')};d.update(mapping[(r['source'],r['target'],r['id_PRIVATE'])]);pub.append(d)
  table(O132/'diagnostics'/(name+'.csv'),pub)
 attempts=[]
 for a in rd(H132/'TARGET_B3_CANDIDATES_PRIVATE.json'):
  sel=rd(B132/'selections_PRIVATE'/('TARGET_'+a['A']['frame_PRIVATE']+'.json'));alias=f'FRAME{frames.index(a["A"]["frame_PRIVATE"])+1:04d}'
  for at in sel['attempts']:
   r=rd(R132/at['path']);d={'target':a['A']['platform'],'frame_cluster':alias,'rank':at['rank'],'status':at['status'],'passed':at['geometry_cycle_supported'],'reused':at['reused'],'new_seconds':at['new_seconds'],'selected':sel['selected_rank']==at['rank']}
   for tag in ['forward','reverse']:
    x=r.get(tag,{});d.update({tag+'__'+k:x.get(k) for k in ['status','n_pairs','n_fit','n_holdout']})
    for key in ['RGB_validation','holdout_residuals']:
     d.update({tag+'__'+key+'__'+k:v for k,v in x.get(key,{}).items() if k in ['n','median_px','median_m','inlier_fraction']})
   for i,x in enumerate(r.get('instrumentation_forward_reverse',[])):
    for k,v in x.items():d[('forward' if i==0 else 'reverse')+'__'+k]=v
   attempts.append(d)
 table(O132/'diagnostics/TARGET_REGISTRATION_ATTEMPTS.csv',attempts)
 source=[]
 for i,r in enumerate(rd(H132/'SOURCE_CALIBRATION_ROWS_PRIVATE.json')):source.append({'source':r['source'],'source_row':f'SRCROW{i+1:04d}','B_available':r['B_available'],'S_groups':r['S_groups'],'eligible':r['eligible'],'gap_A':r['view_gaps'][0] if r['view_gaps'] else None,'gap_B':r['view_gaps'][1] if r['view_gaps'] else None})
 table(O132/'diagnostics/SOURCE_CALIBRATION_ROWS.csv',source)
 for n in ['INPUT_SUMMARY','DIAGNOSIS_SUMMARY','DIAGNOSIS_NO_GT_SEAL','REGISTRATION_SUMMARY','SOURCE_NATIVE_AUDIT','SOURCE_CALIBRATION_AUDIT','LOG_SCOPE_SUMMARY','NATIVE_AUDIT','SCALAR_AUDIT','CASE_SUMMARY','DECISION','ENGINEERING']:
  dump(O132/(n+'.json'),rd(H132/(n+'.json')))
 dump(O132/'SOURCE_LOG_CALIBRATION.json',{k:v for k,v in rd(H132/'SOURCE_LOG_CALIBRATION.json').items() if k!='source_files'})
 for p in (H132/'experiment').glob('*.json'):dump(O132/'config'/p.name,rd(p))
 for name in ['REPORT.md','FINDINGS.md']:(O132/name).write_text((H132/name).read_text())
 for p in list(H132.glob('*.py'))+[R132/'tools/experiment_guard_round132.py']:
  out=O132/'own_code'/p.relative_to(R132);out.parent.mkdir(exist_ok=True,parents=True);out.write_text(p.read_text().replace(str(R132),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 receipts=[]
 for p in (H132/'job_receipts').glob('*.json'):
  r=rd(p);receipts.append({k:r[k] for k in ['resource','started_utc','finished_utc','returncode','max_seconds']}|{'entrypoint':Path(r['command'][1]).name,'engineering':'--engineering' in r['command']})
 dump(O132/'RESOURCE_RECEIPTS.json',receipts)
 (O132/'REPRODUCTION.md').write_text('''# Reproduction and limits

R132 depends on immutable R131 aa2a66ca04d5caf41471e718da2c0e5466d80a12:
studies/3eed_fixed_partial_content_group_expansion_20261117, plus the R130/R129
native source/checkpoint/context dependencies documented there. Upstream3EED:
https://github.com/worldbench/3EED. Licensed media, captions, identifiers,
coordinates, support tensors, features and source/prepared weights are private.

Order: diagnose132 (noGT reason/decision seal, then attach old scalars) ->
prepare132 -> registration code lock -> register132 -> source execution lock ->
source132 -> audit_source132 -> calibrate132 -> scopes132 ->
audit_calibration132 -> target execution lock -> run132 --engineering ->
formal lock -> run132 -> all predictions sealed -> audit132 -> score132 ->
audit_scalars132 -> analyze132 -> report132 -> export132 -> audit_public132.
All models/CPU scoring use experiment_guard_round132.py; one exclusive GPU.

B3 first candidate/old original4results are reused, no old baseline rerun.
Among at most3 preordered distinct candidates, select first passing ORIGINAL
forward/reverse/cycle gates, without scores/GT or threshold relaxation. Every
within-B3 control uses the same selected B. Additional registration costs count.
LOG uses original B ONLY and one source-own real-AB 90percentile log-gap threshold;
no compound B3+LOG arm. Source calibration excludes known protected and R131 A/B
point-content groups before media/model/label reads. All313 source predictions
are frozen and audited before source labels determine kappa. No source weights
are trained; no target GT/calibration/coverage-based parameter choice occurs.

The log policy includes source-AB recalibration; old tau was sourceA-only. A
task difference is not isolated proof that log units alone outperform every
probability-gap calibration. Flat predictions remain unknown; no forced
negative or intersection rule is introduced. Wrong controls preserve S, C
cardinality and group-size counts, but not prior probability mass/q0 inclusion.

All1798 exposed R131 inputs remain, including unknown/no-update/harms. This is
development, not blind independent confirmation or new-scene validation.
Anonymous CONTENT/FRAME/CLUSTER labels preserve dependence across directions;
one Drone recording does not justify scene-population confidence intervals.
Raw target GT/geometry/FINAL are never newly opened or scored; saved R131 all256
IoU vectors are indexed only after the complete target matrix is sealed.
Query switches and IoU switches are separate; geometric groups are not identity
truth. All positive/negative/no-change rows and registration failures are kept.

Only own code, locked protocol and complete anonymous scalar tables are public.
No third-party snapshot, private tensor, raw identifier or credential is exported.
Pre-CAS immutable remote byte/tree/parent audits and archive receipts are retained.
No new experiment, combined configuration or tuning follows automatically.
''')
 dump(O132/'MANIFEST.json',{'prefix':'studies/3eed_update_coverage_B_acquisition_loggap_validation_20261117','files':[{'path':str(p.relative_to(O132)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O132.rglob('*')) if p.is_file()]});print('EXPORT132',len([p for p in O132.rglob('*') if p.is_file()]))
if __name__=='__main__':main()
