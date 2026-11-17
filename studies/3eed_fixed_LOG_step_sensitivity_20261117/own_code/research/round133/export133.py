"""Complete anonymous sensitivity/cost/case tables and own code, no tensor or media export."""
from common133 import *
import csv,io
O133=H133/'publication/public_export_attempt001'
def table(p,rows):
 if not rows:return
 p.parent.mkdir(parents=True,exist_ok=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows);p.write_text(s.getvalue())
def main():
 assert rd(H133/'SCALAR_AUDIT.json')['n']==1798;O133.mkdir(exist_ok=False,parents=True);Q=H133/'scores_attempt001';allrows=[r for p in sorted(Q.glob('*_ALL_ROWS_PRIVATE.json')) for r in rd(p)];frames=sorted({r['frame_PRIVATE'] for r in allrows});contents=sorted({r['content_group_PRIVATE'] for r in allrows});scenes={t:sorted({r['scene_PRIVATE'] for r in allrows if r['target']==t}) for t in ['waymo','drone','quad']};mapping={}
 def anon(r):return {'frame_cluster':f'FRAME{frames.index(r["frame_PRIVATE"])+1:04d}','content_cluster':f'CONTENT{contents.index(r["content_group_PRIVATE"])+1:04d}','scene_cluster':f'CLUSTER{scenes[r["target"]].index(r["scene_PRIVATE"])+1:03d}'}
 for p in sorted(Q.glob('*_ALL_ROWS_PRIVATE.json')):
  rows=[]
  for i,r in enumerate(rd(p)):
   d={k:r[k] for k in ['source','target','sampling_weight','active','seconds_new_joint_work','recoverable_zero_error','S_has_good','C_has_good','zero_already_IoU_oracle','q0_gt_Acc25']};d.update(row=f'ROW{i+1:04d}',**anon(r));mapping[(r['source'],r['target'],r['id_PRIVATE'])]={k:d[k] for k in ['row','frame_cluster','content_cluster','scene_cluster']};d.update({k+'__IoU':v for k,v in r['IoUs'].items()})
   for step,diag in r['diagnostics'].items():d.update({'step'+step+'__'+k:v for k,v in diag.items()})
   rows.append(d)
  table(O133/'tasks'/p.name.replace('_ALL_ROWS_PRIVATE.json','.csv'),rows)
 for n in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS']:table(O133/'results'/(n+'.csv'),rd(Q/(n+'.json')))
 scene=[]
 for r in rd(Q/'ALL_SCENE_ROWS_PRIVATE.json'):
  d=dict(r);d['scene_cluster']=f'CLUSTER{scenes[r["target"]].index(d.pop("scene_PRIVATE"))+1:03d}';scene.append(d)
 table(O133/'results/ALL_SCENE_ROWS.csv',scene)
 for n in ['STEP_SUMMARY','COST_SUMMARY']:table(O133/'diagnostics'/(n+'.csv'),rd(H133/(n+'.json')))
 for n in ['CHANGED_CASES','R132_ALL76_CASES','TIMING_ROWS','PROCESS_ROWS']:
  pub=[]
  for r in rd(H133/(n+'_PRIVATE.json')):
   x={k:v for k,v in r.items() if not k.endswith('_PRIVATE')};x.update(mapping[(r['source'],r['target'],r['id_PRIVATE'])]);pub.append(x)
  table(O133/'diagnostics'/(n+'.csv'),pub)
 for n in ['PREPARATION_SUMMARY','ENGINEERING','NATIVE_AUDIT','SCALAR_AUDIT','CASE_SUMMARY','R132_CASE_SUMMARY','TIMING_SUMMARY','DECISION']:dump(O133/(n+'.json'),rd(H133/(n+'.json')))
 dump(O133/'SOURCE_LOG_CALIBRATION_REUSED.json',{k:v for k,v in rd(R133/'research/round132/SOURCE_LOG_CALIBRATION.json').items() if k!='source_files'})
 for p in (H133/'experiment').glob('*.json'):dump(O133/'config'/p.name,rd(p))
 for n in ['REPORT.md','FINDINGS.md']:(O133/n).write_text((H133/n).read_text())
 for p in list(H133.glob('*.py'))+[R133/'tools/experiment_guard_round133.py']:
  out=O133/'own_code'/p.relative_to(R133);out.parent.mkdir(exist_ok=True,parents=True);out.write_text(p.read_text().replace(str(R133),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 receipts=[]
 for p in sorted((H133/'job_receipts').glob('*.json')):
  r=rd(p);receipts.append({k:r[k] for k in ['resource','started_utc','finished_utc','returncode','max_seconds']}|{'entrypoint':Path(r['command'][1]).name,'engineering':'--engineering' in r['command']})
 dump(O133/'RESOURCE_RECEIPTS.json',receipts)
 (O133/'REPRODUCTION.md').write_text('''# Reproduction and evidence boundary

R133 depends on immutable R132972f5055f5213a8a362ea85c4d7ed9e8f5593d1f,
studies/3eed_update_coverage_B_acquisition_loggap_validation_20261117, plus
R131/R130/R129/R127/R123/R97/R91 dependencies documented there. Upstream3EED:
https://github.com/worldbench/3EED. Licensed input data, source/prepared weights,
features/point indices/coordinates and adaptation states remain private.

Order: prepare133 -> model-code lock -> run133 --engineering -> formal lock ->
run133 -> seal all endpoints -> audit133 -> score133 -> audit_scalars133 ->
analyze133 -> report133 -> export133 -> audit_public133. Models/CPU scoring use
experiment_guard_round133.py; one exclusive GPU, no old paused goal resumed.

Fixed R132 LOG C/S/wrong-C/kappa/original B/interface4896/rho/featurekeep/Armijo.
1step is the actual original first normalized update reconstructed from saved
gradients and GPU norm/accepted fraction. All646x2 original delta3 reassemblies
are bit-exact. 3step readouts/states are inherited, not refitted. Only extra
steps4/5 are differentiated, starting from the actual existing delta3, with no
hidden optimizer state. Original3 readout engineering covers both saved-B and
recomputed-original-B paths. All1798 rows stay,646active/1152inactive.

Primary remains3 by predeclaration. 1/5 are sensitivity, never target-GT parameter
selection, per-case best outputs or per-direction configurations. Same per-step
rho means larger cumulative budget at larger step counts. Actual delta norms,
reconstructed path lengths, accepted radius sums and forward/backward counts are
reported. Incremental measured cache-reuse timing is not standalone TTA latency.
Correct/wrong controls have matched nominal budgets but can accept different
fractions. Identity wrong labels are not counted as effective new supervision.

All16182 task values use R131 saved all256-box IoUs after every new endpoint is
sealed; no raw target GT geometry/source label tuning/FINAL access. Old R132
49good/27bad records are all revisited post-seal, not selected to improve a
configuration. Their set membership/group switches do not certify real object
identity. All new negative/positive/no-op sensitivity outcomes are retained.

Historical target exposure and prior three-source pooled calibration remain;
this is not new blind validation or strict direction-isolated label selection.
Anonymous CONTENT/FRAME/CLUSTER IDs preserve aliases/descriptions/directions;
one Drone recording does not support population scene CIs. Export contains own
code/config and complete anonymous scalars only, no raw IDs/captions/media,
query IDs, tensors, weights, credentials or third-party snapshots. Verify all
immutable remote bytes/trees/parent before CASmain; retain local archive receipt.
No additional tuning, source training, loss or B3 combination is queued.
''')
 dump(O133/'MANIFEST.json',{'prefix':'studies/3eed_fixed_LOG_step_sensitivity_20261117','files':[{'path':str(p.relative_to(O133)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O133.rglob('*')) if p.is_file()]});print('EXPORT133',len([p for p in O133.rglob('*') if p.is_file()]))
if __name__=='__main__':main()
