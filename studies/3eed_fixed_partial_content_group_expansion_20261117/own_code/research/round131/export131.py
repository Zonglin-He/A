"""Public code/config and full anonymous scalar export, no private sensor assets."""
from pathlib import Path
import json,csv,io,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round131';O=H/'publication/public_export_attempt001';Q=H/'scores_attempt001'
rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def table(p,rows):
 if not rows:return
 p.parent.mkdir(parents=True,exist_ok=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows);p.write_text(s.getvalue())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==1798;O.mkdir(parents=True,exist_ok=False);allrows=[r for f in sorted(Q.glob('*_ALL_ROWS_PRIVATE.json')) for r in rd(f)];clusters={t:sorted({r['scene_PRIVATE'] for r in allrows if r['target']==t}) for t in ['waymo','drone','quad']};frames=sorted({r['frame_PRIVATE'] for r in allrows});groups=sorted({r['content_group_PRIVATE'] for r in allrows});mapping={}
 scene=lambda t,s:'CLUSTER'+str(clusters[t].index(s)+1).zfill(3)
 for f in sorted(Q.glob('*_ALL_ROWS_PRIVATE.json')):
  rows=rd(f);out=[]
  for i,r in enumerate(rows):
   keys=['source','target','sampling_weight','protected_alias_fallback','B_available','scope_eligible','partial_eligible','effective_shuffle','seconds'];d={k:r[k] for k in keys};d.update(row=f'ROW{i+1:04d}',scene_cluster=scene(r['target'],r['scene_PRIVATE']),frame_cluster='FRAME'+str(frames.index(r['frame_PRIVATE'])+1).zfill(4),content_cluster='CONTENT'+str(groups.index(r['content_group_PRIVATE'])+1).zfill(4));mapping[(r['source'],r['target'],r['id_PRIVATE'])]={k:d[k] for k in ['row','scene_cluster','frame_cluster','content_cluster']};d.update({k+'__IoU':v['IoU'] for k,v in r['arms'].items()});d.update(r['diagnostics']);out.append(d)
  table(O/'tasks'/f.name.replace('_ALL_ROWS_PRIVATE.json','.csv'),out)
 for n in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS']:table(O/'results'/(n+'.csv'),rd(Q/(n+'.json')))
 scenes=[]
 for r in rd(Q/'ALL_SCENE_ROWS_PRIVATE.json'):
  x=dict(r);x['scene_cluster']=scene(x['target'],x.pop('scene_PRIVATE'));scenes.append(x)
 table(O/'results/ALL_SCENE_ROWS.csv',scenes)
 for n in ['PROCESS_SUMMARY','SUBSET_COMPARISONS','CONTROL_QUALITY']:table(O/'diagnostics'/(n+'.csv'),rd(H/(n+'.json')))
 cases=[]
 for r in rd(H/'CHANGED_CASES_PRIVATE.json'):
  x={k:v for k,v in r.items() if not k.endswith('_PRIVATE')};x.update(mapping[(r['source'],r['target'],r['id_PRIVATE'])]);cases.append(x)
 table(O/'diagnostics/CHANGED_CASES.csv',cases)
 strata=[]
 for r in rd(H/'STRATA_PRIVATE.json'):
  x=dict(r);x['scene_cluster']=scene(x['target'],x.pop('scene_PRIVATE'));strata.append(x)
 table(O/'config/STRATA.csv',strata)
 for n in ['DECISION','CASE_SUMMARY','NATIVE_AUDIT','SCALAR_AUDIT','GEOMETRY_AUDIT','GROUP_AUDIT','ENGINEERING','INPUT_SUMMARY','CONTENT_INVENTORY_SUMMARY','REGISTRATION_SUMMARY','SOURCE_CONTEXT_BRIDGE','SOURCE_PREPARED_PIN_AUDIT','IMAGE_OVERLAP_AUDIT']:dump(O/(n+'.json'),rd(H/(n+'.json')))
 dump(O/'SOURCE_CALIBRATION_REUSED.json',{k:v for k,v in rd(R/'research/round129/SOURCE_CALIBRATION.json').items() if k!='source_files'})
 for f in (H/'experiment').glob('*.json'):dump(O/'config'/f.name,rd(f))
 for n in ['REPORT.md','FINDINGS.md']:(O/n).write_text((H/n).read_text())
 for f in list(H.glob('*.py'))+[R/'tools/experiment_guard_round131.py']:
  p=O/'own_code'/f.relative_to(R);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 receipts=[]
 for f in sorted((H/'job_receipts').glob('*.json')):
  r=rd(f);receipts.append({k:r[k] for k in ['resource','started_utc','finished_utc','returncode','max_seconds']}|{'entrypoint':Path(r['command'][1]).name,'engineering':'--engineering' in r['command']})
 dump(O/'RESOURCE_RECEIPTS.json',receipts)
 dump(O/'ATTEMPTS.json',{'path_only_attempt001':{'status':'CANCELLED_BEFORE_ANY_TASK_SCORING','planned_direction_inputs':1696,'selected_paths_duplicating_old_point_content':239,'partial_predictions_retained_locally':True,'included_in_primary':False,'reason':'directory-path grouping missed identical raw point content; metadata-only correction required'},'content_group_attempt002':{'status':'COMPLETE_ALL_PREDICTIONS_SEALED_BEFORE_SCORING','direction_inputs':1798,'content_groups':384,'old_or_protected_content_groups_selected':0},'no_outcome_based_exclusion_or_reselection':True})
 (O/'REPRODUCTION.md').write_text('''# Reproduction and scope

R131 is one fixed-config expansion of the R129/R130 original-radius unguarded
partial_F recipe. It does not rewrite their historical gates or adopt a new loss.
Own-code dependency: immutable R130 c51169c019131afab7a4b7983884387b3244df98,
studies/3eed_partial_label_fixedS_shuffle_radius_validation_20261117, and the
R129/R127/R123/R97 dependencies documented there. Upstream native 3EED:
https://github.com/worldbench/3EED. Private licensed data, source/prepared weights,
source-A frozen context caches and prior geometry are required, not republished.

Order: content_inventory131 -> select_content131 -> audit_groups131 -> lock
inputs/design -> register131 -> lock model -> run131 --engineering -> lock
execution -> run131 formal -> seal every arm -> audit131 -> score131 ->
audit_scalars131 -> analyze131 -> report131 -> export131 -> audit_public131.
Models and CPU scoring use experiment_guard_round131.py, one exclusive GPU.
select131 is retained only as an allocation/hash utility and failed-attempt
implementation; it is not the selected final input design.

Registration attempts are fixed before gates; UNKNOWN/no-B/no-scope/no-update
cases remain. Complete F_AB matching uses frozen initial-A late/residual context.
No C is generated and no unfitted-view transfer improvement is claimed. Direct
partial readout uses precisely the same frozen candidate sets; wrong labels
preserve S, group-size-bin label counts and budget, not semantic mass/q0. Identity
controls are marked and effective-only results separately reported.

Content grouping uses exact raw point-file hashes, excludes old development A/B/C
and protected-content groups, and includes all aliases/captions/both sources.
Primary inverse-group inclusion weights are fixed by recording population counts.
All target VAL had historical exposure; this is ordinary fixed-config expansion,
not blind independent testing or new-scene generalization. One Drone recording
cannot support a scene-population CI. FINAL has no new inference or scoring.

Task labels enter only after all predictions and states are sealed. Good/bad
coverage and internal ordering are post-seal explanations, never deployment
labels or per-sample/source parameter choices. Every row remains in the results.
Original point-content and per-directory frame dependence is exposed through
anonymous CONTENT/FRAME/CLUSTER table-position labels across source directions.

Export contains only own code/config/anonymous scalars; no raw descriptions,
original identifiers, query IDs, media, coordinates, support arrays, features,
weights, credentials or third-party snapshots. The cancelled path-only attempt
remains documented and retained privately, not silently overwritten.
''')
 dump(O/'MANIFEST.json',{'prefix':'studies/3eed_fixed_partial_content_group_expansion_20261117','files':[{'path':str(f.relative_to(O)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(O.rglob('*')) if f.is_file()]});print('EXPORT131',len([f for f in O.rglob('*') if f.is_file()]))
if __name__=='__main__':main()
