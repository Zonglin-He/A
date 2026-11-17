"""Own code, locked protocol and complete anonymous scalar export only."""
from pathlib import Path
import json,csv,io,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round130';O=H/'publication/public_export_attempt001';Q=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def table(p,rr):
 if not rr:return
 p.parent.mkdir(parents=True,exist_ok=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(dict.fromkeys(k for r in rr for k in r)),lineterminator='\n');w.writeheader();w.writerows(rr);p.write_text(s.getvalue())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==2304;O.mkdir(parents=True,exist_ok=False);mapping={};private=[r for f in Q.glob('*_ALL_ROWS_PRIVATE.json') for r in rd(f)];clusters={t:sorted({r['scene_PRIVATE'] for r in private if r['target']==t}) for t in ['waymo','drone','quad']}
 for f in sorted(Q.glob('*_ALL_ROWS_PRIVATE.json')):
  rr=rd(f);public=[]
  for i,r in enumerate(rr):
   row=f'ROW{i+1:04d}';sc='CLUSTER'+str(clusters[r['target']].index(r['scene_PRIVATE'])+1).zfill(3);mapping[(r['cohort'],r['source'],r['target'],r['id_PRIVATE'])]=(row,sc);p={k:r[k] for k in ['cohort','source','target','protected_alias_fallback','partial_eligible','effective_shuffle']};p.update(row=row,scene_cluster=sc);p.update({k+'__IoU':v['IoU'] for k,v in r['arms'].items()});public.append(p)
  table(O/'tasks'/f.name.replace('_ALL_ROWS_PRIVATE.json','.csv'),public)
 for n in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS','POSTSTRATIFIED_METHOD_ROWS','POSTSTRATIFIED_AGGREGATE_ROWS']:table(O/'results'/(n+'.csv'),rd(Q/(n+'.json')))
 scene=[]
 for r in rd(Q/'ALL_SCENE_ROWS_PRIVATE.json'):s=dict(r);s['scene_cluster']='CLUSTER'+str(clusters[s['target']].index(s.pop('scene_PRIVATE'))+1).zfill(3);scene.append(s)
 table(O/'results/ALL_SCENE_ROWS.csv',scene)
 for n in ['SUBSET_COMPARISONS','PROCESS_SUMMARY','CONTROL_QUALITY']:table(O/'diagnostics'/(n+'.csv'),rd(H/(n+'.json')))
 for name in ['CHANGED_CASES','R129_ALL18_CASES']:
  source=rd(H/(name+'_PRIVATE.json'));frames=sorted({r['physical_frame_PRIVATE'] for r in source if 'physical_frame_PRIVATE' in r});pub=[]
  for r in source:
   d={k:v for k,v in r.items() if not k.endswith('_PRIVATE')};d['row'],d['scene_cluster']=mapping[(r['cohort'],r['source'],r['target'],r['id_PRIVATE'])]
   if 'physical_frame_PRIVATE' in r:d['physical_frame_cluster']='FRAME'+str(frames.index(r['physical_frame_PRIVATE'])+1).zfill(3)
   pub.append(d)
  table(O/'diagnostics'/(name+'.csv'),pub)
 for n in ['DECISION','NATIVE_AUDIT','SCALAR_AUDIT','GEOMETRY_AUDIT','ENGINEERING','CONTROL_SUMMARY','CASE_SUMMARY']:dump(O/(n+'.json'),rd(H/(n+'.json')))
 dump(O/'SOURCE_CALIBRATION_REUSED.json',{k:v for k,v in rd(R/'research/round129/SOURCE_CALIBRATION.json').items() if k!='source_files'})
 for f in (H/'experiment').glob('*.json'):dump(O/'config'/f.name,rd(f))
 for name in ['REPORT.md','FINDINGS.md']:(O/name).write_text((H/name).read_text())
 for f in list(H.glob('*.py'))+[R/'tools/experiment_guard_round130.py']:
  p=O/'own_code'/f.relative_to(R);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 receipts=[]
 for f in (H/'job_receipts').glob('*.json'):
  x=rd(f);receipts.append({k:x[k] for k in ['resource','started_utc','finished_utc','returncode','max_seconds']}|{'entrypoint':Path(x['command'][1]).name,'engineering':'--engineering' in x['command']})
 dump(O/'RESOURCE_RECEIPTS.json',receipts)
 (O/'REPRODUCTION.md').write_text('''# Reproduction and study boundary

R130 fixes R129's unguarded partial_F at original rho as its development primary.
This is a new explicitly authorized follow-up, not a retrospective rewrite of
R129's failed protected primary. Same source-only epoch100/prepared R91 states,
fixed A256/decoder anchors, registered A/B and complete FAB readout. Only4896 Up
parameters, original feature keep/3steps/Armijo. No source training or expert.

Own-code dependency: immutable R129 commit1dd3ff25b0d22ddc34001143d81a282d9f718e13,
studies/3eed_spatial_support_partial_label_TTA_20261117, and its R127/R123/R97
dependencies. Download upstream native3EED from https://github.com/worldbench/3EED;
private licensed data and prior source/prepared weights are not republished.

Order: prepare130 CPU without GT -> lock design -> run130 --engineering ->
lock execution -> run130 formal -> seal all predictions -> audit130 CPU ->
score130 -> audit_scalars130 -> analyze130 -> report130 -> export130 ->
audit_public130 -> immutable remote byte/tree/parent verification -> CAS main.
All models and CPU scoring use experiment_guard_round130.py. No completed
baseline leaderboard is rerun. Original partial states/predictions are reused;
twelve active engineering replays check exact original-state compatibility.
No C readout opens before new states are fixed. C never selects an update.

The wrong set keeps S and per-group-size-bin C counts exactly fixed; swaps
retained/excluded identities maximally where feasible. Five identity cases
retain exactly the same update and are not counted as effective shuffles.
Both full denominator and252-effective control results are retained. Initial
mass/q0 inclusion differ naturally; correct>wrong does not alone isolate
cross-modal identity evidence from semantic-confidence preservation.

DEV/EXP are exposed development, not independent confirmation. rho0 stays
primary; .5/2 endpoints are all reported, without target-specific selection.
The old18 good/bad records are post-seal explanatory audits, not rule design
or deployment filters. Anonymous frame/scene clusters expose dependence.
All negative cases are retained. FINAL receives no new access/scoring.

Export excludes captions/media, original IDs, coordinates/support arrays,
query indexes, features/weights, credentials and third-party snapshots.
Public ROW/FRAME/CLUSTER values are anonymous table-position labels only.
''')
 dump(O/'MANIFEST.json',{'prefix':'studies/3eed_partial_label_fixedS_shuffle_radius_validation_20261117','files':[{'path':str(f.relative_to(O)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(O.rglob('*')) if f.is_file()]});print('EXPORT130',len([f for f in O.rglob('*') if f.is_file()]))
if __name__=='__main__':main()
