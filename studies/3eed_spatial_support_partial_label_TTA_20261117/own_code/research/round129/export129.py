"""Publish only own implementation/configuration and complete anonymous scalars."""
from pathlib import Path
import json,csv,io,hashlib
R=Path(__file__).resolve().parents[2];H=R/'research/round129';O=H/'publication/public_export_attempt001';Q=H/'scores_attempt001';rd=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def table(p,rr):
 if not rr:return
 p.parent.mkdir(parents=True,exist_ok=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(dict.fromkeys(k for r in rr for k in r)),lineterminator='\n');w.writeheader();w.writerows(rr);p.write_text(s.getvalue())
def main():
 assert rd(H/'SCALAR_AUDIT.json')['n']==2304;O.mkdir(parents=True,exist_ok=False);mapping={};clusters={}
 for tgt in ['waymo','drone','quad']:clusters[tgt]=sorted({r['scene_PRIVATE'] for f in Q.glob('*_ALL_ROWS_PRIVATE.json') for r in rd(f) if r['target']==tgt})
 for f in sorted(Q.glob('*_ALL_ROWS_PRIVATE.json')):
  rr=rd(f);public=[]
  for i,r in enumerate(rr):
   row=f'ROW{i+1:04d}';cluster='CLUSTER'+str(clusters[r['target']].index(r['scene_PRIVATE'])+1).zfill(3);mapping[(r['cohort'],r['source'],r['target'],r['id_PRIVATE'])]=(row,cluster);d={k:r[k] for k in ['cohort','source','target','protected_alias_fallback','scope_eligible','partial_eligible']};d.update(row=row,scene_cluster=cluster);d.update({k+'__IoU':v['IoU'] for k,v in r['arms'].items()});public.append(d)
  table(O/'tasks'/f.name.replace('_ALL_ROWS_PRIVATE.json','.csv'),public)
 for name in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS','POSTSTRATIFIED_METHOD_ROWS','POSTSTRATIFIED_AGGREGATE_ROWS']:table(O/'results'/(name+'.csv'),rd(Q/(name+'.json')))
 sc=[]
 for r in rd(Q/'ALL_SCENE_ROWS_PRIVATE.json'):s=dict(r);s['scene_cluster']='CLUSTER'+str(clusters[s['target']].index(s.pop('scene_PRIVATE'))+1).zfill(3);sc.append(s)
 table(O/'results/ALL_SCENE_ROWS.csv',sc)
 for name in ['PROCESS_SUMMARY','SET_QUALITY','SUBSET_COMPARISONS']:table(O/'diagnostics'/(name+'.csv'),rd(H/(name+'.json')))
 changed=[]
 for r in rd(H/'CHANGED_CASES_PRIVATE.json'):
  s={k:v for k,v in r.items() if not k.endswith('_PRIVATE')};s['row'],s['scene_cluster']=mapping[(r['cohort'],r['source'],r['target'],r['id_PRIVATE'])];changed.append(s)
 table(O/'diagnostics/CHANGED_CASES.csv',changed)
 for name in ['DECISION','NATIVE_AUDIT','GEOMETRY_AUDIT','SCALAR_AUDIT','MATH_AUDIT','ENGINEERING','FACTOR_FINDINGS','GPU_RESOURCE_SUMMARY']:dump(O/(name+'.json'),rd(H/(name+'.json')))
 for name in ['SOURCE_CALIBRATION','SCOPE_SEALED']:dump(O/(name+'.json'),{k:v for k,v in rd(H/(name+'.json')).items() if k not in ['source_files','files']})
 table(O/'diagnostics/SOURCE_CALIBRATION_ROWS.csv',rd(H/'SOURCE_CALIBRATION_ROWS.json'))
 for name in ['REPORT.md','LITERATURE_AND_PROTOCOL.md','FINDINGS.md']:(O/name).write_text((H/name).read_text())
 for f in (H/'experiment').glob('*.json'):dump(O/'config'/f.name,rd(f))
 for f in list(H.glob('*.py'))+[R/'tools/experiment_guard_round129.py']:
  p=O/'own_code'/f.relative_to(R);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(f.read_text().replace(str(R),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 receipts=[]
 for f in (H/'job_receipts').glob('*.json'):
  x=rd(f);receipts.append({'resource':x['resource'],'started_utc':x['started_utc'],'finished_utc':x['finished_utc'],'returncode':x['returncode'],'max_seconds':x['max_seconds'],'entrypoint':Path(x['command'][1]).name})
 dump(O/'RESOURCE_RECEIPTS.json',receipts)
 (O/'REPRODUCTION.md').write_text('''# Reproduction boundary

R129 requires the existing licensed 3EED workspace, three epoch100 source-only
checkpoints, prior R91 prepared modules, and the immutable R127 native-foundation
cache/readout interface. It is not a standalone checkpoint distribution.
Prior own-code dependency: studies/3eed_deployment_consistent_complete_readout_TTA_20261117
(R127, commit605d048c189471530c47876c228e06b0f7b14879), and
R123/R125/R97 interfaces linked there. Upstream 3EED comes from
https://github.com/worldbench/3EED and is not republished.

Order: calibrate129 (saved source-only labels/states); prepare129 (A/B only);
lock design and scope; check_math129; run129 --engineering; run129 formal;
seal every prediction; audit129 independent CPU; score129; audit_scalars129;
analyze129; report129; export129; audit_public129. Model, training and task-scoring
steps use experiment_guard_round129.py. No new source training occurs.
No target labels choose thresholds, candidate sets, trial endpoints or steps.
No C arrays are decoded until all parameter states for that input are fixed.
Archived R127 global states/readouts and protected-alias zero results are reused.

Private registry rows, original IDs, captions/media, coordinates, supports,
query indexes, features, weights and third-party snapshots are omitted.
Public ROW/CLUSTER tags identify only anonymous scalar-table positions.
The 25-arm full task tables retain every positive/negative/unchanged observation.
Source-calibration aggregate thresholds and anonymous gap rows are included;
the private source-file mapping is excluded.
''')
 dump(O/'MANIFEST.json',{'prefix':'studies/3eed_spatial_support_partial_label_TTA_20261117','files':[{'path':str(f.relative_to(O)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(O.rglob('*')) if f.is_file()]});print('EXPORT129_FILES',len([p for p in O.rglob('*') if p.is_file()]))
if __name__=='__main__':main()
