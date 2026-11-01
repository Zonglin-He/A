"""Local text/scalar export, no remote mutation and no private feature redistribution."""
import csv,io,json,shutil,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'research/round33';BASE=ROOT/'crossplatform/round33'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if not k.endswith('_PRIVATE') and k not in ['evidence_snapshot','local_snapshot','snapshot_path']}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
def main():
 assert read(R/'independent_audit/AUDIT_RESULT.json')['state']=='PASS_WITH_EXECUTION_SCOPE_LIMIT'
 pub=R/'publication';pub.mkdir(exist_ok=True);out=pub/'public_export_attempt001';out.mkdir(exist_ok=False)
 def cp(p,dest):
  q=out/dest;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
 def dump(p,x):
  q=out/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(json.dumps(clean(x),indent=2,ensure_ascii=False)+'\n')
 def table(p,rr):
  rr=[clean(r) for r in rr];assert rr;buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(rr[0]),lineterminator='\n');w.writeheader();w.writerows(rr);q=out/p;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(buf.getvalue())
 for folder in [BASE,R,R/'independent_audit',R/'gradient_diagnostic']:
  for p in sorted(folder.glob('*.py')):cp(p,p.relative_to(ROOT))
 report=(R/'REPORT.md').read_text().replace('independent_audit/ALL1088_TASK_ROWS_PRIVATE.json','results/ALL1088_TASK_ROWS.csv').replace('independent_audit/audit33_independent.py','research/round33/independent_audit/audit33_independent.py').replace('independent_audit/AUDIT_RESULT.json','audit/AUDIT_RESULT.json')
 (out/'REPORT.md').write_text(report)
 for n in ['PLAN_LOCK','SELECTED_INTERFACE_LOCK','DEV_INTERFACE_TABLE']:dump(Path('protocol')/(n+'.json'),read(R/'experiment'/(n+'.json')))
 ia=R/'independent_audit'
 for n in ['AUDIT_RESULT','GRADIENT_AUDIT_RESULT','SUBSET_SUMMARY','DEV_TABLE_INDEPENDENT']:dump(Path('audit')/(n+'.json'),read(ia/(n+'.json')))
 task=read(ia/'ALL1088_TASK_ROWS_PRIVATE.json');assert len(task)==1088
 aliases={r['id_PRIVATE']:r['alias'] for r in task};table('results/ALL1088_TASK_ROWS.csv',task);table('results/ALL_NEGATIVE_TASK_ROWS.csv',[r for r in task if r['delta_vs_source_pp']<0])
 prefs=read(ia/'ALL88_PREFERENCE_ROWS_PRIVATE.json');assert len(prefs)==88
 table('results/ALL88_PREFERENCE_ROWS.csv',[dict(alias=aliases[r['id_PRIVATE']],**r) for r in prefs])
 dump('results/FIT_RECEIPTS.json',read(BASE/'interface_screen_attempt001/FIT_RECEIPTS.json'))
 dump('results/FIT_SEAL.json',read(BASE/'interface_screen_attempt001/SEALED.json'))
 gd=R/'gradient_diagnostic';cp(gd/'REPORT.md','gradient_diagnostic/REPORT.md')
 for n in ['CPU_AUDIT']:
  dump(Path('gradient_diagnostic')/(n+'.json'),read(gd/(n+'.json')))
 dump('gradient_diagnostic/STATE_SUMMARY.json',read(gd/'attempt001/STATE_SUMMARY.json'))
 for n in ['RESEARCH.md','SOURCES.json']:
  if n.endswith('.json'):dump(Path('literature')/n,read(R/'literature'/n))
  else:cp(R/'literature'/n,Path('literature')/n)
 (out/'README.md').write_text('''# R33 — bounded query/text matching-interface comparison

Read REPORT.md. Same Waymo-only epoch100, frozen source feature caches, all22
R29 preferences from19 observations, originalU96/93 keep rows, beta1, exact
noun log-odds and sourceq0. query/text288-288-288-64 MLP x lr1e-4/5e-4,
AdamW weight_decay0,16steps, platform-independent source reset:8states128steps.
The text_h input is point-conditioned cross-encoder output; no RGB is added.

Only the original R30 query5e-4 satisfies the prespecified feedback feasibility;
it also wins balanced exposedDEV mIoU25.090904 vs best text23.712580. Even
without feasibility the text branch does not improve DEV. No new U128 check,
no extra sweep, no general claim text-side adaptation cannot work.

All1088 A/DEV scalar task outcomes and88 preference states, including negatives
and zeros, are exported. RawIDs, geometry, captions, media, projection states,
features and gradient vectors stay private. The CPU audit does not retrain;
saved-query state bitexactly equals R30 on both platforms. Unselected text
heads have no new full-model GPU closure, only isolated-module CPU replay.

Gradient source/endstate diagnostics find norm-weighted negative-dot energy
<=0.013696percent despite roughly half negative pair angles. This does not
exclude intermediate trajectory conflicts and does not justify a PCGrad trial.
No feedback was filtered from these diagnostic values.

Code composes with prior R24/R29/R30/R32 studies and licensed local 3EED
assets/environment. Rebuild the original private feature schema before running:
interface33.py -> sealed predictions -> select33.py -> independent CPU audit.
No third-party code snapshots, assets or model weights are redistributed.
TPT/RLCF/PCGrad are audited mechanism references, not reproduced baselines.
''')
 dump('DEPENDENCIES.json',{'R30_parent_commit':'ccff83b5383aca03fbb65c879ccb27dd7d06f4d4','R31_commit':'ec608254ca843ecd37cb5fce63e778f0d9f07d81','parent_studies':['R24','R29','R30','R32'],'requires_licensed_private_assets':True,'new_G_calls':0,'new_source_training_steps':0,'new_GPU_fullmodel_closure':False,'unselected_text_heads_only_cached_module_replay':True})
 # Exact sample IDs and current sample payload strings must never be present.
 forbidden=set(aliases)
 sys.path.insert(0,str(BASE))
 import common33
 for split in ['A208','DEV64']:
  for r in common33.rows(split):
   for k in ['id','caption','imagepath','pointpath','metapath','scene']:
    if r.get(k):forbidden.add(str(r[k]))
 files=sorted(p for p in out.rglob('*') if p.is_file())
 for p in files:
  text=p.read_text();assert not any(s and s in text for s in forbidden),p
  assert p.suffix not in ['.npz','.npy','.pkl','.pth','.jpg','.png'],p
 dump('MANIFEST.json',{'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'private_payload_excluded':True,'thirdparty_source_snapshots_excluded':True})
 files=sorted(p for p in out.rglob('*') if p.is_file());prefix='studies/3eed_query_text_matching_interface_probe_20261102'
 (pub/'TREE_PAYLOAD_LOCAL.json').write_text(json.dumps([{'path':prefix+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files],ensure_ascii=False)+'\n')
 (pub/'LOCAL_EXPORT_AUDIT.json').write_text(json.dumps({'n':len(files),'task_rows':1088,'preference_rows':88,'negative_rows':sum(r['delta_vs_source_pp']<0 for r in task),'private_ID_scan_passed':True,'media_weights_features_excluded':True,'prefix':prefix},indent=2)+'\n');print('EXPORT33',len(files))
if __name__=='__main__':main()
