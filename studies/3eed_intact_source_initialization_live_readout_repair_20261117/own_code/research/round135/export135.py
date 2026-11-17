from common135 import *
import csv,io,re,collections
O135=H135/'publication/public_export_attempt001'
def table(p,rows):
 if not rows:return
 p.parent.mkdir(parents=True,exist_ok=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows);p.write_text(s.getvalue())
def main():
 assert rd(H135/'SCALAR_AUDIT.json')['status']=='PASS';O135.mkdir(parents=True,exist_ok=False);Q=H135/'scores_attempt001';rows=[r for p in sorted(Q.glob('*_ALL_ROWS_PRIVATE.json')) for r in rd(p)];frames=sorted({r['frame_PRIVATE'] for r in rows});contents=sorted({r['content_group_PRIVATE'] for r in rows});scenes={t:sorted({r['scene_PRIVATE'] for r in rows if r['target']==t}) for t in ['waymo','drone','quad']};mapping={}
 for p in sorted(Q.glob('*_ALL_ROWS_PRIVATE.json')):
  pub=[]
  for i,r in enumerate(rd(p)):
   ids={'row':f'ROW{i+1:04d}','frame_cluster':f'FRAME{frames.index(r["frame_PRIVATE"])+1:04d}','content_cluster':f'CONTENT{contents.index(r["content_group_PRIVATE"])+1:04d}','scene_cluster':f'CLUSTER{scenes[r["target"]].index(r["scene_PRIVATE"])+1:03d}'};mapping[(r['source'],r['target'],r['id_PRIVATE'])]=ids;d={k:v for k,v in r.items() if not k.endswith('_PRIVATE') and k!='IoUs'};d.update(ids);d.update({k+'__IoU':v for k,v in r['IoUs'].items()});pub.append(d)
  table(O135/'tasks'/p.name.replace('_ALL_ROWS_PRIVATE.json','.csv'),pub)
 for name in ['ALL_METHOD_ROWS','ALL_PAIRED_ROWS','AGGREGATE_ROWS','SUBGROUP_CONTRIBUTIONS']:table(O135/'results'/(name+'.csv'),rd(Q/(name+'.json')))
 pub=[]
 for r in rd(Q/'ALL_SCENE_ROWS_PRIVATE.json'):
  d=dict(r);d['scene_cluster']=f'CLUSTER{scenes[r["target"]].index(d.pop("scene_PRIVATE"))+1:03d}';pub.append(d)
 table(O135/'results/ALL_SCENE_ROWS.csv',pub)
 for name,path in [('ALL_CHANGED_CASES',Q/'ALL_CHANGED_CASES_PRIVATE.json'),('PROCESS_ROWS',H135/'PROCESS_ROWS_PRIVATE.json')]:
  pub=[]
  for r in rd(path):
   d={k:v for k,v in r.items() if not k.endswith('_PRIVATE')};d.update(mapping[(r['source'],r['target'],r['id_PRIVATE'])]);pub.append(d)
  table(O135/'diagnostics'/(name+'.csv'),pub)
 for name in ['SUMMARY','NATIVE_AUDIT','SCALAR_AUDIT','DECISION'] :dump(O135/(name+'.json'),rd(H135/(name+'.json')))
 for name in ['DESIGN_LOCK','BASELINE_PIN','MODEL_CODE_LOCK','FORMAL_LOCK','EVALUATION_LOCK']:dump(O135/'config'/(name+'.json'),rd(H135/'experiment'/(name+'.json')))
 (O135/'REPORT.md').write_text((H135/'REPORT.md').read_text())
 for p in list(H135.glob('*.py'))+[R135/'tools/experiment_guard_round135.py']:
  out=O135/'own_code'/p.relative_to(R135);out.parent.mkdir(exist_ok=True,parents=True);out.write_text(p.read_text().replace(str(R135),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 receipts=[]
 for p in sorted((H135/'job_receipts').glob('*.json')):
  r=rd(p);receipts.append({k:r[k] for k in ['resource','started_utc','finished_utc','returncode','max_seconds']}|{'entrypoint':Path(r['command'][1]).name,'engineering':'--engineering' in r['command']})
 dump(O135/'RESOURCE_RECEIPTS.json',receipts)
 dump(O135/'ATTEMPTS.json',{'engineering_attempt001':'GPU passed but preparation/fit file paths mutated; preserved locally, separated paths before formal run, rerun12 engineering examples','engineering_attempt002':'GPU and CPU passed; same algorithm/settings','CPU_audit_attempt001':'started before engineering seal existed; failed without GT read; rerun after GPU completion','no_target_score_driven_change':True})
 (O135/'REPRODUCTION.md').write_text('''# R135 reproduction

Requires licensed paired data and source-prepared private checkpoints. Reuses immutable R134 1af966d2a07a3be0c3f2e484da7320f395c87f4a / analysis25e63ad141946cc4ab995d2937bc1db5bfb04200, R132/R131/R97/R91 and dependencies documented there. Upstream https://github.com/worldbench/3EED . Private IDs, images, descriptions, coordinates, feature/state caches and weights are intentionally excluded; this is an inspectable code/config/scalar record, not a standalone data release.

Run via tools/experiment_guard_round135.py: run135 --engineering --phase prepare; run135 --engineering --phase fit; audit135 --engineering; run135 --phase prepare; run135 --phase fit; audit135; score135; audit_scalars135; report135; export135. Each GPU job is exclusive and bounded <=10800seconds. CPU jobs also guarded. Immutable preparation and fitted features/records use separate paths. All predictions sealed before any new target task scores, existing fixed-A256 IoU caches only.

Primary tta_A is fixed before scoring, not picked over tta_F. Full trained adapters/gates and source competition remain. Restore full source state, apply zero delta, generate frozen source-A route and live complete h/u/late/local/own/text/relation reads. Only Up4896 changes; feature-keep anchors the intact source-adapter output. Original B, original grouping, kappa and three steps/radius/Armijo are retained. New C/pairs use the restored complete teacher; kappa transfers unchanged from pooled old-reader source calibration. No source retraining/experts/FINAL.

The eight zero-step factorial endpoints isolate reset/readout/input and their interaction. Full A must reproduce existing R97 prepared_frozen for all1798; noB primary/direct/fit controls return completeFrozenA exactly. R134 pair evidence is still initial-FAB reversal with A/B agreement, frozen and shared across A/F student arms. Pair evidence need not initially violate A. Correct-vs-wrong controls keep correctC and permute only pair endpoints, not partial labels. Direct controls exactly project partial-direct scores onto the same ordinal constraints. Geometric grouping and candidate IoU do not establish true object identity.

All1798 rows, all signs and scene/content clusters are retained. Main metrics are six-direction equal means with and without original design weights; repeated physical frames are not independent scenes. No target-conditioned direction/sample output selection, no fresh blind-validation claim, no source-only calibration guarantee per direction. Model engineering restoration and task gain from TTA receive separate decisions. Failed attempts stay in receipts. No automatic continuation is scheduled.
''')
 dump(O135/'MANIFEST.json',{'prefix':'studies/3eed_intact_source_initialization_live_readout_repair_20261117','files':[{'path':str(p.relative_to(O135)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O135.rglob('*')) if p.is_file() and p.name!='MANIFEST.json']})
 # Independent payload privacy/value/cluster audit, no new task computation.
 tokens=set()
 for e in rd(H135/'INPUT_REGISTRY_PRIVATE.json'):
  for k in ['id','caption','scene','imagepath','pointpath','metapath']:
   v=e['row'].get(k)
   if isinstance(v,str) and len(v)>=12:tokens.add(v)
  tokens.add(e['content_group_PRIVATE'])
 count=0;values=0;maps=collections.defaultdict(dict)
 for p in O135.rglob('*'):
  if not p.is_file():continue
  assert p.suffix in ['.csv','.json','.py','.md'];txt=p.read_text();assert 'LOCAL_HOME/' not in txt;assert not any(t in txt for t in tokens),p;assert not re.search(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,}|sk-[A-Za-z0-9]{25,})',txt)
 for p in (O135/'tasks').glob('*.csv'):
  pub=list(csv.DictReader(p.open()));private=rd(Q/(p.stem+'_ALL_ROWS_PRIVATE.json'));assert len(pub)==len(private)
  for x,r in zip(pub,private):
   assert not any(k.endswith('_PRIVATE') for k in x)
   for k,v in r['IoUs'].items():assert abs(float(x[k+'__IoU'])-v)<1e-14;values+=1
   for k,pk in [('content','content_group_PRIVATE'),('frame','frame_PRIVATE'),('scene','scene_PRIVATE')]:
    key=(r['target'],r[pk]);val=x[k+'_cluster'];assert key not in maps[k] or maps[k][key]==val;maps[k][key]=val
   count+=1
 for f in rd(O135/'MANIFEST.json')['files']:p=O135/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
 assert count==1798;dump(H135/'PUBLIC_EXPORT_AUDIT.json',{'status':'PASS','n':count,'task_values':values,'files':len(list(O135.rglob('*'))),'private_tokens_scanned':len(tokens),'raw_data_weights_features_coordinates_descriptions_IDs_exported':False});print({'export':str(O135),'n':count,'task_values':values},flush=True)
if __name__=='__main__':main()
