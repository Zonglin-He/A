from common136 import *
import csv,io,re,collections
O=H136/'publication/public_export_attempt001'
def csvout(p,rows):
 if not rows:return
 p.parent.mkdir(parents=True,exist_ok=True);s=io.StringIO();w=csv.DictWriter(s,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),lineterminator='\n');w.writeheader();w.writerows(rows);p.write_text(s.getvalue())
def main():
 assert rd(H136/'SCALAR_AUDIT.json')['status']=='PASS';O.mkdir(exist_ok=False,parents=True);method=rd(H136/'METHOD_ALL_ROWS_PRIVATE.json');oracle=rd(H136/'ORACLE_ALL_ROWS_PRIVATE.json');frames=sorted({r['frame_PRIVATE'] for r in method});contents=sorted({r['content_group_PRIVATE'] for r in method});scenes={t:sorted({r['scene_PRIVATE'] for r in method if r['target']==t}) for t in ['waymo','drone','quad']};mapping={}
 for name in sorted({r['source']+'_to_'+r['target'] for r in method}):
  old=rd(R136/'research/round135/scores_attempt001'/(name+'_ALL_ROWS_PRIVATE.json'));ids={r['id_PRIVATE']:i+1 for i,r in enumerate(old)}
  for r in [r for r in method if r['source']+'_to_'+r['target']==name]:mapping[(r['source'],r['target'],r['id_PRIVATE'])]={'row':f'ROW{ids[r["id_PRIVATE"]]:04d}','frame_cluster':f'FRAME{frames.index(r["frame_PRIVATE"])+1:04d}','content_cluster':f'CONTENT{contents.index(r["content_group_PRIVATE"])+1:04d}','scene_cluster':f'CLUSTER{scenes[r["target"]].index(r["scene_PRIVATE"])+1:03d}'}
 for block,rows in [('GT_DIAGNOSTIC_ONLY',oracle),('UNLABELED_METHOD',method)]:
  for name in sorted({r['source']+'_to_'+r['target'] for r in rows}):
   pub=[]
   for r in [r for r in rows if r['source']+'_to_'+r['target']==name]:
    d={k:v for k,v in r.items() if not k.endswith('_PRIVATE') and k!='IoUs'};d.update(mapping[(r['source'],r['target'],r['id_PRIVATE'])]);d.update({k+'__IoU':v for k,v in r['IoUs'].items()});pub.append(d)
   csvout(O/block/'tasks'/(name+'.csv'),pub)
 for n in ['ORACLE_METHOD_ROWS','ORACLE_AGGREGATE','METHOD_METHOD_ROWS','METHOD_PAIRED_ROWS','METHOD_AGGREGATE_ROWS','GRADIENT_SUMMARY','SUPERVISION_FLOW','ORACLE_SCOPE_BREAKDOWN']:csvout(O/'results'/(n+'.csv'),rd(H136/(n+'.json')))
 scenespub=[]
 for r in rd(H136/'METHOD_SCENE_ROWS_PRIVATE.json'):
  d=dict(r);d['scene_cluster']=f'CLUSTER{scenes[r["target"]].index(d.pop("scene_PRIVATE"))+1:03d}';scenespub.append(d)
 csvout(O/'results/ALL_SCENE_ROWS.csv',scenespub)
 for n in ['METHOD_CHANGED_CASES_PRIVATE','GRADIENT_ROWS_PRIVATE','METHOD_PROCESS_ROWS_PRIVATE','ORACLE_PROCESS_ROWS_PRIVATE']:
  pub=[]
  for r in rd(H136/(n+'.json')):
   d={k:v for k,v in r.items() if not k.endswith('_PRIVATE')};d.update(mapping[(r['source'],r['target'],r['id_PRIVATE'])]);pub.append(d)
  csvout(O/'diagnostics'/(n.replace('_PRIVATE','')+'.csv'),pub)
 for n in ['PREPARATION','EVIDENCE_AUDIT','ORACLE_AUDIT','METHOD_AUDIT','SCALAR_AUDIT','ORACLE_DECISION','DECISION','SUMMARY']:dump(O/(n+'.json'),rd(H136/(n+'.json')))
 for p in (H136/'experiment').glob('*.json'):dump(O/'config'/p.name,rd(p))
 for p in list(H136.glob('*.py'))+[R136/'tools/experiment_guard_round136.py']:
  q=O/'own_code'/p.relative_to(R136);q.parent.mkdir(exist_ok=True,parents=True);q.write_text(p.read_text().replace(str(R136),'WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME'))
 (O/'REPORT.md').write_text((H136/'REPORT.md').read_text());dump(O/'JOB_RECEIPTS.json',[{k:rd(p)[k] for k in ['resource','started_utc','finished_utc','returncode','max_seconds']}|{'entrypoint':Path(rd(p)['command'][1]).name,'engineering':'--engineering' in rd(p)['command']} for p in sorted((H136/'job_receipts').glob('*.json'))]);dump(O/'ATTEMPTS.json',{'preparation_attempt001':'allunlabeled probabilityfiles generated; summary numpyinteger JSON failure; engineering12 prematurelylaunched/completedbeforeseal, preservednotformal','metadata_fix':'summary builtinintconversion and sealprerequisite only; all1798evidence filesbefore/after bitexact; no label/threshold/method tuning','engineering_attempt002':'rerun afterseal and audited beforeformal','no_GT_delta_or_checkpoint_used_for_unlabeled_initialization':True})
 (O/'REPRODUCTION.md').write_text('''# R136 reproduction

Private licensed data/source-prepared checkpoints and immutable R135 e55002b0840c6c20188d1bc48797c7c6793f7903 with ff216d366c26c793e4254d84c862ba176caf0812 cached diagnosis are prerequisites. Existing R97/R91/R76 dependencies and official https://github.com/worldbench/3EED are not re-vendored. No source weights, raw descriptions/media/coordinates/GTquerymasks/features/IDs/thirdparty snapshots are released.

Use tools/experiment_guard_round136.py for every model/CPU scoring job, one exclusive GPU,<=10800seconds per job. prepare136 (GT-free evidence) -> audit_evidence136 -> oracle136 --engineering -> audit136 --stage oracle --engineering -> oracle136 -> audit136 --stage oracle -> score_oracle136. GT3 uses label-derived best-IoU-set NLL, sourcekeep, original3steps/rho and Up4896. This is diagnostic-only, not a method. A positive control permits the already-locked unlabeled hypothesis test; no oracle state feeds it.

Then method136 --engineering -> audit136 --stage method --engineering -> method136 -> audit136 --stage method -> score_method136 -> analyze136 (postseal GTgradientcosines only) -> audit_scores136 -> report136 -> export136. Targets,wrongpermutations,directreadouts and method design are frozen before initial GT engineering, with metadata-fix byte identity documented. Method starts from source each input/kind; complete liveA scoring, originalA256 and registeredB/S/C. No GT paths in deployed loss/evidence, no source retraining or experts/FINAL. The only new loss is B group-max conditional KL insideC replacing the F-reversal hinge; oldpartial/featurekeep/three-step/radius/Armijo remain. Compare exact target conditional direct readout, completeFrozenF/ABmean/B-only, oldR135 and matched wrongC targets.

All1798rows including1034noB remain. The764commonpool includes15previously zero-task rows but no new view or qualification relaxation. Oracle targets全A256 vs noGT targetsC are intentionally different information, so a positive control establishes limited interface capability, not the precise cause of every remaining gap or a validTTA score. GT3/direct gain fraction is reported. Original data have repeated descriptions/directions/content and prior target-score exposure; all results are development, not independent validation. Existing pooled source-ownkappa not strictly per-direction source-only. All negative metrics/scenes/cases retained. No targetbased per-sample/direction output choice or automatic sweep.
''')
 dump(O/'MANIFEST.json',{'prefix':'studies/3eed_GT_interface_positive_control_Cconditional_BtoA_TTA_20261117','files':[{'path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O.rglob('*')) if p.is_file() and p.name!='MANIFEST.json']})
 # Independently inspect scalar payload and cluster aliases; all GT tensors stay private.
 tokens=set()
 for e in rd(H136/'INPUT_REGISTRY_PRIVATE.json'):
  for k in ['id','caption','scene','imagepath','pointpath','metapath']:
   v=e['row'].get(k)
   if isinstance(v,str) and len(v)>=12:tokens.add(v)
  tokens.add(e['content_group_PRIVATE'])
 for p in O.rglob('*'):
  if not p.is_file():continue
  assert p.suffix in ['.csv','.json','.md','.py'];txt=p.read_text();assert 'LOCAL_HOME/' not in txt and not any(t in txt for t in tokens),p;assert not re.search(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{25,}|sk-[A-Za-z0-9]{25,})',txt)
 n=0;values=0
 for block,private in [('GT_DIAGNOSTIC_ONLY',oracle),('UNLABELED_METHOD',method)]:
  ix={(r['source']+'_to_'+r['target'],mapping[(r['source'],r['target'],r['id_PRIVATE'])]['row']):r for r in private}
  for p in (O/block/'tasks').glob('*.csv'):
   for r in csv.DictReader(p.open()):
    assert not any(k.endswith('_PRIVATE') for k in r);old=ix[(p.stem,r['row'])]
    for k,v in old['IoUs'].items():assert abs(float(r[k+'__IoU'])-v)<1e-14;values+=1
    n+=1
 assert n==3596;dump(H136/'PUBLIC_EXPORT_AUDIT.json',{'status':'PASS','rows':n,'task_scalar_values':values,'private_literals_scanned':len(tokens),'GT_masks_or_raw_data_features_weights_exported':False});print({'files':len([p for p in O.rglob('*') if p.is_file()]),'rows':n,'scalar_values':values},flush=True)
if __name__=='__main__':main()
