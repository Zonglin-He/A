"""Allowlisted local R34/R35 public export. No GPU/score/fit/network or FINAL read."""
import json,hashlib,io,csv,sys,re,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'research/round35';R34=ROOT/'research/round34';EXP=HERE/'experiment';PREFIX='studies/3eed_local_support_partial_output_keep_TTA_20261102'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if not k.endswith('_PRIVATE')}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,str):return x.replace(str(ROOT),'WORKSPACE_ROOT')
 return x
def main():
 assert read(R34/'independent_audit/U_AUDIT_RESULT.json')['state']=='PASS';assert read(HERE/'independent_audit/U_AUDIT_RESULT.json')['state']=='PASS';assert read(EXP/'DECISION.json')['run']=='completed'
 pub=HERE/'publication';pub.mkdir(exist_ok=True);out=pub/'public_export_attempt001';out.mkdir(exist_ok=False)
 def cp(p,dst):
  p=Path(p);q=out/dst;q.parent.mkdir(parents=True,exist_ok=True)
  if p.suffix=='.json':dump(q,clean(read(p)))
  else:q.write_text(p.read_text().replace(str(ROOT),'WORKSPACE_ROOT'))
 def csvwrite(dst,records):
  b=io.StringIO();w=csv.DictWriter(b,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records);q=out/dst;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(b.getvalue())
 # All code here is authored for this project, not a snapshot of upstream dependencies.
 folders=[ROOT/'crossplatform/round34',ROOT/'crossplatform/round35',R34/'key_diagnostic',R34/'independent_audit',HERE/'cpu_checks']
 for folder in folders:
  for p in sorted(folder.glob('*.py')):cp(p,p.relative_to(ROOT))
 for name in ['finalize35_combined.py','publish35_combined.py']:cp(HERE/name,Path('research/round35')/name)
 cp(EXP/'REPORT.md','REPORT.md');cp(R34/'REPORT_DIAGNOSTIC.md','history/R34_DIAGNOSTIC_REPORT_BEFORE_U.md');cp(R34/'independent_audit/REPORT.md','audit/R34_DIAGNOSTIC_REPORT.md');cp(EXP/'RESULTS_ADEV.md','history/R35_RESULTS_ADEV_BEFORE_U.md');cp(HERE/'PLAN.md','protocol/R35_PLAN.md')
 for n in ['PLAN_LOCK','EXECUTION_SCOPE_AMENDMENT_LOCK','CODE_HANDOFF_CHECK','SYNTHETIC_CPU_CHECKS','A_FEASIBILITY_PRIMARY_FIXED']:
  cp(R34/'experiment'/(n+'.json'),Path('protocol/R34')/(n+'.json'))
 for n in ['PLAN_LOCK','CONFIRMATION_INTERFACE_LOCK','MASK_COUNTS_LOCK','INTERFACE_CODE_CHECK','FINAL_CODE_SNAPSHOT']:
  cp(EXP/(n+'.json'),Path('protocol/R35')/(n+'.json'))
 for n in ['U_CONFIRMATION_SCOPE_AMENDMENT_LOCK','SELECTED_NIGHT_CANDIDATE_BEFORE_U_LOCK','FINAL_BOUNDARY_CLARIFICATION_BEFORE_U_SCORES']:
  cp(ROOT/'research/NIGHT_20261101'/(n+'.json'),Path('protocol/night')/(n+'.json'))
 for n in ['DECISION','RUN_STATE','DEV_FEASIBILITY_TABLE','A_PRESERVATION_DRIFT_DECOMPOSITION','PARTIAL_R30_U_QUERY_DIFFERENCES','T_U_POOLED_DESCRIPTIVE','A_ALL_PAIR_EXECUTION_ROWS']:
  cp(EXP/(n+'.json'),Path('results/R35')/(n+'.json'))
 for n in ['diagnostic_SUMMARY','diagnostic_ALL_NEGATIVES','diagnostic_PAIRED_COMPARISONS','diagnostic_SCENE_ROWS','diagnostic_SCORING_RECEIPT','U128_SUMMARY','U128_ALL_NEGATIVES','U128_PAIRED_COMPARISONS','U128_SCENE_ROWS','U128_SCORING_RECEIPT']:
  cp(R34/'experiment'/(n+'.json'),Path('results/R34')/(n+'.json'))
 for n in ['ADEV_SUMMARY','ADEV_ALL_NEGATIVES','ADEV_PAIRED_COMPARISONS','ADEV_SCENE_ROWS','SCORING_RECEIPT','U128_SUMMARY','U128_ALL_NEGATIVES','U128_PAIRED_COMPARISONS','U128_SCENE_ROWS','U128_SCORING_RECEIPT']:
  cp(EXP/(n+'.json'),Path('results/R35')/(n+'.json'))
 for n in ['AUDIT_RESULT','AGGREGATES','ALL220_PREFERENCE_EXECUTION','ALL_NEGATIVES','FEASIBILITY','GATE_STATS','QP19_INDEPENDENT','U_AUDIT_RESULT','U_CONTENT_AUDIT_RESULT','U_ALL_NEGATIVES']:
  cp(R34/'independent_audit'/(n+'.json'),Path('audit/R34')/(n+'.json'))
 for n in ['U_AUDIT_RESULT','A_DRIFT_DECOMPOSITION_INDEPENDENT','U_ALL_NEGATIVES']:
  cp(HERE/'independent_audit'/(n+'.json'),Path('audit/R35')/(n+'.json'))
 cp(R34/'independent_audit/U_tasks_attempt001_strict_closure.log','audit/R34/INITIAL_STRICT_CLOSURE_FAILURE.log')
 for n in ['CPU_AUDIT_diagnostic','CPU_AUDIT_U128']:cp(R34/'experiment'/(n+'.json'),Path('audit/R34')/(n+'.json'))
 for n in ['CPU_AUDIT','CPU_HEAD_AUDIT_ROWS','CPU_PAIR_GAPS','U128_CPU_AUDIT','U128_CPU_REPLAY_ROWS']:cp(EXP/(n+'.json'),Path('audit/R35')/(n+'.json'))
 cp(HERE/'cpu_checks/CPU_CHECK_RESULT.json','audit/R35/CPU_MASK_CHECK_RESULT.json')
 for tag,folder in [('R34_diagnostic',ROOT/'crossplatform/round34/closure_diagnostic_attempt001'),('R34_U128',ROOT/'crossplatform/round34/closure_U128_attempt001'),('R35_U128_new_heads',ROOT/'crossplatform/round35/closureU128_attempt001')]:cp(folder/'CLOSURE_AUDIT.json',Path('audit/fullmodel')/(tag+'.json'))
 cp(ROOT/'crossplatform/round35/partial_keep_attempt001/FIT_RECEIPTS.json','results/R35/ALL6_FIT_RECEIPTS.json')
 for split in ['diagnostic','U128']:cp(ROOT/('crossplatform/round34/scope_'+split+'_attempt001/PROCESS_STATS.json'),Path('results/R34')/(split+'_GATE_PRECISION_STATS.json'))
 # Anonymized, all-sign task rows. These are saved scores; no scorer/GT is run here.
 rowsets={
  'R34_DIAGNOSTIC':read(R34/'experiment/diagnostic_ALL_ROWS_PRIVATE.json'),
  'R34_U128':read(R34/'experiment/U128_ALL_ROWS_PRIVATE.json'),
  'R35_ADEV':read(EXP/'ADEV_ALL_ROWS_PRIVATE.json'),
  'R35_U128':read(EXP/'U128_ALL_ROWS_PRIVATE.json')}
 allrows=sum(rowsets.values(),[]);hist={p:sorted({r['scene_PRIVATE'] for r in allrows if r['platform']==p}) for p in ['drone','quad']}
 split_scenes={(s,p):sorted({r['scene_PRIVATE'] for r in allrows if r.get('split','U128')==s and r['platform']==p}) for s in ['A208','DEV64','R64','S64','T128','U128'] for p in ['drone','quad']}
 counts={}
 for label,rr in rowsets.items():
  scalar=[]
  for r in rr:
   split=r.get('split','U128');ref='R30_ungated' if 'R30_ungated' in r['queries'] else 'R30_U_rows'
   for arm,iou in r['IoU'].items():scalar.append({'split':split,'alias':r['alias'],'platform':r['platform'],'scene_alias':r['platform']+'_scene'+str(split_scenes[split,r['platform']].index(r['scene_PRIVATE'])),'history_scene_alias':r['platform']+'_all_scene'+str(hist[r['platform']].index(r['scene_PRIVATE'])),'arm':arm,'IoU':iou,'query':r['queries'][arm],'query_changed_vs_source':r['queries'][arm]!=r['queries']['source'],'query_changed_vs_R30':r['queries'][arm]!=r['queries'][ref]})
  counts[label]=len(scalar);csvwrite('results/'+label+'_ALL_TASK_SCALARS.csv',scalar)
 assert counts=={'R34_DIAGNOSTIC':5280,'R34_U128':1280,'R35_ADEV':1088,'R35_U128':1536}
 # Prior no-label feature-key diagnosis: own analysis only, no full feature arrays.
 for name in ['CPU_AUDIT.json','REPORT.md','SOURCES.json']:cp(R34/'key_diagnostic'/name,Path('mechanism/key_diagnostic')/name)
 for name in ['SPEC_LOCK','SUMMARY','OBSERVATION_ROWS','WITHIN_CANDIDATE_ROWS','CROSS_OBSERVATION_SAMPLE']:
  cp(R34/'key_diagnostic/attempt001'/(name+'.json'),Path('mechanism/key_diagnostic')/(name+'.json'))
 runtime=[]
 for rel in ['crossplatform/round34/scope_diagnostic_attempt001','crossplatform/round34/scope_U128_attempt001','crossplatform/round34/closure_diagnostic_attempt001','crossplatform/round34/closure_U128_attempt001','crossplatform/round35/partial_keep_attempt001','crossplatform/round35/predictionsU128_attempt001','crossplatform/round35/closureU128_attempt001']:
  x=read(ROOT/rel/'SEALED.json');runtime.append({'attempt':rel,**{k:v for k,v in x.items() if k!='files'}})
 dump(out/'results/RUNTIME_SEAL_METADATA.json',runtime)
 dump(out/'DEPENDENCIES.json',{'prior_R32_R33_study_commit':'d0ae41e3a6e1156133d2bb6fee39e19c59b018c0','R30_initial_head_study_commit':'ccff83b5383aca03fbb65c879ccb27dd7d06f4d4','source_input':'XYZ+text','supervision':'R29 cached Qwen RGB+fulltext, all22 unchanged','source_checkpoint_sha256':'0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd','R34_new_training':False,'R35_training':'same184960 matchingMLP,6states96steps, A only','thirdparty_assets_weights_source_snapshots_redistributed':False,'FINAL384_manifest_media_GT_read_by_export':False,'private_inputs_required_for_full_reproduction':True})
 (out/'README.md').write_text('''# R34/R35 — Local support readout and partial output preservation

Read REPORT.md, protocol locks and independent audit. Fixed pre-U primary is
h_scoped_R30. It remains primary despite U R30 ranking higher. R34 changes only
source feature key for routing reused R30 corrections, no new neural fitting.
R35 trains the same matching head with an expanded keep mask, plus a denominator
control; no new G, Qwen calls, candidate boxes or geometry regression. All22
preferences are retained, including previously known failures. No GT-based
feedback removal. GRACE and output preservation are mechanism inspiration,
not reproductions of upstream algorithms.

U128 decoded RGB/canonical XYZ are distinct from declared used observations,
but all U scenes overlap history (1Drone/3Quad); prior fullVAL exposure remains.
No between-Drone-scene CI is estimable. Scene resampling quantiles are conditional
diagnostics, not population confidence. U has now been exposed and cannot be
used as another independent confirmation. FINAL384 remains entirely unused;
future final evaluation needs a separate immutable method/control lock.

U balanced mIoU: source14.960674, R30 16.313208, fixed h primary15.481331,
noun15.433945, samehgate no-update QP15.063310. Four global amount controls have
source task IoUs but may alter scores/queries. h loses to R30 and is not adopted
as a superior replacement. partial16.313208 is all128 IoUs exact to R30, with
4query differences, and is worse onDEV. Partial reduces newly preserved-region
MSE but does not improve task performance. This mask branch is stopped.

All928? Do not sum repeated tables as distinct observations: R34 diagnostic528,
U128; R35 ADEV272 and U128 repeat shared observations/controls. Scalar files
contain6560R34 arm-task values and2624R35 values, including overlaps. All599/204
R34 diagnostic/U and135/400R35 ADEV/combined-U negative method/reference rows
are kept. Query indices are anonymous candidate indices, not world identities.
scene_alias is per split, history_scene_alias traces reused physical scenes.

Numerical readout uses q0-anchored eps-logodds on official float32 probabilities,
not the exact differentiable training logodds. Actual GPU NN choices are logged;
CPU audits replay them and do not substitute CPU topk. The old R32 numerical
warnings remain part of history. Initial independent U strict bitexact-score
assertion failed; its log is retained. Corrected audit checks original closure
tolerances and exact final queries/boxes without changing predictions.

Actual fullmodel closures: R34 diagnostic10rows20forwards, R34 U128256,
R35 U two newheads256; source/R30 closures reused, not doublecounted. CPU head
replays and cached matching-module passes are not fullmodel forwards. Runtime
receipts are stage measurements, not optimized end-to-end deployment latency.

No original media/captions/coordinates/opaqueIDs/features/weights or thirdparty
source snapshots are redistributed. Own code requires private licensed inputs
and prior study dependencies; do not overwrite sealed attempt directories.
'''.replace('All928? Do not sum','Do not sum'))
 # Scan only data already exposed and used. Do not open FINAL row/media/GT.
 history=read(ROOT/'research/round27/input/ALL528_PRIVATE.json')+read(ROOT/'research/round30/input/S64_PRIVATE.json')+read(ROOT/'research/round29/input/R64_PRIVATE.json')+read(ROOT/'research/NIGHT_20261101/input/T128_PRIVATE.json')+read(ROOT/'research/NIGHT_20261101/input/U128_PRIVATE.json')
 assert len(history)==912
 forbidden={str(r[k]) for r in history for k in ['id','caption','imagepath','pointpath','metapath','scene'] if r.get(k)}
 forbidden|={r['id_PRIVATE'] for r in allrows if 'id_PRIVATE' in r};forbidden|={r['scene_PRIVATE'] for r in allrows}
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];findings=[]
 for p in files:
  text=p.read_text()
  for token in forbidden:
   if token in text:findings.append({'file':p.relative_to(out).as_posix(),'type':'exact_private_token','token_sha256':hashlib.sha256(token.encode()).hexdigest()})
  if re.search('/'+'home/'+r'[^/\s]+/',text):findings.append({'file':str(p.relative_to(out)),'type':'absolute_home_path'})
 assert not findings,findings[:10]
 dump(out/'EXPORT_PRIVACY_AUDIT.json',{'state':'PASS','allowlisted_text_only':True,'private_token_scan_observations':912,'private_token_types':['opaqueID','caption','imagepath','pointpath','metapath','real_scene'],'private_token_findings':findings,'FINAL_manifest_media_GT_not_read':True,'all_negative_method_reference_rows':{'R34diagnostic':599,'R34U':204,'R35ADEV':135,'R35U':400},'all_task_scalar_rows':counts,'no_weights_features_media_coordinates_thirdparty_snapshots':True})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()]
 dump(out/'MANIFEST.json',{'scope':'R34_R35_combined_local_export','prefix':PREFIX,'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'manifest_excludes_itself':True,'private_payload_excluded':True})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];payload=[{'path':PREFIX+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files];dump(pub/'TREE_PAYLOAD_LOCAL.json',payload)
 for p,item in zip(files,payload):assert p.read_text()==item['content'] and item['path']==PREFIX+'/'+p.relative_to(out).as_posix()
 for f in read(out/'MANIFEST.json')['files']:
  p=out/f['path'];assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256']
 total=sum(p.stat().st_size for p in files)
 audit={'state':'PASS','n':len(files),'bytes':total,'payload_bytes':(pub/'TREE_PAYLOAD_LOCAL.json').stat().st_size,'prefix':PREFIX,'private_findings':[],'manifest_hashes_and_payload_exact':True,'all_task_scalar_rows':counts,'all_negative_method_reference_rows':1338,'FINAL_manifest_media_GT_not_read':True,'no_publication_performed':True}
 dump(pub/'LOCAL_EXPORT_AUDIT.json',audit);print(json.dumps(audit))
if __name__=='__main__':main()
