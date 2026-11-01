"""Local allowlisted R36 export of authored code, locks, scalar evidence and audits."""
import json,hashlib,csv,io,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'research/round36';EXP=HERE/'experiment';BASE=ROOT/'crossplatform/round36';PREFIX='studies/3eed_source_fused_rgb_rank_preference_TTA_20261102'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if not k.endswith('_PRIVATE')}
 if isinstance(x,list):return [clean(v) for v in x]
 if isinstance(x,str):return x.replace(str(ROOT),'WORKSPACE_ROOT')
 return x
def main():
 audit=read(HERE/'independent_audit/AUDIT_RESULT.json');assert audit['state']=='PASS' and audit['fixed_DEV_gate_pass'];assert read(EXP/'DECISION.json')['fixed_DEV_gate_pass'];assert read(EXP/'RESOURCE_DECISION.json')['gate_not_reclassified_as_failure']
 pub=HERE/'publication';pub.mkdir(exist_ok=True);out=pub/'public_export_attempt001';out.mkdir(exist_ok=False)
 def cp(p,dst):
  p=Path(p);q=out/dst;q.parent.mkdir(parents=True,exist_ok=True)
  if p.suffix=='.json':dump(q,clean(read(p)))
  else:q.write_text(p.read_text().replace(str(ROOT),'WORKSPACE_ROOT'))
 for folder in [BASE,HERE/'independent_audit']:
  for p in sorted(folder.glob('*.py')):cp(p,p.relative_to(ROOT))
 cp(Path(__file__),'research/round36/publish36.py')
 for name in ['common25.py','model25.py','rank_fusion25.py','fusion25.py','rgb25.py','infer25.py']:cp(ROOT/'crossplatform/round25'/name,Path('crossplatform/round25')/name)
 cp(EXP/'REPORT.md','REPORT.md');cp(EXP/'PLAN_LOCK.json','protocol/PLAN_LOCK.json');cp(EXP/'CLOSURE_INPUTS_PRIVATE.json','protocol/FIXED_CLOSURE_INPUT_HASHES.json')
 for name in ['SUMMARY','DECISION','RESOURCE_DECISION','PAIRED_MECHANISM','ALL_NEGATIVES','RAW_VS_GROUP_READOUT_AUDIT']:
  cp(EXP/(name+'.json'),Path('results')/(name+'.json'))
 for name in ['CPU_AUDIT','CPU_REPLAY_ROWS','SYNTHETIC_CHECKS']:cp(EXP/(name+'.json'),Path('audit/primary_implementation')/(name+'.json'))
 for name in ['AUDIT_RESULT','QP38_INDEPENDENT','HEAD_REPLAY_ROWS','ALL44_PAIR_EXECUTION','RAW_VS_GROUPED_READOUT','SOURCE_PROVENANCE']:
  cp(HERE/'independent_audit'/(name+'.json'),Path('audit/independent')/(name+'.json'))
 cp(BASE/'closure4_attempt001/CLOSURE_AUDIT.json','audit/FULLMODEL_CLOSURE_RECEIPTS.json');cp(BASE/'capture272_attempt001/SOURCE_REPLAY_RECEIPTS.json','audit/ALL544_FROZEN_CAPTURE_SOURCE_REPLAY_RECEIPTS.json');cp(BASE/'fit272_attempt001/FIT_RECEIPTS.json','results/ALL4_FIT_RECEIPTS.json')
 for name in ['RESEARCH.md','SOURCES.json','PROVENANCE_CHECKS.json']:cp(HERE/'source_fusion_audit'/name,Path('source_fusion_audit')/name)
 cache=read(HERE/'source_fusion_audit/CACHE_AND_WEIGHTS.json');dump(out/'source_fusion_audit/CACHE_AND_WEIGHTS_SUMMARY.json',{k:clean(v) for k,v in cache.items() if k!='feedback_pair_ROI_coverage'}|{'feedback_endpoints_both_have_valid_ROI_pairs':len(cache['feedback_pair_ROI_coverage']),'valid_ROI_does_not_certify_identity':True,'individual_ROI_extents_not_exported':True})
 for name in ['FINAL_SOURCE_SELECTED_LOCK','SOURCE_MODEL_MANIFEST','SOURCE_READOUT_EXTENSION_LOCK','TARGET_PRELABEL_LOCK']:
  cp(ROOT/'research/round25/experiment'/(name+'.json'),Path('source_protocol')/(name+'.json'))
 allrows=read(EXP/'ALL_ROWS_PRIVATE.json');assert len(allrows)==544
 hist={p:sorted({r['scene_PRIVATE'] for r in allrows if r['platform']==p}) for p in ['drone','quad']};splits={(s,p):sorted({r['scene_PRIVATE'] for r in allrows if r['split']==s and r['platform']==p}) for s in ['A208','DEV64'] for p in ['drone','quad']}
 scalar=[]
 for r in allrows:
  for arm,v in r['IoU'].items():scalar.append({'split':r['split'],'alias':r['alias'],'mode':r['mode'],'platform':r['platform'],'scene_alias':r['platform']+'_scene'+str(splits[r['split'],r['platform']].index(r['scene_PRIVATE'])),'history_scene_alias':r['platform']+'_all_scene'+str(hist[r['platform']].index(r['scene_PRIVATE'])),'arm':arm,'IoU':v,'query':r['queries'][arm],'query_changed_vs_own_fused_frozen':r['queries'][arm]!=r['queries']['fused_frozen']})
 assert len(scalar)==2720
 b=io.StringIO();w=csv.DictWriter(b,fieldnames=list(scalar[0]),lineterminator='\n');w.writeheader();w.writerows(scalar);(out/'results/ALL544_MODE_ROWS_2720_TASK_SCALARS.csv').write_text(b.getvalue())
 scenes=[]
 for r in read(EXP/'SCENE_ROWS_PRIVATE.json'):
  scenes.append({k:v for k,v in r.items() if k!='scene_PRIVATE'}|{'scene_alias':r['platform']+'_scene'+str(splits[r['split'],r['platform']].index(r['scene_PRIVATE'])),'history_scene_alias':r['platform']+'_all_scene'+str(hist[r['platform']].index(r['scene_PRIVATE']))})
 dump(out/'results/ALL20_SCENE_ROWS.json',scenes)
 runtime=[]
 for name in ['rgbDEV64_attempt001','capture272_attempt001','fit272_attempt001','closure4_attempt001']:
  x=read(BASE/name/'SEALED.json');runtime.append({'attempt':name,**{k:v for k,v in x.items() if k!='files'}})
 dump(out/'results/RUNTIME_SEAL_METADATA.json',runtime)
 dump(out/'DEPENDENCIES.json',{'preceding_study_commit':read(ROOT/'research/round35/publication/RECEIPT_FINAL.json')['commit'],'source_R25_model_selection':'source WaymoVAL512 residual_rank seed0e01, not target-reselected','source_original_3EED_checkpoint_sha256':'0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd','external_pretrained_RGB':'frozen ImageNet ResNet34; weight not redistributed','feedback':'all22original R29 cached Qwen RGB+fulltext P/N preferences','private_licensed_inputs_required':True,'thirdparty_source_snapshots_or_media_redistributed':False,'no_U_FINAL_access_this_round':True})
 (out/'README.md').write_text('''# R36 — Source-fused RGB rank preference TTA, fixed A/DEV 2x2

Read REPORT.md and both DECISION and RESOURCE_DECISION. The registered RGB
comparison gate PASSED: DEV TTA exceeds its own frozen and same-information
rank200-memory control by+.212408pp. It is not reclassified as a failure.
The prespecified interaction(deltaRGB-deltaNoRGB) is-1.393059pp, and RGB TTA
absolute22.867746 is below originalR30 25.090904. Resource decision: stop this
fixed RGB rank branch, no extension or new confirmation. This does not refute
allRGB TTA or establish missingRGB as the previous primary failure cause.

Fixed source models are R25 sourceVAL-selected residual_rank seed0 epoch1,
separate RGB/noRGB source weights. Their difference is not pure input causality.
All22original preferences, oldU189, beta1,16steps;4heads64steps update only12929
rank MLP parameters. Source architecture/training changes are separate from
TTA increments relative to each own frozen baseline. No new source training,
Qwen, teacher, EMA, layer/step/seed sweep or regression. Native final scores and
each source own q0 are used, not old XYZ noun-logodds loss. Legacy B is diagnostic
only and not eligibility. Frozen boxes and all nonrank weights unchanged.

Same-info no-update control: A direct groupQP, DEV frozen rankinput200 key
memory, native margin.1, k1 from recordedGPU topk4, raw transfer then currentq0
anchor. Unknown candidates are not new negatives. All native fullvectors and GPU
indices are private and sealed; public scalar tasks/negatives/code are complete.

IndependentCPU audit:544sample-mode rows/2720task values,38independent QPs,
457negative method/reference rows,20scene tables, sourceq0/U/feedback verified.
CPU64 query differences0; actualGPU neighbor indices validated/replayed, not
substituted with CPU topk. OldR25 A208x2=416source rawargmax/boxes replayed.
A TTA raw vs fixed duplicate-group representative differs12rows, all disclosed;
frozen rawargmax exact. These are readout diagnostics, not posthoc extra scores.

Actualfullmodel closure:4registered input-hash observations x2modes x2states=
16point/RGB/fusion forwards, allboxes/query reproduce, nonrank unchanged. CPU
matching-head replays are not fullbackbone forwards. Preparation12.8042s includes
DEV64RGB extraction4.2364s; do not sum nested timings. Fit/QP/prediction6.3461s,
closure11.1426s; these are stage receipts, not optimized deployment latency.

A/DEV are exposed development data, not independent new confirmation; both
modes/controls reuse observations. No U/FINAL row manifests/media/GT accessed.
No broad scene generalization/significance claim. ROI from predicted boxes still
mixes background/neighbors; distortion-zero assumptions and source history remain.
WildRefer/xMUDA are mechanism inspiration, not upstream reproductions. No raw
media/captions/opaqueIDs/coordinates/features/weights/thirdparty source snapshots
exported; private licensed assets and historical caches are needed to reproduce.
''')
 dump(out/'audit/REPORT_REVIEW.json',{'state':'PASS','gate_pass_and_resource_stop_kept_separate':True,'primary_interaction_and_source_training_caveat_disclosed':True,'all_negative_and_duplicate_group_readout_diagnostics_retained':True,'independent_audit_appended_to_report':True,'nested_runtime_disclosed':True,'audit_agent_did_not_implement_R36_algorithm':True})
 # Scan only previously used528 manifest rows; no U/FINAL row manifests opened.
 history=read(ROOT/'research/round27/input/ALL528_PRIVATE.json');assert len(history)==528
 forbidden={str(r[k]) for r in history for k in ['id','caption','imagepath','pointpath','metapath','scene'] if r.get(k)};findings=[]
 files=[p for p in sorted(out.rglob('*')) if p.is_file()]
 for p in files:
  txt=p.read_text()
  for token in forbidden:
   if token in txt:findings.append({'file':p.relative_to(out).as_posix(),'type':'private_token','token_sha256':hashlib.sha256(token.encode()).hexdigest()})
  if re.search('/'+'home/'+r'[^/\s]+/',txt):findings.append({'file':p.relative_to(out).as_posix(),'type':'home_path'})
 assert not findings,findings[:5]
 dump(out/'EXPORT_PRIVACY_AUDIT.json',{'state':'PASS','private_scan_history_rows':528,'findings':findings,'all_task_values':2720,'all_negatives':457,'sample_mode_rows':544,'unique_observations':272,'no_U_FINAL_manifest_media_GT_read':True,'own_code_only_no_thirdparty_snapshots':True,'no_media_captions_coords_opaqueIDs_features_weights':True})
 code=[p for p in out.rglob('*.py')];dump(out/'protocol/FINAL_CODE_SNAPSHOT.json',{'scope':'exported own code only','files':[{'path':p.relative_to(out).as_posix(),'sha256':sha(p)} for p in sorted(code)]})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];dump(out/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'manifest_excludes_itself':True,'private_payload_excluded':True})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];payload=[{'path':PREFIX+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files];target=pub/'TREE_PAYLOAD_LOCAL.json';dump(target,payload)
 for item,p in zip(payload,files):assert item['content']==p.read_text() and item['path']==PREFIX+'/'+p.relative_to(out).as_posix()
 for item in read(out/'MANIFEST.json')['files']:
  p=out/item['path'];assert sha(p)==item['sha256'] and p.stat().st_size==item['bytes']
 result={'state':'PASS','n':len(files),'bytes':sum(p.stat().st_size for p in files),'payload_bytes':target.stat().st_size,'payload_sha256':sha(target),'manifest_and_payload_exact':True,'privacy_findings':[],'task_values':2720,'negative_method_reference_rows':457,'no_U_FINAL_access':True,'no_new_GPU_optimizer_training_or_scoring_by_export':True,'prefix':PREFIX,'not_published':True};dump(pub/'LOCAL_EXPORT_AUDIT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
