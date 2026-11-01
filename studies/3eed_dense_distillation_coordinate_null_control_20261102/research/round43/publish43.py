"""Local R43 explicit anonymous scalar export. No scores/features/GT/model access."""
import json,csv,io,hashlib,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'research/round43';EXP=HERE/'experiment';PREFIX='studies/3eed_dense_distillation_coordinate_null_control_20261102'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def safe(s):return s.replace('WORKSPACE_ROOT','WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME')
def protocol(x):
 if isinstance(x,dict):return {k:protocol(v) for k,v in x.items() if k!='files'}
 if isinstance(x,list):return [protocol(v) for v in x]
 return safe(x) if isinstance(x,str) else x
def main():
 assert read(HERE/'independent_audit/AUDIT_RESULT.json')['state']=='PASS_INDEPENDENT_R43_ARRAY_LOSS_CONTROL_MEMORY_AUDIT'
 assert read(HERE/'independent_audit/TASK_AUDIT_RESULT.json')['taskvalues']==2512 and read(EXP/'DECISION.json')['state'].startswith('STOP_')
 pub=HERE/'publication';pub.mkdir(exist_ok=True);out=pub/'public_export_attempt001';out.mkdir(exist_ok=False)
 def cp(p,dst):
  q=out/dst;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(safe(Path(p).read_text()))
 def cpj(p,dst):dump(out/dst,protocol(read(p)))
 cp(EXP/'REPORT.md','REPORT.md');cpj(EXP/'DECISION.json','DECISION.json');cpj(HERE/'review/SOURCES.json','SOURCES.json')
 for p in sorted((ROOT/'crossplatform/round43').glob('*.py')):cp(p,p.relative_to(ROOT))
 for p in [HERE/'review/dense43_draft.py',HERE/'independent_audit/audit_score43_independent.py',HERE/'independent_premise/audit43_premise.py',Path(__file__),ROOT/'crossplatform/round24/fit24.py',ROOT/'crossplatform/round24/common24.py',ROOT/'crossplatform/round24/memory24.py']:cp(p,p.relative_to(ROOT))
 for n in ['ROOT_EXECUTION_AUTHORIZATION','PLAN_LOCK','PLAN_LOCK_INITIAL_UNEXECUTED','PLAN_LOCK_REFINEMENT001_UNEXECUTED']:cpj(EXP/(n+'.json'),Path('protocol')/(n+'.json'))
 for n in ['PLAN_LOCK','IMPLEMENTATION_LOCK']:cpj(HERE/'premise'/(n+'.json'),Path('protocol/premise')/(n+'.json'))
 for n in ['PROPOSED_PLAN','DRAFT_STATUS']:cpj(HERE/'review'/(n+'.json'),Path('history')/(n+'.json'))
 cp(HERE/'review/EXPERIMENT_DRAFT.md','history/PREEXECUTION_EXPERIMENT_DRAFT.md')
 for n in ['PREMISE_STATIC_REVIEW','DRAFT_STATIC_REVIEW']:cp(HERE/'independent_review'/(n+'.md'),Path('audit/static')/(n+'.md'))
 for n in ['STATIC_REVIEW_RESULT','DRIVER_STATIC_REVIEW_RESULT']:cpj(HERE/'independent_review'/(n+'.json'),Path('audit/static')/(n+'.json'))
 cp(HERE/'premise/REPORT.md','premise/REPORT.md');cpj(HERE/'premise/SUMMARY.json','premise/SUMMARY.json');cp(HERE/'independent_premise/REPORT.md','audit/premise/REPORT.md');cpj(HERE/'independent_premise/AUDIT_RESULT.json','audit/premise/AUDIT_RESULT.json');cpj(HERE/'independent_premise/INDEPENDENT_REDUCTIONS.json','audit/premise/INDEPENDENT_REDUCTIONS.json')
 # Pure scalar summaries only, excluding source/bank file hashes and raw IDs.
 keys=['F32_range','F32_range_gt1','teacher_peak_CPU64','teacher_peak_CPU32','teacher_peak_gt_ideal_native_cap','teacher_entropy_nats','source_probability_posterior_entropy_nats','source_z_posterior_entropy_nats','KL_teacher_to_source_probability','KL_teacher_to_source_z','null_KL_source_probability_to_source_probability','null_KL_source_probability_to_source_z','deltaG_allzero','deltaG_constant','all_component_evidence_zero','deltaG_range','teacher_argmax_differs_source','initial_native_hinge_fullG','initial_native_hinge_nullG','teacher_peak_Bernoulli_KL_lowerbound','CPU32_CPU64_teacher_posterior_maxerror','CPU32_CPU64_source_posterior_maxerror','F32_interpreted_real_exact_span']
 premise=[]
 for r in read(HERE/'independent_premise/INDEPENDENT_ROWS_PRIVATE.json'):
  signs={k:{sign:r['evidence_sign_counts'][k][sign] for sign in ['positive','negative','zero']} for k in ['vision','position','relation']};premise.append({'alias':r['alias'],'platform':r['platform'],**{k:r[k] for k in keys},'evidence_sign_counts':signs})
 assert len(premise)==208;dump(out/'premise/ALL208_SCALAR_FLAGS.json',premise)
 for n in ['AUDIT_RESULT','TASK_AUDIT_RESULT','HEAD_REPLAY_ROWS','LOSS_REPLAY','CONTROL_REPLAY','MEMORY_REPLAY_ROWS']:cpj(HERE/'independent_audit'/(n+'.json'),Path('audit/runtime')/(n+'.json'))
 for n in ['TASK_SUMMARY','FACTOR_CONTRASTS','SCENE_ROWS']:cpj(EXP/(n+'.json'),Path('results')/(n+'.json'))
 # Every task value and negative comparison, explicit row fields only.
 task=read(EXP/'ALL_ROWS_PRIVATE.json');assert len(task)==272;sceneindex={p:sorted({r['scene_PRIVATE'] for r in task if r['platform']==p}) for p in ['drone','quad']};scalar=[]
 for r in task:
  for a,v in r['IoU'].items():scalar.append({'alias':r['alias'],'platform':r['platform'],'split':r['split'],'scene_alias':r['platform']+'_scene'+str(sceneindex[r['platform']].index(r['scene_PRIVATE'])),'arm':a,'IoU':v,'query':r['queries'][a],'query_changed_vs_source':r['queries'][a]!=r['queries']['source']})
 assert len(scalar)==2512;buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(scalar[0]),lineterminator='\n');w.writeheader();w.writerows(scalar);(out/'results/ALL2512_TASK_SCALARS.csv').write_text(buf.getvalue())
 neg=[{k:r[k] for k in ['alias','platform','split','arm','reference','delta_pp']} for r in read(EXP/'ALL_NEGATIVES.json')];assert len(neg)==2465;dump(out/'results/ALL2465_NEGATIVE_COMPARISONS.json',neg)
 # Fit receipts contain only aggregate losses/costs and no vectors or data IDs.
 base=ROOT/'crossplatform/round43/fit272_attempt001';cpj(base/'FIT_RECEIPTS.json','results/ALL8_FIT_RECEIPTS.json');cpj(base/'MATCHED_CONTROL_REPLAY.json','audit/runtime/MATCHED_CONTROL_REPLAY.json')
 failed=[]
 for n in [1,2,3]:
  p=HERE/'independent_premise'/f'audit_attempt{n:03d}.log';cp(p,Path('history/independent_premise')/p.name);failed.append({'attempt':n,'process_outcome':'output_serialization_failed' if n<3 else 'success','numerical_assertions_failed':False,'author_outputs_modified':False,'kind':'NumPy int64 JSON serialization' if n<3 else 'same numerical audit with output serialization fixed','source_log_sha256':sha(p)})
 dump(out/'history/ALL_PREMISE_AUDIT_ATTEMPTS.json',failed)
 cp(EXP/'FIT_ATTEMPT001.log','history/FIT_ATTEMPT001.log');cp(HERE/'premise/RUN_ATTEMPT001.log','history/PREMISE_RUN_ATTEMPT001.log')
 # All old unexecuted plan refinements retained; no failed new method retry.
 dump(out/'history/ATTEMPT_HISTORY.json',{'fit_attempts':1,'fit_success':True,'method_or_control_failure_retry':False,'unexecuted_plan_snapshots_preserved':['PLAN_LOCK_INITIAL_UNEXECUTED.json','PLAN_LOCK_REFINEMENT001_UNEXECUTED.json'],'refinement':'poststep16 loss/full z/mask/padding audit observability only beforeGPU; no objective/budget change','independent_premise_audit_output_failures':2,'failed_assertions_in_premise_audit':0,'original_two_output_error_logs_preserved':True})
 seals=[]
 for tag,p in [('premise',ROOT/'crossplatform/round43/teacher_premise208_attempt001/SEALED.json'),('fit',base/'SEALED.json')]:
  s=read(p);allow=['state','n','A','V','states','optimizer_steps','same_forward_fullmodel_closure','all3Dboxes_original','no_GT_or_newG_or_FINAL_U_T','no_GPU_model_fit_GT_taskscore','plan_sha256','root_authorization_sha256','implementation_sha256','seconds','peak_bytes','created_utc'];seals.append({'kind':tag,**{k:s[k] for k in allow if k in s},'original_container_sha256':sha(p),'private_file_binding_map_excluded':True})
 dump(out/'protocol/SEALED_RUNTIME_METADATA.json',seals);fitseal=read(base/'SEALED.json');dump(out/'results/COSTS.json',{'cached_query_head_GPU_seconds':fitseal['seconds'],'peak_GPU_allocated_bytes':fitseal['peak_bytes'],'trained_states':8,'optimizer_steps':128,'parameters_per_state':184960,'new_fullgrounder_forward_calls':0,'new_VLM_G_calls':0,'new_source_training':0,'independent_runtime_audit_seconds':read(HERE/'independent_audit/AUDIT_RESULT.json')['seconds'],'independent_premise_audit_seconds':read(HERE/'independent_premise/AUDIT_RESULT.json')['seconds'],'task_geometry_values':2512,'task_score_walltime_not_instrumented':True,'CPU_output_retry_times_not_fabricated':True,'no_fullgrounderclosure_expansion_after_STOP':True})
 dump(out/'DEPENDENCIES.json',{'source':'originalR24Waymo-only100; sameXYZtext sourcecheckpoint andfrozen6Dboxes','original_R24_positive_M64_result_retained_not_rescored':{'queryMLP_delta_vs_source_pp':.466143,'queryMLP_plus_G_delta_vs_direct_G_pp':.478950},'prior_R41_public_commit':'ca01c115b3413492d1dd9d1fe48f5052d21eec08','licensed_originalR24capture336_andbank272_states_required':True,'no_actual_scores_probs_features_weights_captions_coords_or_media_distributed':True,'R30_reference_not_causal_control':True,'V64_already_exposed_development':True,'no_U_T_FINAL_read_or_newmethod_adoption':True})
 (out/'README.md').write_text('''# R43 — dense teacher distribution coordinate × null control

Read REPORT.md and DECISION.json: STOP this fixed z configuration. V64 source
28.063498%, fullprob28.972262%, zfull1.251577%, nullprob27.736370%, znull1.604630%.
z-prob full-27.720685pp/null-26.131740pp, interaction-1.588946pp. DirectG31.178995,
oldprob+currentG31.907429, z+G4.407487, oldmemory26.037296. A also deteriorates.
Both z arms alterall64V queries; allbenefits/harms retained. Null failure means
wrongG is notnecessary for this failure, not proof of auniquegradient mechanism.
OriginalR24positive result is preserved; do not generalize toalllogitsTTA.

8source-initialized queryMLPs184960params,16stepseach128total,4.04sGPU/461MB.
Only teacherKL studentp→pretokenz changes; native sourceKL/hinge/paramanchor,
temperature,finalnative readout remainR24. Full/null teachers fixed acrosscoords.
Bothplatform originalfullprob states bitexact/A+V272queries andtasks exactbefore
attribution. No fit29loaderhook. Sourcecapture andbankbytes checked. No VLM,
newsource/RGB/boxupdate orfullgrounderclosure. GPU released afterfixedhead fit.

Premise94/208teacher spans>1 and20peaks exceednativecap, but109zeroDeltaG rows
have initialKL22.7983 underz,0underprob. All208rows256candidates positive/negative/
zero values retained locally; zero is notsemanticUNKNOWN. Range violation does
not proveGtrue. Rankadapter wasunbounded historicalcounterexample. Full/null2x2
separates fixedrecipeGincrement fromstrongcoordinatecalibration; noGTfilter.

Independentpremise fullnumericreplay PASS; CPU32posteriorsbitexact, SciPyreplay.
Independent8head1088native/zvectors,post16losses/control/memory PASS. CPU64argmax
differences0, oldmemoryall64CPU32queries reproduced. Nativevectors audited not
published. All272predictionrows represented by2512IoUs, all2465negativecompare
rows, everyheadloss and208premisescalarrows public. Repeatedarmcomparisons are
not independentobservations. V64 alreadyexposed,2Drone/3Quadscenes; nonewconfirm.

Two premiseaudit outputserialization TypeErrors retained with sanitizedlogs;
numeric assertions didnotfail, attempt003fixedserialization only. Original
unexecuteddriverplan refinements retained; nosemanticretry orGTchosenwinner.
history/PREEXECUTION* are proposal-timecontext, currentSTOP isauthoritative.

Onlyowncode/protocol/anonymousIoU/query/count/statistic/flags. No actualperquery
scores/probabilities/tokenfeatures,weights,IDs/captions/realpaths/scenes/coords,
media or thirdparty source/paper snapshots. Hinton/RLCF arelinked inspiration,
not reproducedbaselines. Licensedprivateassets needed fororiginalseals.
''')
 # Exact private token scan only original authorized exposed A208/V64 manifests.
 history=read(ROOT/'research/round24/input/D208_PRIVATE.json')+read(ROOT/'research/round24/input/V64_PRIVATE.json');forbidden={str(r[k]) for r in history for k in ['id','caption','imagepath','pointpath','metapath','scene'] if r.get(k)};findings=[]
 for p in sorted(out.rglob('*')):
  if not p.is_file():continue
  txt=p.read_text()
  for token in forbidden:
   if token in txt:findings.append({'file':p.relative_to(out).as_posix(),'type':'private_token','sha256':hashlib.sha256(token.encode()).hexdigest()})
  if re.search('/'+'home/'+r'[^/\s]+/',txt):findings.append({'file':p.relative_to(out).as_posix(),'type':'home_path'})
 assert not findings,findings[:10]
 blocked={'id','id_PRIVATE','caption','scene','scene_PRIVATE','imagepath','pointpath','metapath','scores','source_scores','teacher_pT','source_p0','teacher_F32','teacher_F64','trained_token_logodds','boxes','box','tokens','features','weights','raw_answer','A_ids','V_ids','source_file_sha256','bank_file_sha256'};keyfind=[]
 def walk(x,path):
  if isinstance(x,dict):
   for k,v in x.items():
    if k in blocked:keyfind.append({'file':path,'key':k})
    walk(v,path)
  elif isinstance(x,list):
   for v in x:walk(v,path)
 for p in out.rglob('*.json'):walk(read(p),p.relative_to(out).as_posix())
 assert not keyfind,keyfind[:10]
 dump(out/'EXPORT_PRIVACY_AUDIT.json',{'state':'PASS','policy':'explicit scalar/flag/task allowlists; no originalvector containers copied','taskscalarvalues':2512,'negativecomparisons':2465,'predictionobservationrows':272,'premisescalarrows':208,'headerrorrows':1088,'all8headlossreceipts':True,'all_failed_output_attempts_retained':True,'private_token_findings':findings,'blocked_structured_fields':keyfind,'no_FINAL_U_T_or_newGT_inputs_read_for_export':True})
 dump(out/'protocol/FINAL_CODE_SNAPSHOT.json',{'own_code_only':[{'path':p.relative_to(out).as_posix(),'sha256':sha(p)} for p in sorted(out.rglob('*.py'))]})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];dump(out/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'manifest_excludes_itself':True})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];payload=[{'path':PREFIX+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files];target=pub/'TREE_PAYLOAD_LOCAL.json';dump(target,payload)
 for x in read(out/'MANIFEST.json')['files']:assert sha(out/x['path'])==x['sha256'] and (out/x['path']).stat().st_size==x['bytes']
 for x,p in zip(read(target),files):assert x['content']==p.read_text() and x['path']==PREFIX+'/'+p.relative_to(out).as_posix()
 result={'state':'PASS','n':len(files),'bytes':sum(p.stat().st_size for p in files),'payload_bytes':target.stat().st_size,'payload_sha256':sha(target),'prefix':PREFIX,'all2512taskvalues_all2465negativecomparisons_retained':True,'all272predictionobservations_and208premisesrepresented':True,'all_failed_output_attempts_retained':True,'privacy_findings':[],'manifest_payload_sha_exact':True,'not_published':True};dump(pub/'LOCAL_EXPORT_AUDIT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
