"""R41 explicit anonymous scalar export; no arrays, model, media or task scoring."""
from pathlib import Path
import json,hashlib,csv,io,re
ROOT=Path(__file__).resolve().parents[2];HERE=ROOT/'research/round41';EXP=HERE/'experiment'
PREFIX='studies/3eed_cached_numeric_grounding_readout_diagnostic_20261102'
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
 audit=read(HERE/'independent_audit/AUDIT_RESULT.json');assert audit['state']=='PASS_R41_CACHED_ARITHMETIC_AND_EXACT_OLD_IOU_JOIN'
 join=read(EXP/'TASK_JOIN_RECEIPT.json');assert join['joined_values']==640 and join['interaction_pp']==0
 pub=HERE/'publication';pub.mkdir(exist_ok=True);out=pub/'public_export_attempt001';out.mkdir(exist_ok=False)
 def cp(p,dst):
  q=out/dst;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(safe(Path(p).read_text()))
 def cpj(p,dst):dump(out/dst,protocol(read(p)))
 cp(EXP/'REPORT.md','REPORT.md')
 for p in sorted((ROOT/'crossplatform/round41').glob('*.py')):cp(p,p.relative_to(ROOT))
 for p in [HERE/'review/readout41_draft.py',HERE/'independent_audit/audit41_independent.py',Path(__file__)]:cp(p,p.relative_to(ROOT))
 for n in ['ROOT_EXECUTION_AUTHORIZATION','IMPLEMENTATION_LOCK']:cpj(EXP/(n+'.json'),Path('protocol')/(n+'.json'))
 cpj(HERE/'review/PROPOSED_PLAN.json','history/PREEXECUTION_PROPOSED_PLAN.json');cp(HERE/'review/REVIEW.md','history/PREEXECUTION_REVIEW.md');cpj(HERE/'review/HISTORY_EVIDENCE.json','history/READ_ONLY_HISTORY_EVIDENCE.json');cpj(HERE/'review/SOURCES.json','SOURCES.json')
 for n in ['TASK_SUMMARY','SCENE_ROWS','MECHANISM_DECISION','TASK_JOIN_RECEIPT']:cpj(EXP/(n+'.json'),Path('results')/(n+'.json'))
 cpj(EXP/'CPU_ARITHMETIC_AUDIT.json','audit/PIPELINE_ARITHMETIC_AUDIT.json');cpj(HERE/'independent_audit/AUDIT_RESULT.json','audit/INDEPENDENT_AUDIT_RESULT.json')
 rr=read(EXP/'ALL128_TASK_ROWS_PRIVATE.json');assert len(rr)==128
 scenes={p:sorted({r['scene_PRIVATE'] for r in rr if r['platform']==p}) for p in ['drone','quad']};scalars=[]
 for r in rr:
  for arm,v in r['IoU'].items():scalars.append({'alias':r['alias'],'platform':r['platform'],'scene_alias':r['platform']+'_scene'+str(scenes[r['platform']].index(r['scene_PRIVATE'])),'arm':arm,'IoU':v,'query':r['queries'][arm],'query_changed_vs_sameforward_source':r['queries'][arm]!=r['queries']['source_native_replay']})
 assert len(scalars)==640
 buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(scalars[0]),lineterminator='\n');w.writeheader();w.writerows(scalars);(out/'results/ALL640_TASK_SCALARS.csv').write_text(buf.getvalue())
 neg=read(EXP/'ALL_NEGATIVES.json');allowed=['alias','platform','arm','reference','delta_pp'];neg=[{k:r[k] for k in allowed} for r in neg];assert len(neg)==86;dump(out/'results/ALL86_NEGATIVE_COMPARISONS.json',neg)
 # Explicit scalar diagnostic allowlist; no topset/query-score arrays copied.
 diag=[];fields=['weight','native_query','stable_query','native_exact_top_tie_n','stable_exact_top_tie_n','native_values_exactly_one','native_values_above_one','changed_by_readout','stable_winner_in_native_topset','change_is_native_top_tie_break','change_is_non_tie_numerical_reversal','logmass_topset_equal_logodds','logmass_sorted_monotone','logmass_not_subtracted_near_equal_LSEs','probability_sigmoid_logodds_maxerror']
 for r in read(HERE/'independent_audit/ALL128_NUMERICAL_DIAGNOSTICS.json'):
  bridge={k:r['source_bridge'][k] for k in ['source_replay_query','source_historical_query','query_changed','max_probability_abs_difference','not_TTA_or_readout_factor_gain']}
  diag.append({'alias':r['alias'],'platform':r['platform'],'source_bridge':bridge,'numerical_diagnostics':[{k:d[k] for k in fields} for d in r['numerical_diagnostics']]})
 assert len(diag)==128;dump(out/'results/ALL128_NUMERICAL_SCALARS_FLAGS.json',diag)
 seal=read(ROOT/'crossplatform/round41/cachedU128_attempt001/SEALED.json');allowed=['state','sealed_utc','n','arms','new_GPU_VLM_optimizer_steps','root_authorization_sha256','implementation_sha256','original_R35_seal_sha256','original_sourceU_seal_sha256','within_weight_readout_query_unchanged_all128','seconds'];dump(out/'protocol/PREDICTION_SEAL_METADATA.json',{k:seal[k] for k in allowed}|{'original_container_sha256':sha(ROOT/'crossplatform/round41/cachedU128_attempt001/SEALED.json'),'private_file_binding_map_not_exported':True})
 dump(out/'results/COSTS.json',{'new_model_forwards':0,'new_GPU_jobs':0,'new_VLM_calls':0,'new_optimizer_steps':0,'new_geometry_scores':0,'cached_prediction_seconds':seal['seconds'],'independent_CPU_audit_seconds':audit['seconds'],'whole_readonly_research_and_export_walltime_not_measured':True,'640_taskvalues_exactjoin_not640_forwards':True})
 decision={'run':'completed','measurement':'independent_CPU_arithmetic_and_exact_old_IoU_join_PASS','scientific_support':'no within-weight readout query effect on exposedU128; fixedweight numerical readout null, not absence of TTA parameter effect','resource':'STOP_no_GPU_no_newsplit_no_optimizer','deployment':'no_new_method_or_changed_default','existing_R30_delta_pp':join['native_TTA_delta_pp'],'readout_interaction_pp':0,'source_R30_weights_not_equal_and_query_changes_remain':{'drone':38,'quad':11},'all640taskvalues_all86negativecomparisons_retained':True,'FINAL_untouched':True};dump(out/'DECISION.json',decision)
 prior=ROOT/'research/round35/publication/RECEIPT_FINAL.json';dump(out/'DEPENDENCIES.json',{'prior_R34_R35_receipt_commit':read(prior).get('commit') if prior.exists() else None,'R30_and_R35_licensed_private_caches_and_states_required':True,'source':'sameWaymoOnly epoch100; boxes frozen; XYZ+text student','student_update_this_round':False,'U128_already_exposed_1Drone3Quad_scenes':True,'no_new_fullmodel_stablewinner_closure_or_rescaled_memory_control':True,'current_scope':'STOP_not_adoption','no_original_arrays_media_coords_captions_IDs_weights_or_thirdparty_source':True})
 (out/'README.md').write_text('''# R41 — cached numerical grounding readout diagnostic

Read REPORT.md and DECISION.json. For fixed source/R30 weights, same-GPU-forward
native noun probability versus saved token-logodds changes 0/128 queries in both
weights; historical source probability bridge also 0. Native highest ties and
exact1 probabilities both0. STOP the numerical readout branch, no newGPU or split.
R30 still changes38Drone/11Quad queries relative tosource; this is not TTA-zero.
Source14.960674%,R3016.313208%,same +1.352534pp underbothreadouts,interaction0.
Those taskvalues restate old sealedU128,not a new method gain/confirmation.

Independent reconstruction640complete vectors/640frozenboxes, immutablehead check,
seal beforetask analysis; exactjoin640existingIoUs byalias+query+box. No newtaskGT
container orgeometry scoring/model/VLM/optimizer. All640taskscalars,86negative
method/reference comparisons,128anonymousnumericalflags,4sceneaggregates retained.
Duplicate identical arms and repeated negatives are not independent evidence.
U is alreadyexposed,1Drone/3Quadscenes; no independent-sceneCI/newconfirmation.

Exact meansclosedformGPU32 LSE(P)-LSE(N),not arbitraryprecision orlogit(roundedp).
Mask,validtokens,.07,normalization,specials,sentinel andargmax tiecontractfixed.
source_replay_p pairedwithsourcez;historicalstoredp is onlybridge. Stablelogmass
computedfrom-logaddexp(0,-z),topsets/monotonicitychecked,notanotherGTselectedarm.
No newstablewinnerfullmodelclosure/rescaledmemorybenchmark ordeploymentclaim.
This fixedweight null does not negate priorlogodds training-gradient evidence.

history/PREEXECUTION_* are immutable proposal-time context,not currentrunstate.
Owncode/protocol/anonymousscalars only. Original probability/logodds vectors,
features,3Dboxes,positions,rawIDs/scenes/captions/paths/media/weights andthirdparty
source are excluded. Reproduction oforiginalseals requires licensedprivateinputs.
''')
 # Privacy scan reads only the already-authorized exposedU128 manifest as data.
 history=read(ROOT/'research/NIGHT_20261101/input/U128_PRIVATE.json');forbidden={str(r[k]) for r in history for k in ['id','caption','imagepath','pointpath','metapath','scene'] if r.get(k)};findings=[]
 for p in sorted(out.rglob('*')):
  if not p.is_file():continue
  txt=p.read_text()
  for token in forbidden:
   if token in txt:findings.append({'file':p.relative_to(out).as_posix(),'type':'private_token','sha256':hashlib.sha256(token.encode()).hexdigest()})
  if re.search('/'+'home/'+r'[^/\s]+/',txt):findings.append({'file':p.relative_to(out).as_posix(),'type':'home_path'})
 assert not findings,findings[:5]
 blocked={'id','id_PRIVATE','caption','scene','scene_PRIVATE','imagepath','pointpath','metapath','scores','complete_scores','probability_scores','exact_token_logodds','boxes','box','h','tokens','features','weights','raw_answer','raw_answer_PRIVATE','native_topset','stable_topset'};keyfind=[]
 def walk(x,path):
  if isinstance(x,dict):
   for k,v in x.items():
    if k in blocked:keyfind.append({'path':path,'key':k})
    walk(v,path)
  elif isinstance(x,list):
   for v in x:walk(v,path)
 for p in out.rglob('*.json'):walk(read(p),p.relative_to(out).as_posix())
 assert not keyfind,keyfind[:5]
 dump(out/'EXPORT_PRIVACY_AUDIT.json',{'state':'PASS','policy':'explicit alias/scalar/status allowlists; no original vector containers copied','taskscalars':640,'negativecomparisons':86,'anonymousnumericalobservationrows':128,'private_token_findings':findings,'blocked_json_field_findings':keyfind,'no_FINAL_or_new_taskGT_containers_read':True,'privacy_scan_manifest':'already_exposed_authorizedU128only'})
 dump(out/'protocol/FINAL_CODE_SNAPSHOT.json',{'own_code_only':[{'path':p.relative_to(out).as_posix(),'sha256':sha(p)} for p in sorted(out.rglob('*.py'))]})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];dump(out/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'manifest_excludes_itself':True})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];target=pub/'TREE_PAYLOAD_LOCAL.json';payload=[{'path':PREFIX+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files];dump(target,payload)
 for a in read(out/'MANIFEST.json')['files']:assert sha(out/a['path'])==a['sha256'] and (out/a['path']).stat().st_size==a['bytes']
 for x,p in zip(read(target),files):assert x['content']==p.read_text() and x['path']==PREFIX+'/'+p.relative_to(out).as_posix()
 result={'state':'PASS','n':len(files),'bytes':sum(p.stat().st_size for p in files),'payload_bytes':target.stat().st_size,'payload_sha256':sha(target),'prefix':PREFIX,'all640taskscalars_all86negativecomparisons_retained':True,'privacy_findings':[],'manifest_payload_sha_exact':True,'not_published':True};dump(pub/'LOCAL_EXPORT_AUDIT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
