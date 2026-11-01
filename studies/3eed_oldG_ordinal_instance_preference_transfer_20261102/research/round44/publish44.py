"""R44 both phases, scalar allowlists only; local export never publishes."""
from pathlib import Path
import json,csv,io,hashlib,re
ROOT=Path(__file__).resolve().parents[2];H=ROOT/'research/round44';P1=H/'independent_phase1';P2=H/'independent_phase2';PREFIX='studies/3eed_oldG_ordinal_instance_preference_transfer_20261102'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def safe(s):return s.replace('WORKSPACE_ROOT','WORKSPACE_ROOT').replace('LOCAL_HOME','LOCAL_HOME')
def protocol(x):
 if isinstance(x,dict):return {k:protocol(v) for k,v in x.items() if k!='files'}
 if isinstance(x,list):return [protocol(v) for v in x]
 return safe(x) if isinstance(x,str) else x
def main():
 assert read(P1/'SCORING_RECEIPT.json')['task_values']==624 and read(P2/'SCORING_RECEIPT.json')['task_values']==1424
 pub=H/'publication';pub.mkdir(exist_ok=True);out=pub/'public_export_attempt001';out.mkdir(exist_ok=False)
 def cp(p,dst):q=out/dst;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(safe(Path(p).read_text()))
 def cpj(p,dst):dump(out/dst,protocol(read(p)))
 cp(H/'COMBINED_REPORT.md','REPORT.md');cpj(H/'COMBINED_DECISION.json','DECISION.json')
 for p in sorted((ROOT/'crossplatform/round44').glob('*.py')):cp(p,p.relative_to(ROOT))
 for p in [P1/'audit_score44.py',P2/'audit_score44_phase2.py',ROOT/'research/round43/independent_audit/audit_score43_independent.py',H/'finalize44_combined.py',Path(__file__)]:cp(p,p.relative_to(ROOT))
 for ph in ['phase1','phase2']:
  cp(H/ph/'REPORT.md',Path('history')/(ph+'_REPORT.md'));cpj(H/ph/'DECISION.json',Path('results')/ph/'DECISION.json')
  for n in ['ROOT_EXECUTION_AUTHORIZATION','PLAN_LOCK']:cpj(H/ph/(n+'.json'),Path('protocol')/ph/(n+'.json'))
 cpj(H/'phase2/IMPLEMENTATION_LOCK.json','protocol/phase2/IMPLEMENTATION_LOCK.json');cpj(H/'phase2/PROPOSED_PLAN.json','history/PREEXECUTION_PHASE2_PLAN.json')
 for n in ['INDEPENDENT_STATIC_REVIEW.md','ACTUAL_DRIVER_STATIC_REVIEW.json']:
  p=H/'phase2'/n;cpj(p,Path('audit/static')/n) if p.suffix=='.json' else cp(p,Path('audit/static')/n)
 cp(P2/'AUDIT_PLAN.md','audit/static/INDEPENDENT_AUDIT_PLAN.md')
 cp(H/'review/REVIEW.md','history/READ_ONLY_MECHANISM_CARD.md');cpj(H/'review/HISTORY_EVIDENCE.json','history/HISTORY_EVIDENCE.json');cpj(H/'review/SOURCES.json','SOURCES.json');cpj(H/'review/PROPOSED_TWO_STAGE_PLAN.json','history/PREEXECUTION_TWO_STAGE_PLAN.json')
 for n in ['RESULT','CPU_GATE_PLAN_LOCK','CPU_GATE_IMPLEMENTATION_LOCK']:cpj(H/'qualification'/(n+'.json'),Path('qualification')/(n+'.json'))
 gate=[]
 for r in read(ROOT/'crossplatform/round44/source_exact_gate_attempt001/ALL208_EXACT_COMPARISONS_PRIVATE.json'):
  keys=['exact','shape_R24','shape_R37','dtype_R24','dtype_R37','numeric_equal','raw_bytes_equal','finite','max_abs_difference_diagnostic_only'];gate.append({'alias':r['alias'],'platform':r['platform'],'pass':r['pass'],'source_box_stats':{k:r['boxes'][k] for k in keys},'source_score_stats':{k:r['scores'][k] for k in keys}})
 assert len(gate)==208;dump(out/'qualification/ALL208_EXACT_SOURCE_SCALARS.json',gate)
 for n in ['READOUT_AUDIT_RESULT','SCORING_RECEIPT','TASK_SUMMARY']:cpj(P1/(n+'.json'),Path('audit/phase1')/(n+'.json'))
 for n in ['AUDIT_RESULT','SCORING_RECEIPT','HEAD_REPLAY_ROWS','LOSS_REPLAY','ALL32_FEEDBACK_EXECUTION','ALL128_MEMORY_REPLAY_ROWS']:cpj(P2/(n+'.json'),Path('audit/phase2')/(n+'.json'))
 for n in ['TASK_SUMMARY','SCENE_ROWS']:cpj(H/'phase2'/(n+'.json'),Path('results/phase2')/(n+'.json'))
 sets={'phase1':read(P1/'ALL208_TASK_ROWS_PRIVATE.json'),'phase2':read(H/'phase2/ALL_ROWS_PRIVATE.json')};allr=sets['phase1']+sets['phase2'];scenes={pl:sorted({r['scene_PRIVATE'] for r in allr if r['platform']==pl}) for pl in ['drone','quad']};counts={};negcounts={};lookup2={r['id_PRIVATE']:r for r in sets['phase2'] if r['split']=='D208'}
 for ph,rows in sets.items():
  scalar=[]
  for r in rows:
   split=r.get('split','D208')
   for a,v in r['IoU'].items():scalar.append({'alias':r['alias'],'platform':r['platform'],'split':split,'scene_alias':r['platform']+'_scene'+str(scenes[r['platform']].index(r['scene_PRIVATE'])),'arm':a,'IoU':v,'query':r['queries'][a],'changed_vs_source':r['queries'][a]!=r['queries']['source']})
  buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(scalar[0]),lineterminator='\n');w.writeheader();w.writerows(scalar);p=out/'results'/ph/'ALL_TASK_SCALARS.csv';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(buf.getvalue());counts[ph]=len(scalar)
  npath=P1/'ALL_NEGATIVES.json' if ph=='phase1' else H/'phase2/ALL_NEGATIVES.json';neg=[{k:r[k] for k in ['alias','platform','arm','reference','delta_pp']}|({'split':r['split']} if 'split' in r else {'split':'D208'}) for r in read(npath)];dump(out/'results'/ph/'ALL_NEGATIVE_COMPARISONS.json',neg);negcounts[ph]=len(neg)
 assert counts=={'phase1':624,'phase2':1424} and negcounts=={'phase1':62,'phase2':622}
 p1scenes=[{'platform':r['platform'],'scene_alias':r['platform']+'_scene'+str(scenes[r['platform']].index(r['scene_PRIVATE'])),'n':r['n'],'eligible_n':r['eligible_n'],'mIoU_percent':r['mIoU_percent'],'few_existing_scenes_descriptive_only':True} for r in read(P1/'SCENE_SUMMARIES_PRIVATE.json')];dump(out/'results/phase1/SCENE_SUMMARIES.json',p1scenes)
 # Preserve all qualification statuses including169no-change/7unknown, not groups.
 flags=[];feedback=[];pairs={r['alias']:r for r in read(P2/'ALL32_FEEDBACK_EXECUTION.json')}
 for r in sets['phase1']:
  d=r['decision'];flags.append({'alias':r['alias'],'platform':r['platform'],**{k:d[k] for k in ['source_q0','raw_G_q','G_F64_gap_vs_q0','source_group_rep','G_group_rep','eligible','reason','query','distinct_mask_hypothesis_not_gold_identity','unknown_not_labelled_negative']},'source_unique_group_count':len(d['source_membership_masks']),'G_unique_group_count':len(d['G_membership_masks']),'negative_group_size':len(d['source_group_members']),'positive_group_size':len(d['G_group_members'])})
  if d['eligible']:
   delta=100*(r['IoU']['eligible_fixed_source_rep']-r['IoU']['source']);q=lookup2[r['id_PRIVATE']];feedback.append({'alias':r['alias'],'platform':r['platform'],'phase1_delta_pp_offline_only':delta,'phase1_outcome':'harm' if delta< -1e-8 else 'benefit' if delta>1e-8 else 'neutral','positive_group_size':len(d['G_group_members']),'negative_group_size':len(d['source_group_members']),'all_used_for_fit_without_GT_filter':True,'phase2_group_delta_pp_offline_only':100*(q['IoU']['TTA_group_fixed_source_rep']-q['IoU']['source']),**{k:pairs[r['alias']][k] for k in ['initial_gap','final_gap','pair_satisfied','final_in_positive','final_in_negative','third_unjudged']}})
 assert len(flags)==208 and len(feedback)==32 and sum(r['phase1_outcome']=='harm' for r in feedback)==1 and sum(r['phase1_outcome']=='neutral' for r in feedback)==6;dump(out/'results/phase1/ALL208_QUALIFICATION_SCALAR_FLAGS.json',flags);dump(out/'results/ALL32_FEEDBACK_OFFLINE_TASK_DIAGNOSTIC.json',feedback)
 bridge=[{k:r[k] for k in ['alias','platform','F64_query','F32_query','F64_queryscore_gap','diagnostic_only_no_F32_readout_substitution']} for r in read(P1/'F64_F32_DIFFERENCES_PRIVATE.json')];assert len(bridge)==3;dump(out/'audit/phase1/ALL3_F64_F32_ARGMAX_DIAGNOSTICS.json',bridge)
 # Pure own fit scalar trace; explicit preference-scalar allowlist, no memberships.
 receipts=[]
 for r in read(ROOT/'crossplatform/round44/fit272_attempt001/FIT_RECEIPTS.json'):
  top=['platform','trainable_params','steps','lr','beta','pair_temperature','E','U','source_probability_replay_maxerrors','source_query_replay_exact','z0_no_grad','trace','initial_terms','final_terms_after_step16','parameter_delta_L2'];fkeys=['alias','row_index','initial_gap_z','final_gap_z','pair_satisfied_z','raw_query','fixed_source_group_query','winner_in_positive','winner_in_negative','third_unjudged_winner','third_not_automatically_negative'];receipts.append({**{k:r[k] for k in top},'preferences':[{k:f[k] for k in fkeys} for f in r['preferences']]})
 dump(out/'results/phase2/ALL2_FIT_RECEIPTS.json',receipts)
 runtime=[]
 for tag,path in [('qualification',ROOT/'crossplatform/round44/source_exact_gate_attempt001'),('phase1',ROOT/'crossplatform/round44/direct208_attempt001'),('phase2',ROOT/'crossplatform/round44/fit272_attempt001')]:runtime.append({'stage':tag,**protocol(read(path/'SEALED.json')),'original_container_sha256':sha(path/'SEALED.json'),'private_filemap_not_redistributed':True})
 dump(out/'protocol/SEALED_RUNTIME_METADATA.json',runtime)
 costs=read(ROOT/'crossplatform/round44/fit272_attempt001/SEALED.json');dump(out/'results/COSTS.json',{'cached_head_and_memory_process_seconds':costs['seconds'],'peak_GPU_allocated_bytes':costs['peak_bytes'],'states':2,'optimizer_steps':32,'new_fullgrounder_or_VLM_or_G_calls':0,'independent_phase2_fullNN_audit_seconds':read(P2/'AUDIT_RESULT.json')['seconds'],'source_gate_no_G_or_rank_prediction':True,'no_additional_fullgrounderclosure_or_FINAL_U_T':True,'all503316480_CPU32_similarity_cells_recomputed':True})
 dump(out/'history/ATTEMPT_HISTORY.json',{'source_exact_gate_attempts':1,'phase1_prediction_attempts':1,'phase2_training_attempts':1,'independent_phase1_and_phase2_audits_first_attempt_PASS':True,'failed_method_or_technical_retry':False,'all_unknown_negative_neutral_feedback_retained':True,'no_postscore_steps_temperature_layers_or_GT_filter':True})
 cp(H/'qualification/RUN_ATTEMPT001.log','history/QUALIFICATION_RUN_ATTEMPT001.log');cp(H/'phase1/PREDICT_ATTEMPT001.log','history/PHASE1_PREDICT_ATTEMPT001.log')
 for n in ['AUDIT_ATTEMPT001.log','SCORE_ATTEMPT001.log']:cp(P2/n,Path('history')/n)
 (out/'README.md').write_text('''# R44 — old-G ordinal instance preferences, direct versus shared-head transfer

Phase1directeligibleG passesprelockedN>0/wholeAgain>0: source19.663563% ->24.332121%
(+4.668558pp), unrestrictedG24.727373%. E32=25benefit/1harm/6neutral, allusedunchanged;
169nocorrectivenomination+7unknownmappingretained. Differentmaskgroupsnotgold.
Source256boxes/scores R24/R37exact208; unrestrictedGhistorical208q/boxesexact;
3F64/F32argmaxdifferencesdiagnosticonly,noposthocprecisionswap.

Phase2STOP:sourceinitializedqueryMLP2heads32steps,R30RankNetz/.1+beta1relq0keep,
E14/18,U90/86,sourceq0/z0frozen,allgroupsfixedsourcerepA/nativepV. No denseKL/extra
hinge/paramanchor/newteacherarchitecture. OldGstillcontainsfrozenexternalVLM
knowledge,notteacher-free supervision. Newfeedback/group/keep/denomcoupled,
notpuresupervisioncausalityornewRankNet. OriginalR24/R30positiveresultsretained.

A TTAgroup19.649967%, Vsource28.063498/TTA25.400646/E-memory23.273275/allA0U-memory
25.866269%. TTA V-2.662852pp,vsmemory-.465623pp. CurrentGdirect31.178995 vsTTA+sameG
28.761805(extra-currentinfo,notpuretransfer).19/32pairsgappositive,notfullyfitted;
noaddsteps/temperature/layers/fullclosure/FINAL. DirectGsignalremainspositive.

Independent272head/loss/sourcefreeze auditPASS,p1.24e-6/z6.82e-6,qdiff0;full503m
CPU32NNcells/max/top4/ties/values/weights/correctionsrecomputed,128retrievalsPASS.
2boundaryties,actualvsreplayedindices0diff. Memory native-p units only, no z/QP
coordinate substitution. Actualneighborvectorsprivate; scalarproofs public.

All624+1424IoUs/62+622negativecomparisons kept. RepeatedA/controlvaluesnotindependent,
uniqueA208+V64=272exposedobservations,fewscenes,nonewconfirmationorCIclaim.
Phase1gatePASSmerelyallowedseparatefinitephase2,itwasnotmethodadoption.
Owncode/protocol/anonymousquery/IoU/count/error/status/flags only; no actualF/p/z,
keys/features/weights/groups/maskIDs/coordinates/captions/rawIDs/scenes/paths/media.
OriginallicensedR24/R37/privatecachesneeded. Stagehistoricalreports reflecttheir
then-currentauthorizationstatus;combinedREPORT/DECISION governcompletedstudy.
''')
 history=read(ROOT/'research/round24/input/D208_PRIVATE.json')+read(ROOT/'research/round24/input/V64_PRIVATE.json');forbidden={str(r[k]) for r in history for k in ['id','caption','imagepath','pointpath','metapath','scene'] if r.get(k)};findings=[]
 for p in out.rglob('*'):
  if not p.is_file():continue
  txt=p.read_text()
  for token in forbidden:
   if token in txt:findings.append({'file':p.relative_to(out).as_posix(),'type':'private_token','sha256':hashlib.sha256(token.encode()).hexdigest()})
  if re.search('/'+'home/'+r'[^/\s]+/',txt):findings.append({'file':p.relative_to(out).as_posix(),'type':'home_path'})
 assert not findings,findings[:6]
 blocked={'id','id_PRIVATE','caption','scene','scene_PRIVATE','imagepath','pointpath','metapath','scores','boxes','keys','A_keys','V_keys','weights','source_group_members','G_group_members','full_groups','positive_queries','negative_queries','positive_mask','negative_mask','adapted_native_probability','adapted_token_logodds','source_A_z','source_scores','oldG_F64','deltaG_F64','mask_index','A_ids','V_ids'};keyfind=[]
 def walk(x,path):
  if isinstance(x,dict):
   for k,v in x.items():
    if k in blocked:keyfind.append({'file':path,'key':k})
    walk(v,path)
  elif isinstance(x,list):
   for v in x:walk(v,path)
 for p in out.rglob('*.json'):walk(read(p),p.relative_to(out).as_posix())
 assert not keyfind,keyfind[:6]
 dump(out/'EXPORT_PRIVACY_AUDIT.json',{'state':'PASS','explicit_task_status_feedback_scalar_allowlists':True,'task_values':counts,'negative_comparisons':negcounts,'all32_feedback_including_1harm6neutral':True,'all169nocorrective7unknown_retained':True,'all208compatibility_rows':True,'private_token_findings':findings,'blocked_json_field_findings':keyfind,'no_original_vectors_features_weights_groups_coords_rawIDs_media':True,'no_FINAL_U_T_or_new_GT_read_for_export':True})
 dump(out/'protocol/FINAL_CODE_SNAPSHOT.json',{'own_code_only':[{'path':p.relative_to(out).as_posix(),'sha256':sha(p)} for p in sorted(out.rglob('*.py'))]});files=[p for p in sorted(out.rglob('*')) if p.is_file()];dump(out/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'manifest_excludes_itself':True})
 files=[p for p in sorted(out.rglob('*')) if p.is_file()];target=pub/'TREE_PAYLOAD_LOCAL.json';dump(target,[{'path':PREFIX+'/'+p.relative_to(out).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files])
 for r in read(out/'MANIFEST.json')['files']:assert sha(out/r['path'])==r['sha256'] and (out/r['path']).stat().st_size==r['bytes']
 for r,p in zip(read(target),files):assert r['content']==p.read_text() and r['path']==PREFIX+'/'+p.relative_to(out).as_posix()
 result={'state':'PASS','n':len(files),'bytes':sum(p.stat().st_size for p in files),'payload_bytes':target.stat().st_size,'payload_sha256':sha(target),'prefix':PREFIX,'all2048taskvalues_684negativecomparisons_retained':True,'all32feedback_1harm6neutral_and_unknown_preserved':True,'privacy_findings':[],'manifest_payload_allSHAexact':True,'not_published':True};dump(pub/'LOCAL_EXPORT_AUDIT.json',result);print(json.dumps(result))
if __name__=='__main__':main()
