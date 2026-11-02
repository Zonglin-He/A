"""Unified local R46 release. No GitHub writes, models, GT scoring or new fitting."""
import importlib.util,json,hashlib,re,csv,io
from pathlib import Path
R=Path(__file__).resolve().parents[3];F=R/'research/round46/final';P=R/'research/round46/publication';PREFIX='studies/3eed_locked_crossplatform_TTA_final_validation_20261102'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def module(name,path):s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def opaque_hex_matches(text):
 for m in re.finditer(r'(?<![A-Za-z0-9])[0-9a-f]{16}(?![A-Za-z0-9])',text):
  before=text[:m.start()];after=text[m.end():];decimal_prefix=bool(re.search(r'(?<![A-Za-z0-9_.])[0-9]+\.$',before));number_end=bool(re.match(r'(?=$|[^A-Za-z0-9_./])',after));scientific_end=bool(re.match(r'[+\-−][0-9]+(?=$|[^A-Za-z0-9_./])',after))
  if decimal_prefix and (m.group().isdigit() and number_end or m.group().endswith('e') and m.group()[:-1].isdigit() and scientific_end):continue
  yield m
def main():
 P.mkdir(exist_ok=True);O=P/'public_export_attempt003';base=module('base46',F/'prepare_export_base46.py');result=module('results46',F/'prepare_result_export46.py');base_receipt=base.build_base(O);counts=result.build_results(O)
 q=O/'code'/Path(__file__).relative_to(R);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(result.safe(Path(__file__).read_text()))
 (O/'README.md').write_text('''# R46 — locked cross-platform grounding TTA evaluation

The predeclared primary R30 and secondary R24 do not improve overall frozen
source performance on this locked384-observation pool. Balanced mIoU source
14.973844%, R3014.802516%(-.171328pp), R2414.917954%(-.055891pp).
R30 is+.163085pp above its all-A/zero-U no-update memory; R24+currentG is
+.158719pp above direct currentG, but each system remains below bare source.
These descriptive positive contrasts are retained; no primary reselection.
Direct currentG itself is14.670101%, below source. SAR's+.000487pp changes
neither query choices nor Acc25/50; it is not a substantive task gain.

REPORT.md contains all6 frozen-transfer systems,6 matched currentG counterparts,
and12 episodic controls in separate tables. ENGINEERING_REPORT.md explains
the exact source/ports and fresh-frame qualification. OVERNIGHT_REPORT.md
summarizes all completed mechanism iterations with public evidence links.

All9216IoUs and37622ordered negative comparisons are exported, with new
anonymous F0001..F0384 row aliases and anonymous scenes; no private linking map.
The negative CSV uses lossless A01..A24 method IDs: see ARM_DICTIONARY.json.
Those IDs label methods, not object/query IDs. All384source-pool Top1/5/10 and
256oracle diagnostic rows,1536method update/skip rows,384contract flags,
768head arithmetic and1152memory audit scalar rows are included.
TopK/oracle are GT-assisted offline diagnostics, not deployed results; source
queries can duplicate instances, and this bound excludes external new boxes.

Methods and hyperparameters were locked before reading this pool. All24outputs
were sealed, independently audited, then scored once under separate label
authorization. A second reviewer recomputed saved scalar aggregates only.
Historical full-validation exposure and1Drone/3Quad existing scenes remain
disclosed; this is locked new-content evaluation, not independent new-scene
gold confirmation. No cross-scene CI or postscore method change is claimed.

Final execution/audit/score succeeded in one pass. Joint24output collection
cost17minutes includes model load,CPU preparation,persistence and closure;
it is not per-method deployment latency. Separate oldA2 engineering I/O
failures and bounded raw reuse are preserved as anonymous history.

Only own source,anonymous protocol/scalars/flags and provenance are distributed.
No rawUIDs/oldaliases/captions/images/pointclouds/boxes/masks/token tensors,
query indices/features/weights/raw answers, licensed cohort registration or
thirdparty source snapshots are included. Overlay own code on the prior
published chain and provide licensed private assets to reproduce the run.
''')
 # Exact-token scan reads only already-authorized result/input registrations.
 finalrows=read(R/'research/NIGHT_20261101/input/FINAL384_PRIVATE.json');oldrows=read(R/'research/round46/engineering2/INPUT2_PRIVATE.json');forbidden={str(r[k]) for r in finalrows+oldrows for k in ['id','alias','scene','caption','imagepath','pointpath','metapath','frame_alias'] if r.get(k)};findings=[];blocked_extensions={'.npz','.npy','.pth','.pt','.safetensors','.jpg','.jpeg','.png','.gif','.webp','.pdf','.bin'}
 for p in sorted(O.rglob('*')):
  if not p.is_file():continue
  if p.suffix.lower() in blocked_extensions:findings.append({'file':p.relative_to(O).as_posix(),'kind':'blocked_binary_extension'})
  text=p.read_text()
  for token in forbidden:
   if token in text:findings.append({'file':p.relative_to(O).as_posix(),'kind':'private_token','token_sha256':hashlib.sha256(token.encode()).hexdigest()})
  if re.search('/'+'home/'+r'[^/\s]+/',text):findings.append({'file':p.relative_to(O).as_posix(),'kind':'home_path'})
  if list(opaque_hex_matches(text)):findings.append({'file':p.relative_to(O).as_posix(),'kind':'private_style16hex_identifier'})
 assert not findings,findings[:10]
 blocked={'id','id_PRIVATE','alias','caption','scene','scene_PRIVATE','pointpath','metapath','imagepath','frame_alias','queries','query','mask_index','mask_IoU','raw','raw_answer_PRIVATE','generated_token_ids','input_ids','pixel_values','source_scores','source_boxes','masks_bool','box_pools','A_keys','A_row_indices','A_query_indices','shared_deltaG_F64','oldG_F64','token_mask','tokens','query_override','box_override','reusable_mask_files'};keyfind=[]
 def walk(x,name):
  if isinstance(x,dict):
   for k,v in x.items():
    if k in blocked and not (k in {'box_pools','query_override','box_override'} and isinstance(v,str)):keyfind.append({'file':name,'key':k})
    walk(v,name)
  elif isinstance(x,list):
   for v in x:walk(v,name)
 for p in O.rglob('*.json'):walk(read(p),p.relative_to(O).as_posix())
 assert not keyfind,keyfind[:10]
 # Prove CSV counts and lossless negative method-ID decoding, without new IoUs.
 task=list(csv.DictReader(io.StringIO((O/'results/ALL9216_ANONYMOUS_TASK_SCALARS.csv').read_text())));negative=list(csv.DictReader(io.StringIO((O/'results/ALL37622_ORDERED_NEGATIVE_COMPARISONS.csv').read_text())));assert len(task)==9216 and len(negative)==37622;armmap={x['arm_id']:x['arm'] for x in read(O/'results/ARM_DICTIONARY.json')['arms']};rawnegative=read(R/'research/round46/independent_final/score384_attempt001/ALL_ORDERED_NEGATIVE_COMPARISONS.json');assert all(armmap[x['arm_id']]==r['arm'] and armmap[x['reference_id']]==r['reference'] and float(x['delta_pp'])==r['delta_pp'] for x,r in zip(negative,rawnegative))
 for name,n in [('ALL384_SOURCE_POOL_DIAGNOSTIC_SCALARS',384),('ALL1536_EXTERNAL_UPDATE_AND_SKIP_SCALARS',1536)]:assert len(list(csv.DictReader(io.StringIO((O/'results'/(name+'.csv')).read_text()))))==n
 for name,n in [('ALL768_HEAD_ARITHMETIC_SCALARS',768),('ALL1152_MEMORY_ARITHMETIC_SCALARS',1152)]:assert len(list(csv.DictReader(io.StringIO((O/'audit'/(name+'.csv')).read_text()))))==n
 privacy={'state':'PASS','explicit_scalar_flag_protocol_allowlists':True,'new_anonymous_rows_no_linking_map':True,'private_token_findings':findings,'blocked_structured_field_findings':keyfind,'no_media_rawanswers_coords_scores_features_models_thirdparty_snapshots':True,'negative_CSV_arm_dictionary_lossless_exact':True,'all9216_and37622_preserved':True,'counts':counts,'private16hex_heuristic_excludes_explicit_decimal_numeric_mantissas_only':True,'registered_UID_exactscan_has_no_numeric_exception':True,'no_new_model_optimization_taskIoU_or_method_changes_for_export':True};dump(O/'EXPORT_PRIVACY_AUDIT.json',privacy)
 dump(O/'history/EXPORT_PREPARATION_HISTORY.json',{'attempt001':'Preserved local partial export and log. Heuristic mistook a scientific decimal mantissa for a16hex UID; no actual registeredUID finding and no payload published.','attempt002':'Complete local package preserved with its original manifest/payload/audit. Root requested clearer historical fullVAL and known-scene wording in overnight appendix before publication.','attempt003':'Fresh package with matching local/public overnight wording and source hashes. Numeric literal exception only in generic heuristic; exact registered UID/path/caption scans unchanged.','no_scientific_metrics_or_experiment_changes':True,'all_negative_rows_preserved_lossless_method_dictionary_encoding':True})
 dump(O/'protocol/FINAL_PUBLIC_CODE_SNAPSHOT.json',{'own_code_only':[{'path':p.relative_to(O).as_posix(),'public_sha256':sha(p)} for p in sorted(O.rglob('*.py'))]})
 files=[p for p in sorted(O.rglob('*')) if p.is_file()];dump(O/'MANIFEST.json',{'prefix':PREFIX,'files':[{'path':p.relative_to(O).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files],'manifest_excludes_itself':True});files=[p for p in sorted(O.rglob('*')) if p.is_file()];target=P/'TREE_PAYLOAD_LOCAL.json';dump(target,[{'path':PREFIX+'/'+p.relative_to(O).as_posix(),'mode':'100644','type':'blob','content':p.read_text()} for p in files])
 for f in read(O/'MANIFEST.json')['files']:assert sha(O/f['path'])==f['sha256'] and (O/f['path']).stat().st_size==f['bytes']
 for x,p in zip(read(target),files):assert x['path']==PREFIX+'/'+p.relative_to(O).as_posix() and x['content']==p.read_text()
 final={'state':'PASS','files':len(files),'bytes':sum(p.stat().st_size for p in files),'payload_bytes':target.stat().st_size,'payload_sha256':sha(target),'prefix':PREFIX,'counts':counts,'privacy_findings':[],'manifest_payload_allSHAexact':True,'base_engineering_files':len(base_receipt['files']),'not_published':True};dump(P/'LOCAL_EXPORT_AUDIT.json',final);print(json.dumps(final))
if __name__=='__main__':main()
