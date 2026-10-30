"""Deterministic R8 input selection from R7 input-only whitelist, no GT/I/O assets.
No metadata/index/model/caches/metrics, point arrays or images are opened.
"""
from pathlib import Path
import ast,collections,hashlib,json,re
B=Path(__file__).resolve().parents[3];E=Path(__file__).resolve().parent
SALT='R8_G_prereq_v1'
KEYS={'id','platform','scene','pointpath','metapath','imagepath','caption','suffix','seed'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
 source=B/'research/round7/experiment/full_gpu_inputs.json';oldsmoke=B/'research/round7/experiment/smoke_gpu_inputs.json';parser=B/'crossplatform/round7/multimodal_pseudo.py'
 full=read(source);old=read(oldsmoke);assert len(full)==5912 and len(old)==20
 assert all(set(x)==KEYS for x in full+old) and len({r['id'] for r in full})==5912
 assert dict(collections.Counter(x['platform'] for x in full))=={'drone':2984,'quad':2928}
 groups=sorted({(r['platform'],r['scene']) for r in full});assert len(groups)==5
 # Extract only the audited descriptive regex; it never participates in selection.
 tree=ast.parse(parser.read_text());nodes=[]
 for n in tree.body:
  if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('COLORS','NOUNS') for t in n.targets):nodes.append(n)
  if isinstance(n,ast.FunctionDef) and n.name=='target_color':nodes.append(n)
 ns={'re':re};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(parser),'exec'),ns);target_color=ns['target_color'];colors=ns['COLORS']
 plan={'purpose':'R8_GTfree_fixed_smoke40_and_complete5912_target_input_manifest','selection_salt':SALT,'expected_full_n':5912,'expected_smoke_n':40,'per_platform_scene_smoke_n':8,'expected_physical_scene_n':5,'exclude':'All20 original R7smoke IDs; no old sourcecache/model outputs or scores read','rank':'ascending SHA256(UTF8(R8_G_prereq_v1| + id)), tie-break ascending id','frame_identity':'exact pointpath string; per-scene no repeat; imagepath uniqueness independently checked','selection':'For each sorted(platform,scene), iterate eligible rows in rank order, take first8 distinct pointpaths. No color/old score/GT/geometry filtering.','full_manifest':'Exact bytes of R7 strict full_gpu_inputs.json, all5912 retained; source input seed/suffix and ninekeys unchanged','R7_full_input_sha256':sha(source),'R7_smoke20_input_sha256':sha(oldsmoke),'descriptive_color_parser_module_sha256':sha(parser),'color_statistics':'Only after fixed selection; supported/no-finite-color-lexeme/unsupported fractions do not gate sampling or model updates','source_model_requirement':'Each R8case requires fresh same-source strictFP32 forward. R7model features/boxes/scores cannot stand in for R8source.','historical_exposure':'TargetVAL already R5/R7 researched; this is engineering smoke with new fixed sample IDs, not fresh research confirmation','no_GT_index_metadata_or_task_score_access':True,'CPU_manifest_only_no_model_GPU_or_media_read':True}
 plan_path=E/'INPUT_PLAN_LOCK.json';assert not plan_path.exists();plan_path.write_text(json.dumps(plan,indent=2)+'\n')
 excluded={r['id'] for r in old};selected=[];by_scene=[]
 rank=lambda r:(hashlib.sha256((SALT+'|'+r['id']).encode()).hexdigest(),r['id'])
 color_regex=re.compile(r'\b(?:'+'|'.join(colors)+r')\b',re.I)
 def describe(rows):
  parsed=[target_color(r['caption']) for r in rows];n=len(rows)
  supported=sum(c is not None for c,reason in parsed);no_color=sum(not color_regex.search(r['caption']) for r in rows)
  return {'n':n,'supported_opening_count':supported,'supported_opening_fraction':supported/n,'no_finite_color_lexeme_count':no_color,'no_finite_color_lexeme_fraction':no_color/n,'unsupported_opening_count':n-supported,'unsupported_opening_fraction':(n-supported)/n,'supported_colors':dict(collections.Counter(c for c,reason in parsed if c is not None)),'parser_reasons':dict(collections.Counter(reason for c,reason in parsed))}
 for platform,scene in groups:
  rows=[r for r in full if (r['platform'],r['scene'])==(platform,scene)];eligible=[r for r in rows if r['id'] not in excluded];chosen=[];frames=set()
  for row in sorted(eligible,key=rank):
   if row['pointpath'] in frames:continue
   chosen.append(row);frames.add(row['pointpath'])
   if len(chosen)==8:break
  assert len(chosen)==len(frames)==len({r['imagepath'] for r in chosen})==8
  selected.extend(chosen);by_scene.append({'platform':platform,'scene':scene,'all_inputs':len(rows),'excluded_R7smoke_ids':sum(r['id'] in excluded for r in rows),'eligible_inputs':len(eligible),'smoke_unique_pointpaths':len(frames),'smoke_unique_imagepaths':len({r['imagepath'] for r in chosen}),'full_text_only_color_description':describe(rows),'smoke_text_only_color_description':describe(chosen)})
 assert len(selected)==len({r['id'] for r in selected})==40 and not excluded&{r['id'] for r in selected}
 assert not {r['pointpath'] for r in old}&{r['pointpath'] for r in selected}
 full_path=E/'full5912_gpu_inputs.json';smoke_path=E/'smoke40_gpu_inputs.json';assert not full_path.exists() and not smoke_path.exists()
 full_path.write_bytes(source.read_bytes());smoke_path.write_text(json.dumps(selected,indent=2,ensure_ascii=False)+'\n')
 assert sha(full_path)==sha(source) and all(next(r for r in full if r['id']==x['id'])==x for x in selected)
 receipt={'state':'fixed40_and_complete5912_input_only_manifests_sealed_no_GT_model_or_media_access','input_plan_sha256':sha(plan_path),'selection_program_sha256':sha(__file__),'R7_full_source_sha256':sha(source),'R7_smoke20_source_sha256':sha(oldsmoke),'descriptive_parser_sha256':sha(parser),'output_files':{full_path.name:sha(full_path),smoke_path.name:sha(smoke_path)},'ninekey_schema':sorted(KEYS),'full_n':5912,'full_by_platform':dict(collections.Counter(r['platform'] for r in full)),'full_bytes_identical_to_R7_strict_inputs':True,'smoke_n':40,'smoke_by_platform':dict(collections.Counter(r['platform'] for r in selected)),'smoke_scene_n':len(groups),'smoke_all_distinct_pointpaths':len({r['pointpath'] for r in selected})==40,'smoke_all_distinct_imagepaths':len({r['imagepath'] for r in selected})==40,'R7smoke20_ID_overlap':0,'R7smoke20_pointpath_overlap':0,'all_selected_source_rows_including_seed_suffix_preserved_exactly':True,'selection_independent_of_color_and_GT':True,'full_text_only_color_description':describe(full),'smoke_text_only_color_description':describe(selected),'by_scene_PRIVATE':by_scene,'no_scoringindex_GT_metadata_boxes_class_positive_map_or_scores_read':True,'no_target_points_RGB_or_modelcache_read':True,'no_model_or_GPU_execution':True,'target_research_exposure':'historical R5/R7 targetVAL; not fresh confirmation','new_R8_source_forward_required_no_R7_modelcache_reuse':True}
 (E/'INPUT_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='by_scene_PRIVATE'},indent=2))
if __name__=='__main__':main()
