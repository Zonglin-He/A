"""R9 fixed-q0 diagnostic input cohort, strict input-only CPU preparation.
No annotation payload, scoring index, old metrics/source outputs, images, point
arrays, GPU/model or identity/GT subset is opened. All output IDs remain private.
"""
from pathlib import Path
import collections,hashlib,json
B=Path(__file__).resolve().parents[3];E=Path(__file__).resolve().parent
SALT='R9_fixedq0_geometry_v1';KEYS={'id','platform','scene','pointpath','metapath','imagepath','caption','suffix','seed'}
SOURCE_FILES={'R8_full5912':'research/round8/experiment/full5912_gpu_inputs.json','R8_smoke40':'research/round8/experiment/smoke40_gpu_inputs.json','R7_smoke20':'research/round7/experiment/smoke_gpu_inputs.json'}
EXPECTED_SHA={'R8_full5912':'d6296d2577300796055fd255cbde2e768b9c493254e5ee610296a095d90afbbd','R8_smoke40':'951af09f44b526d79550e7b2792e014b989ea858dcc19a5fc1c900b9aa3995f2','R7_smoke20':'4c210baca0c145b329ec98fb3b2c5277f6fd076d4a3ccca7b17c5929044e0b40'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write_new(name,value):
 path=E/name;assert not path.exists(),('Refuse overwrite',name);path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');return path

def main():
 data={}
 for key,rel in SOURCE_FILES.items():
  assert sha(B/rel)==EXPECTED_SHA[key],('Source input changed',key);data[key]=read(B/rel)
 assert [len(data[k]) for k in SOURCE_FILES]==[5912,40,20]
 assert all(set(r)==KEYS for vv in data.values() for r in vv)
 full=data['R8_full5912'];old=data['R8_smoke40']+data['R7_smoke20'];excluded={r['id'] for r in old};excluded_frames={r['pointpath'] for r in old}
 assert len({r['id'] for r in full})==5912 and len(excluded)==len(excluded_frames)==60
 assert {r['id'] for r in old}<={r['id'] for r in full}
 scenes=sorted({(r['platform'],r['scene']) for r in full});assert len(scenes)==5
 counts=[];eligible_by_scene={}
 for i,key in enumerate(scenes,1):
  all_scene=[r for r in full if (r['platform'],r['scene'])==key];eligible=[r for r in all_scene if r['id'] not in excluded and r['pointpath'] not in excluded_frames];eligible_by_scene[key]=eligible
  counts.append({'scene_alias':'scene_'+str(i).zfill(2),'platform':key[0],'full_n':len(all_scene),'excluded_prior_smoke_n':len(all_scene)-len(eligible),'eligible_n':len(eligible),'eligible_distinct_pointpaths':len({r['pointpath'] for r in eligible}),'eligible_distinct_imagepaths':len({r['imagepath'] for r in eligible}),'requested_n':32,'enough_distinct_frames':len({r['pointpath'] for r in eligible})>=32})
 # Count/eligibility prerequisite is recorded before selecting or freezing cohort.
 counts_path=write_new('COHORT_AVAILABILITY.json',{'state':'input_only_preselection_counts_no_GT_or_metrics','source_input_sha256':EXPECTED_SHA,'scene_counts':counts,'all_scenes_have32':all(x['enough_distinct_frames'] for x in counts),'no_target_annotation_contents_or_prior_scores_opened':True})
 if not all(x['enough_distinct_frames'] for x in counts):raise RuntimeError('Insufficient scene frames: preserve count audit and ask root; never fabricate or resample extras')
 plan={'state':'fixed_input_cohort_plan_before_R9_model_outputs_or_identity_GT_diagnosis','purpose':'R9_fixedq0_geometry_unlabelled_input_selection_and_later_sealed_GT_capability_diagnosis','selection_salt':SALT,'rank':'ascending SHA256(UTF8(R9_fixedq0_geometry_v1| + id)), then ascending id tie-break','per_scene':32,'scene_count':5,'expected_n':160,'scene_order':'ascending(platform,original_scene) from strict input manifest; public counts use ordinal scene_alias only','exclude':'all R8smoke40 and R7smoke20 IDs and exact pointpaths, union60','frame_identity':'exact pointpath; selected imagepaths independently unique','no_score_color_class_GT_or_geometry_filtering':True,'short_scene_rule':'If fewer32 distinct frames, preserve all availability counts and stop for root decision; no replacement, imaginary examples or score-guided substitution','source_input_paths':SOURCE_FILES,'source_input_sha256':EXPECTED_SHA,'preselection_counts_sha256':sha(counts_path),'strict_schema':sorted(KEYS),'preserve_all_source_row_values_including_seed_suffix':True,'full_population':'Original5912 VAL inputs retained as population; this diagnosticcohort is balanced32-per-scene, not populationmicro representative','historical_target_research_exposure':'All targetVAL previously R5/R7/R8 researched/scored; excluded smoke IDs are not a fresh research holdout','source_provenance':'Genuine Waymo-only epoch100 source. No reuse or relabeling of prior R8 predictions as R9 outputs. New fixedq0 experiment outputs must follow parentlocked source/precision plan.','identity_subset':'Defined only by separately authorized scorer after R9 predictions/GT-dependent diagnostics are sealed; never read old scores or identity labels for cohort selection','private_public_boundary':'Manifest9keys include private IDs/paths/fullcaptions and stay local. Public export must use aggregates/ordinalaliases without original IDs or mapping.','preparation_access':'Three input-only JSONs and selected file existence stat only; no targetannotationJSON payload/index/GT, oldscore/modelcache, pixel/pointarray/header read','preparation_program_sha256':sha(__file__)}
 planpath=write_new('INPUT_PLAN_LOCK.json',plan)
 rank=lambda r:(hashlib.sha256((SALT+'|'+r['id']).encode()).hexdigest(),r['id'])
 cohort=[]
 for i,key in enumerate(scenes):
  seen=set();chosen=[]
  for row in sorted(eligible_by_scene[key],key=rank):
   if row['pointpath'] in seen:continue
   seen.add(row['pointpath']);chosen.append(row)
   if len(chosen)==32:break
  assert len(chosen)==len(seen)==len({r['imagepath'] for r in chosen})==32
  cohort.extend(chosen)
 assert len(cohort)==len({r['id'] for r in cohort})==len({r['pointpath'] for r in cohort})==len({r['imagepath'] for r in cohort})==160
 assert not excluded&{r['id'] for r in cohort} and not excluded_frames&{r['pointpath'] for r in cohort}
 assert all(set(r)==KEYS for r in cohort)
 original={r['id']:r for r in full};assert all(original[r['id']]==r for r in cohort)
 for row in cohort:
  assert isinstance(row['seed'],int) and not isinstance(row['seed'],bool)
  for key in ('pointpath','imagepath','metapath'):assert Path(row[key]).is_file(),('Missing selected file',key)
 path=write_new('diagnostic160_gpu_inputs.json',cohort)
 receipt={'state':'fixed160_strict_input_cohort_sealed_CPU_only_no_GT_model_or_scoring','input_plan_sha256':sha(planpath),'availability_counts_sha256':sha(counts_path),'selection_program_sha256':sha(__file__),'source_input_sha256':EXPECTED_SHA,'output_file':path.name,'output_sha256':sha(path),'n':160,'per_scene':32,'scenes':5,'by_platform':dict(collections.Counter(r['platform'] for r in cohort)),'by_scene_aggregate':counts,'prior_R8smoke40_overlap_IDs':0,'prior_R7smoke20_overlap_IDs':0,'prior_combined_smoke_pointpath_overlap':0,'selected_unique_pointpaths':160,'selected_unique_imagepaths':160,'selected_source_rows_exact_9key_seed_suffix_preserved':True,'selected_file_existence_stat_pass':True,'no_short_scene_or_replacement':True,'selection_did_not_use_color_lexicon_or_scores_or_GT':True,'no_target_annotation_contents_scoringindex_GT_class_positive_map_objectindex_old_scores_or_modelcache_access':True,'no_media_pointarray_header_model_GPU_or_new_scoring_access':True,'target_exposure':'Previously researched R5/R7/R8 VAL; not fresh confirmation. Scene-balanced engineering/GTcapability diagnosticcohort only.','identity_subset_not_defined_in_preparation':True,'public_export_original_identifiers_prohibited':True}
 receiptpath=write_new('INPUT_RECEIPT.json',receipt)
 print(json.dumps({'state':receipt['state'],'n':160,'by_platform':receipt['by_platform'],'manifest_sha256':sha(path),'plan_sha256':sha(planpath),'receipt_sha256':sha(receiptpath),'count_receipt_sha256':sha(counts_path)},indent=2))
if __name__=='__main__':main()
