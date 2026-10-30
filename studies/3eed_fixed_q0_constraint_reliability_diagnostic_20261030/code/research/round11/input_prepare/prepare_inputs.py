"""R11 input-only320; prior manifest JSON text and file stat, never labels/media."""
from pathlib import Path
import collections,hashlib,json
B=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent
SALT='R11_constraint_reliability_v1';KEYS={'id','platform','scene','pointpath','metapath','imagepath','caption','suffix','seed'}
SOURCES={'population':'research/round8/experiment/full5912_gpu_inputs.json','R9_160':'research/round9/experiment/diagnostic160_gpu_inputs.json','R10_160':'research/round10/input_prepare/diagnostic160_gpu_inputs.json','R8_40':'research/round8/experiment/smoke40_gpu_inputs.json','R7_20':'research/round7/experiment/smoke_gpu_inputs.json'}
HASHES={'population':'d6296d2577300796055fd255cbde2e768b9c493254e5ee610296a095d90afbbd','R9_160':'7da939a81f35e71de90e780c9e25787f8d245c577b66d727f200a3538279e0bd','R10_160':'b72c64157f83e93e11db10953a5721c07a981a0726b387de4248b6e954d27425','R8_40':'951af09f44b526d79550e7b2792e014b989ea858dcc19a5fc1c900b9aa3995f2','R7_20':'4c210baca0c145b329ec98fb3b2c5277f6fd076d4a3ccca7b17c5929044e0b40'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):
 p=D/name;assert not p.exists();p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');return p
def main():
 data={k:json.loads((B/v).read_text()) for k,v in SOURCES.items()}
 assert all(sha(B/v)==HASHES[k] for k,v in SOURCES.items()) and all(set(r)==KEYS for rr in data.values() for r in rr)
 full=data['population'];old=[r for k,v in data.items() if k!='population' for r in v];oldids={r['id'] for r in old};oldframes={r['pointpath'] for r in old};assert len(full)==5912 and len(old)==len(oldids)==len(oldframes)==380
 groups=sorted({(r['platform'],r['scene']) for r in full});assert len(groups)==5;counts=[];pools={}
 for i,key in enumerate(groups,1):
  rr=[r for r in full if (r['platform'],r['scene'])==key];pool=[r for r in rr if r['id'] not in oldids and r['pointpath'] not in oldframes];pools[key]=pool;counts.append({'scene_alias':f'scene_{i:02d}','platform':key[0],'full_n':len(rr),'prior_excluded_n':len(rr)-len(pool),'eligible_n':len(pool),'eligible_unique_pointpaths':len({r['pointpath'] for r in pool}),'eligible_unique_imagepaths':len({r['imagepath'] for r in pool}),'requested_n':64})
 availability=save('COHORT_AVAILABILITY.json',{'state':'R11_input_only_preselection_availability','source_input_sha256':HASHES,'excluded_unique_IDs':380,'excluded_unique_pointpaths':380,'scene_counts':counts,'all_scenes_have64':all(r['eligible_unique_pointpaths']>=64 for r in counts),'no_GT_oldmetrics_media_points_or_model_access':True})
 assert all(r['eligible_unique_pointpaths']>=64 for r in counts),'Stop forroot if insufficient frames; no fabricated replacements'
 plan=save('INPUT_PLAN_LOCK.json',{'state':'R11_fixed320_input_only_plan_before_new_source_or_G','selection_salt':SALT,'rank':'ascending SHA256(UTF8(salt|id)), tie id, within each platform-scene','per_scene':64,'scenes':5,'n':320,'exclude':'R9manifest160 + R10manifest160 + R8smoke40 + R7smoke20 =380 unique IDs and pointpaths','frame_identity':'exactpointpath, additionally imagepath uniqueness checked','all9inputvalues_seed_suffix_preserved':True,'no_color_class_GT_score_oldmetric_pixels_or_geometry_screening':True,'source_input_paths':SOURCES,'source_input_sha256':HASHES,'availability_sha256':sha(availability),'preparation_program_sha256':sha(__file__),'scope':'R11 input-frame-disjoint from selected prior380; entire targetVAL historically researched. Not unseen-domain/freshresearchheldout confirmation. Source training Waymo only, R11 stillcrossplatform.','cache_policy':'No caches read or reused. Later fresh forwards or reuse require exact input/source/model/precision/token/point/index/seed binding byroot.','ownership':'This agent only writes new research/round11/input_prepare; no oldfiles/GPU/download/modelweights/GT.'})
 selected=[]
 for key in groups:
  seen=set();chosen=[]
  for r in sorted(pools[key],key=lambda r:(hashlib.sha256((SALT+'|'+r['id']).encode()).hexdigest(),r['id'])):
   if r['pointpath'] in seen:continue
   chosen.append(r);seen.add(r['pointpath'])
   if len(chosen)==64:break
  assert len(chosen)==len(seen)==len({r['imagepath'] for r in chosen})==64;selected.extend(chosen)
 assert len(selected)==len({r['id'] for r in selected})==len({r['pointpath'] for r in selected})==len({r['imagepath'] for r in selected})==320
 assert not {r['id'] for r in selected}&oldids and not {r['pointpath'] for r in selected}&oldframes
 for r in selected:
  for key in ('pointpath','metapath','imagepath'):assert Path(r[key]).is_file()
 manifest=save('diagnostic320_gpu_inputs.json',selected)
 receipt=save('INPUT_RECEIPT.json',{'state':'R11_strict9key320_input_only_cohort_sealed','n':320,'by_platform':dict(collections.Counter(r['platform'] for r in selected)),'scene_counts':counts,'input_manifest_sha256':sha(manifest),'input_plan_sha256':sha(plan),'availability_sha256':sha(availability),'preparation_program_sha256':sha(__file__),'source_input_sha256':HASHES,'old380_ID_overlap':0,'old380_pointpath_overlap':0,'selected_unique_pointpaths':320,'selected_unique_imagepaths':320,'all_rows_exactly_preserved_from_population':all(next(x for x in full if x['id']==r['id'])==r for r in selected),'selected_point_image_camera_container_stat_pass':True,'no_annotation_payload_GT_index_sidecar_oldmetrics_pixels_points_headers_cache_weights_model_download_or_GPU_access':True,'historical_targetresearch_exposure_not_fresh_GT_claim':True})
 print(json.dumps({'n':320,'platform_counts':dict(collections.Counter(r['platform'] for r in selected)),'manifest_sha256':sha(manifest),'plan_sha256':sha(plan),'receipt_sha256':sha(receipt)},indent=2))
if __name__=='__main__':main()
