"""R10 strictly input-only cohort construction; no GT/media/point arrays/caches.
Existing9key input text and selected-file stat only. No score/color selection.
"""
from pathlib import Path
import collections,hashlib,json
B=Path(__file__).resolve().parents[3];E=Path(__file__).resolve().parent
SALT='R10_rgb_support_geometry_v1';KEYS={'id','platform','scene','pointpath','metapath','imagepath','caption','suffix','seed'}
SOURCES={'population':'research/round8/experiment/full5912_gpu_inputs.json','R9_160':'research/round9/experiment/diagnostic160_gpu_inputs.json','R8_40':'research/round8/experiment/smoke40_gpu_inputs.json','R7_20':'research/round7/experiment/smoke_gpu_inputs.json'}
HASHES={'population':'d6296d2577300796055fd255cbde2e768b9c493254e5ee610296a095d90afbbd','R9_160':'7da939a81f35e71de90e780c9e25787f8d245c577b66d727f200a3538279e0bd','R8_40':'951af09f44b526d79550e7b2792e014b989ea858dcc19a5fc1c900b9aa3995f2','R7_20':'4c210baca0c145b329ec98fb3b2c5277f6fd076d4a3ccca7b17c5929044e0b40'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(name,value):
 p=E/name;assert not p.exists(),('Refuse overwrite',name);p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');return p

def main():
 data={k:read(B/v) for k,v in SOURCES.items()};assert all(sha(B/v)==HASHES[k] for k,v in SOURCES.items())
 assert all(set(r)==KEYS for rows in data.values() for r in rows)
 full=data['population'];old=data['R9_160']+data['R8_40']+data['R7_20'];exclude={r['id'] for r in old};oldframes={r['pointpath'] for r in old};assert len(full)==5912 and len(exclude)==len(oldframes)==220
 groups=sorted({(r['platform'],r['scene']) for r in full});assert len(groups)==5;counts=[];pools={}
 for i,key in enumerate(groups,1):
  rr=[r for r in full if (r['platform'],r['scene'])==key];pool=[r for r in rr if r['id'] not in exclude and r['pointpath'] not in oldframes];pools[key]=pool;counts.append({'scene_alias':'scene_'+str(i).zfill(2),'platform':key[0],'full_n':len(rr),'excluded_prior_n':len(rr)-len(pool),'eligible_n':len(pool),'eligible_unique_pointpaths':len({r['pointpath'] for r in pool}),'eligible_unique_imagepaths':len({r['imagepath'] for r in pool}),'requested_n':32})
 availability=write('COHORT_AVAILABILITY.json',{'state':'R10_input_only_preselection_availability','source_sha256':HASHES,'excluded_unique_IDs':220,'excluded_unique_pointpaths':220,'scene_counts':counts,'all_scenes_have32':all(r['eligible_unique_pointpaths']>=32 for r in counts),'no_GT_score_index_sidecar_media_points_or_model_read':True})
 assert all(r['eligible_unique_pointpaths']>=32 for r in counts),'Stop for root if any scene short; never fabricate replacements'
 plan=write('INPUT_PLAN_LOCK.json',{'state':'R10_fixed160_input_only_cohort_before_new_G_or_source_capture','selection_salt':SALT,'rank':'ascending SHA256(UTF8(salt|id)), then id','per_scene':32,'scenes':5,'n':160,'exclude':'R9manifest160 + R8smoke40 + R7smoke20 union220 IDs and exactpointpaths','frame_identity':'pointpath exactstring; imagepath uniqueness crosschecked','all_source9keys_seed_suffix_preserved':True,'no_color_GT_score_class_objectindex_or_geometry_selection':True,'source_input_paths':SOURCES,'source_input_sha256':HASHES,'availability_sha256':sha(availability),'preparation_program_sha256':sha(__file__),'scope':'64Drone/96Quad, fiveactualscenes; frame-disjoint prior220, but entire targetVAL historically researched R5/R7/R8/R9; no freshheldout researchclaim','cache_policy':'No cache opened/reused bypreparation. Any later reuse requires exact source/model/precision/inputtext/tokenization/point/index/seed binding and root approval; expected freshforwards.','ownership':'This agent owns input_prepare and crossplatform/round10/rgb_support.py; geometricG/runner owned by crossplatformagent. Oldfiles immutable.','point_observation_split':'Separate deterministic per-frame coordinatehash parity plan saved; no pointarray accessed now, not independent views/scenes or immediate validation claim.'})
 selected=[]
 for key in groups:
  ordered=sorted(pools[key],key=lambda r:(hashlib.sha256((SALT+'|'+r['id']).encode()).hexdigest(),r['id']));seen=set();chosen=[]
  for r in ordered:
   if r['pointpath'] in seen:continue
   chosen.append(r);seen.add(r['pointpath'])
   if len(chosen)==32:break
  assert len(chosen)==len(seen)==len({r['imagepath'] for r in chosen})==32;selected.extend(chosen)
 assert len(selected)==len({r['id'] for r in selected})==len({r['pointpath'] for r in selected})==160
 assert not{r['id'] for r in selected}&exclude and not{r['pointpath'] for r in selected}&oldframes
 for r in selected:
  for key in ('pointpath','imagepath','metapath'):assert Path(r[key]).is_file()
 manifest=write('diagnostic160_gpu_inputs.json',selected)
 split=write('POINT_OBSERVATION_SPLIT_PLAN.json',{'state':'observation_split_scaffold_before_any_new_point_observation','seed_rule':'low31bits little-endian first4bytes SHA256(UTF8(R10_visible_point_split_v1|private_case_id))','point_rule':'SHA256(seed8bytes_unsigned_littleendian || canonical_littleendian_float32_modelXYZ_bytes), firstbyte parity0=fit,parity1=eval','canonicalization':'FiniteNx3 modelcoords float32, signedzero canonical+0; exactduplicateXYZ always samepartition, order invariant; no splitting duplicate samples acrossroles','ratio':'hashparity expectedhalf/half, actualcounts stored; no forced balancing or fallback if short','no_label_or_coordinate_quality_selection':True,'current_scope':'Rootfirststagewhole-observationDirectG may save this split only; it is not automatically used or claimed as independent validation. G/runner must declare actually consumed points.','future_if_heldout_claimed':'GrabCut seeds/visibility/targetfit use fitpoints only; evaldepth/geometry cannot enter segmentation/acceptance. Build visibility separately forfit/eval; sourceq0/backbone/sharedRGB may still use sharedinformation, so not independentdomain/view evidence.','statistical_units':'Within-frame pointpartitions share sourceprediction/RGB/camera/scenenormalization; not extra independent scenes orsamples.'})
 receipt=write('INPUT_RECEIPT.json',{'state':'R10_fixed160_strict9key_input_manifest_sealed_CPU_only','n':160,'by_platform':dict(collections.Counter(r['platform'] for r in selected)),'scene_counts':counts,'input_manifest_sha256':sha(manifest),'input_plan_sha256':sha(plan),'availability_sha256':sha(availability),'point_split_plan_sha256':sha(split),'selection_program_sha256':sha(__file__),'source_input_sha256':HASHES,'selected_unique_pointpaths':160,'selected_unique_imagepaths':160,'old220_ID_overlap':0,'old220_frame_overlap':0,'source_rows_exactly_preserved':all(next(x for x in full if x['id']==r['id'])==r for r in selected),'selectedfile_stat_pass':True,'no_target_annotation_payload_GT_index_sidecar_scores_caches_media_pointarray_or_header_read':True,'no_GPU_model_or_new_scoring':True,'public_boundary':'Manifest IDs/paths/captions private; exportonlyaggregate/ordinalaliases.','ownership':'input_prepare+rgb_support only; oldscientific files notmodified.'})
 print(json.dumps({'n':160,'by_platform':dict(collections.Counter(r['platform'] for r in selected)),'manifest_sha256':sha(manifest),'plan_sha256':sha(plan),'receipt_sha256':sha(receipt),'split_plan_sha256':sha(split)},indent=2))
if __name__=='__main__':main()
