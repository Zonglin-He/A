"""R15 root-prescribed64fresh+140regression,96source/64masks/32unusedB.
Metadata+authorized content hashes only. No model/G/GTscore or sensor decode.
"""
from pathlib import Path
import json,hashlib,collections
R=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
SALTS={'fresh':'R15_fresh64_content_v1','unusedB':'R15_unusedB32_content_v1','supportA':'R15_conditionalA32_content_v1'}
def rank(role,row):return hashlib.sha256((SALTS[role]+'|'+row['platform']+'|'+row['pair_group_sha256']).encode()).hexdigest()
def save(name,value):
 p=D/name
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');return p

def main():
 auth=R/'research/round15/experiment/AUTHORIZATION.json';assert auth.is_file()
 receipt=read(D/'DECODED_CONTENT_INDEPENDENCE_RECEIPT.json');inventory=read(D/'DECODED_CONTENT_HASHES_PRIVATE.json');assert receipt['hash_inventory_sha256']==sha(D/'DECODED_CONTENT_HASHES_PRIVATE.json');assert receipt['collision_table_sha256']==sha(D/'CONTENT_COLLISIONS_PRIVATE.json')
 rr=inventory['rows_PRIVATE'];assert len(rr)==1753
 full=read(R/'research/round8/experiment/full5912_gpu_inputs.json');fm={r['id']:r for r in full};byid={r['canonical_id_PRIVATE']:r for r in rr}
 avail=[r for r in rr if not r['component_touches_previous_recent_input'] and r['finite_XYZ']]
 chosen=[];scenes={}
 for platform,quota in [('quad',16),('drone',32)]:
  ss=sorted({r['scene_PRIVATE'] for r in avail if r['platform']==platform});assert len(ss)==(2 if platform=='quad' else 1);scenes[platform]=ss
  for scene in ss:
   candidates=sorted([r for r in avail if r['platform']==platform and r['scene_PRIVATE']==scene],key=lambda r:(rank('fresh',r),r['canonical_id_PRIVATE']))
   # OR-component and both per-modality uniqueness enforced even if inventory has collisions.
   selected=[];components=set();images=set();points=set()
   for row in candidates:
    if row['content_component_sha256'] in components or row['decoded_pixel_sha256'] in images or row['canonical_XYZ_sha256'] in points:continue
    selected.append(row);components.add(row['content_component_sha256']);images.add(row['decoded_pixel_sha256']);points.add(row['canonical_XYZ_sha256'])
    if len(selected)==quota:break
   assert len(selected)==quota;chosen.extend(selected)
 assert len(chosen)==64
 usedcomponents={r['content_component_sha256'] for r in chosen};usedimages={r['decoded_pixel_sha256'] for r in chosen};usedpoints={r['canonical_XYZ_sha256'] for r in chosen}
 unused=[];a_support=[];pairs=[]
 for plat in ['quad','drone']:
  for scene in scenes[plat]:
   n=8 if plat=='quad' else 16
   aa=sorted([r for r in chosen if r['platform']==plat and r['scene_PRIVATE']==scene],key=lambda r:(rank('supportA',r),r['canonical_id_PRIVATE']))[:n];assert len(aa)==n
   candidates=sorted([r for r in avail if r['platform']==plat and r['scene_PRIVATE']==scene],key=lambda r:(rank('unusedB',r),r['canonical_id_PRIVATE']));bb=[]
   for row in candidates:
    if row['content_component_sha256'] in usedcomponents or row['decoded_pixel_sha256'] in usedimages or row['canonical_XYZ_sha256'] in usedpoints:continue
    bb.append(row);usedcomponents.add(row['content_component_sha256']);usedimages.add(row['decoded_pixel_sha256']);usedpoints.add(row['canonical_XYZ_sha256'])
    if len(bb)==n:break
   assert len(bb)==n;a_support.extend(aa);unused.extend(bb)
   for a,b in zip(aa,bb):pairs.append({'pair_alias':f'conditional_pair_{len(pairs)+1:05d}','platform':plat,'scene_PRIVATE':scene,'A_id_PRIVATE':a['canonical_id_PRIVATE'],'B_id_PRIVATE':b['canonical_id_PRIVATE'],'A_raw_source_query_not_crossinput_slot_identity':True,'same_instance_correspondence_verified':False})
 assert len(unused)==len(a_support)==len(pairs)==32
 fresh=[fm[r['canonical_id_PRIVATE']] for r in chosen];binputs=[fm[r['canonical_id_PRIVATE']] for r in unused];support=[fm[r['canonical_id_PRIVATE']] for r in a_support]
 qreg=read(R/'research/round14/input/QUAD_B76_GPU_INPUTS.json');dreg=read(R/'research/round14/input/DRONE_ALL64_GPU_INPUTS.json');reg=qreg+dreg;formal=fresh+reg;source=fresh+binputs
 assert len(formal)==204 and len(source)==96 and len({r['id'] for r in source+reg})==236
 # Actual prior values are reused, never new sampler seeds/captions.
 assert all(r==fm[r['id']] and set(r)=={'id','platform','scene','pointpath','metapath','imagepath','caption','suffix','seed'} for r in source+reg)
 metadata=read(R/'research/round12/data_audit/TARGET5912_INPUT_FIELDS_PRIVATE.json');meta={r['meta_path_PRIVATE']:r for r in metadata['rows_PRIVATE']}
 catalog={r['id_PRIVATE']:r for r in read(R/'research/round12/data_audit/PHYSICAL_INPUT_HASH_INDEX_PRIVATE.json')['rows_PRIVATE']}
 r12frames=read(R/'research/round12/data_audit/FRAME_INDEX_PRIVATE.json');p12={r['anchor_id_PRIVATE']:r for r in read(R/'research/round12/data_audit/PAIRS_PRIVATE.json')['rows_PRIVATE']};dcase={r['id']:r for r in read(R/'research/round14/input/DRONE_CASE64_PRIVATE.json')};r14frames=read(R/'research/round14/input/DRONE_FRAME_INDEX_PRIVATE.json')
 fi={};cases=[];maskin=[];counter=collections.Counter()
 for i,row in enumerate(formal):
  plat=row['platform'];isfresh=i<64;counter[('fresh_' if isfresh else 'reg_')+plat]+=1;kind=('fresh_' if isfresh else 'reg_')+plat;alias=f'{kind}_{counter[kind]:05d}'
  if isfresh:
   frame=f'input_frame_{1000+i:05d}';h=catalog[row['id']];m=meta[row['metapath']];white=m['input_fields_PRIVATE'];assert sha(row['metapath'])==m['meta_sha256']
   # Only K/E/D are extracted; no pose/timestamp/ground_info/class/positive_map.
   item={'platform':plat,'scene_PRIVATE':row['scene'],'pointpath':row['pointpath'],'metapath':row['metapath'],'imagepath':row['imagepath'],'image_sha256':h['image_sha256'],'point_sha256':h['point_sha256'],'metadata_sha256':m['meta_sha256'],'physical_group_sha256':h['physical_group'],'image_intrinsic':white['image_intrinsic'],'image_extrinsic':white['image_extrinsic'],'camera_distortion':white.get('camera_distortion')};src='crossplatform/round15/source96_attempt001';mask='crossplatform/round15/masks64_attempt001';legacy=None;maskin.append({'frame_alias':frame,'imagepath':row['imagepath']})
  elif plat=='quad':frame=p12[row['id']]['anchor_frame_alias'];item=r12frames[frame];src='crossplatform/round12/source146_attempt001';mask='crossplatform/round12/masks501_attempt001';legacy='crossplatform/round12/feedback146_attempt001'
  else:old=dcase[row['id']];frame=old['frame_alias'];item=r14frames[frame];src=old['source_dir'];mask=old['mask_dir'];legacy='crossplatform/round14/drone_association64_attempt004'
  if frame in fi:assert fi[frame]==item
  fi[frame]=item;cases.append({'id':row['id'],'case_alias':alias,'role':'fresh64_recent_content_relative_only' if isfresh else 'R14_regression140_exposed','platform':plat,'scene_PRIVATE':row['scene'],'frame_alias':frame,'source_dir':src,'mask_dir':mask,'legacy_endpoint_dir':legacy,'legacy_scope':'No oldG on fresh inputs; primary arms require same fullquery information.' if isfresh else ('R12legacy methods only, no Quad R14combined' if plat=='quad' else 'Actual R14Dronecombined historical background')})
 assert len(fi)==204 and len(maskin)==64
 # B never enters the G frame/case index or mask input list.
 assert not {r['id'] for r in binputs}&{c['id'] for c in cases}
 allcontent=[catalog[r['id']] for r in source+reg];decoded={r['pair_group_sha256']:r for r in rr};actual=[decoded[r['physical_group']] for r in allcontent]
 assert len({r['decoded_pixel_sha256'] for r in actual})==len({r['canonical_XYZ_sha256'] for r in actual})==236
 for name,rows in [('FRESH64_GPU_INPUTS.json',fresh),('REGRESSION140_GPU_INPUTS.json',reg),('FORMAL204_GPU_INPUTS.json',formal),('SOURCE96_GPU_INPUTS.json',source),('UNUSED_B32_GPU_INPUTS_PRIVATE.json',binputs),('CONDITIONAL_A32_GPU_INPUTS_PRIVATE.json',support),('CONDITIONAL_PAIRS32_PRIVATE.json',pairs),('FORMAL_CASE204_PRIVATE.json',cases),('G_FRAME_INDEX_PRIVATE.json',fi),('MASK64_FRAME_INPUTS.json',maskin)]:save(name,rows)
 save('INPUT_PLAN_LOCK.json',{'state':'R15_input_registration_only_no_model_GT_or_G_execution','program_sha256':sha(__file__),'authorization_sha256':sha(auth),'decoded_content_receipt_sha256':sha(D/'DECODED_CONTENT_INDEPENDENCE_RECEIPT.json'),'decoded_hash_inventory_sha256':sha(D/'DECODED_CONTENT_HASHES_PRIVATE.json'),'decoded_collision_table_sha256':sha(D/'CONTENT_COLLISIONS_PRIVATE.json'),'selection_salts':SALTS,'ranking':'SHA256(salt|platform|pair_group_sha256), canonicalID tie; canonicalID=min opaqueID within exact prior pair. No caption/model/G/label/performance selection.','fresh64_counts':{'quad':32,'drone':32},'fresh64_scene_counts':{'quad':[16,16],'drone':[32]},'regression140_counts':{'quad':76,'drone':64},'regression140_scene_counts':{'quad':[32,12,32],'drone':[64]},'formal_total':204,'fresh_and_regression_denominators_must_remain_separate':True,'new_source_count':96,'new_mask_frame_count':64,'conditional_A_counts':{'quad':16,'drone':16},'conditional_B_counts':{'quad':16,'drone':16},'conditional_B_scene_counts':{'quad':[8,8],'drone':[16]},'conditional_TTA_not_execution_authorized_by_inputplan':True,'B_has_no_Gcase_or_frame_index_or_mask_input':True,'B_not_used_for_current_G_fit_gate_or_hyperparameter_selection':True,'no_same_instance_or_ego_transform_claim':True,'selected236_observations_have_separately_unique_decoded_pixel_and_canonicalXYZ':True,'previous_recent_content_group_exclusion_n':1054,'all5912_historical_model_GT_exposure_explicit':True,'fresh_means_recent_content_relative_not_pristine_holdout':True,'source_checkpoint_sha256':'0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd','source_model_no_retraining':True,'calibration_read_scope':'Only K/E/optionalD existinginput whitelist extracted; no pose/timestamp/targetbbox/class/positive_map. MissingD preservedNone, algorithm must freeze explicit policy.','reuse_source_mask_pairs':[['crossplatform/round12/source146_attempt001','crossplatform/round12/masks501_attempt001'],['crossplatform/round13/source108_attempt001','crossplatform/round13/masks108_attempt001'],['crossplatform/round14/drone_source32_attempt001','crossplatform/round14/drone_masks32_attempt001']],'primary_legacy_endpoint_not_unified_R14combined_on_Quad':True,'no_GPU_model_GT_payload_new_taskIoU_or_new_rawdecode_in_this_preparer':True,'files':{p.name:sha(p) for p in D.glob('*.json')}})
 print(json.dumps({'fresh':64,'regression':140,'source_new':96,'mask_new':64,'unusedB':32,'input_plan_sha256':sha(D/'INPUT_PLAN_LOCK.json')}))
if __name__=='__main__':main()
