"""Authorized R15 independent pixel/geometry hashing only; no display/GT/model."""
from pathlib import Path
import hashlib,json,time,collections
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parents[3];D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
catfile=R/'research/round12/data_audit/PHYSICAL_INPUT_HASH_INDEX_PRIVATE.json';cat=read(catfile)['rows_PRIVATE'];byid={r['id_PRIVATE']:r for r in cat};groups=collections.defaultdict(list)
for r in cat:groups[r['physical_group']].append(r)
canonical=[min(v,key=lambda r:r['id_PRIVATE']) for _,v in sorted(groups.items())];assert len(canonical)==1753
used_paths=['research/round9/experiment/diagnostic160_gpu_inputs.json','research/round10/input_prepare/diagnostic160_gpu_inputs.json','research/round11/input_prepare/diagnostic320_gpu_inputs.json','research/round8/experiment/smoke40_gpu_inputs.json','research/round7/experiment/smoke_gpu_inputs.json','research/round13/input/ALL116_GPU_INPUTS.json','research/round14/input/DRONE_ALL64_GPU_INPUTS.json','research/round14/input/QUAD_A76_GPU_INPUTS.json','research/round14/input/QUAD_B76_GPU_INPUTS.json']
used={byid[r['id']]['physical_group'] for p in used_paths for r in read(R/p)}
for p in ['research/round12/data_audit/FRAME_INDEX_PRIVATE.json','research/round13/input/FRAME_INDEX_PRIVATE.json','research/round14/input/DRONE_FRAME_INDEX_PRIVATE.json']:used.update(r['physical_group_sha256'] for r in read(R/p).values())
assert len(used)==1054
outpath=D/'DECODED_CONTENT_HASHES_PRIVATE.json';receipt=D/'DECODED_CONTENT_INDEPENDENCE_RECEIPT.json';collfile=D/'CONTENT_COLLISIONS_PRIVATE.json'
assert not any(p.exists() for p in [outpath,receipt,collfile]);rows=[];start=time.monotonic()
for i,r in enumerate(canonical):
 image=Path(r['image_path_PRIVATE']);point=Path(r['point_path_PRIVATE']);assert sha(image)==r['image_sha256'] and sha(point)==r['point_sha256']
 with Image.open(image) as im:pixels=np.array(im.convert('RGB'),np.uint8)
 assert pixels.ndim==3 and pixels.shape[2]==3
 imagehash=hashlib.sha256(json.dumps(list(pixels.shape)).encode()+b'RGB_uint8_c_order'+pixels.tobytes()).hexdigest()
 if point.suffix=='.npy':raw=np.load(point,allow_pickle=False);assert raw.ndim==2 and raw.shape[1]>=3
 else:raw=np.fromfile(point,dtype='<f4');assert raw.size%4==0;raw=raw.reshape(-1,4)
 xyz=np.array(raw[:,:3],dtype='<f4',order='C',copy=True);finite=bool(np.isfinite(xyz).all());geomhash=None;unique_n=None
 if finite:
  xyz[xyz==0]=0.;u=np.ascontiguousarray(np.unique(xyz,axis=0),dtype='<f4');unique_n=len(u);geomhash=hashlib.sha256(str(len(u)).encode()+b'canonical_uniqueXYZ_float32_xyz_lexicographic'+u.tobytes()).hexdigest()
 rows.append({'canonical_id_PRIVATE':r['id_PRIVATE'],'platform':r['platform'],'scene_PRIVATE':r['scene_PRIVATE'],'pair_group_sha256':r['physical_group'],'previous_recent_input':r['physical_group'] in used,'image_file_sha256':r['image_sha256'],'point_file_sha256':r['point_sha256'],'decoded_pixel_sha256':imagehash,'canonical_XYZ_sha256':geomhash,'finite_XYZ':finite,'raw_point_n':len(xyz),'unique_XYZ_n':unique_n,'original_HW':list(pixels.shape[:2])})
 if (i+1)%250==0:print(json.dumps({'content_groups_hashed':i+1,'of':len(canonical),'seconds':time.monotonic()-start}),flush=True)
parent=list(range(len(rows)))
def find(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
def union(a,b):
 a,b=find(a),find(b)
 if a!=b:parent[b]=a
collisions={}
for field in ['decoded_pixel_sha256','canonical_XYZ_sha256']:
 bins=collections.defaultdict(list)
 for i,r in enumerate(rows):
  if r[field] is not None:bins[r[field]].append(i)
 clusters=[ix for ix in bins.values() if len(ix)>1];collisions[field]=clusters
 for ix in clusters:
  for j in ix[1:]:union(ix[0],j)
comps=collections.defaultdict(list)
for i in range(len(rows)):comps[find(i)].append(i)
for ix in comps.values():
 comp=hashlib.sha256('|'.join(sorted(rows[i]['pair_group_sha256'] for i in ix)).encode()).hexdigest();touches=any(rows[i]['previous_recent_input'] for i in ix)
 for i in ix:rows[i]['content_component_sha256']=comp;rows[i]['component_touches_previous_recent_input']=touches
private={'state':'R15_ACTUAL1753_DECODED_HASHES_ONLY_NO_GT_MODEL','program_sha256':sha(__file__),'source_catalog_sha256':sha(catfile),'rows_PRIVATE':rows};outpath.write_text(json.dumps(private,indent=2)+'\n')
collision_out={'state':'ACTUAL_CANONICAL_COLLISION_TABLE_BEFORE_INPUT_SELECTION','program_sha256':sha(__file__),'hash_inventory_sha256':sha(outpath),'collisions_PRIVATE':{k:[[rows[i]['canonical_id_PRIVATE'] for i in ix] for ix in clusters] for k,clusters in collisions.items()},'invalid_XYZ_ids_PRIVATE':[r['canonical_id_PRIVATE'] for r in rows if not r['finite_XYZ']]};collfile.write_text(json.dumps(collision_out,indent=2)+'\n')
summary=[]
for plat in ['drone','quad']:
 for j,scene in enumerate(sorted({r['scene_PRIVATE'] for r in rows if r['platform']==plat})):
  rr=[r for r in rows if r['platform']==plat and r['scene_PRIVATE']==scene];available=[r for r in rr if not r['component_touches_previous_recent_input'] and r['finite_XYZ']];summary.append({'platform':plat,'scene_alias':f'{plat}_scene_{j+1}','all_pair_groups':len(rr),'byte_pair_unused_n':sum(not r['previous_recent_input'] for r in rr),'decoded_eligible_rows':len(available),'decoded_distinct_components':len({r['content_component_sha256'] for r in available})})
x={'state':'PASS_R15_ACTUAL_DECODED_INVENTORY_COLLISIONS_RECORDED_NOT_INPUT_LOCK','program_sha256':sha(__file__),'source_catalog_sha256':sha(catfile),'hash_inventory_sha256':sha(outpath),'collision_table_sha256':sha(collfile),'all_groups_decoded':1753,'previous_recent_pairgroups':1054,'metadata_remaining_pairgroups':699,'unique_decoded_pixels':len({r['decoded_pixel_sha256'] for r in rows}),'unique_canonical_XYZ':len({r['canonical_XYZ_sha256'] for r in rows if r['finite_XYZ']}),'pixel_collision_groups':len(collisions['decoded_pixel_sha256']),'XYZ_collision_groups':len(collisions['canonical_XYZ_sha256']),'invalid_XYZ_n':sum(not r['finite_XYZ'] for r in rows),'eligibility_by_scene':summary,'seconds':time.monotonic()-start,'canonicalization':'RGB PIL.convertRGB uint8/noEXIFtranspose; XYZ first3 only float32 finite/signedzero normalized/lexicographicunique, reflectance order multiplicity excluded; image and geometry collisions joined by OR components.','source_selection_uses_caption_G_model_GT_or_performance':False,'raw_pixels_or_coordinates_saved_or_displayed':False,'GT_payload_new_IoU_model_or_GPU_access':False,'historical_5912_model_GT_exposure_not_erased':True,'next':'Report collisions then deterministic root-prescribed SHA selection, no frozenplan overwritten.'};receipt.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({'state':x['state'],'seconds':x['seconds'],'pixel_collision_groups':x['pixel_collision_groups'],'XYZ_collision_groups':x['XYZ_collision_groups'],'eligibility':summary,'receipt_sha256':sha(receipt)}),flush=True)
