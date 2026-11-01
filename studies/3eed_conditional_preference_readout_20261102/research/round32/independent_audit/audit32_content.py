"""Read actual T128 RGB/XYZ only. Reserve pools are checked by hash lists only."""
import sys,json,hashlib,time,numpy as np
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];P=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def actual(r):
 with Image.open(r['imagepath']) as im:pix=np.array(im.convert('RGB'),np.uint8)
 ih=hashlib.sha256(json.dumps(list(pix.shape)).encode()+b'RGB_uint8_c_order'+pix.tobytes()).hexdigest();p=Path(r['pointpath'])
 if p.suffix=='.npy':raw=np.load(p,allow_pickle=False)
 else:raw=np.fromfile(p,dtype='<f4').reshape(-1,4)
 xyz=np.array(raw[:,:3],dtype='<f4',order='C',copy=True);assert np.isfinite(xyz).all();xyz[xyz==0]=0.;u=np.ascontiguousarray(np.unique(xyz,axis=0),dtype='<f4');gh=hashlib.sha256(str(len(u)).encode()+b'canonical_uniqueXYZ_float32_xyz_lexicographic'+u.tobytes()).hexdigest();return ih,gh,len(u)
start=time.perf_counter();T=read(ROOT/'research/NIGHT_20261101/input/T128_PRIVATE.json');th=read(ROOT/'research/NIGHT_20261101/input/T128_CONTENT_HASHES_PRIVATE.json');assert len(T)==len(th)==128;results=[]
for r,h in zip(T,th):
 assert r['id']==h['canonical_id_PRIVATE'];assert sha(r['imagepath'])==h['image_file_sha256'] and sha(r['pointpath'])==h['point_file_sha256'];i,g,n=actual(r);assert i==h['decoded_pixel_sha256'] and g==h['canonical_XYZ_sha256'] and n==h['unique_XYZ_n'];results.append({'alias':r['alias'],'platform':r['platform'],'RGB_sha256':i,'XYZ_sha256':g,'canonical_points':n})
assert len({x['RGB_sha256'] for x in results})==len({x['XYZ_sha256'] for x in results})==128
# Previous inventory has authenticated canonical content hashes; verify current byte bindings.
inv=read(ROOT/'research/round15/input/DECODED_CONTENT_HASHES_PRIVATE.json')['rows_PRIVATE'];byimage={x['image_file_sha256']:x for x in inv};prior=sum((read(ROOT/p) for p in ['research/round27/input/ALL528_PRIVATE.json','research/round29/input/R64_PRIVATE.json','research/round30/input/S64_PRIVATE.json']),[]);assert len(prior)==656;history=list(prior)
for p in ['research/round15/input/SOURCE96_GPU_INPUTS.json','research/round15/input/REGRESSION140_GPU_INPUTS.json','research/round18/input/ALL96_PRIVATE.json','research/round18/input/C32_PRIVATE.json']:history+=read(ROOT/p)
prev=[];checked={}
for r in history:
 for field in ['imagepath','pointpath']:
  if r[field] not in checked:checked[r[field]]=sha(r[field])
 h=byimage[checked[r['imagepath']]];assert h['point_file_sha256']==checked[r['pointpath']];prev.append(h)
Timages={x['RGB_sha256'] for x in results};Tpoints={x['XYZ_sha256'] for x in results};assert not Timages&{h['decoded_pixel_sha256'] for h in prev};assert not Tpoints&{h['canonical_XYZ_sha256'] for h in prev}
reserve={}
for name in ['U128','FINAL384']:
 # Deliberately NOT reading *_PRIVATE row manifests, raw media or annotation/GT.
 hh=read(ROOT/'research/NIGHT_20261101/input'/(name+'_CONTENT_HASHES_PRIVATE.json'));images={x['decoded_pixel_sha256'] for x in hh};points={x['canonical_XYZ_sha256'] for x in hh};assert len(images)==len(points)==len(hh);assert not Timages&images and not Tpoints&points;reserve[name]={'n_hash_records_only':len(hh),'RGB_disjoint_T128':True,'XYZ_disjoint_T128':True};
U=read(ROOT/'research/NIGHT_20261101/input/U128_CONTENT_HASHES_PRIVATE.json');F=read(ROOT/'research/NIGHT_20261101/input/FINAL384_CONTENT_HASHES_PRIVATE.json');assert not {x['decoded_pixel_sha256'] for x in U}&{x['decoded_pixel_sha256'] for x in F};assert not {x['canonical_XYZ_sha256'] for x in U}&{x['canonical_XYZ_sha256'] for x in F}
result={'state':'PASS','actual_T128_decoded_RGB_and_canonical_XYZ_recomputed':128,'T128_unique_RGB':128,'T128_unique_XYZ':128,'prior656_plus_declared_history_exact_content_disjoint':True,'previous_asset_bytes_bound_to_independently_decoded_inventory':True,'reserve_hash_checks':reserve,'U128_and_FINAL384_row_manifests_media_and_GT_not_read':True,'T128_own_task_GT_not_read_by_this_content_audit':True,'historical_scene_fullVAL_exposure_not_erased':True,'T_scene_counts':{p:len({r['scene'] for r in T if r['platform']==p}) for p in ['drone','quad']},'seconds':time.perf_counter()-start};(P/'CONTENT_AUDIT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');(P/'T128_CONTENT_RECONSTRUCTED_PRIVATE.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(result,indent=2))
