"""Authorized U RGB/XYZ content audit. Never opens task GT or FINAL row/media."""
import sys,time,json,hashlib,collections,numpy as np
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'crossplatform/round30'))
from common30 import verify
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def actual(r):
 with Image.open(r['imagepath']) as im:pix=np.array(im.convert('RGB'),dtype=np.uint8)
 ih=hashlib.sha256(json.dumps(list(pix.shape)).encode()+b'RGB_uint8_c_order'+pix.tobytes()).hexdigest()
 point=Path(r['pointpath']);raw=np.load(point,allow_pickle=False) if point.suffix=='.npy' else np.fromfile(point,dtype='<f4').reshape(-1,4)
 xyz=np.array(raw[:,:3],dtype='<f4',order='C',copy=True);assert xyz.ndim==2 and xyz.shape[1]==3 and np.isfinite(xyz).all();xyz[xyz==0]=0.
 u=np.ascontiguousarray(np.unique(xyz,axis=0),dtype='<f4');gh=hashlib.sha256(str(len(u)).encode()+b'canonical_uniqueXYZ_float32_xyz_lexicographic'+u.tobytes()).hexdigest()
 return ih,gh,len(u),list(pix.shape)

def main():
 start=time.perf_counter();inp=ROOT/'research/NIGHT_20261101/input';U=read(inp/'U128_PRIVATE.json');declared=read(inp/'U128_CONTENT_HASHES_PRIVATE.json');assert len(U)==len(declared)==128 and collections.Counter(r['platform'] for r in U)=={'drone':64,'quad':64}
 lock=ROOT/'research/NIGHT_20261101/SELECTED_NIGHT_CANDIDATE_BEFORE_U_LOCK.json';selection=read(lock);assert selection['U_not_for_selection'] and selection['selected_primary']=='h_scoped_R30'
 verify(ROOT/'crossplatform/round33/sourceU128_attempt001')
 resultrows=[]
 for r,h in zip(U,declared):
  assert r['id']==h['canonical_id_PRIVATE'] and r['stage']=='U'
  assert sha(r['imagepath'])==h['image_file_sha256'] and sha(r['pointpath'])==h['point_file_sha256']
  ih,gh,n,shape=actual(r);assert ih==h['decoded_pixel_sha256'] and gh==h['canonical_XYZ_sha256'] and n==h['unique_XYZ_n']
  assert (ROOT/'crossplatform/round33/sourceU128_attempt001/samples_PRIVATE'/(r['id']+'.npz')).exists()
  resultrows.append({'alias':r['alias'],'platform':r['platform'],'id_PRIVATE':r['id'],'RGB_sha256':ih,'XYZ_sha256':gh,'unique_XYZ_n':n,'decoded_RGB_shape':shape})
 I={r['RGB_sha256'] for r in resultrows};P={r['XYZ_sha256'] for r in resultrows};assert len(I)==len(P)==128
 # Historical canonical content inventory was independently decoded previously.
 # Bind its entries to the current exact historical asset bytes, without GT.
 inv=read(ROOT/'research/round15/input/DECODED_CONTENT_HASHES_PRIVATE.json')['rows_PRIVATE'];byimage={x['image_file_sha256']:x for x in inv}
 prior=sum((read(ROOT/p) for p in ['research/round27/input/ALL528_PRIVATE.json','research/round29/input/R64_PRIVATE.json','research/round30/input/S64_PRIVATE.json']),[]);assert len(prior)==656
 history=list(prior)
 for p in ['research/round15/input/SOURCE96_GPU_INPUTS.json','research/round15/input/REGRESSION140_GPU_INPUTS.json','research/round18/input/ALL96_PRIVATE.json','research/round18/input/C32_PRIVATE.json']:history+=read(ROOT/p)
 old=[];checked={}
 for r in history:
  for f in ['imagepath','pointpath']:
   if r[f] not in checked:checked[r[f]]=sha(r[f])
  h=byimage[checked[r['imagepath']]];assert h['point_file_sha256']==checked[r['pointpath']];old.append(h)
 assert not I&{x['decoded_pixel_sha256'] for x in old};assert not P&{x['canonical_XYZ_sha256'] for x in old}
 # T128 is already exposed; use its independently verified actual-content results.
 T=read(inp/'T128_PRIVATE.json');th=read(inp/'T128_CONTENT_HASHES_PRIVATE.json');tr=read(ROOT/'research/round32/independent_audit/T128_CONTENT_RECONSTRUCTED_PRIVATE.json');assert len(T)==len(th)==len(tr)==128
 assert {x['RGB_sha256'] for x in tr}=={x['decoded_pixel_sha256'] for x in th} and {x['XYZ_sha256'] for x in tr}=={x['canonical_XYZ_sha256'] for x in th}
 assert not I&{x['RGB_sha256'] for x in tr};assert not P&{x['XYZ_sha256'] for x in tr}
 # Only the pre-existing FINAL content-hash list is authorized, not its rows/media.
 final=read(inp/'FINAL384_CONTENT_HASHES_PRIVATE.json');assert len(final)==384 and len({x['decoded_pixel_sha256'] for x in final})==len({x['canonical_XYZ_sha256'] for x in final})==384
 assert not I&{x['decoded_pixel_sha256'] for x in final};assert not P&{x['canonical_XYZ_sha256'] for x in final}
 scenecounts={pl:len({r['scene'] for r in U if r['platform']==pl}) for pl in ['drone','quad']}
 scene_overlap={pl:{'U_scenes_overlapping_T128':len({r['scene'] for r in U if r['platform']==pl}&{r['scene'] for r in T if r['platform']==pl}),'U_scenes_overlapping_prior656':len({r['scene'] for r in U if r['platform']==pl}&{r['scene'] for r in prior if r['platform']==pl})} for pl in ['drone','quad']}
 result={'state':'PASS','actual_U128_RGB_decode_and_canonical_XYZ_recomputed':128,'unique_RGB':128,'unique_XYZ':128,'platform_observations':dict(collections.Counter(r['platform'] for r in U)),'scene_counts':scenecounts,'scene_overlap_with_exposed_history':scene_overlap,'content_disjoint_prior656_plus_declared_history':True,'previous_asset_bytes_bound_to_independent_content_inventory':True,'T128_content_disjoint':True,'T128_independent_content_audit_reused_not_redecoded':True,'FINAL384_hashonly_content_disjoint':True,'source_U_capture_seal_verified':True,'source_U_capture_seal_sha256':sha(ROOT/'crossplatform/round33/sourceU128_attempt001/SEALED.json'),'U_primary_preselection_lock_sha256':sha(lock),'U_manifest_sha256':sha(inp/'U128_PRIVATE.json'),'U_content_declaration_sha256':sha(inp/'U128_CONTENT_HASHES_PRIVATE.json'),'U_task_GT_not_read':True,'FINAL_manifest_media_GT_not_read':True,'no_GPU_inference_or_training':True,'limitations':['New content is disjoint at decoded pixels and canonical XYZ; this does not establish independent scenes or remove earlier official full-validation exposure.','Prior media are checked by current byte hashes against a previously independently decoded content inventory; prior656/history are not all decoded again.','T128 comparison reuses its existing independent decoded-content audit; only U actual media are newly decoded here.','No U task correctness, IoU, label quality, or adaptation benefit is assessed in this content-only audit.'],'seconds':time.perf_counter()-start}
 (OUT/'U128_CONTENT_RECONSTRUCTED_PRIVATE.json').write_text(json.dumps(resultrows,indent=2)+'\n');(OUT/'U_CONTENT_AUDIT_RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
