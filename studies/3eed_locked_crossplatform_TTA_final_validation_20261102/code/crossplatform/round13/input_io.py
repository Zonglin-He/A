"""R13 strict input/sealed-cache adapter. Never opens annotation containers."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from PIL import Image
B=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(B/'crossplatform/round7'),str(B/'crossplatform/round7/experiment')]
from camera_projection import CameraSpec
MASK_KEYS=('boxes_xyxy','class_ids','scores','masks_bool','instance_indices')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def verify_seal(folder):
 folder=Path(folder);s=read(folder/'SEALED.json')
 for name,digest in s['files'].items():
  p=Path(name);assert not p.is_absolute() and '..' not in p.parts
  assert sha(folder/p)==digest,(folder,name)
 return s
def load_case(row,case,index):
 assert set(row)=={'id','platform','scene','pointpath','metapath','imagepath','caption','suffix','seed'}
 assert row['id']==case['id'] and row['platform']==case['platform'] and row['scene']==case['scene_PRIVATE']
 fr=index[case['frame_alias']]
 for key,hashkey in [('imagepath','image_sha256'),('pointpath','point_sha256')]:
  assert row[key]==fr[key] and sha(row[key])==fr[hashkey]
 # Metadata identity is checked without parsing its label-bearing payload.
 assert row['metapath']==fr['metapath'] and sha(row['metapath'])==fr['metadata_sha256']
 src=B/case['source_dir'];mask=B/case['mask_dir'];ss=read(src/'SEALED.json');ms=read(mask/'SEALED.json')
 cp=src/'source_cache'/(row['id']+'.npz');mp=mask/(case['frame_alias']+'.npz')
 assert ss['files'][str(cp.relative_to(src))]==sha(cp) and ms['files'][mp.name]==sha(mp)
 assert ss['source_unchanged'] and ss['source_repeat_bitexact'] and ss['source_checkpoint_sha256']=='0d8766abaf06c9f3e2500fcb187337982d391f9e4d46ca854f3b3bb70fd33bbd'
 assert ms['trained_or_updated'] is False and ms['mask_returned_original_image_pixels']
 with np.load(cp,allow_pickle=False) as z:c={k:z[k].copy() for k in z.files}
 assert str(c['sample_id'])==row['id']
 assert c['manifest_row_sha256'].tobytes()==hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).digest()
 text=row['caption']+read(src/'registered_plan.json')['sentinel']
 assert c['actual_model_text_sha256'].tobytes()==hashlib.sha256(text.encode()).digest()
 assert c['boxes_raw6'].shape==(1,256,6) and c['text_scores'].shape==(1,256)
 assert int(c['query_index'])==int(c['text_scores'][0].argmax())
 rgb=np.array(Image.open(row['imagepath']).convert('RGB'),copy=True);p=Path(row['pointpath'])
 raw=np.load(p,allow_pickle=False) if p.suffix=='.npy' else np.fromfile(p,np.float32).reshape(-1,4)
 assert raw.ndim==2 and raw.shape[1]>=3 and len(raw)>0 and np.isfinite(raw[:,:3]).all()
 raw=np.ascontiguousarray(raw[:,:3],np.float32)
 fields={k:fr[k] for k in ('image_intrinsic','image_extrinsic')}
 if fr['camera_distortion'] is not None:fields['camera_distortion']=fr['camera_distortion']
 cam=CameraSpec.from_camera_fields(row['platform'],fields,missing_distortion_policy='assume_zero_declared');cam.validated()
 with np.load(mp,allow_pickle=False) as z:
  ins={k:z[k].copy() for k in MASK_KEYS};assert tuple(z['original_hw'])==rgb.shape[:2]
 assert ins['masks_bool'].dtype==bool and ins['masks_bool'].shape==(len(ins['scores']),*rgb.shape[:2])
 assert np.isfinite(ins['scores']).all() and np.all(ins['scores']>=.5)
 return {'frame_alias':case['frame_alias'],'rgb':rgb,'raw_points':raw,'camera':cam,'instances':ins},c,{'source_cache_sha256':sha(cp),'mask_cache_sha256':sha(mp),'image_sha256':fr['image_sha256'],'point_sha256':fr['point_sha256'],'source_seal_sha256':sha(src/'SEALED.json'),'mask_seal_sha256':sha(mask/'SEALED.json')}
