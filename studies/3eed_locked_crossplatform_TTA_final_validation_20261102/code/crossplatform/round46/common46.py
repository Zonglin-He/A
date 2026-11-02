"""R46 data-free import helpers. Protected input access requires root authorization."""
from pathlib import Path
import json,hashlib,os,datetime
import numpy as np
from decimal import Decimal
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def serial(x):
 if isinstance(x,dict):return {str(k):serial(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [serial(v) for v in x]
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,Decimal):return float(x)
 return x
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(serial(v),indent=2,allow_nan=False)+'\n');os.replace(t,p)
def npz(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def seal(folder,state,**kw):
 folder=Path(folder);dump(folder/'SEALED.json',dict(state=state,files={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p!=folder/'SEALED.json'},created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw))
def verify(folder):
 folder=Path(folder);s=read(folder/'SEALED.json')
 for rel,h in s['files'].items():
  assert not Path(rel).is_absolute() and '..' not in Path(rel).parts;assert sha(folder/rel)==h,(folder,rel)
 return s
def context(a):
 # Authorization and all code/state hashes are verified BEFORE manifest access.
 p=Path(a.plan).resolve();auth=read(a.authorization)
 assert auth['state'] in ('ROOT_AUTHORIZED_OLD_A2_ENGINEERING_ONLY_NOT_FINAL_ACCESS','ROOT_AUTHORIZED_R46_FINAL_PREDICTIONS_NO_TASKSCORE')
 plan=read(p)
 assert plan['execution_mode']==('engineering_oldA2' if 'OLD_A2' in auth['state'] else 'final_once')
 if plan['execution_mode']=='final_once':assert auth['plan_sha256']==sha(p) and auth['output_path']==str(Path(a.output).resolve())
 else:assert plan['root_authorization_sha256']==sha(a.authorization) and auth['proposal_sha256']==sha(ROOT/'research/round46/review/ENGINEERING2_PROPOSAL.json')
 assert plan['output_path']==str(Path(a.output).resolve())
 review=read(plan['static_review_path']);assert review['state']=='PASS_R46_ACTUAL_STATIC_REVIEW' and review['plan_sha256']==sha(p)
 for rel,h in plan['program_sha256'].items():assert sha(ROOT/rel)==h,rel
 for rel,h in plan['old_inputs_sha256'].items():assert sha(ROOT/rel)==h,rel
 mp=Path(plan['manifest_path']);assert sha(mp)==plan['manifest_sha256']
 if plan['execution_mode']=='final_once':assert plan['manifest_sha256']==auth['manifest_sha256']
 rows=read(mp);assert len(rows)==plan['expected_n'] and len({r['id'] for r in rows})==len(rows)
 required={'id','platform','scene','pointpath','metapath','imagepath','caption','suffix','seed'};optional={'stage','alias','frame_alias'}
 for r in rows:
  assert required<=set(r)<=required|optional and r['platform'] in ('drone','quad')
  assert isinstance(r['caption'],str) and r['caption'] and isinstance(r['seed'],int) and not isinstance(r['seed'],bool)
  assert isinstance(r['id'],str) and all(c.isalnum() or c in '_-' for c in r['id'])
 if plan['execution_mode']=='engineering_oldA2':
  assert len(rows)==2 and {r['platform'] for r in rows}=={'drone','quad'}
  old=read(ROOT/'research/round24/input/D208_PRIVATE.json');chosen=[min([r for r in old if r['platform']==pl],key=lambda r:hashlib.sha256(('R46engineering|'+r['id']).encode()).hexdigest()) for pl in ('drone','quad')]
  assert rows==chosen,'Only the predeclared old-A2 input-hash cohort is authorized'
 counts={pl:sum(r['platform']==pl for r in rows) for pl in ('drone','quad')}
 assert counts==plan['platform_counts'];return plan,rows,Path(a.output).resolve()
def args(description):
 import argparse
 p=argparse.ArgumentParser(description=description)
 for key in ('plan','authorization','output'):p.add_argument('--'+key,required=True)
 return p.parse_args()
def fresh_observation(r,out):
 """No old501 lookup: same camera/scene builder with fresh authorized masks."""
 import sys
 from PIL import Image
 sys.path[:0]=[str(ROOT/'crossplatform/round15'),str(ROOT/'crossplatform/round7')]
 from camera_projection import CameraSpec
 from scene_observation import build_scene_observation
 folder=Path(out)/'frames';index={x['id_PRIVATE']:x for x in read(folder/'FRAME_BINDINGS_PRIVATE.json')};b=index[r['id']]
 assert sha(r['imagepath'])==b['image_sha256'] and sha(r['pointpath'])==b['point_sha256'] and sha(r['metapath'])==b['metadata_sha256']
 mp=folder/'masks_PRIVATE'/(r['id']+'.npz');assert sha(mp)==b['mask_cache_sha256'];ins=npz(mp);rgb=np.array(Image.open(r['imagepath']).convert('RGB'));p=Path(r['pointpath']);raw=np.load(p,allow_pickle=False) if p.suffix=='.npy' else np.fromfile(p,np.float32).reshape(-1,4);raw=np.ascontiguousarray(raw[:,:3],np.float32);fields=read(folder/'camera_PRIVATE'/(r['id']+'.json'));camera=CameraSpec.from_camera_fields(r['platform'],fields,missing_distortion_policy='assume_zero_declared');assert tuple(ins['original_hw'])==rgb.shape[:2]
 frame={'rgb':rgb,'raw_points':raw,'camera':camera,'instances':{k:ins[k] for k in ('boxes_xyxy','class_ids','scores','masks_bool','instance_indices')}};h,w=rgb.shape[:2];binding={'image_sha256':b['image_sha256'],'point_sha256':b['point_sha256'],'mask_cache_sha256':sha(mp)};scene=build_scene_observation(raw_points=raw,camera_spec=camera,image_size_wh=(w,h),instances=frame['instances'],image_sha256=b['image_sha256']);return frame,scene,binding
