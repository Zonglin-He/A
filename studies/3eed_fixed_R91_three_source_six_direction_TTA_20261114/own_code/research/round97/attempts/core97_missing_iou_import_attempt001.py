"""Transfer wrappers only: existing R76/R91 math, source-pinned 3EED weights."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import sys,json,hashlib,time
R97=Path(__file__).resolve().parents[2];H97=R97/'research/round97';B97=R97/'crossplatform/round97'
sys.path.insert(0,str(R97/'crossplatform/round96'))
import factory as factory96
sys.path[:0]=[str(R97/'research/round91'),str(R97/'research/round76'),str(R97/'research/round66'),str(R97/'crossplatform/round51'),str(R97/'crossplatform/round7/experiment')]
from targets91 import *
from interface66 import capture,replay
from prepare76 import geometry
from runner_helpers import sample_points
from model51 import fingerprint
sys.path.insert(0,str(R97/'research/round5/audit'));from independent_core_v2 import candidate_ious,transform_gt,yaw_matrix
def jread(p):return json.loads(Path(p).read_text())
def jdump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def source97(pl):return next(x for x in jread(H97/'experiment/DESIGN_LOCK.json')['sources'] if x['source']==pl)
def strict97():
 torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
def base97(pl):
 factory96.configure({'seed':0,'torch_num_threads':4});strict97();c=source97(pl);assert sha(c['checkpoint_path'])==c['checkpoint_sha256'];m=factory96.model();v=torch.load(c['checkpoint_path'],map_location='cpu',weights_only=True);assert v['epoch']==100 and v['trained_platforms']==[pl];m.load_state_dict(v['model'],strict=True);del v;return m.cuda().eval().requires_grad_(False)
def rows97(pl,split):return jread(H97/'input'/((('TARGET_'+pl.upper()) if split=='target' else ('SOURCE_'+pl.upper()+'_'+split))+ '_PRIVATE.json'))
def sample97(base,r):
 pts,ix,_=sample_points(r)
 if r['platform']=='waymo':
  from utils.transform_waymo import transform_to_front_view
  pts,_=transform_to_front_view(pts,np.array([[0.,0.,0.,1.,1.,1.,0.]]),['F','FL','FR','SL','SR'][r['suffix']]);pts=np.ascontiguousarray(pts,np.float32)
 z,rec=capture(base,pts,ix,r['caption']);rec['source_checkpoint_sha256']=source97(CURRENT_SOURCE97)['checkpoint_sha256'];rec['source_platform']=CURRENT_SOURCE97;g,proof=geometry(r,z);return z,g,rec,proof
CURRENT_SOURCE97=None
def setsource97(pl):global CURRENT_SOURCE97;CURRENT_SOURCE97=pl
def role97(base,m,r,z0,g,z,d):
 role=jread(B97/'roles_PRIVATE'/(r['id']+'.json'));graph,doc=role['graph'],role['doc'];assert role['caption_sha256']==hashlib.sha256(r['caption'].encode()).hexdigest();m.route=None;m.use_interface=False;aa=[];hk=base.cross_encoder.layers[-1].cross_layer.cross_lv.register_forward_hook(lambda _m,_a,o:aa.append(o[1].detach().cpu().numpy()[0]))
 try:
  with torch.no_grad():p0,_,_=m.main(base,z,d,'bidirectional')
 finally:hk.remove()
 roles,offset=token_roles88(base,r,z0,graph,doc);rt,info=route89(r,{**z0,'frozen_probability':p0.cpu().numpy()},g,graph,roles,aa[0],offset,doc);return rt,info
def model97(base,pl):
 torch.manual_seed(0);m=Fusion91(base.contrastive_align_projection_image).cuda().eval().requires_grad_(False)
 if pl=='waymo':
  selected=jread(R97/'crossplatform/round91/trained_source_attempt001/SELECTED_SOURCE_ONLY.json')['models']['R76_full_disc'];S=load(R97/selected['state_path'])
 else:S=load(B97/pl/'SELECTED_SOURCE_ONLY.npz')
 restore(m,S);m.use_interface=True;m.pool_mode='strict';m.variant='full';return m,S
def views97(m,base,z,d,route):
 m.route=route;pm,im=maskpair79({'id':CURRENT_ID97},d)
 with torch.no_grad():vv=[native_feature91(m,base,z,d,a,b) for a,b in [(None,None)]+list(zip(pm,im))]
 return vv,pm,im
CURRENT_ID97=None
def competitor97(r,z0,rt,y):
 meta=jread(r['metapath']);target=meta['ground_info'][r['object_index']];cl=str(target['class']).lower().strip();gt=transform_gt(target['bbox_3d'],r['platform'],r['suffix']);raw=[(x['bbox_3d'],str(x['class']).strip().lower()) for x in meta['ground_info']]
 for x in meta['ground_info']:raw.extend((o['bbox_3d_other'],str(o['class_other']).strip().lower()) for o in x.get('others',[]))
 objects=[];seen=set()
 for b,c in raw:
  if c!=cl:continue
  b=transform_gt(b,r['platform'],r['suffix']);key=tuple(np.round(b.astype(float),5))
  if key in seen:continue
  seen.add(key)
  if oriented_iou(gt[:3],gt[3:6],yaw_matrix(gt[6]),b[:3],b[3:6],yaw_matrix(b[6]))<.05:objects.append(b)
 boxes=z0['boxes'].copy();boxes[:,3:]=np.maximum(boxes[:,3:],1e-6);own=rt['own_support'].astype(bool);i=int(y.argmax());positive=bool(y[i]>.25 and own[i].sum()>=2);other=np.stack([candidate_ious(boxes,b)[0] for b in objects],1) if objects else np.zeros((256,0));assign=np.full(256,-1,np.int64)
 if positive and objects:
  best=other.argmax(1);top=other[np.arange(256),best];altern=other.copy();altern[np.arange(256),best]=-1;second=np.maximum(y,altern.max(1));lo=boxes[:,:3]-boxes[:,3:]/2;hi=boxes[:,:3]+boxes[:,3:]/2;iv=np.maximum(0,np.minimum(hi,hi[i])-np.maximum(lo,lo[i])).prod(1);vol=boxes[:,3:].prod(1);iou=iv/np.maximum(1e-12,vol+vol[i]-iv);jac=(own&own[i]).sum(1)/np.maximum((own|own[i]).sum(1),1);ok=(top>.25)&(top-second>=.1)&(y<.05)&(own.sum(1)>=2)&(jac<=.1)&(iou<=.1);assign[ok]=best[ok]
 return i,assign,{'annotated_distinct_same_class':len(objects),'valid_positive':positive,'eligible_negative_queries':int((assign>=0).sum())}
