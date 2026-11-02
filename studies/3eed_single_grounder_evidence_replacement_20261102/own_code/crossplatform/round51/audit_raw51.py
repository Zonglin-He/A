"""UNEXECUTED two-case independent raw-support / sparse-RGB CPU audit.

Does not import instances51/attributes51 or any model/evaluator. The R7 camera
whitelist and projection are shared, explicitly limiting independence to their
consumers: unique raw-point support, components, ownership, z-buffer and colors.
No semantic instance labels, complete-object masks or physical calibration truth
are inferred by these checks.
"""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
MANIFEST='research/round24/input/ALL336_PRIVATE.json'
sys.path.insert(0,str(ROOT/'crossplatform/round7'))
from camera_projection import CameraSpec,read_camera_allowlist,project_model_points,raw_to_model_points

CONFIG={'minimum_depth_m':.1,'minimum_component_points':6,'connectivity_neighbors':8,
 'radius_nn_multiplier':2.5,'minimum_radius_m':.08,'maximum_radius_m':.45,
 'dominant_component_fraction':.6,'maximum_runner_fraction':.25,
 'deduplicate_jaccard':.7,'deduplicate_each_coverage':.75,'maximum_shared_fraction':.25,
 'support_box_tolerance_m':0.,'association_unknown_pseudocount':6.}
PROTOTYPES={'red':(190,40,40),'orange':(225,120,35),'yellow':(225,210,40),
 'green':(50,140,65),'cyan':(50,180,190),'blue':(45,80,180),'purple':(130,65,160),
 'pink':(225,145,175),'brown':(120,75,40),'beige':(200,180,140),
 'white':(235,235,235),'gray':(135,135,135),'black':(30,30,30)}


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def dump(path,x):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
 temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');temp.replace(path)
def path(value):
 p=(ROOT/value).resolve();assert p.is_relative_to(ROOT) and 'FINAL' not in str(p) and 'round46/final' not in str(p)
 return p
def array_sha(a):
 a=np.ascontiguousarray(a);return hashlib.sha256(a.dtype.str.encode()+str(a.shape).encode()+a.tobytes()).hexdigest()
def equal(a,b,label):
 a=np.asarray(a);b=np.asarray(b)
 assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),label
def close(a,b,label,bound=1e-11):
 a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape and np.allclose(a,b,rtol=0,atol=bound,equal_nan=True),label
def scalarclose(a,b,label,bound=1e-11):
 if a is None or b is None:assert a is b,label
 else:assert np.isfinite(a) and np.isfinite(b) and abs(a-b)<=bound,label
def pack(parts):
 offsets=np.zeros(len(parts)+1,np.int64);offsets[1:]=np.cumsum([len(x) for x in parts])
 return offsets,np.concatenate(parts) if len(parts) else np.empty(0,np.int64)


def components(xyz,indices):
 """Independent adjacency-list DFS, not producer sparse-graph components."""
 if len(indices)<6:return [],None
 points=xyz[indices];dist,neigh=cKDTree(points).query(points,k=min(len(indices),9),workers=1)
 positive=dist[:,1][dist[:,1]>1e-8]
 scale=float(np.median(positive)) if len(positive) else .08
 radius=float(np.clip(2.5*scale,.08,.45));adj=[set() for _ in indices]
 for i in range(len(indices)):
  for d,j in zip(dist[i,1:],neigh[i,1:]):
   if np.isfinite(d) and d<=radius:adj[i].add(int(j));adj[int(j)].add(i)
 visited=np.zeros(len(indices),bool);found=[]
 for start in range(len(indices)):
  if visited[start]:continue
  stack=[start];visited[start]=True;members=[]
  while stack:
   q=stack.pop();members.append(int(indices[q]))
   for j in adj[q]:
    if not visited[j]:visited[j]=True;stack.append(j)
  if len(members)>=6:found.append(np.array(sorted(members),np.int64))
 found.sort(key=lambda x:(-len(x),int(x[0])))
 return found,radius


def compatible(a,b):
 intersection=len(a&b)
 return intersection/len(a|b)>=.7 and intersection/len(a)>=.75 and intersection/len(b)>=.75


def independent_geometry(raw,boxes,camera,wh):
 unique,first=np.unique(raw.astype(np.float64),axis=0,return_index=True)
 model=unique.astype(np.float32)
 if camera.platform=='drone':model[:,2]+=np.float32(1.8)
 elif camera.platform!='quad':raise ValueError('DEV target platforms only')
 supports=[];parts=[];radii=np.full(256,np.nan);candidates=[]
 for q,box in enumerate(boxes):
  support=(np.flatnonzero(np.all(np.abs(model.astype(np.float64)-box[:3])<=box[3:]/2,axis=1))
   if (box[3:]>0).all() else np.empty(0,np.int64))
  cs,r=components(model,support);supports.append(support);parts.append(cs)
  if r is not None:radii[q]=r
  for c,component in enumerate(cs):candidates.append((q,c,component))
 candidates.sort(key=lambda v:(-len(v[2]),int(v[2][0]),v[0],v[1]))
 representatives=[];member_sets=[];assignment={};ambiguous=set()
 for q,c,component in candidates:
  pointset=set(component.tolist())
  possible=[i for i,g in enumerate(representatives) if pointset&set(g.tolist())]
  matches=[i for i in possible if all(compatible(pointset,s) for s in member_sets[i])]
  if len(matches)==1:g=matches[0];member_sets[g].append(pointset)
  else:
   g=len(representatives);representatives.append(component.copy());member_sets.append([pointset])
   if len(matches)>1:ambiguous.add((q,c))
  assignment[q,c]=g
 memberships=np.zeros(len(unique),np.int32);owner=np.full(len(unique),-1,np.int64)
 for i,ids in enumerate(representatives):memberships[ids]+=1;owner[ids]=i
 owner[memberships>1]=-2
 strict=np.full(256,-1,np.int64);fractions=np.zeros((256,3))
 for q,cs in enumerate(parts):
  if not cs:continue
  f0=len(cs[0])/len(supports[q]);f1=len(cs[1])/len(supports[q]) if len(cs)>1 else 0.
  shared=float(np.mean(memberships[cs[0]]>1));fractions[q]=[f0,f1,shared]
  tied=len(cs)>1 and len(cs[0])==len(cs[1])
  if not tied and f0>=.6 and f1<=.25 and shared<=.25 and (q,0) not in ambiguous:strict[q]=assignment[q,0]
 # Projection deliberately shares the original camera API; everything after
 # projection below is independently reconstructed from raw numerical values.
 projection=project_model_points(raw_to_model_points(unique,camera.platform,0),camera,wh,min_depth=.1)
 valid=np.flatnonzero(projection['in_image']);xy=np.floor(projection['uv'][valid]).astype(np.int64)
 keys=xy[:,1]*wh[0]+xy[:,0]
 records=np.rec.fromarrays([keys,projection['depth'][valid],unique[valid,0],unique[valid,1],unique[valid,2],first[valid]],
  names=['pixel','depth','x','y','z','raw_id'])
 order=np.argsort(records,order=['pixel','depth','x','y','z','raw_id'],kind='stable')
 valid=valid[order];keys=keys[order]
 keep=np.r_[True,keys[1:]!=keys[:-1]] if len(keys) else np.empty(0,bool)
 visible=valid[keep];pixels=np.floor(projection['uv'][visible]).astype(np.int64)
 visible_owner=owner[visible];counts=np.zeros((256,len(representatives)),np.int64)
 total=np.zeros(256,np.int64);unassigned=np.zeros(256,np.int64);overlap=np.zeros(256,np.int64)
 for q,ids in enumerate(supports):
  local=visible_owner[np.isin(visible,ids,assume_unique=True)];total[q]=len(local)
  unassigned[q]=int((local==-1).sum());overlap[q]=int((local==-2).sum())
  if len(representatives):counts[q]=np.bincount(local[local>=0],minlength=len(representatives))
 conditional=np.divide(counts,counts.sum(1)[:,None],out=np.zeros_like(counts,dtype=np.float64),where=counts.sum(1)[:,None]>0)
 subprob=counts/(total[:,None]+6.)
 result={'query_instance_ids':strict,'query_support_counts':np.array(list(map(len,supports)),np.int64),
  'query_component_counts':np.array(list(map(len,parts)),np.int64),'query_component_radius_m':radii,
  'query_dominant_runner_shared_fractions':fractions,'visible_raw_point_indices':first[visible].astype(np.int64),
  'visible_pixels_xy':pixels,'visible_uv':projection['uv'][visible],
  'visible_depth_camera_m':projection['depth'][visible],'visible_point_owner_instance_ids':visible_owner,
  'query_instance_visible_counts':counts,'query_allvisible_counts':total,'query_visible_unassigned_counts':unassigned,
  'query_visible_ambiguous_counts':overlap,'query_instance_conditional_weights':conditional,
  'query_instance_weights':conditional.copy(),'query_instance_subprob_weights':subprob,'query_unknown_mass':1-subprob.sum(1)}
 for prefix,sets in [('instance_point',[first[g].astype(np.int64) for g in representatives]),
  ('instance_exclusive_point',[first[g[owner[g]==i]].astype(np.int64) for i,g in enumerate(representatives)]),
  ('instance_visible_point',[first[visible[visible_owner==i]].astype(np.int64) for i in range(len(representatives))]),
  ('query_support_point',[first[g].astype(np.int64) for g in supports])]:
  off,values=pack(sets);result[prefix+'_offsets']=off;result[prefix+'_indices']=values
 pixelsets=[pixels[visible_owner==i] for i in range(len(representatives))]
 off,_=pack(pixelsets);result['instance_crop_mask_pixel_offsets']=off
 result['instance_crop_mask_pixel_xy']=np.concatenate(pixelsets) if pixelsets else np.empty((0,2),np.int64)
 result['instance_sparse_mask_values']=np.ones(off[-1],bool)
 rects=np.zeros((len(representatives),4),np.int64)
 xyz=np.full((len(representatives),3),np.nan);uv=np.full((len(representatives),2),np.nan)
 for i,px in enumerate(pixelsets):
  if len(px):
   rects[i]=np.r_[px.min(0),px.max(0)+1]
   xyz[i]=model[visible[visible_owner==i]].astype(np.float64).mean(0)
   uv[i]=projection['uv'][visible[visible_owner==i]].mean(0)
 result['instance_rect_xyxy']=rects;result['instance_crop_shape_hw']=(rects[:,2:]-rects[:,:2])[:,::-1]
 result['instance_crop_local_mask_pixel_xy']=(np.concatenate([px-rects[i,:2] for i,px in enumerate(pixelsets)])
  if pixelsets else np.empty((0,2),np.int64))
 result['instance_visible_centroid_xyz']=xyz;result['instance_visible_centroid_uv']=uv
 return result,{'raw_point_count':len(raw),'unique_point_count':len(unique),'canonical_hypotheses':len(representatives),
  'visible_pixels':len(pixels),'projection_common_dependency':'LOCKED_R7_CAMERA_API_NOT_INDEPENDENT_PHYSICAL_CALIBRATION',
  'independent_components_DFS':True,'independent_complete_link_sets':True,'independent_record_sort_zbuffer':True}


def hsv(pixels):
 x=np.asarray(pixels,np.float64)/255.;maximum=x.max(1);minimum=x.min(1);span=maximum-minimum
 hue=np.zeros(len(x));sat=np.divide(span,maximum,out=np.zeros_like(span),where=maximum>0)
 maximum_channel=x.argmax(1)
 for channel,(a,b) in enumerate([(1,2),(2,0),(0,1)]):
  chosen=(span>0)&(maximum_channel==channel)
  hue[chosen]=((x[chosen,a]-x[chosen,b])/span[chosen]+2*channel)/6
 return np.c_[hue%1,sat,maximum]
def lab(pixels):
 x=np.asarray(pixels,np.float64)/255.
 linear=np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
 xyz=linear@np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]]).T
 xyz=xyz/np.array([.95047,1.,1.08883]);d=6/29
 f=np.where(xyz>d**3,np.cbrt(xyz),xyz/(3*d*d)+4/29)
 return np.c_[116*f[:,1]-16,500*(f[:,0]-f[:,1]),200*(f[:,1]-f[:,2])]
def colors(pixels):
 if not len(pixels):return {k:None for k in PROTOTYPES}
 h,s,v=hsv(pixels).T;l=lab(pixels);out={}
 for name,prototype in PROTOTYPES.items():
  ph,ps,_=hsv(np.array([prototype]))[0];distance=np.minimum(abs(h-ph),1-abs(h-ph))
  chromatic=np.exp(-.5*(distance/.075)**2)*np.clip(s/.35,0,1)
  if name=='black':support=np.clip((.35-v)/.25,0,1)
  elif name=='white':support=np.clip((v-.60)/.30,0,1)*np.clip((.30-s)/.25,0,1)
  elif name=='gray':support=np.clip((.30-s)/.25,0,1)*np.exp(-.5*((v-.52)/.25)**2)
  elif name=='brown':support=chromatic*np.exp(-.5*((l[:,0]-40)/20)**2)
  elif name in ('pink','beige'):support=chromatic*np.exp(-.5*((s-ps)/.25)**2)*np.exp(-.5*((l[:,0]-75)/25)**2)
  else:support=chromatic*np.clip((v-.08)/.25,0,1)
  pl=lab(np.array([prototype]))[0];labsupport=np.exp(-.5*np.sum(((l-pl)/30.)**2,axis=1))
  out[name]=float(np.mean((support+labsupport)*.5))
 return out


def audit_colors(rgb,raw,a,attrs):
 assert attrs['minimum_color_pixels']==attrs['minimum_geometry_points']==8
 assert attrs['color_prototypes_rgb']=={k:list(v) for k,v in PROTOTYPES.items()}
 assert attrs['color_logodds_beta_pseudocount']==.5 and attrs['color_logodds_clip']==4.
 assert attrs['rgb_sha256']==hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
 xy=a['visible_pixels_xy'];nscene=len(xy);scene=colors(rgb[xy[:,1],xy[:,0]])
 assert nscene==attrs['n_scene_visible_pixels']
 for name in PROTOTYPES:scalarclose(scene[name],attrs['scene_visible_color_soft_support'][name],'scene_RGB_'+name)
 assert len(attrs['instances'])==len(a['instance_point_offsets'])-1
 eligible=unknown=0
 for i,row in enumerate(attrs['instances']):
  lo,hi=a['instance_crop_mask_pixel_offsets'][i:i+2];pixels=a['instance_crop_mask_pixel_xy'][lo:hi]
  po,ph=a['instance_visible_point_offsets'][i:i+2];ids=a['instance_visible_point_indices'][po:ph]
  assert i==row['instance_id'] and len(pixels)==len(ids)==row['n_unique_visible_rgb_pixels']
  assert len(np.unique(pixels,axis=0))==len(pixels)
  samples=rgb[pixels[:,1],pixels[:,0]];enough=len(pixels)>=8
  expected=colors(samples);med=np.median(samples,axis=0).tolist() if len(samples) else None
  if med is None:assert row['rgb_median'] is None
  else:close(med,row['rgb_median'],'direct_RGB_median',0.)
  assert row['color_status']==('MEASURED_sparse_visible_support' if enough else 'UNKNOWN_insufficient_visible_pixels')
  scalarclose(row['color_evidence_reliability'],len(pixels)/(len(pixels)+8) if enough else 0.,'color_reliability')
  for name in PROTOTYPES:
   scalarclose(row['color_soft_support'][name],expected[name] if enough else None,'instance_RGB_'+name)
   corrected=None
   if enough and nscene>=8:
    p=(len(pixels)*expected[name]+.5)/(len(pixels)+1.)
    b=(nscene*scene[name]+.5)/(nscene+1.)
    corrected=float(np.clip(np.log(p/(1-p))-np.log(b/(1-b)),-4.,4.))
   scalarclose(row['color_logodds_relative_scene_visible'][name],corrected,'scene_relative_color_'+name)
  geometry=row['geometry'];assert geometry['complete_object_size'] is geometry['physical_object_center'] is None
  assert geometry['span_coordinate_frame']=='raw_pointcloud_not_complete_model_box'
  if len(ids)>=8:
   points=raw[ids].astype(np.float64)
   assert geometry['status']=='MEASURED_visible_surface_only'
   close(np.ptp(points,axis=0),geometry['visible_surface_span_xyz_m'],'visible_surface_span')
   close(np.linalg.eigvalsh(np.cov(points.T))[::-1],geometry['visible_surface_pca_eigenvalues'],'visible_surface_eigenvalues')
   close(np.quantile(a['visible_depth_camera_m'][a['visible_point_owner_instance_ids']==i],[.1,.5,.9]),geometry['visible_depth_camera_q10_q50_q90_m'],'visible_depth_quantiles')
  else:
   assert geometry['status']=='UNKNOWN_insufficient_visible_points'
   assert geometry['visible_surface_span_xyz_m'] is geometry['visible_surface_pca_eigenvalues'] is geometry['visible_depth_camera_q10_q50_q90_m'] is None
  assert row['semantic_category'] is row['complex_appearance'] is None
  assert row['absence_of_attribute_evidence_is_not_negative'] is True
  eligible+=enough;unknown+=not enough
 return {'instances_color_known':eligible,'instances_color_UNKNOWN_below_8':unknown,
  'colors_reconstructed_from_original_sparse_RGB':True,'visible_surface_not_complete_box':True}


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',required=True);parser.add_argument('--authorization',required=True)
 args=parser.parse_args();pp=path(args.plan);plan=read(pp);grant=read(path(args.authorization))
 assert pp.parent==ROOT/'research/round51/experiment'
 assert grant['state']=='ROOT_AUTHORIZED_R51_INDEPENDENT_RAW2_AUDIT_NO_GT' and grant['plan_sha256']==sha(pp) and grant['program_sha256']==sha(__file__)
 assert plan['state']=='ROOT_LOCKED_R51_INDEPENDENT_RAW2_AUDIT' and plan['n']==2 and plan['n_per_platform']==1
 assert plan['manifest']==MANIFEST and plan['no_GT_external_models_or_method_changes'] is True
 assert plan['selection_rule']=='HASH_MIN_REGISTERED_ID_PER_PLATFORM'
 expected={MANIFEST,plan['prepared_root']+'/SEALED.json'};assert set(plan['inputs_sha256'])==expected
 for group in ('inputs_sha256','programs_sha256'):
  for rel,digest in plan[group].items():assert sha(path(rel))==digest,rel
 assert {'crossplatform/round51/audit_raw51.py','crossplatform/round7/camera_projection.py'}<=set(plan['programs_sha256'])
 prep=path(plan['prepared_root']);assert prep.parent==ROOT/'crossplatform/round51'
 seal=read(prep/'SEALED.json');assert seal['state']=='R51_RAW_SAFE_INSTANCE_ATTRIBUTE_PREPARATION_COMPLETE_NO_GT' and seal['n']==336
 assert seal['external_vision_outputs_or_mask_derived_cache_read'] is False
 rows=read(ROOT/MANIFEST);assert len(rows)==336 and len({r['id'] for r in rows})==336
 chosen=[min([r for r in rows if r['platform']==platform],key=lambda r:hashlib.sha256((plan['selection_salt']+'|'+r['id']).encode()).hexdigest()) for platform in ('drone','quad')]
 out=path(plan['output_root']);assert out.parent==ROOT/'crossplatform/round51' and not out.exists();out.mkdir()
 dump(out/'SELECTED_BEFORE_RAW_READ_PRIVATE.json',{'rows':[{'id_PRIVATE':r['id'],'platform':r['platform']} for r in chosen],'selection_uses_GT':False})
 bindings=[];results=[];start=time.perf_counter()
 def bind(p,digest,role):
  actual=sha(p);bindings.append({'path_PRIVATE':str(p),'sha256':actual,'role':role});dump(out/'INPUT_LEDGER_PRIVATE.json',{'files':bindings})
  assert actual==digest,role
  return p
 def sealed(rel,role):return bind(prep/rel,seal['files'][rel],role)
 try:
  for row in chosen:
   sid=row['id'];pc=read(sealed('cases_PRIVATE/'+sid+'.json','PREPARED_CASE'))
   assert pc['config']==CONFIG and pc['platform']==row['platform']
   ap=sealed('arrays_PRIVATE/'+sid+'.npz','PREPARED_ARRAYS');atp=sealed('attributes_PRIVATE/'+sid+'.json','PREPARED_ATTRIBUTES')
   assert sha(ap)==pc['arrays_sha256'] and sha(atp)==pc['attributes_sha256']
   with np.load(ap,allow_pickle=False) as z:a={k:z[k] for k in z.files}
   attrs=read(atp)
   ip=bind(Path(row['imagepath']),pc['raw_bindings']['image_sha256'],'ORIGINAL_RGB')
   xp=bind(Path(row['pointpath']),pc['raw_bindings']['point_sha256'],'ORIGINAL_XYZ')
   mp=bind(Path(row['metapath']),pc['raw_bindings']['metadata_sha256'],'ORIGINAL_CAMERA_WHITELIST_ONLY')
   with Image.open(ip) as image:rgb=np.asarray(image.convert('RGB')).copy()
   raw=np.load(xp,allow_pickle=False) if xp.suffix=='.npy' else np.fromfile(xp,np.float32).reshape(-1,4)
   assert raw.ndim==2 and raw.shape[1]>=3;raw=np.ascontiguousarray(raw[:,:3],np.float32)
   assert array_sha(raw)==pc['raw_points_sha256'] and array_sha(a['source_boxes'])==pc['source_boxes_sha256']
   assert [rgb.shape[1],rgb.shape[0]]==pc['image_size_wh']
   camera=CameraSpec.from_camera_fields(row['platform'],read_camera_allowlist(mp),waymo_view_suffix=0,missing_distortion_policy='assume_zero_declared')
   rebuilt,diagnostics=independent_geometry(raw,a['source_boxes'],camera,tuple(pc['image_size_wh']))
   assert diagnostics['raw_point_count']==pc['n_raw_points'] and diagnostics['unique_point_count']==pc['n_unique_raw_points']
   assert diagnostics['canonical_hypotheses']==pc['n_instances'] and diagnostics['visible_pixels']==pc['n_visible_points']
   assert attrs['external_vision_model_or_mask_consumed'] is False and attrs['point_support_pixels_only'] is True
   for key,value in rebuilt.items():
    if value.dtype.kind=='f':close(value,a[key],key)
    else:equal(value,a[key],key)
   color=audit_colors(rgb,raw,rebuilt,attrs)
   results.append({'id_PRIVATE':sid,'platform':row['platform'],'state':'PASS','arrays_verified':list(rebuilt),**diagnostics,**color})
   dump(out/'PROGRESS_PRIVATE.json',{'completed':len(results),'results':results})
  result={'state':'PASS_R51_INDEPENDENT_RAW_POINT_AND_SPARSE_RGB_AUDIT','n':2,'results':results,
   'GT_read':False,'external_model_cache_read':False,'model_forwards':0,'optimizer_steps':0,
   'independent_scope':'RAW_ID_SUPPORT_COMPONENTS_CARRIERS_OWNERSHIP_ZBUFFER_SPARSE_RGB_DESCRIPTORS',
   'shared_projection_dependency':'R7_CAMERA_WHITELIST_AND_PROJECTION; NOT_INDEPENDENT_CALIBRATION_GROUND_TRUTH',
   'semantic_identity_or_complete_object_mask_not_validated':True,'seconds':time.perf_counter()-start}
  dump(out/'RESULT_PRIVATE.json',result)
  dump(out/'SEALED.json',{'state':'R51_INDEPENDENT_RAW2_AUDIT_COMPLETE_NO_GT','n':2,'plan_sha256':sha(pp),
   'files':{p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
 except BaseException as error:
  dump(out/'FAILED.json',{'error_type':type(error).__name__,'error':str(error),'completed':len(results),'automatic_retry':False,'GT_read':False});raise


if __name__=='__main__':main()
