"""Build frozen pair contradictions from actual RGB/XYZ/camera and literal roles."""
import argparse,sys,time,itertools,re
from pathlib import Path
import numpy as np
from PIL import Image
from common55 import *
sys.path.insert(0,str(ROOT/'crossplatform/round7'));from camera_projection import CameraSpec,read_camera_allowlist,project_model_points,raw_to_model_points
sys.path.insert(0,str(ROOT/'research/round53'));from support53 import pixel_descriptors,entity_masks

def rawcloud(path):
 p=Path(path);a=np.load(p,allow_pickle=False) if p.suffix=='.npy' else np.fromfile(p,np.float32).reshape(-1,4)
 return np.ascontiguousarray(a[:,:3],np.float32)

def measure(boxes,xyz,uv,rgb,depth,camera,wh,sid):
 n=len(xyz);W,H=wh;desc=pixel_descriptors(rgb);perm=rng('R55_RGB_visible_permutation:'+sid).permutation(n);dsh=desc[perm]
 corner_sign=np.array(list(itertools.product([-1.,1.],repeat=3)))
 corners=boxes[:,:3,None] if False else boxes[:,None,:3]+corner_sign[None]*boxes[:,None,3:]/2
 proj=project_model_points(corners.reshape(-1,3),camera,wh);cu=proj['uv'].reshape(256,8,2);cd=proj['depth'].reshape(256,8)
 good=np.isfinite(cu).all((1,2))&(cd>.1).all(1)&(boxes[:,3:]>0).all(1)
 rect=np.nan_to_num(np.c_[cu[:,:,0].min(1),cu[:,:,1].min(1),cu[:,:,0].max(1),cu[:,:,1].max(1)])
 good&=(rect[:,0]>=0)&(rect[:,1]>=0)&(rect[:,2]<W)&(rect[:,3]<H)
 center=(rect[:,:2]+rect[:,2:])/2;size=np.maximum(rect[:,2:]-rect[:,:2],1.)
 counts=np.zeros((3,256),np.int64);means=np.zeros((3,256,13));shmeans=means.copy();cover=np.zeros((3,256));interval=np.zeros((256,2));valid=np.zeros(256,bool);supports=[]
 for q,box in enumerate(boxes):
  diff=np.abs(xyz.astype(float)-box[:3]);base=None
  for v,scale in enumerate([.85,1.,1.15]):
   take=(diff<=box[3:]*scale/2).all(1) if good[q] else np.zeros(n,bool);ix=np.flatnonzero(take);counts[v,q]=len(ix)
   if len(ix):
    means[v,q]=desc[ix].mean(0);shmeans[v,q]=dsh[ix].mean(0)
    coverage=np.ptp(uv[ix],axis=0)/size[q];cover[v,q]=float(np.minimum(coverage[0],coverage[1]))
   if v==1:base=ix
  supports.append(base)
  if len(base)>=8 and good[q]:
   low,high=np.quantile(uv[base,0],[.1,.9]);mid=center[q,0]
   interval[q]=[min(low,mid)-.05*size[q,0],max(high,mid)+.05*size[q,0]];interval[q]/=W;valid[q]=True
 # Shared visible raw-point support, not box IoU, blocks same-surface pair contradictions.
 incidence=np.zeros((256,n),np.float32)
 for q,s in enumerate(supports):incidence[q,s]=1
 shared=incidence@incidence.T;den=np.minimum(counts[1,:,None],counts[1,None,:]);overlap=np.divide(shared,den,out=np.ones_like(shared),where=den>0)
 return {'color_means':means,'shuffled_color_means':shmeans,'counts':counts,'cover':cover,'intervals':interval,'geometry_valid':valid,'support_overlap':overlap,'visible_RGB_permutation':perm,'projection_rects':rect,'visible_rgb':rgb,'visible_xyz':xyz,'visible_uv':uv}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p)
 out=ROOT/p['output_root'];assert not out.exists();out.mkdir(parents=True);rows=read(ROOT/p['manifest']);ls=read(ROOT/p['latent_root']/'SEALED.json');ps=read(ROOT/p['prepared_root']/'SEALED.json');gs=read(ROOT/p['graph_root']/'SEALED.json');stats=[];cache={};start=time.perf_counter()
 for row in rows:
  sid=row['id'];lp=ROOT/p['latent_root']/('samples_PRIVATE/'+sid+'.npz');assert sha(lp)==ls['files']['samples_PRIVATE/'+sid+'.npz'];z=load(lp,['boxes','scores','tokens','token_mask'])
  gp=ROOT/p['graph_root']/('attributes_PRIVATE/'+sid+'.json');assert sha(gp)==gs['files']['attributes_PRIVATE/'+sid+'.json'];attrs=read(gp);graph=attrs['text_graph_unchanged'];case=read(ROOT/p['latent_root']/('cases_PRIVATE/'+sid+'.json'))
  masks,maskstatus=entity_masks(graph,row['caption'],np.asarray(case['token_offsets_PRIVATE']));names=['T']+[x['entity'] for x in graph['references'] if masks[x['entity']].any()]
  # Source readout is preserved for T. Reference spans use literal full-sentence token offsets.
  role_masks=np.stack([z['token_mask']]+[masks[e] for e in names[1:]])
  reqdict={v['entity']:v['recognized_color_words'] for v in attrs['entity_attribute_requests']};requests=[reqdict.get(e,[]) for e in names]
  phrases={'T':graph['base_text']['subject'].get('text',''),**{e['entity']:e['phrase'] for e in graph['references']}}
  for i,e in enumerate(names):
   if re.search(r'\b(?:not|no|without|except)\b',phrases[e],re.I):requests[i]=[]
  cp=ROOT/p['prepared_root']/('cases_PRIVATE/'+sid+'.json');assert sha(cp)==ps['files']['cases_PRIVATE/'+sid+'.json'];pc=read(cp);binding=pc['raw_bindings'];key=row['imagepath']
  if key not in cache:
   assert sha(row['imagepath'])==binding['image_sha256'] and sha(row['pointpath'])==binding['point_sha256'] and sha(row['metapath'])==binding['metadata_sha256']
   raw=rawcloud(row['pointpath']);camera=CameraSpec.from_camera_fields(row['platform'],read_camera_allowlist(row['metapath']),waymo_view_suffix=0,missing_distortion_policy='assume_zero_declared')
   with Image.open(row['imagepath']) as im:image=np.array(im.convert('RGB'))
   cache={key:(raw,camera,image)}
  raw,camera,image=cache[key];ar=ROOT/p['prepared_root']/('arrays_PRIVATE/'+sid+'.npz');assert sha(ar)==ps['files']['arrays_PRIVATE/'+sid+'.npz'];vis=load(ar,['visible_raw_point_indices','visible_uv','visible_pixels_xy','visible_depth_camera_m']);ids=vis['visible_raw_point_indices'];pix=vis['visible_pixels_xy'];xyz=raw_to_model_points(raw[ids],row['platform']);rgb=image[pix[:,1],pix[:,0]]
  m=measure(z['boxes'].astype(float),xyz,vis['visible_uv'],rgb,vis['visible_depth_camera_m'],camera,(image.shape[1],image.shape[0]),sid)
  c=colors_conflict(m['color_means'],m['counts'],m['cover'],requests);cs=colors_conflict(m['shuffled_color_means'],m['counts'],m['cover'],requests);g=relation_conflict(m['intervals'],m['geometry_valid'],m['support_overlap'],graph['reference_relations'],names)
  ep=rng('R55_reference_endpoint_permutation:'+sid).permutation(256);conf=np.stack([joint_conflicts(c,g),joint_conflicts(cs,g),joint_conflicts(c,g[:,:,ep])])
  target_mask_ok=np.array_equal(role_masks[0],z['token_mask']);assert target_mask_ok
  np.savez_compressed(out/'dummy.npz',a=np.array([1])) if False else None
  (out/'samples_PRIVATE').mkdir(exist_ok=True);np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),conflicts=conf,role_masks=role_masks,entity_names=np.asarray(names),color_conflict=c,shuffled_color_conflict=cs,geometry_conflict=g,endpoint_permutation=ep,**m)
  record={'id_PRIVATE':sid,'entity_names_PRIVATE':names,'requests_PRIVATE':requests,'edges_PRIVATE':graph['reference_relations'],'maskstatus_PRIVATE':maskstatus,'RGB_visibility_source':'sealed R51 all-cloud z-buffer; no R51 grouping/owner read','source_boxes_unchanged':True,'coordinate_frame':'camera_image_only; R->T endpoint direction explicit','n_roles':len(names),'nonzero_pairs_by_variant':[int((v>0).sum()) for v in conf],'geometry_nonzero_pairs':int((g>0).sum()),'color_conflict_queries':int((c>0).sum()),'geometry_known_queries':int(m['geometry_valid'].sum()),'color_stable_support_queries':int(((m['counts']>=16).all(0)&(m['cover']>=.35).all(0)).sum()),'visible_points':len(ids),'referent_missing_does_not_create_conflict':True,'UNKNOWN_not_trainable':True}
  dump(out/'cases_PRIVATE'/(sid+'.json'),record);stats.append({k:v for k,v in record.items() if not k.endswith('_PRIVATE')})
  if len(stats)%32==0:print('PREPARE55',len(stats),336,flush=True)
 dump(out/'COVERAGE.json',stats);dump(out/'COSTS.json',{'n':336,'seconds':time.perf_counter()-start,'new_models':0,'GT_reads':0,'raw_media_hashes_verified':True})
 seal(out,'R55_OBSERVATION_CONFLICTS_FROZEN_NO_GT',n=336,plan_sha256=sha(a.plan));print('SEALED55',len(stats),flush=True)
if __name__=='__main__':main()
