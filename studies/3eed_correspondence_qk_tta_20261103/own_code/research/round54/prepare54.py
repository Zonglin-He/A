"""Raw RGB/seed observations and competing local hypotheses; no GT/models."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round53'))
from support53 import pixel_descriptors,entity_masks
COLORS=['red','orange','yellow','green','cyan','blue','purple','pink','brown','beige','white','gray','black']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,x):p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def geometry(boxes,xyz):
 valid=(boxes[:,3:]>0).all(1);half=np.maximum(boxes[:,3:].astype(float)/2,.05);centers=boxes[:,:3].astype(float)
 delta=np.maximum(np.abs(xyz[None].astype(float)-centers[:,None])-half[:,None],0)/half[:,None]
 cost=np.minimum(np.sum(delta*delta,axis=-1),4);cost[~valid]=4
 expanded=(np.abs(xyz[None].astype(float)-centers[:,None])<=1.5*half[:,None]).all(-1)&valid[:,None]
 allowed=expanded.copy();neighbors=[]
 for q in range(256):
  selected=[q]
  # Diverse nearby proposal boxes are hypotheses, never certified instances.
  order=np.argsort(np.sum((centers-centers[q])**2,axis=1),kind='stable')
  for other in order:
   if other==q or not valid[other]:continue
   admissible=True
   for old in selected:
    inter=np.maximum(np.minimum(centers[other]+half[other],centers[old]+half[old])-np.maximum(centers[other]-half[other],centers[old]-half[old]),0).prod();union=(2*half[other]).prod()+(2*half[old]).prod()-inter
    if inter/max(union,1e-12)>.5:admissible=False;break
   if admissible:selected.append(int(other))
   if len(selected)==4:break
  neighbors.append(selected);allowed[q]=expanded[selected].any(0) if valid[q] else False
 return allowed,cost,neighbors
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan)
 for rel,h in {**p['programs_sha256'],**p['inputs_sha256']}.items():assert sha(ROOT/rel)==h,rel
 rows=read(ROOT/p['manifest']);out=ROOT/p['output_root'];assert not out.exists();out.mkdir(parents=True)
 for d in ['samples_PRIVATE','cases_PRIVATE']:(out/d).mkdir()
 latent=ROOT/p['latent_root'];ls=read(latent/'SEALED.json');prepared=ROOT/p['prepared_root'];ps=read(prepared/'SEALED.json');group=ROOT/p['pure_graph_root'];gs=read(group/'SEALED.json');start=time.perf_counter();summary=[];media={}
 for row in rows:
  sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(latent/rel)==ls['files'][rel]
  with np.load(latent/rel,allow_pickle=False) as z:xyz=z['fp2_xyz'];rawids=z['seed_raw_indices'];boxes=z['boxes'];scores=z['scores'];queries=z['query'];tokens=z['tokens']
  assert np.isfinite(xyz).all();pr='arrays_PRIVATE/'+sid+'.npz';assert sha(prepared/pr)==ps['files'][pr]
  with np.load(prepared/pr,allow_pickle=False) as z:visible=z['visible_raw_point_indices'];pixels=z['visible_pixels_xy']
  cr='cases_PRIVATE/'+sid+'.json';assert sha(prepared/cr)==ps['files'][cr];case=read(prepared/cr)
  ar='attributes_PRIVATE/'+sid+'.json';assert sha(group/ar)==gs['files'][ar];attrs=read(group/ar);graph=attrs['text_graph_unchanged']
  if row['imagepath'] not in media:
   assert sha(row['imagepath'])==case['raw_bindings']['image_sha256']
   with Image.open(row['imagepath']) as im:media={row['imagepath']:np.array(im.convert('RGB'))}
  rgb=media[row['imagepath']];mapping={int(i):j for j,i in enumerate(visible)};assert len(mapping)==len(visible)
  locations=np.asarray([mapping.get(int(i),-1) for i in rawids]);known=locations>=0;seedrgb=np.zeros((1024,3),np.uint8);seedrgb[known]=rgb[pixels[locations[known],1],pixels[locations[known],0]]
  descriptors=np.zeros((1024,13),float);descriptors[known]=pixel_descriptors(seedrgb[known]);scene=np.array([attrs['scene_visible_color_soft_support'][c] for c in COLORS]);evidence=np.zeros_like(descriptors)
  if known.any():
   obs=np.clip(descriptors[known],1e-4,1-1e-4);base=np.clip(scene,1e-4,1-1e-4);evidence[known]=np.clip(np.log(obs/(1-obs))-np.log(base/(1-base)),-4,4)
  lc=read(latent/'cases_PRIVATE'/(sid+'.json'));masks,maskstatus=entity_masks(graph,row['caption'],np.asarray(lc['token_offsets_PRIVATE']));entities=list(masks);logits=queries.astype(float)@tokens.astype(float).T/.07;logits-=logits.max(1,keepdims=True);prob=np.exp(logits);prob/=prob.sum(1,keepdims=True)
  responsibility=np.stack([prob[:,masks[e]].sum(1) for e in entities],axis=1);responsibility/=np.maximum(responsibility.sum(1,keepdims=True),1)
  req={r['entity']:r['recognized_color_words'] for r in attrs['entity_attribute_requests']};entity_cost=np.zeros((len(entities),1024))
  for i,e in enumerate(entities):
   cols=[COLORS.index(c) for c in req.get(e,[]) if c in COLORS]
   if cols:entity_cost[i]=-evidence[:,cols].mean(1)
  colorcost=responsibility@entity_cost
  allow,geocost,neighbors=geometry(boxes,xyz)
  distances,nn=cKDTree(xyz.astype(float)).query(xyz.astype(float),k=9,workers=1);edges=np.unique(np.sort(np.stack([np.repeat(np.arange(1024),8),nn[:,1:].ravel()],axis=1),axis=1),axis=0);edges=edges[edges[:,0]!=edges[:,1]];length=np.linalg.norm(xyz[edges[:,0]].astype(float)-xyz[edges[:,1]],axis=1);scale=float(np.median(length[length>1e-8])) if (length>1e-8).any() else 1.;space=np.exp(-.5*(length/max(scale,1e-3))**2)
  from attributes51 import rgb_to_lab
  lab=rgb_to_lab(seedrgb);pairknown=known[edges[:,0]]&known[edges[:,1]];colordiff=np.linalg.norm(lab[edges[:,0]]-lab[edges[:,1]],axis=1);positive=colordiff[pairknown&(colordiff>1e-8)];cscale=float(np.median(positive)) if len(positive) else 30.;affinity=space.copy();affinity[pairknown]*=np.exp(-.5*(colordiff[pairknown]/max(cscale,1e-3))**2)
  rng=np.random.default_rng(int.from_bytes(hashlib.sha256(('R54_RGB_mapping_control_v1:'+sid).encode()).digest()[:8],'little'));perm=np.arange(1024);where=np.flatnonzero(known);perm[where]=rng.permutation(where);shuffleevidence=evidence[perm];shuffle_entity=np.zeros_like(entity_cost)
  for i,e in enumerate(entities):
   cols=[COLORS.index(c) for c in req.get(e,[]) if c in COLORS]
   if cols:shuffle_entity[i]=-shuffleevidence[:,cols].mean(1)
  shufflecost=responsibility@shuffle_entity;slab=lab[perm];sd=np.linalg.norm(slab[edges[:,0]]-slab[edges[:,1]],axis=1);shuffleaffinity=space.copy();shuffleaffinity[pairknown]*=np.exp(-.5*(sd[pairknown]/max(cscale,1e-3))**2)
  subsets=[]
  for v in ['A','B']:
   r=np.random.default_rng(int.from_bytes(hashlib.sha256(('R54_key_subsample_'+v+':'+sid).encode()).digest()[:8],'little'));subsets.append(np.sort(r.choice(1024,768,replace=False)))
  prior=np.maximum(scores.astype(float),1e-6);prior=prior/prior.sum()*.8
  np.savez_compressed(out/'samples_PRIVATE'/(sid+'.npz'),allowed=allow,geometry_cost=geocost,RGB_cost=colorcost,RGB_shuffle_cost=shufflecost,edge_pairs=edges,edge_affinity=affinity,edge_shuffle_affinity=shuffleaffinity,point_RGB_known=known,point_RGB=seedrgb,entity_responsibility=responsibility,seed_raw_indices=rawids,seed_xyz=xyz,source_boxes=boxes,source_scores=scores,query_prior=prior,view_A=subsets[0],view_B=subsets[1],RGB_permutation=perm)
  r={'id_PRIVATE':sid,'platform':row['platform'],'visible_seed_RGB':int(known.sum()),'known_RGB_graph_edges':int(pairknown.sum()),'total_edges':len(edges),'queries_without_local_seed':int((~allow.any(1)).sum()),'point_domain_coverage':int(allow.any(0).sum()),'target_attribute_available':bool(req.get('T')),'entities':len(entities),'entity_masks':maskstatus,'spatial_bandwidth':scale,'RGB_bandwidth':cscale,'neighbors_PRIVATE':neighbors,'key_views_shared_points':int(len(np.intersect1d(*subsets))),'true_rgb_point_identity_not_nearest_color_imputation':True};dump(out/'cases_PRIVATE'/(sid+'.json'),r);summary.append({k:v for k,v in r.items() if k not in ['id_PRIVATE','neighbors_PRIVATE','entity_masks']})
  if len(summary)%32==0:print('PREPARE54',len(summary),336,flush=True)
 dump(out/'COVERAGE_PRIVATE.json',summary);dump(out/'COSTS.json',{'n':len(summary),'seconds':time.perf_counter()-start,'new_models':0,'GT_reads':0});dump(out/'SEALED.json',{'state':'R54_RAW_OBSERVATIONS_LOCAL_SOFT_COMPETITION_NO_GT','n':len(summary),'plan_sha256':sha(a.plan),'files':{f.relative_to(out).as_posix():sha(f) for f in sorted(out.rglob('*')) if f.is_file()}});print(json.dumps({'state':'SEALED','n':len(summary),'seconds':time.perf_counter()-start}),flush=True)
if __name__=='__main__':main()
