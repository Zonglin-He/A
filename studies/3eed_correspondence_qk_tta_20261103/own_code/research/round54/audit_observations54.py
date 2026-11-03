"""Independent all336 RGB identity, local-domain, graph and text-cost audit."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'research/round53'));sys.path.insert(0,str(ROOT/'crossplatform/round51'))
from support53 import pixel_descriptors,entity_masks
from attributes51 import rgb_to_lab,color_support,COLOR_RGB_PROTOTYPES
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def check(a,b,tol=1e-11):assert np.asarray(a).shape==np.asarray(b).shape and np.allclose(a,b,atol=tol,rtol=0)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan)
 for rel,h in {**p['programs_sha256'],**p['inputs_sha256']}.items():assert sha(ROOT/rel)==h
 rows=read(ROOT/p['manifest']);obsroot=ROOT/p['observation_root'];seal=read(obsroot/'SEALED.json');latent=ROOT/p['latent_root'];ls=read(latent/'SEALED.json');prep=ROOT/p['prepared_root'];ps=read(prep/'SEALED.json');graphroot=ROOT/p['pure_graph_root'];gs=read(graphroot/'SEALED.json');start=time.perf_counter();pixel_checks=0;all_masks=0;media={};counts=[]
 for row in rows:
  sid=row['id'];rel='samples_PRIVATE/'+sid+'.npz';assert sha(obsroot/rel)==seal['files'][rel] and sha(latent/rel)==ls['files'][rel]
  with np.load(obsroot/rel,allow_pickle=False) as z:o={k:z[k].copy() for k in z.files}
  with np.load(latent/rel,allow_pickle=False) as z:xyz=z['fp2_xyz'];ids=z['seed_raw_indices'];boxes=z['boxes'];scores=z['scores'];q=z['query'];t=z['tokens']
  assert np.array_equal(o['seed_xyz'],xyz) and np.array_equal(o['source_boxes'],boxes) and np.array_equal(o['source_scores'],scores) and np.array_equal(ids,o['seed_raw_indices'])
  pr='arrays_PRIVATE/'+sid+'.npz';assert sha(prep/pr)==ps['files'][pr]
  with np.load(prep/pr,allow_pickle=False) as z:vis=z['visible_raw_point_indices'];xy=z['visible_pixels_xy'];d={int(v):j for j,v in enumerate(vis)}
  pc=read(prep/'cases_PRIVATE'/(sid+'.json'))
  if row['imagepath'] not in media:
   assert sha(row['imagepath'])==pc['raw_bindings']['image_sha256']
   with Image.open(row['imagepath']) as im:media={row['imagepath']:np.array(im.convert('RGB'))}
  known=np.array([int(v) in d for v in ids]);assert np.array_equal(known,o['point_RGB_known']);rgb=np.zeros((1024,3),np.uint8)
  for k in np.flatnonzero(known):pix=xy[d[int(ids[k])]];rgb[k]=media[row['imagepath']][pix[1],pix[0]]
  assert np.array_equal(rgb,o['point_RGB']);pixel_checks+=int(known.sum())
  ar='attributes_PRIVATE/'+sid+'.json';assert sha(graphroot/ar)==gs['files'][ar];attrs=read(graphroot/ar);graph=attrs['text_graph_unchanged'];offsets=read(latent/'cases_PRIVATE'/(sid+'.json'))['token_offsets_PRIVATE'];masks,_=entity_masks(graph,row['caption'],np.asarray(offsets));entities=list(masks);logits=q.astype(float)@t.astype(float).T/.07;z=np.exp(logits-logits.max(axis=1,keepdims=True));z/=z.sum(1,keepdims=True);r=np.column_stack([z[:,masks[e]].sum(1) for e in entities]);r/=np.maximum(r.sum(1,keepdims=True),1);check(r,o['entity_responsibility'])
  names=list(COLOR_RGB_PROTOTYPES);features=pixel_descriptors(rgb[known]);scene=np.array([attrs['scene_visible_color_soft_support'][c] for c in names]);signal=np.zeros((1024,13));x=np.clip(features,1e-4,1-1e-4);b=np.clip(scene,1e-4,1-1e-4);signal[known]=np.clip(np.log(x)-np.log1p(-x)-np.log(b)+np.log1p(-b),-4,4)
  for j in np.flatnonzero(known)[:4]:
   single=color_support(rgb[j:j+1]);k=np.flatnonzero(np.flatnonzero(known)==j)[0];check(np.array([single[c] for c in names]),features[k])
  req={x['entity']:x['recognized_color_words'] for x in attrs['entity_attribute_requests']};unary=np.zeros((len(entities),1024))
  for i,e in enumerate(entities):
   cols=[names.index(c) for c in req.get(e,[]) if c in names]
   if cols:unary[i]=-signal[:,cols].mean(1)
  check(r@unary,o['RGB_cost'])
  half=np.maximum(boxes[:,3:].astype(float)/2,.05);center=boxes[:,:3].astype(float);valid=(boxes[:,3:]>0).all(1);outside=np.maximum(np.abs(xyz[None].astype(float)-center[:,None])-half[:,None],0)/half[:,None];cost=np.clip((outside**2).sum(2),0,4);cost[~valid]=4;check(cost,o['geometry_cost']);expanded=(np.abs(xyz[None].astype(float)-center[:,None])<=1.5*half[:,None]).all(2)&valid[:,None];domain=np.zeros_like(expanded)
  intersections=np.maximum(np.minimum((center+half)[:,None],(center+half)[None])-np.maximum((center-half)[:,None],(center-half)[None]),0).prod(2);vol=(2*half).prod(1);iou=intersections/np.maximum(vol[:,None]+vol[None]-intersections,1e-12)
  for qid in range(256):
   neighbors=[qid]
   for other in np.argsort(((center-center[qid])**2).sum(1),kind='stable'):
    if other==qid or not valid[other] or (iou[other,neighbors]>.5).any():continue
    neighbors.append(int(other))
    if len(neighbors)==4:break
   if valid[qid]:domain[qid]=expanded[neighbors].any(0)
  assert np.array_equal(domain,o['allowed']);all_masks+=256
  _,knn=cKDTree(xyz.astype(float)).query(xyz.astype(float),k=9,workers=1);edges=np.sort(np.column_stack([np.repeat(np.arange(1024),8),knn[:,1:].reshape(-1)]),axis=1);edges=np.unique(edges,axis=0);edges=edges[edges[:,0]!=edges[:,1]];assert np.array_equal(edges,o['edge_pairs']);length=np.sqrt(((xyz[edges[:,0]].astype(float)-xyz[edges[:,1]])**2).sum(1));scale=float(np.median(length[length>1e-8])) if (length>1e-8).any() else 1.;space=np.exp(-.5*(length/max(scale,1e-3))**2);lab=rgb_to_lab(rgb);pair=known[edges[:,0]]&known[edges[:,1]];diff=np.linalg.norm(lab[edges[:,0]]-lab[edges[:,1]],axis=1);positive=diff[pair&(diff>1e-8)];sig=float(np.median(positive)) if len(positive) else 30.;aff=space.copy();aff[pair]*=np.exp(-.5*(diff[pair]/max(sig,1e-3))**2);check(aff,o['edge_affinity'])
  perm=np.arange(1024);at=np.flatnonzero(known);rng=np.random.default_rng(int.from_bytes(hashlib.sha256(('R54_RGB_mapping_control_v1:'+sid).encode()).digest()[:8],'little'));perm[at]=rng.permutation(at);assert np.array_equal(perm,o['RGB_permutation']) and np.array_equal(known[perm],known)
  for i,e in enumerate(entities):
   cols=[names.index(c) for c in req.get(e,[]) if c in names]
   unary[i]=-signal[perm][:,cols].mean(1) if cols else 0.
  check(r@unary,o['RGB_shuffle_cost']);ll=lab[perm];diff=np.linalg.norm(ll[edges[:,0]]-ll[edges[:,1]],axis=1);aff=space.copy();aff[pair]*=np.exp(-.5*(diff[pair]/max(sig,1e-3))**2);check(aff,o['edge_shuffle_affinity'])
  prior=np.maximum(scores.astype(float),1e-6);check(prior/prior.sum()*.8,o['query_prior'])
  for key in ['view_A','view_B']:
   ix=o[key];assert len(ix)==768 and len(np.unique(ix))==768 and (np.diff(ix)>0).all();assert np.array_equal(ids[ix],o['seed_raw_indices'][ix])
  counts.append({'platform':row['platform'],'visible_RGB_seeds':int(known.sum()),'local_domain_seed_count':int(domain.any(0).sum()),'RGB_graph_pairs':int(pair.sum())})
 result={'state':'PASS_R54_INDEPENDENT_ALL_CASE_OBSERVATION_AND_DOMAIN_AUDIT','n':len(rows),'actual_visible_seed_RGB_samples_checked':pixel_checks,'candidate_local_domains_checked':all_masks,'no_R51_grouping_or_GT_used':True,'source_geometry_and_RGB_shuffle_invariants_checked':True,'raw_RGB_point_identity_checked':True,'library_color_prototypes_reused_and_single_pixel_checked':True,'observations_seal_sha256':sha(obsroot/'SEALED.json'),'seconds':time.perf_counter()-start,'new_model_or_GT_calls':0};out=ROOT/p['output_directory'];assert not out.exists();out.mkdir();dump(out/'AUDIT_RESULT.json',result);dump(out/'COVERAGE.json',counts);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
