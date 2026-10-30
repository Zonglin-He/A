"""GT-free cohort selection by coordinate-only parent grouping and image reuse components.
NPY containers include annotation columns. We map that container, but copy only [:,:3]
into geometry hashes/caches; annotation columns never enter grouping, selection or inference.
"""
import pathlib,json,hashlib,collections,itertools,time
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
ROOT=pathlib.Path(__file__).resolve().parent;BASE=ROOT.parent;DATA=BASE/'datasets/PIADv2_Unseen_aff';SPLIT=DATA/'Unseen_aff'
SEED='GREAT_ROUND2_PHYSICAL_GROUPS_V1';NEAR_RMS=.01;NEAR_Q95=.025
class UF:
 def __init__(self,n):self.p=list(range(n))
 def find(self,x):
  while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
  return x
 def union(self,a,b):a,b=self.find(a),self.find(b);self.p[max(a,b)]=min(a,b)
def sha(b):return hashlib.sha256(b).hexdigest()
def path(s):return DATA/('Unseen_aff/'+s.replace('\\','/').split('Unseen_aff/',1)[1])
def geometry(p):
 a=np.load(p,mmap_mode='r',allow_pickle=False)
 assert a.ndim==2 and a.shape[0]==2048 and a.shape[1]>=4,(p,a.shape)
 x=np.array(a[:,:3],dtype=np.float64,copy=True);assert np.isfinite(x).all()
 n=x-x.mean(0);n/=np.linalg.norm(n,axis=1).max()
 def key(y,digits):
  q=np.rint(y*10**digits).astype('<i8');q=q[np.lexsort((q[:,2],q[:,1],q[:,0]))];return sha(q.tobytes())
 exact=key(n,5);eig,vec=np.linalg.eigh(n.T@n/len(n));eig=eig[::-1];projected=n@vec[:,::-1]
 versions=[]
 for signs in itertools.product([-1,1],repeat=3):
  z=projected*np.array(signs);q=np.rint(z*1e4).astype('<i8');q=q[np.lexsort((q[:,2],q[:,1],q[:,0]))];versions.append((q.tobytes(),z))
 b,z=min(versions,key=lambda pair:pair[0]);return {'xyz':x,'normalized':n,'pca':z,'eigenvalues':eig,'exact_hash':exact,'pca_hash':sha(b),'raw_dtype':str(a.dtype),'container_columns':a.shape[1],'pca_axis_near_degenerate':bool(np.min(np.abs(np.diff(eig)))<1e-3)}
def main():
 out=ROOT/'cohorts';out.mkdir(exist_ok=True);assert not (out/'COHORT_LOCK.json').exists(),'Never overwrite an existing cohort lock'
 read=lambda n:(SPLIT/n).read_text().splitlines();images,points,human,obj=[read(n) for n in ['Img_test.txt','Point_test.txt','hk_test.txt','ok_test.txt']];assert len(images)==len(points)==len(human)==len(obj)==2013
 unique=sorted(set(points));point_index={s:i for i,s in enumerate(unique)};g=[]
 for i,s in enumerate(unique):
  z=geometry(path(s));z.update(path=s,object_class=path(s).parts[-4]);g.append(z)
  if i%250==0:print('geometry',i,'/',len(unique),flush=True)
 uf=UF(len(g));seen={}
 for i,z in enumerate(g):
  for kind in ['exact_hash','pca_hash']:
   k=(kind,z[kind]);uf.union(i,seen[k]) if k in seen else seen.setdefault(k,i)
 # Conservative near-shape grouping, not proof of shared physical instance.
 # Candidate descriptor neighbors are bounded to 3 per member within object category.
 near=[];pairs=set();by_obj=collections.defaultdict(list)
 for i,z in enumerate(g):by_obj[z['object_class']].append(i)
 for cls,ids in by_obj.items():
  feats=np.stack([g[i]['eigenvalues'] for i in ids]);tree=cKDTree(feats);_,nn=tree.query(feats,k=min(4,len(ids)))
  if nn.ndim==1:nn=nn[:,None]
  for local,row in enumerate(nn):
   for other in row:
    i,j=ids[local],ids[int(other)]
    if i>=j or uf.find(i)==uf.find(j):continue
    if np.linalg.norm(g[i]['eigenvalues']-g[j]['eigenvalues'])>.01:continue
    pairs.add((i,j))
 trees={}
 for pairn,(i,j) in enumerate(sorted(pairs)):
  a,b=g[i]['pca'],g[j]['pca'];ta=trees.setdefault(i,cKDTree(a));best=(float('inf'),float('inf'))
  for signs in itertools.product([-1,1],repeat=3):
   bb=b*np.array(signs);d1=ta.query(bb,k=1)[0];d2=cKDTree(bb).query(a,k=1)[0];d=np.r_[d1,d2];score=(float(np.sqrt((d*d).mean())),float(np.quantile(d,.95)))
   if score<best:best=score
  if best[0]<=NEAR_RMS and best[1]<=NEAR_Q95:uf.union(i,j);near.append({'left':i,'right':j,'rms':best[0],'q95':best[1]})
  if pairn%200==0:print('near comparisons',pairn,'/',len(pairs),flush=True)
 groups=collections.defaultdict(list)
 for i in range(len(g)):groups[uf.find(i)].append(i)
 parent_id={i:sha('|'.join(sorted(g[j]['exact_hash'] for j in groups[uf.find(i)])).encode()) for i in range(len(g))}
 # Image reuse is grouped across rows; perceptual dHash <=2 within class is conservative.
 image_meta=[];image_bytes={}
 for i,s in enumerate(images):
  p=path(s);raw=p.read_bytes();digest=sha(raw)
  if digest not in image_bytes:
   with Image.open(p) as im:y=np.asarray(im.convert('L').resize((9,8),Image.Resampling.BILINEAR))
   bits=(y[:,1:]>y[:,:-1]).reshape(-1);ph=sum(int(v)<<k for k,v in enumerate(bits));image_bytes[digest]=ph
  image_meta.append({'sha256':digest,'dhash':image_bytes[digest]})
 rowuf=UF(len(images));seen={}
 for i,s in enumerate(points):
  pid=parent_id[point_index[s]]
  for key in [('geometry',pid),('image',image_meta[i]['sha256'])]:rowuf.union(i,seen[key]) if key in seen else seen.setdefault(key,i)
 obj_rows=collections.defaultdict(list)
 for i,s in enumerate(points):obj_rows[path(s).parts[-4]].append(i)
 image_near=0
 for ids in obj_rows.values():
  for ai,i in enumerate(ids):
   for j in ids[ai+1:]:
    if (image_meta[i]['dhash']^image_meta[j]['dhash']).bit_count()<=2:rowuf.union(i,j);image_near+=1
 components=collections.defaultdict(list)
 for i in range(len(images)):components[rowuf.find(i)].append(i)
 compid={i:sha('|'.join(sorted(set(parent_id[point_index[points[j]]]+'|'+image_meta[j]['sha256'] for j in components[rowuf.find(i)]))).encode()) for i in range(len(images))}
 old=json.loads((BASE/'data/inference_manifest.json').read_text());old_keys=set();old_image=set()
 for row in old:
  z=geometry(pathlib.Path(row['geometry'])) if False else None
  x=np.load(row['geometry'],allow_pickle=False).astype(np.float64);n=x-x.mean(0);n/=np.linalg.norm(n,axis=1).max();q=np.rint(n*1e5).astype('<i8');q=q[np.lexsort((q[:,2],q[:,1],q[:,0]))];old_keys.add(sha(q.tobytes()));old_image.add(sha(pathlib.Path(row['image']).read_bytes()))
 excluded=set()
 for i,z in enumerate(g):
  if z['exact_hash'] in old_keys:
   for j,s in enumerate(points):
    if parent_id[point_index[s]]==parent_id[i]:excluded.add(compid[j])
 for i,m in enumerate(image_meta):
  if m['sha256'] in old_image:excluded.add(compid[i])
 eligible=[i for i in range(len(images)) if compid[i] not in excluded]
 affs=sorted({path(s).parts[-2] for s in images});assert len(affs)==6
 def choose(n,tag,blocked):
  quotas={a:n//len(affs) for a in affs}
  for a in sorted(affs,key=lambda a:sha((SEED+'|'+tag+'|'+a).encode()))[:n%len(affs)]:quotas[a]+=1
  selected=[];used=set(blocked);counts=collections.Counter()
  for i in sorted(eligible,key=lambda i:sha((SEED+'|'+tag+'|'+parent_id[point_index[points[i]]]+'|'+image_meta[i]['sha256']).encode())):
   a=path(images[i]).parts[-2]
   if compid[i] in used or counts[a]>=quotas[a]:continue
   selected.append(i);used.add(compid[i]);counts[a]+=1
  assert len(selected)==n,(tag,counts,quotas);return selected,used,dict(counts)
 confirm,used,conf_counts=choose(32,'confirm',set());dev,used,dev_counts=choose(16,'dev',used)
 assert not {compid[i] for i in confirm}&{compid[i] for i in dev};assert not {parent_id[point_index[points[i]]] for i in confirm}&{parent_id[point_index[points[i]]] for i in dev}
 # Confirm selection is fixed before any dev task scoring. Only XYZ is written for inference.
 xyzdir=out/'xyz';xyzdir.mkdir(exist_ok=True)
 lock={'schema_version':1,'seed_string':SEED,'selection_uses_ground_truth_masks':False,'annotation_container_access':'NPY file has >=4 columns including GT; mmap metadata/container opened, only first3 coordinates logically copied/hashed/exported. No annotation column used for grouping or selection.','geometry_grouping':{'normalized_sorted_quantization':1e-5,'pca_sign_canonical_quantization':1e-4,'near_shape_candidate_neighbors':3,'near_shape_rms_radius_units':NEAR_RMS,'near_shape_q95_radius_units':NEAR_Q95,'limitations':'Canonical/near shape groups are conservative geometry leakage controls, not verified physical-object IDs. Different point resamplings, axis degeneracy, same-category similar copies, and near-duplicate images can still escape grouping.'},'native_rows':len(images),'unique_point_paths':len(unique),'geometry_groups':len(groups),'image_geometry_components':len(components),'near_geometry_merged_pairs':len(near),'near_image_pairs':image_near,'excluded_old_components':len(excluded),'old_geometry_exact_keys':len(old_keys),'dev_n':len(dev),'confirm_n':len(confirm),'dev_affordance_counts':dev_counts,'confirm_affordance_counts':conf_counts,'no_gpu_or_task_scores_yet':True,'manifests':{}}
 for tag,selected in [('confirm',confirm),('dev',dev)]:
  inference=[];scoring=[];metadata=[]
  for i in selected:
   pointid=point_index[points[i]];z=g[pointid];ident=sha((SEED+'|'+str(i)+'|'+parent_id[pointid]).encode());geo=xyzdir/(ident+'.npy');np.save(geo,z['xyz'])
   inference.append({'id':ident,'parent_id':parent_id[pointid],'component_id':compid[i],'image':str(path(images[i])),'geometry':str(geo),'human':human[i],'object':obj[i],'seed':int(ident[:8],16)%(2**31),'native_index':i})
   scoring.append({'id':ident,'parent_id':parent_id[pointid],'native_annotation_container':str(path(points[i])),'target_columns':'3:','object_class':path(points[i]).parts[-4],'affordance':path(images[i]).parts[-2],'native_index':i})
   metadata.append({'id':ident,'parent_id':parent_id[pointid],'component_id':compid[i],'object_class':path(points[i]).parts[-4],'affordance':path(images[i]).parts[-2],'image_sha256':image_meta[i]['sha256'],'geometry_exact_hash':z['exact_hash'],'pca_axis_near_degenerate':z['pca_axis_near_degenerate'],'source_raw_coordinate_dtype':z['raw_dtype'],'container_columns':z['container_columns'],'human_phrase_count':len(human[i].split(','))})
  for typ,data in [('inference',inference),('scoring',scoring),('metadata',metadata)]:
   p=out/f'{tag}_{typ}.json';p.write_text(json.dumps(data,indent=2));lock['manifests'][p.name]=sha(p.read_bytes())
 (out/'geometry_group_audit_private.json').write_text(json.dumps({'point_rows':[{'path':z['path'],'object_class':z['object_class'],'parent_id':parent_id[i],'exact_hash':z['exact_hash'],'pca_hash':z['pca_hash'],'raw_dtype':z['raw_dtype'],'pca_axis_near_degenerate':z['pca_axis_near_degenerate']} for i,z in enumerate(g)],'near_pairs':near},indent=2))
 text=json.dumps(lock,indent=2);(out/'COHORT_LOCK.json').write_text(text);(out/'COHORT_LOCK.sha256').write_text(sha(text.encode())+'  COHORT_LOCK.json\n');print(json.dumps(lock,indent=2),flush=True)
if __name__=='__main__':main()
