"""Fixed spatial ROI supports; no object/grounding pseudo-labels or expert outputs."""
from common74 import *
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0,str(ROOT/'research/round66'));from groups66 import support,overlap
def geometry_labels(s):
 x=s['xyz'];n=len(x)
 if n<12:return np.tile(np.arange(n),(2,1))
 dist,kn=cKDTree(x).query(x,k=12,workers=1);local=x[kn]-x[kn].mean(1,keepdims=True);eig,vec=np.linalg.eigh(np.einsum('nki,nkj->nij',local,local)/12);norm=vec[:,:,0];stable=(eig[:,1]-eig[:,0])/np.maximum(eig[:,2],1e-9)>.15;src=np.repeat(np.arange(n),11);dst=kn[:,1:].reshape(-1);spacing=np.maximum(dist[:,1],.02);limit=np.clip(3*np.maximum(spacing[src],spacing[dst]),.06,.8);ok=(dist[:,1:].reshape(-1)<=limit)&(np.linalg.norm(s['uv'][src]-s['uv'][dst],axis=1)<=32)&(np.abs(s['depth'][src]-s['depth'][dst])<.15+.025*np.minimum(s['depth'][src],s['depth'][dst]))&stable[src]&stable[dst];dot=np.abs((norm[src]*norm[dst]).sum(1));labels=[]
 for cos in [np.cos(np.pi/6),np.cos(np.pi/4)]:
  good=ok&(dot>=cos);mat=coo_matrix((np.ones(int(good.sum())),(src[good],dst[good])),shape=(n,n)).tocsr();_,lab=connected_components(mat,directed=False);labels.append(lab)
 return np.array(labels)
def region_pool(z,boxes):
 mask=support(z['query_anchor_xyz'].astype(float),boxes);counts=mask.sum(1);pool=mask.astype(float)/np.maximum(counts[:,None],1);return pool,counts
def sampled_support(raw,ids,boxes):
 unique=np.unique(ids);mask=support(raw[unique],boxes);return [unique[m] for m in mask]
def select_regions(z,raw,group):
 reps=np.array([np.flatnonzero(group==g)[0] for g in np.unique(group)]);boxes=z['A']['boxes'][reps].astype(float);pa,na=region_pool(z['A'],boxes);pb,nb=region_pool(z['B'],boxes);sa=sampled_support(raw,z['A']['point_indices'],boxes);sb=sampled_support(raw,z['B']['point_indices'],boxes);common=np.array([len(np.intersect1d(a,b)) for a,b in zip(sa,sb)]);valid=(na>0)&(nb>0)&(common>=4)&np.isfinite(boxes).all(1)&(boxes[:,3:]>0).all(1);ix=np.flatnonzero(valid)
 # Geometry-only farthest-region coverage, independent of language/CLIP scores.
 chosen=[]
 if len(ix):
  first=sorted(ix,key=lambda k:(-common[k],int(reps[k])))[0];chosen=[int(first)];left=set(map(int,ix));left.remove(int(first))
  while left and len(chosen)<16:
   k=max(sorted(left),key=lambda k:(float(np.linalg.norm(boxes[k,:3]-boxes[chosen,:3],axis=1).min()),int(common[k]),-int(reps[k])));chosen.append(k);left.remove(k)
 q=np.array(chosen,int);return boxes[q],reps[q],pa[q],pb[q],common[q]
def negative_masks(raw,boxes,s):
 n=len(boxes);rgb=np.zeros((n,n),bool);geom=np.zeros_like(rgb)
 if not n:return rgb,geom,np.zeros(n,int)
 bags=support(raw,boxes);inter=(bags.astype(float)@bags.astype(float).T);sizes=bags.sum(1);contain=np.divide(inter,np.maximum(np.minimum(sizes[:,None],sizes[None]),1));eligible=(overlap(boxes,boxes)<.05)&(contain<=.05)&~np.eye(n,dtype=bool);visible=support(s['xyz'],boxes);counts=visible.sum(1);eligible&=(counts[:,None]>=4)&(counts[None,:]>=4)
 # Removing RGB changes only the line-boundary edge condition. Both masks keep
 # the same calibrated visibility, normals, depth, supports and numeric gates.
 for labels,dest in [(s['labels'],rgb),(geometry_labels(s),geom)]:
  dest[:]=eligible
  for lab in labels:
   hist=np.array([np.bincount(lab[m],minlength=int(lab.max()+1)) for m in visible]) if len(lab) else np.zeros((n,0))
   for i in range(n):
    for j in range(i):
     shared=np.minimum(hist[i],hist[j]).sum()/max(min(counts[i],counts[j]),1)
     if shared>.05:dest[i,j]=dest[j,i]=False
 return rgb,geom,counts
def derangement(n,salt):
 if n<2:return np.arange(n)
 order=perm(n,salt);p=np.empty(n,int);p[order]=np.roll(order,1);assert (p!=np.arange(n)).all();return p
