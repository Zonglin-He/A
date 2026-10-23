"""Exact singleton truncated-mean search; branch bounds, same objective/ties.

No beam width, approximate stopping, or changed calibration. Worst case remains
exponential. A node budget raises an error and preserves failure, never returns
an approximate value. Only NEW variable-K research can use this module.
"""
import numpy as np

def estimate(points,scale,cutoff,max_nodes=2000000):
 points=np.asarray(points,np.float64);scale=np.asarray(scale,np.float64);n=len(points)
 assert points.shape==(n,4)and n>0 and scale.shape==(4,)and(scale>0).all()and cutoff>0
 best=None;nodes=0;leaves=0;pruned=0;indices=np.arange(n)
 def evaluate(active):
  nonlocal best,leaves
  delta=points[active].mean(0)if len(active)else np.zeros(4);geom=np.square((points-delta)/scale).sum(1);selected=tuple(0 if v<cutoff else -1 for v in geom);obj=float(sum(min(cutoff,float(v))for v in geom));key=(obj,selected)
  if best is None or key<best[0]:best=(key,delta.copy())
  leaves+=1
 # Incumbent generation affects speed only; exhaustive branch proof still used.
 evaluate(indices)
 for i in range(n):
  active=np.array([i])
  for _ in range(n+1):
   evaluate(active);delta=points[active].mean(0);new=indices[np.square((points-delta)/scale).sum(1)<cutoff]
   if np.array_equal(new,active):break
   active=new
 def visit(i,active,rejected):
  nonlocal nodes,pruned
  nodes+=1
  if nodes>max_nodes:raise RuntimeError('Exact singleton search node budget exceeded; no approximate output')
  if active:
   x=points[active];mean=x.mean(0);lower=float(np.square((x-mean)/scale).sum())+rejected*cutoff
  else:lower=rejected*cutoff
  if active and i<n and lower<=best[0][0]+1e-8:
   # Any competitive center lies within this ball around the selected mean.
   # Independent distance-to-ball minima give a VALID lower bound for all
   # undecided points; no subset/consensus approximation is introduced.
   radius=np.sqrt(max(0.,(best[0][0]-lower)/len(active)))
   distance=np.linalg.norm((points[i:]-mean)/scale,axis=1)
   lower+=float(np.minimum(np.maximum(distance-radius,0.)**2,cutoff).sum())
  # Conservative slack protects rounded SSE lower bound; final keys use exact original operations.
  if lower>best[0][0]+1e-8*(1+n+abs(lower)):pruned+=1;return
  if i==n:evaluate(np.asarray(active,dtype=int));return
  visit(i+1,active+[i],rejected);visit(i+1,active,rejected+1)
 visit(0,[],0)
 selected=best[0][1];active=[i for i,v in enumerate(selected)if v==0];assert active
 assert np.max(abs(best[1]-points[active].mean(0)))<1e-10
 return dict(delta=best[1],selected=list(selected),objective=best[0][0],nodes=nodes,leaves=leaves,pruned=pruned,exact_search_completed=True,worst_case_exponential=True)
