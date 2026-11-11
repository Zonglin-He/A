"""Same fixed-region evidence, no model-weight update and no C input."""
from common74 import *
from scipy.special import softmax,logsumexp
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def direct_region(z,e):
 qa=z['A']['query'].astype(float).copy();qb=z['B']['query'].astype(float).copy();eligible=e['RGB_negatives'].any(1);PA=e['A_pool'];PB=e['B_pool'];ma=PA@qa;mb=PB@qb;changed=0
 for q,P,delta in [(qa,PA,(mb-ma)/2),(qb,PB,(ma-mb)/2)]:
  memberships=(P>0)&eligible[:,None];unique=memberships.sum(0)==1;cols=np.flatnonzero(unique)
  if len(cols):
   owner=memberships[:,cols].argmax(0);q[cols]=unit(q[cols]+delta[owner]);changed+=len(cols)
 pp={}
 for v,q in [('A',qa),('B',qb)]:
  mat=q@z[v]['tokens'].astype(float).T/.07;mask=z[v]['token_mask'];pp[v]=softmax(logsumexp(mat[:,mask],1)-logsumexp(mat[:,~mask],1))
 W=e['B_W'];b=W@pp['B'];mass=float(e['B_fixed_mass']);b=mass*b/max(b.sum(),1e-300)+(1-mass)*e['base'];return (pp['A']+b)/2,{'unique_region_query_updates':changed,'region_rows_eligible':int(eligible.sum()),'gradient_or_model_updates':0}
