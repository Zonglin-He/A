"""Source-only episode construction and exact physical-grid embedding."""
import numpy as np

def window(i,j,n=65):
 w=max(49,j-i+1);lo=max(0,min((i+j-w+1)//2,n-w));shift=0 if lo>=(n-w)/2 else n-w
 assert 0<=lo<=i<=j<lo+w<=n
 return lo,lo+w-1,shift

def slices(lo,hi,n=65):
 order=[lo%2,1-lo%2];out=[]
 for off in order:
  positions=[i for i in range(off,n,2)if lo<=i<=hi];out.append((off,positions,[(i-off)//2 for i in positions]))
 assert [out[i%2][1][i//2]for i in range(hi-lo+1)]==list(range(lo,hi+1))
 return out

def embed(q,lo,hi,n=65):
 assert q.shape==(hi-lo+1,)*2 and abs(q.sum()-1)<1e-10
 out=np.zeros((n,n),dtype=q.dtype);out[lo:hi+1,lo:hi+1]=q
 assert abs(out.sum()-q.sum())<1e-12
 return out
