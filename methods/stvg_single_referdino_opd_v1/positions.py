"""Original physical observation rule, with no detector or detector cache."""
import numpy as np

def observation_positions(frame_ids,extent):
    ids=np.asarray(frame_ids,dtype=float);a,b=map(int,extent)
    assert len(ids)>=8 and np.all(np.diff(ids)>0)and 0<=a<=b<len(ids)
    k=min(4,b-a+1);edges=np.linspace(ids[a],ids[b]+1,k+1);out=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        pool=[i for i in range(a,b+1)if lo<=ids[i]<hi]
        if pool:out.append(min(pool,key=lambda i:(abs(ids[i]-(lo+hi)/2),i)))
    while len(out)<k:
        rest=[i for i in range(a,b+1)if i not in out]
        out.append(max(rest,key=lambda i:(min(abs(ids[i]-ids[j])for j in out),-i)))
    s=set(out)
    while len(s)<8:
        s.add(max((i for i in range(len(ids))if i not in s),key=lambda i:(min(abs(ids[i]-ids[j])for j in s),-ids[i])))
    return sorted(s)
