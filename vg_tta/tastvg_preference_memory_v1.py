"""Nonparametric, orientation-invariant critic pair memory, fixed k=3."""
import hashlib
import numpy as np


def state_hash(memory):
    h=hashlib.sha256()
    for z in memory:
        h.update(np.asarray(z['direction'],dtype='<f8').tobytes());h.update(str((z['position'],z['i'],z['j'],z['label'])).encode())
    return h.hexdigest()


def append(memory,phi,teacher,position):
    phi=np.asarray(phi,dtype=np.float64);new=list(memory)
    for i in range(len(phi)):
        for j in range(i+1,len(phi)):
            gap=teacher[i]-teacher[j]
            if abs(gap)<=1e-12:continue
            d=phi[i]-phi[j];norm=np.linalg.norm(d)
            if norm==0:continue
            new.append(dict(position=position,i=i,j=j,direction=d/norm,label=1 if gap>0 else -1))
    return new


def select(memory,phi,base):
    phi=np.asarray(phi,dtype=np.float64);wins=np.zeros(len(phi),dtype=int);details=[]
    vectors=np.stack([z['direction'] for z in memory]) if memory else np.empty((0,phi.shape[1]))
    for i in range(len(phi)):
        for j in range(i+1,len(phi)):
            d=phi[i]-phi[j];n=np.linalg.norm(d);neighbors=[];vote=0.
            if n!=0 and memory:
                cos=vectors@(d/n)
                # Each historical pair has two equivalent directed views (d,y),(-d,-y).
                # Retain only its closer view, so it can contribute at most once.
                ids=np.argsort(-np.abs(cos),kind='stable')[:3]
                for k in ids:
                    z=memory[k];v=float(cos[k])*z['label'];vote+=v
                    neighbors.append(dict(memory_index=int(k),source_position=z['position'],i=z['i'],j=z['j'],cosine=float(cos[k]),label=z['label'],contribution=v))
            y=1 if vote>0 else (-1 if vote<0 else 0)
            if y>0:wins[i]+=1
            elif y<0:wins[j]+=1
            details.append(dict(i=i,j=j,pair_norm=float(n),vote=vote,predicted_preference=y,neighbors=neighbors))
    selected=max(range(len(phi)),key=lambda k:(int(wins[k]),float(base[k]),-k))
    return selected,wins.tolist(),details
