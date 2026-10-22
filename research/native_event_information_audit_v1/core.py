"""Offline supervised diagnostic only. Not an admissible inference algorithm."""
import numpy as np

def moments(x,y):
 n=len(x);w=1/n
 return dict(total=1.,sx=x.sum(0)*w,sy=float(y.mean()),xx=x.T@x*w,xy=x.T@y*w)

def combine(rows):
 return {k:sum(x[k]for x in rows)for k in rows[0]}

def fit_predict(stat,x):
 n=stat['total'];xm=stat['sx']/n;ym=stat['sy']/n;G=stat['xx']-np.outer(stat['sx'],stat['sx'])/n;scale=max(float(np.sqrt(max(np.trace(G)/n,0))),1e-12);W=np.linalg.solve(G/scale**2+np.eye(G.shape[0]),(stat['xy']-stat['sx']*ym)/scale);return (x-xm)@W/scale+ym

def holdout_stats(stats,parents,index):
 exclude={j for j,p in enumerate(parents)if p==parents[index]};return combine([s for j,s in enumerate(stats)if j not in exclude]),exclude

def auc(score,y):
 y=np.asarray(y,dtype=bool);a=score[y];b=score[~y]
 if not len(a)or not len(b):return None
 d=a[:,None]-b[None,:];return float(((d>0)+.5*(d==0)).mean())
