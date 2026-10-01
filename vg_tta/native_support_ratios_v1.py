"""Source-paired normalized oracle headroom; model/GT I/O lives elsewhere."""
import numpy as np

def recoverable(native,candidates):
 native=np.asarray(native,dtype=np.float64);c=np.asarray(candidates,dtype=np.float64)
 assert c.shape==native.shape+(6,) and np.isfinite(native).all() and np.isfinite(c).all()
 assert ((native>=0)&(native<=1)).all() and ((c>=0)&(c<=1)).all()
 assert (c==native[...,None]).any(-1).all(),'native absent from support'
 delta=c.max(-1)-native;eligible=native<1;ratio=np.full(native.shape,np.nan)
 ratio[eligible]=100*delta[eligible]/(1-native[eligible]);assert (delta>=0).all();return delta,ratio,eligible

def paired_summary(native_t,candidates_t,native_s,candidates_s,seed=20260930,replicates=10000):
 # shape sources x conditions; zero-remaining-error handling keeps paired masks.
 dt,rt,et=recoverable(native_t,candidates_t);ds,rs,es=recoverable(native_s,candidates_s)
 assert dt.ndim==2 and dt.shape==ds.shape
 pair=et&es;used=pair.any(1);values=np.array([[rt[i,pair[i]].mean(),rs[i,pair[i]].mean()] for i in np.flatnonzero(used)])
 out=dict(sources=len(dt),paired_sources=int(used.sum()),paired_cells=int(pair.sum()),temporal_perfect_cells=int((~et).sum()),spatial_perfect_cells=int((~es).sum()),excluded_paired_sources=int((~used).sum()),raw_temporal_gain_pp=float(100*dt.mean(1).mean()),raw_spatial_gain_pp=float(100*ds.mean(1).mean()),fraction_sources_temporal_gain_gt5pp=float((dt.mean(1)>.05).mean()),fraction_sources_spatial_gain_gt5pp=float((ds.mean(1)>.05).mean()))
 if not len(values):out.update(paired=None);return out
 values=np.column_stack([values,values[:,0]-values[:,1]]);rng=np.random.default_rng(seed);boot=[]
 for at in range(0,replicates,500):boot.append(values[rng.integers(0,len(values),(min(500,replicates-at),len(values)))].mean(1))
 ci=np.quantile(np.concatenate(boot),[.025,.975],axis=0)
 out['paired']={k:dict(mean=float(values[:,j].mean()),ci95=ci[:,j].tolist()) for j,k in enumerate(['temporal','spatial','asymmetry'])};out['temporal_over_spatial_supported']=bool(ci[0,2]>0);return out
