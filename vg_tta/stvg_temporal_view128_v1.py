"""Physical observation grids and common-action-space posterior fusion, no labels."""
import numpy as np
from vg_tta.stvg_posterior_decision_tree_v1 import expected_iou_fast

def views(ids,interval):
 ids=list(map(int,ids));a,b=map(int,interval);L=b-a;lo=max(ids[0],int(np.floor(a-L/2)));hi=min(ids[-1]+1,int(np.ceil(b+L/2)))
 coarse=ids[::2]
 if coarse[-1]!=ids[-1]:coarse.append(ids[-1])
 window=ids if lo==ids[0]and hi==ids[-1]+1 else np.unique(np.rint(np.linspace(lo,hi-1,min(len(ids),hi-lo))).astype(int)).tolist()
 assert len(coarse)>=4 and len(window)>=4 and lo<=window[0]<window[-1]<hi
 return dict(Coarse=coarse,Window=window),dict(context=[lo,hi],original_interval=[a,b],new_Window_RGB_frames=len(set(window)-set(ids)),new_Coarse_RGB_frames=len(set(coarse)-set(ids)),coarse_nodes=len(coarse),window_nodes=len(window),same_full_coverage_coarse=coarse[0]==ids[0]and coarse[-1]==ids[-1],full_window_reuses_original=window==ids)

def embed(grid,q,U):
 q=np.asarray(q,float);ix=np.searchsorted(U,grid);assert np.array_equal(np.asarray(U)[ix],grid);out=np.zeros((len(U),len(U)));out[np.ix_(ix,ix)]=q;assert abs(out.sum()-1)<1e-10;return out

def readout(original_ids,components):
 U=np.unique(np.concatenate([np.asarray(original_ids)]+[np.asarray(g)for w,g,q in components])).astype(int);q=sum(w*embed(g,x,U)for w,g,x in components);assert abs(sum(w for w,g,x in components)-1)<1e-12
 utility=expected_iou_fast(U,q);ix=np.searchsorted(U,original_ids);restricted=utility[np.ix_(ix,ix)];k=int(np.argmax(restricted));a,b=divmod(k,len(ix));assert a<b
 return dict(interval=[int(original_ids[a]),int(original_ids[b])+1],hypothesis_ids=U.tolist(),q=q,decision_utility=restricted,utility=float(restricted[a,b]),original_decision_grid=list(original_ids))

def properties():
 from vg_tta.stvg_posterior_decision_tree_v1 import final_mbr
 ids=[0,3,7,12,20];q=np.zeros((5,5));q[0,4]=.6;q[1,3]=.4
 a=readout(ids,[(1.,ids,q)]);assert a['interval']==final_mbr(ids,q)['interval']
 g=[1,4,9,15];v=np.zeros((4,4));v[0,3]=.3;v[1,2]=.7;b=readout(ids,[(.5,ids,q),(.5,g,v)]);c=readout(ids,[(1.,g,v)]);mask=np.triu(np.ones((5,5),bool),1);assert np.max(abs(b['decision_utility'][mask]-(a['decision_utility'][mask]+c['decision_utility'][mask])/2))<1e-12
 vs,meta=views(list(range(65)),[20,40]);assert vs['Window'][0]==10 and vs['Window'][-1]==49 and len(vs['Coarse'])==33
 return dict(status='pass',physical_mass_preserved=True,mixture_utility_linear=True,common_original_decisions_only=True,context_rule_no_GT=True)
