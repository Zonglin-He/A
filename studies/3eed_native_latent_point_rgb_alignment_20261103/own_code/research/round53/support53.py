"""Frozen geometry-constrained latent support and unchanged color descriptors."""
import hashlib,sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'crossplatform/round51'))
from attributes51 import rgb_to_hsv,rgb_to_lab,COLOR_RGB_PROTOTYPES,color_support

def pixel_descriptors(pixels):
 hsv=rgb_to_hsv(pixels);lab=rgb_to_lab(pixels);h,s,v=hsv.T;L=lab[:,0];out=[]
 for name,proto in COLOR_RGB_PROTOTYPES.items():
  ph,ps,pv=rgb_to_hsv(np.array([proto]))[0];d=np.minimum(np.abs(h-ph),1-np.abs(h-ph));ch=np.exp(-.5*(d/.075)**2)*np.clip(s/.35,0,1)
  if name=='black':hs=np.clip((.35-v)/.25,0,1)
  elif name=='white':hs=np.clip((v-.60)/.30,0,1)*np.clip((.30-s)/.25,0,1)
  elif name=='gray':hs=np.clip((.30-s)/.25,0,1)*np.exp(-.5*((v-.52)/.25)**2)
  elif name=='brown':hs=ch*np.exp(-.5*((L-40)/20)**2)
  elif name in ('pink','beige'):hs=ch*np.exp(-.5*((s-ps)/.25)**2)*np.exp(-.5*((L-75)/25)**2)
  else:hs=ch*np.clip((v-.08)/.25,0,1)
  pl=rgb_to_lab(np.array([proto]))[0];ls=np.exp(-.5*np.sum(((lab-pl)/[30.,30.,30.])**2,axis=1));out.append(.5*hs+.5*ls)
 return np.stack(out,axis=1) if len(pixels) else np.empty((0,13))

def entity_masks(graph,caption,offsets):
 subject=graph['base_text']['subject'];specs={'T':(subject.get('span'),subject.get('text',''))}
 specs.update({r['entity']:(r.get('span'),r.get('phrase','')) for r in graph['references']});result={};receipts={}
 for ent,(span,literal) in specs.items():
  mask=np.zeros(len(offsets),bool);reason='UNKNOWN_invalid_span';narrowed=False
  if isinstance(span,list) and len(span)==2 and 0<=span[0]<span[1]<=len(caption) and literal:
   a,b=span;fragment=caption[a:b]
   starts=[i for i in range(len(fragment)) if fragment.startswith(literal,i)]
   if len(starts)==1:
    x=a+starts[0];y=x+len(literal);rest=caption[a:x]+caption[y:b]
    if not any(c.isalnum() for c in rest):
     narrowed=(x,y)!=(a,b);a,b=x,y;mask=(offsets[:,1]>a)&(offsets[:,0]<b)&(offsets[:,1]>offsets[:,0]);reason='EXACT_LITERAL_IN_ORIGINAL_SPAN'
   elif fragment==literal:mask=(offsets[:,1]>a)&(offsets[:,0]<b)&(offsets[:,1]>offsets[:,0]);reason='EXACT_LITERAL_IN_ORIGINAL_SPAN'
  result[ent]=mask;receipts[ent]={'available':bool(mask.any()),'reason':reason,'narrowed_separator_only':narrowed,'tokens':int(mask.sum())}
 return result,receipts

def seed_weights(latent,masks,medoid):
 f=latent['fp2_features'].T.astype(float);f/=np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-12);anchor=int(latent['query_points_sample_inds'][medoid]);pre=np.clip((f@f[anchor]+1)/2,0,1)
 qv=latent['query_point_5'][medoid].astype(float);qt=latent['query_token_5'][medoid].astype(float);tp=latent['token_point_2'].astype(float);semantic={}
 for ent,mask in masks.items():
  if not mask.any() or qt[mask].sum()<=0:semantic[ent]=None;continue
  token_point=(qt[mask]/qt[mask].sum())@tp[mask];w=np.sqrt(np.maximum(qv*token_point,0));semantic[ent]=w/(w.sum() or 1)*len(w)
 return pre,semantic

def local_lift(seed_xyz,raw_xyz,box):
 inside=(np.abs(seed_xyz.astype(float)-box[:3])<=box[3:]/2).all(1) if (box[3:]>0).all() else np.zeros(len(seed_xyz),bool)
 candidates=np.flatnonzero(inside)
 if not len(candidates):return None,None
 distance,ix=cKDTree(seed_xyz[candidates].astype(float)).query(raw_xyz.astype(float),workers=1)
 return candidates[ix],distance

def weighted_measurement(desc,weights,scene_color,nscene,instance_index):
 n=len(desc);w=np.asarray(weights,float);assert w.shape==(n,) and np.isfinite(w).all() and (w>=0).all()
 total=float(w.sum());effective=total*total/float(w@w) if total>0 else 0.;known=effective>=8.-1e-10 and nscene>=8
 observed=w@desc/total if total>0 else np.zeros(13);corrected={name:None for name in COLOR_RGB_PROTOTYPES}
 if known:
  for k,name in enumerate(COLOR_RGB_PROTOTYPES):
   p=(effective*observed[k]+.5)/(effective+1);b=(nscene*scene_color[name]+.5)/(nscene+1);corrected[name]=float(np.clip(np.log(p/(1-p))-np.log(b/(1-b)),-4,4))
 return {'instance_id':int(instance_index),'n_unique_visible_rgb_pixels':n,'effective_visible_pixels':effective,'color_soft_support':{name:float(observed[k]) if known else None for k,name in enumerate(COLOR_RGB_PROTOTYPES)},'color_logodds_relative_scene_visible':corrected,'color_evidence_reliability':effective/(effective+8) if known else 0.,'color_status':'WEIGHTED_VISIBLE_ONLY' if known else 'UNKNOWN_insufficient_effective_pixels'}

def permutation(sid,medoid):return np.random.default_rng(int.from_bytes(hashlib.sha256(('R53_seed_mapping_control_v1:'+sid+':'+str(medoid)).encode()).digest()[:8],'little')).permutation(1024)
