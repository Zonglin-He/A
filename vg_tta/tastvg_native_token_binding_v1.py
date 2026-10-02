"""Uncalibrated native branch/ROI cosine diagnostic, not a learned verifier."""
import numpy as np

def cosine(a,b):
 a,b=np.asarray(a,float),np.asarray(b,float);d=float(np.linalg.norm(a)*np.linalg.norm(b))
 return float(np.dot(a,b)/d) if d>0 and np.isfinite(d) else None

def text_query(H,weights,mask,count,branch):
 H,weights,mask=np.asarray(H,float),np.asarray(weights,float),np.asarray(mask,bool)
 text=H[count:-count].transpose(1,0,2);length=text.shape[1]
 assert branch in ['spatial','temporal'] and weights.shape==(text.shape[0],1,count+length)
 w=weights[:,0,count:count+length] if branch=='spatial' else weights[:,0,:length]
 keep=~mask[:,count:-count];w=np.where(keep,w,0.);mass=w.sum(1)
 assert np.all(mass>0),'No valid branch text-attention mass'
 w=w/mass[:,None];q=np.sum(text*w[:,:,None],axis=1)
 return q,w,mass

def roi_weights(box,h,w,mask):
 box=np.asarray(box,float);mask=np.asarray(mask,bool).reshape(h,w)
 assert box.shape==(4,) and np.isfinite(box).all() and np.all(box[2:]>0)
 low=np.maximum(box[:2]-box[2:]/2,0);high=np.minimum(box[:2]+box[2:]/2,1)
 wx=np.maximum(0,np.minimum((np.arange(w)+1)/w,high[0])-np.maximum(np.arange(w)/w,low[0]))
 wy=np.maximum(0,np.minimum((np.arange(h)+1)/h,high[1])-np.maximum(np.arange(h)/h,low[1]))
 area=np.outer(wy,wx).reshape(-1);area[mask.reshape(-1)]=0.;total=float(area.sum())
 return area/total if total>0 else None

def score(views,attention,boxes,interval):
 boxes=np.asarray(boxes,float);assert boxes.ndim==3 and boxes.shape[-1]==4
 n=boxes.shape[1];inside=np.zeros(n,bool);inside[interval[0]:interval[1]+1]=True
 assert 0<=interval[0]<interval[1]<n
 qs={};aw={};mass={};spatial=[];motion=[];masks=[];counts=[]
 for offset,v in enumerate(views):
  H=np.asarray(v['H'],float);h,w=v['info']['fea_map_size'];count=h*w;counts.append(count)
  m=np.asarray(v['info']['encoded_mask'],bool)
  spatial.append(H[:count].transpose(1,0,2));motion.append(H[-count:].transpose(1,0,2));masks.append(m[:,:count])
  for branch in ['spatial','temporal']:
   q,a,z=text_query(H,attention[branch][offset],m,count,branch)
   qs.setdefault(branch,[]).append(q);aw.setdefault(branch,[]).append(a);mass.setdefault(branch,[]).append(z)
 def weave(xs):return np.stack([xs[i%2][i//2] for i in range(n)])
 qobj=weave(qs['spatial']).mean(0);qevt=weave(qs['temporal']).mean(0)
 # Different frame-map sizes between offsets are not silently resampled.
 assert counts[0]==counts[1] and views[0]['info']['fea_map_size']==views[1]['info']['fea_map_size']
 a,m,pad=weave(spatial),weave(motion),weave(masks);h,w=views[0]['info']['fea_map_size'];objects=[];bindings=[];frames=[];available=[];vectors=[]
 for tube in boxes:
  za=[];zm=[];valid=[]
  for t,b in enumerate(tube):
   weights=roi_weights(b,h,w,pad[t]);valid.append(weights is not None)
   za.append(np.zeros(256) if weights is None else weights@a[t]);zm.append(np.zeros(256) if weights is None else weights@m[t])
  za,zm,valid=np.stack(za),np.stack(zm),np.asarray(valid,bool);vi=valid&inside;vo=valid&~inside
  obj=cosine(za[vi].mean(0),qobj) if vi.any() else None
  cin=cosine(zm[vi].mean(0),qevt) if vi.any() else None;cout=cosine(zm[vo].mean(0),qevt) if vo.any() else None
  binding=cin-cout if cin is not None and cout is not None else None
  objects.append(obj);bindings.append(binding);frames.append([cosine(zm[t],qevt) if valid[t] else None for t in range(n)])
  available.append(dict(inside_frames=int(vi.sum()),outside_frames=int(vo.sum()),roi_frames=int(valid.sum())))
  vectors.append(dict(appearance_inside=za[vi].mean(0) if vi.any() else None,motion_inside=zm[vi].mean(0) if vi.any() else None,motion_outside=zm[vo].mean(0) if vo.any() else None))
 return dict(object_scores=objects,binding_scores=bindings,frame_event_cosines=frames,availability=available,
  q_obj=qobj,q_evt=qevt,query_cosine=cosine(qobj,qevt),text_weights={k:weave(v) for k,v in aw.items()},text_attention_mass={k:weave(v) for k,v in mass.items()},
  text_weights_mean_L1=float(np.abs(weave(aw['spatial'])-weave(aw['temporal'])).sum(1).mean()),vectors=vectors)
