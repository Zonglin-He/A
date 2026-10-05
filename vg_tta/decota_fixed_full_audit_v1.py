"""Independent CPU NumPy arithmetic; no labels, model forward or selection."""
import hashlib
import numpy as np

def vector(state,names):return np.concatenate([state[n].detach().cpu().numpy().reshape(-1) for n in names])
def ahash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def top1_support(expert):
 out=[]
 for (_,pos),item in sorted(expert['observations'].items()):
  p=item['probe'];b=np.asarray(p['boxes'],np.float32).reshape(-1,4);s=np.asarray(p['target_scores'],np.float32)
  assert len(b)==len(s)<=3 and np.isfinite(b).all() and np.isfinite(s).all()
  valid=(b[:,2:]>0).all(1);b,s=b[valid],s[valid]
  if len(s) and p['accepted']:out.append((pos,b[int(s.argmax())]))
 return out
def overlap(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float)
 lo=np.maximum(a[:2]-a[2:]/2,b[:2]-b[2:]/2);hi=np.minimum(a[:2]+a[2:]/2,b[:2]+b[2:]/2)
 x=np.maximum(hi-lo,0).prod();return x/max(a[2:].prod()+b[2:].prod()-x,1e-7)
def loss(boxes,support):return -float(np.mean([overlap(boxes[p],b) for p,b in support])) if support else 0.
def audit(z,expert):
 names=list(z['initial']);hist=z['path'];support=top1_support(expert)
 assert len(vector(z['initial'],names))==1792 and z['arm']=='top1' and not z['GT_used']
 assert z['empty']==(not support) and len(hist)==(11 if support else 1)
 assert z['gradient_calls']==len(hist)-1 and z['selected_step']==min(range(len(hist)),key=lambda j:hist[j]['loss'])
 assert np.array_equal(vector(z['state'],names),vector(hist[z['selected_step']]['state'],names))
 assert np.array_equal(z['before'].numpy(),hist[0]['boxes'].numpy()) and np.array_equal(z['final'].numpy(),hist[z['selected_step']]['boxes'].numpy())
 m=np.zeros(1792);v=np.zeros(1792);mxloss=mxadam=0.;steps=[];checks=4+1792
 for k,h in enumerate(hist):
  a=vector(h['state'],names);boxes=h['boxes'].numpy();err=abs(loss(boxes,support)-h['loss']);assert err<8e-6;mxloss=max(mxloss,err);checks+=1
  rec=dict(step=k,state_vector_sha256=ahash(a),loss=h['loss'])
  if 'update' in h:
   u=h['update'];assert u['lr']==.03 and u['optimizer']=='adam' and u['names']==names
   g=u['gradient'].numpy();raw=u['raw'].numpy();assert np.isfinite(g).all() and np.isfinite(raw).all()
   m=.9*m+.1*g.astype(float);v=.999*v+.001*g.astype(float)**2
   expected=-.03*(m/(1-.9**(k+1)))/(np.sqrt(v/(1-.999**(k+1)))+1e-8)
   err=float(np.max(abs(expected-raw)));assert err<3e-6;mxadam=max(mxadam,err)
   assert np.array_equal(vector(hist[k+1]['state'],names)-a,raw);checks+=1792*3
   rec.update(gradient_sha256=ahash(g),raw_sha256=ahash(raw),gradient_norm=float(np.linalg.norm(g)),step_norm=float(np.linalg.norm(raw)))
  steps.append(rec)
 return dict(status='pass',checks=checks,max_loss_error=mxloss,max_Adam_error=mxadam,
  admitted_support_frames=len(support),steps=steps,NumPy_loss_and_Adam=True,GT_read=False,decoder_Jacobian_independently_replayed=False)
