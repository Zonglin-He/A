"""Independent final readout, objective, Jacobian and native-feature checks; no GT."""
from core99 import *
def jacobian_check99(m,views,r0):
 put94(m,ORIGINAL_HEAD99);aa,gg,_=zip(*(components_solve(m,f,r0) for f in views));a=torch.cat(aa);g=torch.cat(gg);nv=len(views);j=torch.cat([(g[:,:,None]*a[:,None,:]).reshape(nv*256,-1),g],1).reshape(nv,256,-1);j=j-j.mean(1,keepdim=True);J=j.reshape(nv*256,-1);fast=(g@g.T)*(a@a.T+1);fast=fast.reshape(nv,256,nv,256);fast=fast-fast.mean(1,keepdim=True)-fast.mean(3,keepdim=True)+fast.mean((1,3),keepdim=True);err=float(abs(fast.reshape(nv*256,nv*256)-J@J.T).max());assert err<1e-7
 W=m.match[-1].weight.detach().double().requires_grad_(True);b=m.match[-1].bias.detach().double().requires_grad_(True);maximum=0.
 for v,f in enumerate(views):
  raw=F.linear(aa[v],W,b)+f['late'].double();qq=F.normalize(raw,dim=-1);scores=native_scores(qq,f['u'].double(),f['mask'].bool())+r0;scores=scores-scores.mean()
  for q in [0,128,255]:
   dw,db=torch.autograd.grad(scores[q],(W,b),retain_graph=True);row=torch.cat([dw.flatten(),db]);maximum=max(maximum,float(abs(row-J[v*256+q]).max()));assert maximum<1e-9
 return {'explicit_Gram_max_error':err,'autograd_Jacobian_max_error':maximum}
ORIGINAL_HEAD99=None
def main():
 global ORIGINAL_HEAD99
 n=endpoints=trials=native=0;maximum=0.;jac=[];seals={}
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);ORIGINAL_HEAD99=patch94(m)
  for tgt in ['waymo','drone','quad']:
   if tgt==src:continue
   O=B99/(src+'_to_'+tgt);seal=jread(O/'SEALED.json');rows=jread(H99/f'input/TARGET_{tgt.upper()}128_PRIVATE.json');assert seal['n']==len(rows)==128
   for p,h in seal['files'].items():assert sha(O/p)==h
   for ix,r in enumerate(rows):
    sid=r['id'];out=load(O/'samples_PRIVATE'/(sid+'.npz'));old=load(B98/(src+'_to_'+tgt)/'samples_PRIVATE'/(sid+'.npz'));ev=load(B98/(src+'_to_'+tgt)/'evidence_PRIVATE'/(sid+'.npz'));assert np.array_equal(out['source_boxes'],old['source_boxes']);oldp=dict(zip(old['arm_names'],old['scores']));newp=dict(zip(out['arm_names'],out['scores']));assert all(np.array_equal(newp[k],oldp[k]) for k in REFS99);ff=load(O/'features_PRIVATE'/(sid+'.npz'));views=[{k:torch.as_tensor(ff[str(i)+'_'+k],device='cuda') for k in ['h','u','late','mask']} for i in range(3)];cc=load(B98/(src+'_to_'+tgt)/'features_PRIVATE'/(sid+'.npz'));fc={k:torch.as_tensor(cc['C_'+k],device='cuda') for k in ['h','u','late','mask']};r0=torch.as_tensor(ev['r0'],device='cuda');rec=jread(O/'records_PRIVATE'/(sid+'.json'));delta=torch.as_tensor(ev['delta'],device='cuda');correct=torch.as_tensor(ev['correct'],device='cuda')
    for name in FITS99:
     st=load(O/'states_PRIVATE'/(sid+'_'+name+'.npz'));put94(m,st)
     with torch.no_grad():
      pf=read94(m,views[0],r0);pc=read94(m,fc,r0)
      for v,p in [('F',pf),('C',pc),('FC',(pf+pc)/2)]:
       e=float(abs(p.cpu().numpy()-newp[name+'_'+v]).max());maximum=max(maximum,e);assert e<1e-7 and int(p.argmax())==int(newp[name+'_'+v].argmax());endpoints+=1
      dd=delta if name=='joint3' else delta[torch.as_tensor(qperm94(r),device='cuda')];target=(correct+r0+dd).log_softmax(1);e,_=errors99(m,views,r0,target);assert abs(float(e.square().mean()/2)-rec[name]['final_error'])<1e-12
     assert rec[name]['accepted_steps']==sum(t['accepted'] for t in rec[name]['trials'])
     for t in rec[name]['trials']:
      rho=(t['before']-t['after'])/max(t['predicted_reduction'],1e-30);assert abs(rho-t['rho'])<1e-8 and t['accepted']==(t['predicted_reduction']>0 and t['after']<t['before'] and rho>.1);trials+=1
    if ix==0:
     restore(m,S)
     with torch.no_grad():z0,g,_=capture98(base,r,src)
     z=tensor_prefix(z0);d=tensor_data(g,r);rt,_=role97(base,m,r,z0,g,z,d);route=tensor_route89(rt);pm,im=maskpair79(r,d)
     with torch.no_grad():vv=[feature91(m,base,z,d,route,S,a,b) for a,b in [(None,None)]+list(zip(pm,im))]
     for f,ref in zip(vv,views):
      for k in ['h','u','late','mask']:assert torch.equal(f[k],ref[k])
     for name in FITS99:
      dd=delta if name=='joint3' else delta[torch.as_tensor(qperm94(r),device='cuda')];st,_,_=lmviews99(m,vv,r0,(correct+r0+dd).log_softmax(1),ORIGINAL_HEAD99);oldst=load(O/'states_PRIVATE'/(sid+'_'+name+'.npz'));assert all(np.array_equal(st[k],oldst[k]) for k in st)
     jac.append({'source':src,'target':tgt,**jacobian_check99(m,vv,r0)});native+=1
    n+=1
   seals[src+'_to_'+tgt]=sha(O/'SEALED.json');print('AUDIT99',src,tgt,len(rows),flush=True)
  assert fingerprint(base)==fp;del m,base;torch.cuda.empty_cache()
 assert n==768 and endpoints==4608 and native==6;jdump(H99/'NATIVE_AUDIT.json',{'state':'PASS_R99_ALL_ENDPOINTS_LOSS_TRIALS_NATIVE_FEATURES_EXPLICIT_JACOBIAN','n':n,'endpoints':endpoints,'trials':trials,'native':native,'max_probability_error':maximum,'Jacobian_checks':jac,'GT_access':False,'FINAL_access':False});jdump(B99/'MATRIX_SEALED.json',{'directions':seals,'n':n,'arms':ARMS99,'primary':'joint3_FC'});print('AUDIT99_PASS',n,endpoints)
if __name__=='__main__':main()
