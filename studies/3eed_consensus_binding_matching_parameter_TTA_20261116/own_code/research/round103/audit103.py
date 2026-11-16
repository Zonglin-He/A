"""Independent source-cache, target algebra, endpoint and acceptance replay, no GT."""
from core103 import *
sys.path.insert(0,str(R103/'research/round100'));import audit100 as jaccheck
def main():
 matrix=jread(B103/'MATRIX_SEALED.json');assert matrix['n']==768;n=endpoints=trials=0;maximum=0.;checks=[]
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);head=patch94(m);jaccheck.ORIGINAL_HEAD99=head
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;O=B103/name;old=R103/'crossplatform/round102'/name;seal=jread(O/'SEALED.json');assert sha(O/'SEALED.json')==matrix['directions'][name];oldseal=jread(old/'SEALED.json');froot=R103/'crossplatform/round100'/name;fseal=jread(froot/'SEALED.json')
   for p,h in seal['files'].items():assert sha(O/p)==h
   for ix,r in enumerate(jread(R103/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json')):
    sid=r['id'];f=froot/'features_PRIVATE'/(sid+'.npz');assert sha(f)==fseal['files']['features_PRIVATE/'+sid+'.npz'];ff=load(f);views=[{k:torch.as_tensor(ff[str(i)+'_'+k],device='cuda') for k in ['h','u','late','mask']} for i in range(3)];ev=load(B98/name/'evidence_PRIVATE'/(sid+'.npz'));d=(ev['correct'][None]-ev['wrong']).mean(1);d-=d.mean(1,keepdims=True);mean=d.mean(0);c=np.where(d[0]*d[1]>0,np.sign(d[0])*np.minimum(abs(d[0]),abs(d[1])),0.);c-=c.mean();ratio=np.linalg.norm(c)/max(np.linalg.norm(mean),1e-30);shifts=[c,mean*ratio,c[qperm94(r)]];r0=torch.as_tensor(ev['r0'],device='cuda');correct=torch.as_tensor(ev['correct'],device='cuda');out=load(O/'samples_PRIVATE'/(sid+'.npz'));of=old/'samples_PRIVATE'/(sid+'.npz');assert sha(of)==oldseal['files']['samples_PRIVATE/'+sid+'.npz'];ref=load(of);assert np.array_equal(out['source_boxes'],ref['source_boxes']);p=dict(zip(out['arm_names'],out['scores']));pp=dict(zip(ref['arm_names'],ref['scores']));assert all(np.array_equal(p[k],pp[k]) for k in REFS103)
    if ix==0:checks.append({'source':src,'target':tgt,**jaccheck.jacobian_check99(m,views,r0)})
    cc=load(B98/name/'features_PRIVATE'/(sid+'.npz'));C={k:torch.as_tensor(cc['C_'+k],device='cuda') for k in ['h','u','late','mask']};rec=jread(O/'records_PRIVATE'/(sid+'.json'))
    for label,delta in zip(FITS103,shifts):
     target=(correct+r0+torch.as_tensor(delta,device='cuda')).log_softmax(1);put94(m,load(O/'states_PRIVATE'/(sid+'_'+label+'.npz')))
     with torch.no_grad():
      Fprob=read94(m,views[0],r0);Cprob=read94(m,C,r0);err,_=errors99(m,views,r0,target);after=float(err.square().mean()/2);assert abs(after-rec[label]['final_error'])<1e-12
      for v,z in [('F',Fprob),('C',Cprob),('FC',(Fprob+Cprob)/2)]:
       err=float(abs(z.cpu().numpy()-p[label+'_'+v]).max());maximum=max(maximum,err);assert err<1e-7 and int(z.argmax())==int(p[label+'_'+v].argmax());endpoints+=1
     for t in rec[label]['trials']:
      rho=(t['before']-t['after'])/max(t['predicted_reduction'],1e-30);assert abs(rho-t['rho'])<1e-8 and t['accepted']==(t['predicted_reduction']>0 and np.isfinite(t['after']) and t['after']<t['before'] and rho>.1);trials+=1
    n+=1
   print('AUDIT103',name,flush=True)
  restore(m,S);assert fingerprint(base)==fp;del m,base;torch.cuda.empty_cache()
 assert n==768 and endpoints==6912;jdump(H103/'NATIVE_AUDIT.json',{'state':'PASS_R103_SOURCE_HASH_TARGET_RECONSTRUCTION_ALL_MATCHING_ENDPOINTS_TRIAL_RULES_EXPLICIT_JACOBIAN','n':n,'endpoints':endpoints,'trials':trials,'max_probability_error':maximum,'checks':checks,'GT_access':False,'FINAL_access':False});print('AUDIT103_PASS',n,endpoints)
if __name__=='__main__':main()
