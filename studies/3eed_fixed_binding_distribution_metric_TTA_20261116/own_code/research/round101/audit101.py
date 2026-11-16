"""Check every sealed prediction, source-cache bytes, exact endpoints and KL trials."""
from core101 import *
def main():
 matrix=jread(B101/'MATRIX_SEALED.json');assert matrix['n']==768;n=endpoints=trials=0;maximum=0.;checks=[]
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);head=patch94(m)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;O=B101/name;old=R101/'crossplatform/round100'/name;seal=jread(O/'SEALED.json');assert sha(O/'SEALED.json')==matrix['directions'][name];oldseal=jread(old/'SEALED.json');rows=jread(R101/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json')
   for p,h in seal['files'].items():assert sha(O/p)==h
   for ix,r in enumerate(rows):
    sid=r['id'];ffile=old/'features_PRIVATE'/(sid+'.npz');assert sha(ffile)==oldseal['files']['features_PRIVATE/'+sid+'.npz'];ff=load(ffile);views=[{k:torch.as_tensor(ff[str(i)+'_'+k],device='cuda') for k in ['h','u','late','mask']} for i in range(3)];ev=load(B98/name/'evidence_PRIVATE'/(sid+'.npz'));r0=torch.as_tensor(ev['r0'],device='cuda');delta=torch.as_tensor(ev['delta'],device='cuda');correct=torch.as_tensor(ev['correct'],device='cuda');out=load(O/'samples_PRIVATE'/(sid+'.npz'));ref=load(old/'samples_PRIVATE'/(sid+'.npz'));assert np.array_equal(out['source_boxes'],ref['source_boxes']);pp=dict(zip(out['arm_names'],out['scores']));qq=dict(zip(ref['arm_names'],ref['scores']));assert all(np.array_equal(pp[k],qq[k]) for k in REFS101);assert list(out['arm_names'])==ARMS101
    put94(m,head)
    with torch.no_grad():assert float(abs(probabilities101(m,views,r0)-(correct+r0).softmax(1)).max())<1e-7
    if ix==0:checks.append({'source':src,'target':tgt,**check101(m,views,r0,(correct+r0+delta).softmax(1),head)})
    cc=load(B98/name/'features_PRIVATE'/(sid+'.npz'));C={k:torch.as_tensor(cc['C_'+k],device='cuda') for k in ['h','u','late','mask']};rec=jread(O/'records_PRIVATE'/(sid+'.json'))
    for label in FITS101:
     st=load(O/'states_PRIVATE'/(sid+'_'+label+'.npz'));put94(m,st);target=(correct+r0+(delta if label=='kl3' else delta[torch.as_tensor(qperm94(r),device='cuda')])).softmax(1)
     with torch.no_grad():
      p=read94(m,views[0],r0);c=read94(m,C,r0)
      for v,z in [('F',p),('C',c),('FC',(p+c)/2)]:
       err=float(abs(z.cpu().numpy()-pp[label+'_'+v]).max());maximum=max(maximum,err);assert err<1e-9 and int(z.argmax())==int(pp[label+'_'+v].argmax());endpoints+=1
      assert abs(float(loss101(probabilities101(m,views,r0),target))-rec[label]['final_error'])<1e-10
     accepted=0;previous=rec[label]['initial_error']
     for tr in rec[label]['trials']:
      ok=tr['predicted_reduction']>0 and tr['after']<tr['before'] and tr['rho']>.1;assert tr['accepted']==ok;assert abs((tr['before']-tr['after'])/max(tr['predicted_reduction'],1e-30)-tr['rho'])<1e-8;assert abs(tr['before']-previous)<1e-10
      if ok:previous=tr['after'];accepted+=1
      trials+=1
     assert accepted==rec[label]['accepted_steps'] and abs(previous-rec[label]['final_error'])<1e-10
    n+=1
   print('AUDIT101',name,len(rows),flush=True)
  restore(m,S);assert fingerprint(base)==fp and all(np.array_equal(v,state(m)[k]) for k,v in S.items());del m,base;torch.cuda.empty_cache()
 assert n==768 and endpoints==4608;jdump(H101/'NATIVE_AUDIT.json',{'state':'PASS_R101_ALL_ENDPOINTS_KL_TRIALS_EXPLICIT_FISHER_AUTOGRAD_SOURCE_CACHE_HASH','n':n,'endpoints':endpoints,'trials':trials,'max_probability_error':maximum,'checks':checks,'native_feature_provenance':'R100 native feature audit and per-file SHA256 checked; no new point backbone execution needed','GT_access':False,'FINAL_access':False});print('AUDIT101_PASS',n,endpoints)
if __name__=='__main__':main()
