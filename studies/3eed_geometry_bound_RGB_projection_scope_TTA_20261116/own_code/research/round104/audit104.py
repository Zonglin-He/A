"""Independent late-projection state replay, source-prefix and unmapped-logit checks."""
from core104 import *
def main():
 matrix=jread(B104/'MATRIX_SEALED.json');assert matrix['n']==768;n=endpoints=trials=unmapped=0;maximum=0.;checks=[]
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);rgb0=patch104(m)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;O=B104/name;seal=jread(O/'SEALED.json');assert sha(O/'SEALED.json')==matrix['directions'][name]
   for p,h in seal['files'].items():assert sha(O/p)==h
   for ix,r in enumerate(jread(R104/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json')):
    sid=r['id'];restore(m,S);f=load(O/'features_PRIVATE'/(sid+'.npz'));allviews=[{k:torch.as_tensor(f[v+'_'+k],device='cuda') for k in ['h','u','late','mask','rgb','known','gate','query_raw']} for v in ['0','1','2','C']];views=allviews[:3];C=allviews[-1];out=load(O/'samples_PRIVATE'/(sid+'.npz'));refroot=R104/'crossplatform/round100'/name;old=load(refroot/'samples_PRIVATE'/(sid+'.npz'));oldf=load(refroot/'features_PRIVATE'/(sid+'.npz'));oldc=load(B98/name/'features_PRIVATE'/(sid+'.npz'));ev=load(B98/name/'evidence_PRIVATE'/(sid+'.npz'));r0=torch.as_tensor(ev['r0'],device='cuda');correct=torch.as_tensor(ev['correct'],device='cuda');delta=torch.as_tensor(ev['delta'],device='cuda');p=dict(zip(out['arm_names'],out['scores']));pold=dict(zip(old['arm_names'],old['scores']));assert np.array_equal(out['source_boxes'],old['source_boxes']) and all(np.array_equal(p[k],pold[k]) for k in REFS104);baseline=[]
    with torch.no_grad():
     for j,v in enumerate(allviews):
      for key in ['h','u','late','mask']:
       ref=oldf[str(j)+'_'+key] if j<3 else oldc['C_'+key];assert np.array_equal(v[key].cpu().numpy(),ref)
      assert torch.equal(v['query_raw'],m.match(v['h'][None])[0]) and torch.equal(v['gate'],torch.tanh(m.gate_late));assert torch.equal(v['late'],v['gate']*m.late(v['rgb'])*v['known'][:,None]);baseline.append(scores104(m,v,r0).clone());unmapped+=int((~v['known']).sum())
    if ix==0:checks.append({'source':src,'target':tgt,**check104(m,views,r0,rgb0)})
    rec=jread(O/'records_PRIVATE'/(sid+'.json'))
    for label in FITS104:
     st=load(O/'states_PRIVATE'/(sid+'_'+label+'.npz'));assert sum(x.size for x in st.values())==4160;put104(m,st);d=delta if label=='late' else delta[torch.as_tensor(qperm94(r),device='cuda')];targets=(correct+r0+d).log_softmax(1)
     with torch.no_grad():
      e,_=errors104(m,views,r0,targets);assert abs(float(e.square().mean()/2)-rec[label]['final_error'])<1e-12;pf=read104(m,views[0],r0);pc=read104(m,C,r0)
      for v,q in [('F',pf),('C',pc),('FC',(pf+pc)/2)]:
       err=float(abs(q.cpu().numpy()-p[label+'_'+v]).max());maximum=max(maximum,err);assert err<1e-7 and int(q.argmax())==int(p[label+'_'+v].argmax());endpoints+=1
      for v,b in zip(allviews,baseline):assert torch.equal(scores104(m,v,r0)[~v['known']],b[~v['known']])
     for t in rec[label]['trials']:
      rho=(t['before']-t['after'])/max(t['predicted_reduction'],1e-30);assert abs(rho-t['rho'])<1e-8 and t['accepted']==(t['predicted_reduction']>0 and t['after']<t['before'] and rho>.1);trials+=1
    if ix==0:
     restore(m,S)
     with torch.no_grad():z0,g,_=capture98(base,r,src)
     z=tensor_prefix(z0);d=tensor_data(g,r);rt,_=role97(base,m,r,z0,g,z,d);route=tensor_route89(rt);pm,im=maskpair79(r,d);cp,ci=masks(r,d,76003)
     fresh=[features104(m,z,d,a,b,v) for a,b,v in zip([None]+pm+[cp],[None]+im+[ci],allviews)]
     for x,y in zip(fresh,allviews):assert all(torch.equal(x[k],y[k]) for k in ['rgb','known','gate','query_raw'])
     for label in FITS104:
      ds=delta if label=='late' else delta[torch.as_tensor(qperm94(r),device='cuda')];st,_,_=lm104(m,fresh[:3],r0,(correct+r0+ds).log_softmax(1),rgb0);saved=load(O/'states_PRIVATE'/(sid+'_'+label+'.npz'));assert all(np.array_equal(st[k],saved[k]) for k in st)
      for f,a,b in [(fresh[0],None,None),(fresh[3],cp,ci)]:
       with torch.no_grad():v=feature91(m,base,z,d,route,S,a,b);q=F.normalize(m.match(v['h'][None])[0]+v['late'],dim=-1);pp=(native_scores(q,v['u'],v['mask'].bool()).double()+r0).softmax(0);assert float(abs(pp-read104(m,f,r0)).max())<1e-7
    n+=1
   print('AUDIT104',name,flush=True)
  restore(m,S);assert fingerprint(base)==fp;end=state(m);assert all(np.array_equal(v,end[k]) for k,v in S.items());del m,base;torch.cuda.empty_cache()
 assert n==768 and endpoints==4608;jdump(H104/'NATIVE_AUDIT.json',{'state':'PASS_R104_ALL_NATIVE_RGB_FEATURES_PATCHED_OUTPUTS_JACOBIANS_UNMAPPED_LOGITS_TRIALS','n':n,'endpoints':endpoints,'trials':trials,'unmapped_query_view_checks':unmapped,'unmapped_raw_score_max_error':0.,'max_probability_error':maximum,'checks':checks,'native_refits':12,'GT_used':False,'FINAL_access':False});print('AUDIT104_PASS',n,endpoints)
if __name__=='__main__':main()
