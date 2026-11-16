"""Native frozen RGB prefix, fixed G and late-projection-only LM; no task labels."""
from core104 import *
import argparse
def main():
 p=argparse.ArgumentParser();p.add_argument('--engineering',action='store_true');args=p.parse_args();n=0;checks=[];seals={}
 if not args.engineering:
  assert jread(H104/'ENGINEERING.json')['n']==12
  for f,h in jread(H104/'experiment/CODE_LOCK.json')['code_sha256'].items():assert sha(R104/f)==h
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);rgb0=patch104(m)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;rows=jread(R104/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');rows=rows[:2] if args.engineering else rows;O=B104/(('engineering_' if args.engineering else '')+name)
   for sub in ['samples_PRIVATE','states_PRIVATE','records_PRIVATE','features_PRIVATE']:(O/sub).mkdir(exist_ok=True,parents=True)
   for ix,r in enumerate(rows):
    sid=r['id'];restore(m,S);refroot=R104/'crossplatform/round100'/name;old=load(refroot/'samples_PRIVATE'/(sid+'.npz'));ev=load(B98/name/'evidence_PRIVATE'/(sid+'.npz'));cached=load(refroot/'features_PRIVATE'/(sid+'.npz'))
    with torch.no_grad():z0,g,_=capture98(base,r,src)
    assert np.array_equal(z0['boxes'],old['source_boxes']);z=tensor_prefix(z0);d=tensor_data(g,r);pm,im=maskpair79(r,d);refs=[{k:torch.as_tensor(cached[str(v)+'_'+k],device='cuda') for k in ['h','u','late','mask']} for v in range(3)];views=[features104(m,z,d,a,b,f) for a,b,f in zip([None]+pm,[None]+im,refs)];r0=torch.as_tensor(ev['r0'],device='cuda');delta=torch.as_tensor(ev['delta'],device='cuda');correct=torch.as_tensor(ev['correct'],device='cuda');targets={'late':(correct+r0+delta).log_softmax(1),'late_shuffle':(correct+r0+delta[torch.as_tensor(qperm94(r),device='cuda')]).log_softmax(1)}
    with torch.no_grad():
     initial_scores=[scores104(m,f,r0).clone() for f in views]
     for v,f in enumerate(views):assert float(abs(read104(m,f,r0)-(correct[v]+r0).softmax(0)).max())<1e-7
    if args.engineering:checks.append({'source':src,'target':tgt,**check104(m,views,r0,rgb0)})
    states={};records={}
    for label in FITS104:
     st,trials,info=lm104(m,views,r0,targets[label],rgb0);states[label]=st;records[label]={**info,'trials':trials}
     with torch.no_grad():
      for f,before in zip(views,initial_scores):assert torch.equal(scores104(m,f,r0)[~f['known']],before[~f['known']]),'unmapped raw score changed'
    # Unfitted C is generated only after both fits and step acceptances are over.
    put104(m,rgb0);cp,ci=masks(r,d,76003);cf=load(B98/name/'features_PRIVATE'/(sid+'.npz'));refC={k:torch.as_tensor(cf['C_'+k],device='cuda') for k in ['h','u','late','mask']};C=features104(m,z,d,cp,ci,refC);pold=dict(zip(old['arm_names'],old['scores']));scores={k:pold[k] for k in REFS104}
    with torch.no_grad():
     for label,st in states.items():
      put104(m,st);Fprob=read104(m,views[0],r0);Cprob=read104(m,C,r0)
      for v,q in [('F',Fprob),('C',Cprob),('FC',(Fprob+Cprob)/2)]:scores[label+'_'+v]=q.cpu().numpy()
    if args.engineering:
     restore(m,S);rt,_=role97(base,m,r,z0,g,z,d);route=tensor_route89(rt)
     for label,st in states.items():
      put104(m,st)
      for f,a,b in [(views[0],None,None),(C,cp,ci)]:
       with torch.no_grad():live=feature91(m,base,z,d,route,S,a,b);raw=F.normalize(m.match(live['h'][None])[0]+live['late'],dim=-1);prob=(native_scores(raw,live['u'],live['mask'].bool()).double()+r0).softmax(0);err=float(abs(prob-read104(m,f,r0)).max());assert err<1e-7
       checks.append({'source':src,'target':tgt,'native_patched_late_probability_error':err})
    arr=np.stack([scores[k] for k in ARMS104]);assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-12);np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS104),scores=arr,source_boxes=old['source_boxes'],queries=arr.argmax(1),anno_index=old['anno_index']);jdump(O/'records_PRIVATE'/(sid+'.json'),records)
    for label,st in states.items():np.savez_compressed(O/'states_PRIVATE'/(sid+'_'+label+'.npz'),**st)
    keep={v+'_'+k:f[k].detach().cpu().numpy() for v,f in [(str(i),f) for i,f in enumerate(views)]+[('C',C)] for k in ['h','u','late','mask','rgb','known','gate','query_raw']};np.savez_compressed(O/'features_PRIVATE'/(sid+'.npz'),**keep);restore(m,S);n+=1
    if args.engineering or (ix+1)%16==0:print('RUN104',name,ix+1,len(rows),flush=True)
   jdump(O/'SEALED.json',{'n':len(rows),'primary':'late_FC','C_only_after_fits':True,'no_GT':True,'files':{str(f.relative_to(O)):sha(f) for f in sorted(O.rglob('*')) if f.is_file()}});seals[name]=sha(O/'SEALED.json')
  end=state(m);assert fingerprint(base)==fp and all(np.array_equal(v,end[k]) for k,v in S.items());del m,base;torch.cuda.empty_cache()
 if args.engineering:jdump(H104/'ENGINEERING.json',{'state':'PASS_NATIVE_SOURCE_AND_PATCHED_OUTPUT_EXPLICIT_RGB_JACOBIAN_UNMAPPED_LOGIT_CONSERVATION','n':n,'checks':checks,'FINAL_access':False})
 else:jdump(B104/'MATRIX_SEALED.json',{'n':n,'directions':seals,'arms':ARMS104,'primary':'late_FC','no_GT':True})
 print('RUN104_COMPLETE',n,flush=True)
if __name__=='__main__':main()
