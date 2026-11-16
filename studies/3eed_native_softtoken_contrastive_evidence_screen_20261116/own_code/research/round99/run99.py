"""No target labels. All fits precede access to cached held-out C features."""
from core99 import *
import argparse
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',choices=['waymo','drone','quad'],required=True);p.add_argument('--target',choices=['waymo','drone','quad'],required=True);p.add_argument('--engineering',action='store_true');args=p.parse_args();src,tgt=args.source,args.target;assert src!=tgt;setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);Shead=patch94(m);rows=jread(H99/f'input/TARGET_{tgt.upper()}128_PRIVATE.json');rows=rows[:2] if args.engineering else rows;O=B99/(('engineering_' if args.engineering else '')+src+'_to_'+tgt);O.mkdir(exist_ok=False)
 for sub in ['samples_PRIVATE','states_PRIVATE','records_PRIVATE','features_PRIVATE']:(O/sub).mkdir()
 if not args.engineering:
  for f,h in jread(H99/'experiment/CODE_LOCK.json')['code_sha256'].items():assert sha(R99/f)==h
 start=time.perf_counter();engineering=[]
 for ix,r in enumerate(rows):
  restore(m,S);oldroot=B98/(src+'_to_'+tgt);old=load(oldroot/'samples_PRIVATE'/(r['id']+'.npz'));ev=load(oldroot/'evidence_PRIVATE'/(r['id']+'.npz'))
  with torch.no_grad():z0,g,_=capture98(base,r,src)
  assert np.array_equal(old['source_boxes'],z0['boxes']);z=tensor_prefix(z0);d=tensor_data(g,r);rt,_=role97(base,m,r,z0,g,z,d);route=tensor_route89(rt);pm,im=maskpair79(r,d)
  with torch.no_grad():views=[feature91(m,base,z,d,route,S,a,b) for a,b in [(None,None)]+list(zip(pm,im))];correct=torch.stack([parts92(m,f,route,S)[0] for f in views]);r0=torch.as_tensor(ev['r0'],device='cuda');delta=torch.as_tensor(ev['delta'],device='cuda');assert float(abs(correct-torch.as_tensor(ev['correct'],device='cuda')).max())<1e-7
  targets=(correct+r0+delta).log_softmax(1);shuffled=(correct+r0+delta[torch.as_tensor(qperm94(r),device='cuda')]).log_softmax(1);states={};records={}
  if args.engineering:
   # Single-view reduction must reproduce the exact saved original R95 endpoint.
   st0,_,_,rec0=lm95(m,views[0],r0,targets[0],Shead,'lm');st1,_,_=lmviews99(m,views[:1],r0,targets[:1],Shead);err=max(float(abs(st0[k]-st1[k]).max()) for k in st0);assert err<2e-6;engineering.append({'source':src,'target':tgt,'one_view_generalization_weight_max_error':err})
  for name,target in [('joint3',targets),('joint3_shuffle',shuffled)]:
   st,trials,rec=lmviews99(m,views,r0,target,Shead);states[name]=st;records[name]={**rec,'trials':trials}
  # Only now read C. The cache was produced under the same immutable source state.
  ff=load(oldroot/'features_PRIVATE'/(r['id']+'.npz'));fc={k:torch.as_tensor(ff['C_'+k],device='cuda') for k in ['h','u','late','mask']};oldarms=dict(zip(old['arm_names'],old['scores']));scores={k:oldarms[k] for k in REFS99}
  with torch.no_grad():
   for name,st in states.items():
    put94(m,st);pF=read94(m,views[0],r0);pC=read94(m,fc,r0)
    for view,q in [('F',pF),('C',pC),('FC',(pF+pC)/2)]:scores[name+'_'+view]=q.cpu().numpy()
  arrays=np.stack([scores[k] for k in ARMS99]);assert np.isfinite(arrays).all() and np.allclose(arrays.sum(1),1,atol=1e-10);np.savez_compressed(O/'samples_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(ARMS99),scores=arrays,source_boxes=z0['boxes'],queries=arrays.argmax(1),anno_index=np.array(r['anno_index']));jdump(O/'records_PRIVATE'/(r['id']+'.json'),records)
  for name,st in states.items():np.savez_compressed(O/'states_PRIVATE'/(r['id']+'_'+name+'.npz'),**st)
  keep={str(i)+'_'+k:f[k].detach().cpu().numpy() for i,f in enumerate(views) for k in ['h','u','late','mask']};np.savez_compressed(O/'features_PRIVATE'/(r['id']+'.npz'),**keep);restore(m,S)
  if (ix+1)%16==0 or args.engineering:print('RUN99',src,tgt,ix+1,len(rows),round(time.perf_counter()-start,1),flush=True)
 endstate=state(m);assert fingerprint(base)==fp and all(np.array_equal(v,endstate[k]) for k,v in S.items());files=sorted(f for f in O.rglob('*') if f.is_file());jdump(O/'SEALED.json',{'state':'R99_FIXED_G_THREE_FIT_VIEWS_SEALED_BEFORE_NEW_TASK_SCORE','source':src,'target':tgt,'n':len(rows),'arms':ARMS99,'primary':'joint3_FC','seconds':time.perf_counter()-start,'files':{str(f.relative_to(O)):sha(f) for f in files},'FINAL_access':False,'no_target_GT':True,'C_read_only_after_both_fits':True,'engineering':engineering});print('RUN99_COMPLETE',src,tgt,len(rows),round(time.perf_counter()-start,1),flush=True)
if __name__=='__main__':main()
