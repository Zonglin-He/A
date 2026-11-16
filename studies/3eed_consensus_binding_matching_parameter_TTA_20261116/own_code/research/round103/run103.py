"""No target labels; all targets fixed before fits; C read only afterwards."""
from core103 import *
import argparse
def main():
 p=argparse.ArgumentParser();p.add_argument('--engineering',action='store_true');args=p.parse_args();seals={};count=0;checks=[]
 if not args.engineering:
  assert jread(H103/'ENGINEERING.json')['n']==12
  for f,h in jread(H103/'experiment/CODE_LOCK.json')['code_sha256'].items():assert sha(R103/f)==h
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);head=patch94(m)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;rows=jread(R103/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');rows=rows[:2] if args.engineering else rows;O=B103/(('engineering_' if args.engineering else '')+name)
   for sub in ['samples_PRIVATE','states_PRIVATE','records_PRIVATE']:(O/sub).mkdir(exist_ok=True,parents=True)
   for ix,r in enumerate(rows):
    sid=r['id'];restore(m,S);f=load(R103/'crossplatform/round100'/name/'features_PRIVATE'/(sid+'.npz'));views=[{k:torch.as_tensor(f[str(i)+'_'+k],device='cuda') for k in ['h','u','late','mask']} for i in range(3)];ev=load(B98/name/'evidence_PRIVATE'/(sid+'.npz'));r0=torch.as_tensor(ev['r0'],device='cuda');correct=torch.as_tensor(ev['correct'],device='cuda');e=load(R103/'crossplatform/round102'/name/'evidence_PRIVATE'/(sid+'.npz'));common=e['common_delta'];scale=np.linalg.norm(common)/max(np.linalg.norm(e['mean_delta']),1e-30);shifts={'common3':common,'scaledmean3':e['mean_delta']*scale,'shufflecommon3':e['shuffled_delta']};targets={k:(correct+r0+torch.as_tensor(d,device='cuda')).log_softmax(1) for k,d in shifts.items()}
    if args.engineering:
     st0,_,_,_=lm95(m,views[0],r0,targets['common3'][0],head,'lm');st1,_,_=lmviews99(m,views[:1],r0,targets['common3'][:1],head);err=max(float(abs(st0[k]-st1[k]).max()) for k in st0);assert err<2e-6;checks.append({'source':src,'target':tgt,'one_view_reduction_weight_error':err})
    records={};states={}
    for label in FITS103:
     st,trials,info=lmviews99(m,views,r0,targets[label],head);states[label]=st;records[label]={**info,'trials':trials}
    # C is not used to construct targets, acceptance, or the number of updates.
    cc=load(B98/name/'features_PRIVATE'/(sid+'.npz'));C={k:torch.as_tensor(cc['C_'+k],device='cuda') for k in ['h','u','late','mask']};old=load(R103/'crossplatform/round102'/name/'samples_PRIVATE'/(sid+'.npz'));pp=dict(zip(old['arm_names'],old['scores']));scores={k:pp[k] for k in REFS103}
    with torch.no_grad():
     for label,st in states.items():
      put94(m,st);Fprob=read94(m,views[0],r0);Cprob=read94(m,C,r0)
      for v,z in [('F',Fprob),('C',Cprob),('FC',(Fprob+Cprob)/2)]:scores[label+'_'+v]=z.cpu().numpy()
    arr=np.stack([scores[k] for k in ARMS103]);assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-12);np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS103),scores=arr,source_boxes=old['source_boxes'],queries=arr.argmax(1),anno_index=old['anno_index']);jdump(O/'records_PRIVATE'/(sid+'.json'),records)
    for label,st in states.items():np.savez_compressed(O/'states_PRIVATE'/(sid+'_'+label+'.npz'),**st)
    count+=1
    if (ix+1)%16==0 or args.engineering:print('RUN103',name,ix+1,len(rows),flush=True)
   jdump(O/'SEALED.json',{'n':len(rows),'primary':'common3_FC','no_GT':True,'C_only_after_fits':True,'files':{str(f.relative_to(O)):sha(f) for f in sorted(O.rglob('*')) if f.is_file()}});seals[name]=sha(O/'SEALED.json')
  restore(m,S);assert fingerprint(base)==fp;end=state(m);assert all(np.array_equal(v,end[k]) for k,v in S.items());del m,base;torch.cuda.empty_cache()
 if args.engineering:jdump(H103/'ENGINEERING.json',{'state':'PASS_REAL_TWELVE_INPUT_TARGET_CHANGE_OLD_SOLVER_REDUCTION','n':count,'checks':checks})
 else:jdump(B103/'MATRIX_SEALED.json',{'n':count,'directions':seals,'primary':'common3_FC','arms':ARMS103,'no_GT':True})
 print('RUN103_COMPLETE',count,flush=True)
if __name__=='__main__':main()
