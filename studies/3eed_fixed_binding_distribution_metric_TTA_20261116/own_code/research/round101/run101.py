"""Fixed cached observations; no labels; C is opened after both fits."""
from core101 import *
import argparse
OLD=R101/'crossplatform/round100'
def main():
 p=argparse.ArgumentParser();p.add_argument('--engineering',action='store_true');args=p.parse_args();seals={};n=0;checks=[]
 if not args.engineering:
  assert jread(H101/'ENGINEERING.json')['n']==12
  for f,h in jread(H101/'experiment/CODE_LOCK.json')['code_sha256'].items():assert sha(R101/f)==h
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);head=patch94(m)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;rows=jread(R101/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');rows=rows[:2] if args.engineering else rows;O=B101/(('engineering_' if args.engineering else '')+name);O.mkdir(exist_ok=True)
   for sub in ['samples_PRIVATE','states_PRIVATE','records_PRIVATE']:(O/sub).mkdir(exist_ok=True)
   for ix,r in enumerate(rows):
    sid=r['id'];restore(m,S);ff=load(OLD/name/'features_PRIVATE'/(sid+'.npz'));views=[{k:torch.as_tensor(ff[str(i)+'_'+k],device='cuda') for k in ['h','u','late','mask']} for i in range(3)];ev=load(B98/name/'evidence_PRIVATE'/(sid+'.npz'));r0=torch.as_tensor(ev['r0'],device='cuda');delta=torch.as_tensor(ev['delta'],device='cuda');correct=torch.as_tensor(ev['correct'],device='cuda');targets=(correct+r0+delta).softmax(1);shuffled=(correct+r0+delta[torch.as_tensor(qperm94(r),device='cuda')]).softmax(1)
    if args.engineering:checks.append({'source':src,'target':tgt,**check101(m,views,r0,targets,head)})
    states={};records={}
    for label,t in [('kl3',targets),('kl3_shuffle',shuffled)]:
     st,trials,rec=fit101(m,views,r0,t,head);states[label]=st;records[label]={**rec,'trials':trials};assert rec['final_error']<=rec['initial_error']+1e-12
    cf=load(B98/name/'features_PRIVATE'/(sid+'.npz'));c={k:torch.as_tensor(cf['C_'+k],device='cuda') for k in ['h','u','late','mask']};old=load(OLD/name/'samples_PRIVATE'/(sid+'.npz'));scores=dict(zip(old['arm_names'],old['scores']))
    with torch.no_grad():
     for label,st in states.items():
      put94(m,st);pf=read94(m,views[0],r0);pc=read94(m,c,r0)
      for v,q in [('F',pf),('C',pc),('FC',(pf+pc)/2)]:scores[label+'_'+v]=q.cpu().numpy()
    arr=np.stack([scores[k] for k in ARMS101]);assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-10);np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS101),scores=arr,source_boxes=old['source_boxes'],queries=arr.argmax(1),anno_index=old['anno_index']);jdump(O/'records_PRIVATE'/(sid+'.json'),records)
    for label,st in states.items():np.savez_compressed(O/'states_PRIVATE'/(sid+'_'+label+'.npz'),**st)
    restore(m,S);n+=1
    if args.engineering or (ix+1)%16==0:print('R101',name,ix+1,len(rows),flush=True)
   files=sorted(f for f in O.rglob('*') if f.is_file() and f.name!='SEALED.json');jdump(O/'SEALED.json',{'n':len(rows),'files':{str(f.relative_to(O)):sha(f) for f in files},'no_GT':True,'C_only_after_fits':True,'arms':ARMS101});seals[name]=sha(O/'SEALED.json')
  assert fingerprint(base)==fp;restore(m,S);assert all(np.array_equal(x,state(m)[k]) for k,x in S.items());del m,base;torch.cuda.empty_cache()
 if args.engineering:jdump(H101/'ENGINEERING.json',{'state':'PASS_EXPLICIT_FISHER_AUTOGRAD_REAL_TWELVE_INPUT_FITS','n':n,'checks':checks})
 else:jdump(B101/'MATRIX_SEALED.json',{'n':n,'directions':seals,'arms':ARMS101,'primary':'kl3_FC','no_GT':True})
 print('RUN101_COMPLETE',n,flush=True)
if __name__=='__main__':main()
