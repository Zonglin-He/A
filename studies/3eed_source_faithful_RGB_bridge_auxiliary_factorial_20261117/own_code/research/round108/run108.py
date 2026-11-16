"""Frozen readout screen on immutable R106 states, all examples retained."""
from core108 import *
def main():
 strict97();assert jread(H108/'ENGINEERING.json')['n']==12
 for p,h in jread(H108/'experiment/CODE_LOCK.json')['files'].items():assert sha(R108/p)==h
 for p,h in jread(H108/'experiment/CODE_LOCK.json')['dependency_sha256'].items():assert sha(R108/p)==h
 B108.mkdir(exist_ok=False,parents=True);seals={};n=0
 for src in ['waymo','drone','quad']:
  head=loadce108(src);np.savez_compressed(B108/(src+'_CE_WEIGHTS_PRIVATE.npz'),**statece108(head))
  for tgt in ['waymo','drone','quad']:
   if tgt==src:continue
   name=src+'_to_'+tgt;old=B106/name;O=B108/name;assert not O.exists()
   for sub in ['samples_PRIVATE','records_PRIVATE']:(O/sub).mkdir(parents=True,exist_ok=True)
   for r in jread(R108/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json'):
    tick=time.perf_counter();sid=r['id'];po=load(old/'samples_PRIVATE'/(sid+'.npz'));op=dict(zip(po['arm_names'],po['scores']));rec=jread(old/'records_PRIVATE'/(sid+'.json'));scores={k:op[k] for k in REF108};alias=rec['protected_alias_fallback'];proof={}
    if alias:scores.update({p+'_'+v:op['prepared_frozen'] for p in POLICIES108 for v in VIEWS108})
    else:
     with np.load(old/'features_PRIVATE'/(sid+'.npz')) as z:
      with torch.no_grad():
       for v in VIEWS108:
        h=torch.as_tensor(z['real_'+v+'_h'],device='cuda');mask=torch.as_tensor(z['real_'+v+'_mask'],device='cuda');q,logits=ce108(head,h,mask);p=torch.as_tensor(z['real_'+v+'_p'],device='cuda');read=pool108(p,q,r);scores.update({k+'_'+v:x.cpu().numpy() for k,x in read.items()});proof[v]={'input_h_sha256':hashlib.sha256(z['real_'+v+'_h'].tobytes()).hexdigest(),'mask_sha256':hashlib.sha256(z['real_'+v+'_mask'].tobytes()).hexdigest(),'logits_sha256':hashlib.sha256(logits.cpu().numpy().tobytes()).hexdigest(),'positive_tokens':int(mask.sum()),'native256denominator':True}
    arr=np.stack([scores[k] for k in ARMS108]);assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-12);np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS108),scores=arr,source_boxes=po['source_boxes'],queries=arr.argmax(1),anno_index=po['anno_index']);jdump(O/'records_PRIVATE'/(sid+'.json'),{'protected_alias_fallback':alias,'proof':proof,'seconds':time.perf_counter()-tick,'no_update':True,'GT_used':False});n+=1
   jdump(O/'SEALED.json',{'n':128,'files':{str(f.relative_to(O)):sha(f) for f in sorted(O.rglob('*')) if f.is_file()},'no_GT':True});seals[name]=sha(O/'SEALED.json');print('RUN108',name,n,flush=True)
  del head;torch.cuda.empty_cache()
 jdump(B108/'MATRIX_SEALED.json',{'n':n,'arms':ARMS108,'primary':PRIMARY108,'directions':seals,'CE_weights':{p.name:sha(p) for p in B108.glob('*WEIGHTS_PRIVATE.npz')},'GT_used':False,'FINAL_access':False});print('RUN108_COMPLETE',n,len(ARMS108),flush=True)
if __name__=='__main__':main()
