"""Episodic exact running-stat transport; same-info direct readers and wrong anchors."""
from core110 import *
REF110=REF108+[p+'_'+v for p in POLICIES108 for v in VIEWS108]
DIRECT110=['native_original_CE','original_source_matchedcoverage','original_native_pair_geometric','original_native_pair_arithmetic','original_CE_with_prepared_geometric_A','original_CE_with_prepared_arithmetic_A','original_CE_with_fused_geometric','original_CE_with_fused_arithmetic','original_CE_with_C_geometric','direct_three_geometric','direct_three_arithmetic']
LABEL110=['transport','wrong_anchor'];END110=['A','C','fusedABC'];POL110=['CE','geometric','arithmetic'];ARMS110=REF110+DIRECT110+[k+'_'+p+'_'+v for k in LABEL110 for v in END110 for p in POL110]
def gm110(*xs):return torch.stack([x.log() for x in xs]).mean(0).softmax(0)
def main():
 assert jread(H110/'ENGINEERING.json')['n']==12;lock=jread(H110/'experiment/CODE_LOCK.json')
 for p,h in lock['files'].items():assert sha(R110/p)==h
 wrong=jread(H110/'WRONG_ANCHOR_PRIVATE.json');seals={};n=0
 for src in ['waymo','drone','quad']:
  base=base97(src);fp=fingerprint(base);head=base.prediction_heads[-1].sem_cls_scores_head;S=statece108(head);np.savez_compressed(B110/(src+'_CE_SOURCE_PRIVATE.npz'),**S)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;D=B110/name;assert not D.exists()
   for sub in ['samples_PRIVATE','records_PRIVATE','states_PRIVATE','transport_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
   rows=jread(R110/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json')
   for ix,r in enumerate(rows):
    tick=time.perf_counter();sid=r['id'];alias=jread(B108/name/'records_PRIVATE'/(sid+'.json'))['protected_alias_fallback'];record={'protected_alias_fallback':alias,'calibrations':{},'events':[],'no_gradient_or_pseudo_labels':True,'actual_C_available':False};scores={}
    if record['protected_alias_fallback']:
     old=load(B108/name/'samples_PRIVATE'/(sid+'.npz'));oldp=dict(zip(old['arm_names'],old['scores']));scores=dict(oldp);scores.update({k:oldp['prepared_frozen'] for k in ARMS110 if k not in scores})
    else:
     oz=load(B110/'original_PRIVATE'/name/(sid+'.npz'));O=torch.as_tensor(oz['h'],device='cuda');mask=torch.as_tensor(oz['mask'],device='cuda');qO=torch.as_tensor(oz['pce'],device='cuda');pO=torch.as_tensor(oz['p'],device='cuda');wz=load(B110/'original_PRIVATE'/name/(wrong[name][sid]+'.npz'));wrongO=torch.as_tensor(wz['h'],device='cuda')
     with np.load(B106/name/'features_PRIVATE'/(sid+'.npz')) as cache:
      P=torch.as_tensor(cache['real_A_h'],device='cuda');record['events'].append('read_original_and_prepared_A_only');states={}
      for label,ref in [('transport',O),('wrong_anchor',wrongO)]:
       st,rec,info=transport110(head,ref,P,S);states[label]=st;record['calibrations'][label]=info;record['events'].append('calibrated_'+label);np.savez_compressed(D/'states_PRIVATE'/(sid+'_'+label+'.npz'),**st);np.savez_compressed(D/'transport_PRIVATE'/(sid+'_'+label+'.npz'),**{str(j)+'_'+k:np.asarray(v) for j,z in enumerate(rec) for k,v in z.items()})
      old=load(B108/name/'samples_PRIVATE'/(sid+'.npz'));oldp=dict(zip(old['arm_names'],old['scores']));scores=dict(oldp);record['actual_C_available']=jread(B106/name/'records_PRIVATE'/(sid+'.json'))['families']['real']['C_available']
      record['events'].append('first_C_and_fusedABC_read');views={v:{'h':torch.as_tensor(cache['real_'+v+'_h'],device='cuda'),'mask':torch.as_tensor(cache['real_'+v+'_mask'],device='cuda'),'p':torch.as_tensor(cache['real_'+v+'_p'],device='cuda')} for v in END110};A=views['A']['p'];C=views['C']['p'];F=views['fusedABC']['p'];qF=torch.as_tensor(oldp['softtoken_fusedABC'],device='cuda')
      direct={'native_original_CE':qO,'original_source_matchedcoverage':pO,'original_native_pair_geometric':gm110(pO,qO),'original_native_pair_arithmetic':(pO+qO)/2,'original_CE_with_prepared_geometric_A':gm110(A,qO),'original_CE_with_prepared_arithmetic_A':(A+qO)/2,'original_CE_with_fused_geometric':gm110(F,qO),'original_CE_with_fused_arithmetic':(F+qO)/2,'original_CE_with_C_geometric':gm110(C,qO),'direct_three_geometric':gm110(F,qO,qF),'direct_three_arithmetic':(F+qO+qF)/3}
      scores.update({k:v.cpu().numpy() for k,v in direct.items()})
      for label,st in states.items():
       putce110(head,st)
       with torch.no_grad():
        for v,f in views.items():
         q,_=ce108(head,f['h'],f['mask']);scores[label+'_CE_'+v]=q.cpu().numpy();scores[label+'_geometric_'+v]=gm110(f['p'],q).cpu().numpy();scores[label+'_arithmetic_'+v]=((f['p']+q)/2).cpu().numpy()
    putce110(head,S);assert all(np.array_equal(statece108(head)[k],v) for k,v in S.items());arr=np.stack([scores[k] for k in ARMS110]);assert arr.shape==(58,256) and np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-12);record['seconds']=time.perf_counter()-tick;np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS110),scores=arr,source_boxes=old['source_boxes'],queries=arr.argmax(1),anno_index=old['anno_index']);jdump(D/'records_PRIVATE'/(sid+'.json'),record);n+=1
    if (ix+1)%32==0:print('RUN110',name,ix+1,len(rows),flush=True)
   jdump(D/'SEALED.json',{'n':len(rows),'no_GT':True,'primary':PRIMARY110,'files':{str(p.relative_to(D)):sha(p) for p in sorted(D.rglob('*')) if p.is_file()}});seals[name]=sha(D/'SEALED.json')
  assert fingerprint(base)==fp;del base,head;torch.cuda.empty_cache()
 jdump(B110/'MATRIX_SEALED.json',{'n':n,'arms':ARMS110,'directions':seals,'primary':PRIMARY110,'updated_values':'only1152running-stat buffers, no learned weights','no_GT':True});print('RUN110_COMPLETE',n,len(ARMS110))
if __name__=='__main__':main()
