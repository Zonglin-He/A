"""Guarded six-source bridge physical TTA, predictions sealed before scoring."""
from case113 import *
def main():
 lock=jread(H113/'experiment/CODE_LOCK.json')
 for p,h in lock['files'].items():assert sha(R113/p)==h,p
 assert jread(H113/'ENGINEERING.json')['n']==12;seals={};n=0
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);np.savez_compressed(B113/(src+'_SOURCE_READOUT_PRIVATE.npz'),**{k:v for k,v in S.items() if k.startswith(('match.','interface.out.'))});names=[k for k,_ in m.tta_parameters()];dims=[int(p.numel()) for _,p in m.tta_parameters()];jdump(B113/(src+'_THETA_SCHEMA_PRIVATE.json'),{'names':names,'sizes':dims,'source_module_sha256':hashlib.sha256(b''.join(k.encode()+v.tobytes() for k,v in sorted(S.items()))).hexdigest()})
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;D=B113/name
   for k in ['samples_PRIVATE','features_PRIVATE','records_PRIVATE','fits_PRIVATE']: (D/k).mkdir(parents=True,exist_ok=True)
   rows=jread(R113/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');start=time.perf_counter()
   for j,r in enumerate(rows):
    sid=r['id'];sf=D/'samples_PRIVATE'/(sid+'.npz');rf=D/'records_PRIVATE'/(sid+'.json')
    if sf.exists():assert rf.exists();continue
    oldmeta=jread(B112/name/'records_PRIVATE'/(sid+'.json'));alias=oldmeta['protected_alias_fallback'];new={};features={};fm={};meta={'protected_alias_fallback':alias,'fits':{},'TTA_run':not alias,'C_neural_after_all_fits':True};ss=time.perf_counter()
    if alias:
     # No prefix, media, feature, model or protected annotation access.
     old=load(B112/name/'samples_PRIVATE'/(sid+'.npz'));pr=dict(zip(old['arm_names'],old['scores']));new={k:pr['prepared_frozen'].copy() for k in NEW113}
    else:
     c=fitcase113(base,m,S,src,r);new,features,fm,ev=evaluate113(base,m,S,src,r,c);meta.update(fits=c['updates'],gradients=c['gradient_records'],source_inputs=c['proof'],role_info=c['role'],wrong_changed_visible_seeds=c['wrong_changed_seeds'],C_boundary_stamp=c['C_boundary_stamp'],evaluation=ev,cpu_readouts=fm)
     states={k+'_'+t+'_theta':np.concatenate([v[n].reshape(-1) for n in names]) for k,st in c['fits'].items() for t,v in st.items()};states.update({'gradient_'+k:v for k,v in c['gradients'].items()});np.savez_compressed(D/'fits_PRIVATE'/(sid+'.npz'),**states);np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**features)
     # Only now may old records containing C/F scores be accessed.
     old=load(B112/name/'samples_PRIVATE'/(sid+'.npz'));pr=dict(zip(old['arm_names'],old['scores']));frozenA=features['frozen_full_A_h'][0];assert np.array_equal(frozenA,load(B106/name/'features_PRIVATE'/(sid+'.npz'))['real_A_h'])
     for key,arm in [('full','real_frozen_fusedABC'),('anchor','neither_correct_fusedABC')]:
      aa={k[len('frozen_'+key+'_fusedABC_'):]:v for k,v in features.items() if k.startswith('frozen_'+key+'_fusedABC_')};pp,_=cpu113(aa,{k:v for k,v in S.items() if k.startswith(('match.','interface.out.'))});assert abs(pp['single']-pr[arm]).max()<5e-6
    scores={k:pr[k] for k in REFS113};scores.update(new);arr=np.stack([scores[a] for a in ARMS113]);assert arr.shape==(89,256) and np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-10);np.savez_compressed(sf,source_boxes=old['source_boxes'],arm_names=np.array(ARMS113),scores=arr,queries=arr.argmax(1));meta['seconds']=time.perf_counter()-ss;jdump(rf,meta);restore(m,S);freeze91(m);assert all(np.array_equal(state(m)[k],v) for k,v in S.items());n+=1
    if (j+1)%16==0:print('RUN113',name,j+1,len(rows),round(time.perf_counter()-start,1),'last',round(meta['seconds'],2),flush=True)
   files=[p for k in ['samples_PRIVATE','features_PRIVATE','records_PRIVATE','fits_PRIVATE'] for p in (D/k).glob('*')];jdump(D/'SEALED.json',{'n':len(rows),'arms':ARMS113,'primary':PRIMARY113,'files':{str(p.relative_to(D)):sha(p) for p in files},'source_original_unchanged':fingerprint(base)==fp,'seconds':time.perf_counter()-start,'peak_GPU_bytes':int(torch.cuda.max_memory_allocated()),'GT_read':False,'FINAL_access':False});seals[name]=sha(D/'SEALED.json');print('DIRECTION113_COMPLETE',name,flush=True)
  assert fingerprint(base)==fp;del base,m;torch.cuda.empty_cache()
 jdump(B113/'MATRIX_SEALED.json',{'n':768,'arms':ARMS113,'primary':PRIMARY113,'directions':seals,'GT_read':False,'FINAL_access':False,'deadline_utc':jread(H113/'experiment/DESIGN_LOCK.json')['deadline_utc']});print('RUN113_COMPLETE',768,len(ARMS113),flush=True)
if __name__=='__main__':B113.mkdir(exist_ok=True);main()
