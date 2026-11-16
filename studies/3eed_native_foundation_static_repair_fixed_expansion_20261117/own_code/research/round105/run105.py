"""Fixed actual-observation self-distillation, all development inputs retained."""
from engineering105 import *
HIST105=['original_source','prepared_frozen','frozen_four','direct_FC','joint3_FC']
NEW105=['frozen_A_AB','frozen_AC','frozen_ensemble3','frozen_jointABC','TTA_A','TTA_AB','TTA_AC','TTA_jointABC','TTA_ensemble3','teacher_shuffle_AC','teacher_shuffle_jointABC','wrong_frozen_AC','wrong_frozen_jointABC','wrong_frozen_ensemble3','wrong_TTA_AC','wrong_TTA_jointABC','wrong_TTA_ensemble3']
ARMS105=HIST105+NEW105
def draw105(m,views,state):
 put94(m,state)
 with torch.no_grad():
  pp={k:read105(m,f) for k,f in views.items()};pp['ensemble3']=torch.stack([pp[k] for k in ['A','AB','AC']]).mean(0);return pp
def main():
 assert jread(H105/'ENGINEERING.json')['n']==12
 for p,h in jread(H105/'experiment/CODE_LOCK.json')['files'].items():assert sha(R105/p)==h
 anchors={x['A']['metapath']:x for x in jread(H105/'FULL_ANCHORS_PRIVATE.json')};pairs={x['pair_PRIVATE']:x for x in jread(H105/'FULL_PAIRS_PRIVATE.json')};n=0;seals={}
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);head=patch94(m)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;rows=jread(R105/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');O=B105/name;assert not O.exists()
   for sub in ['samples_PRIVATE','features_PRIVATE','states_PRIVATE','records_PRIVATE']:(O/sub).mkdir(parents=True,exist_ok=False)
   for ix,r in enumerate(rows):
    tick=time.perf_counter();old=load(R105/'crossplatform/round100'/name/'samples_PRIVATE'/(r['id']+'.npz'));oldp=dict(zip(old['arm_names'],old['scores']));probs={k:oldp[k] for k in HIST105};ar=anchors[str(Path(r['metapath']).resolve())];status={'eligible_A':ar['A']['eligible_for_extra_media'],'B':{'status':'not_available'},'C':{'status':'not_available'},'source_state_reset':True,'C_neural_only_after_fits':True};records={};states={};features={};restore(m,S)
    if not ar['A']['eligible_for_extra_media']:
     probs.update({k:oldp['prepared_frozen'] for k in NEW105});status['fallback']='protected_RGB_content_alias_existing_frozen_prediction_no_decode_or_model'
    else:
     with torch.no_grad():z0,g,_=capture98(base,r,src)
     assert np.array_equal(z0['boxes'],old['source_boxes']);z=tensor_prefix(z0);d=tensor_data(g,r)
     with torch.no_grad():rt,_=role97(base,m,r,z0,g,z,d);route=tensor_route89(rt);A=native105(m,base,z,d,route,S,[])
     assert float(abs(A['p']-torch.as_tensor(oldp['prepared_frozen'],device='cuda')).max())<1e-7
     B=None
     if ar['B_pair']:B,status['B']=safe_extra105(ar['A'],pairs[ar['B_pair']],z0)
     wB=geo.shuffled_extra(B,int(hashlib.sha256(('R105Bwrong|'+r['id']).encode()).hexdigest()[:16],16)) if B is not None else None
     def feature(gs):
      if not gs:return A
      with torch.no_grad():return native105(m,base,z,data105(g,gs,r),route,S,[tensor_image105(x) for x in gs])
     AB=feature([B] if B is not None else []);wAB=feature([wB] if wB is not None else []);views={'A':A,'AB':AB};wrong={'A':A,'AB':wAB}
     if B is not None:
      target=(A['p']+AB['p'])/2;qp=torch.as_tensor(qperm94(r),device='cuda');shuftarget=(A['p']+AB['p'][qp])/2;wt=(A['p']+wAB['p'])/2
      for label,vv,tt in [('real',views,target),('teacher_shuffle',views,shuftarget),('wrong',wrong,wt)]:
       st,trials,info=lm105(m,list(vv.values()),tt.log()[None].repeat(2,1),head);states[label]=st;records[label]={**info,'trials':trials,'target':tt.cpu().numpy().tolist()}
     # No C image, C features or C probabilities entered any fit/acceptance.
     put94(m,head);C=None
     if ar['C_pair']:C,status['C']=safe_extra105(ar['A'],pairs[ar['C_pair']],z0)
     wC=geo.shuffled_extra(C,int(hashlib.sha256(('R105Cwrong|'+r['id']).encode()).hexdigest()[:16],16)) if C is not None else None
     views.update(AC=feature([C] if C is not None else []),jointABC=feature([x for x in [B,C] if x is not None]));wrong.update(AC=feature([wC] if wC is not None else []),jointABC=feature([x for x in [wB,wC] if x is not None]));frozen=draw105(m,views,head);wfrozen=draw105(m,wrong,head);transfer=bool(B is not None and C is not None);status['adaptation_attempted']=B is not None;status['heldout_new_observation_available']=C is not None;status['TTA_output_enabled']=transfer;status['fallback']='none' if transfer else 'no_B_no_fit_or_no_C_zero_update_final_readout'
     fitted={label:draw105(m,wrong if label=='wrong' else views,st) for label,st in states.items()};real=fitted.get('real',frozen) if transfer else frozen;shuf=fitted.get('teacher_shuffle',frozen) if transfer else frozen;wreal=fitted.get('wrong',wfrozen) if transfer else wfrozen
     out={'frozen_A_AB':(frozen['A']+frozen['AB'])/2,'frozen_AC':frozen['AC'],'frozen_ensemble3':frozen['ensemble3'],'frozen_jointABC':frozen['jointABC'],'TTA_A':real['A'],'TTA_AB':real['AB'],'TTA_AC':real['AC'],'TTA_jointABC':real['jointABC'],'TTA_ensemble3':real['ensemble3'],'teacher_shuffle_AC':shuf['AC'],'teacher_shuffle_jointABC':shuf['jointABC'],'wrong_frozen_AC':wfrozen['AC'],'wrong_frozen_jointABC':wfrozen['jointABC'],'wrong_frozen_ensemble3':wfrozen['ensemble3'],'wrong_TTA_AC':wreal['AC'],'wrong_TTA_jointABC':wreal['jointABC'],'wrong_TTA_ensemble3':wreal['ensemble3']};probs.update({k:p.cpu().numpy() for k,p in out.items()});features=feature_save105(views)|feature_save105({'wrong_'+k:f for k,f in wrong.items()});status.update(B_visible_queries=status['B'].get('mapped_queries',0),C_visible_queries=status['C'].get('mapped_queries',0));put94(m,head)
    arr=np.stack([probs[k] for k in ARMS105]);assert arr.shape==(len(ARMS105),256) and np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-12);status['seconds']=time.perf_counter()-tick;np.savez_compressed(O/'samples_PRIVATE'/(r['id']+'.npz'),arm_names=np.array(ARMS105),scores=arr,source_boxes=old['source_boxes'],queries=arr.argmax(1),anno_index=old['anno_index']);jdump(O/'records_PRIVATE'/(r['id']+'.json'),{'status':status,'fits':records})
    if features:np.savez_compressed(O/'features_PRIVATE'/(r['id']+'.npz'),**features)
    for label,st in states.items():np.savez_compressed(O/'states_PRIVATE'/(r['id']+'_'+label+'.npz'),**st)
    restore(m,S);n+=1
    if (ix+1)%16==0:print('RUN105',name,ix+1,len(rows),round(status['seconds'],2),flush=True)
   jdump(O/'SEALED.json',{'n':len(rows),'no_GT':True,'primary':'TTA_jointABC','files':{str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file()}});seals[name]=sha(O/'SEALED.json')
  assert fingerprint(base)==fp and all(np.array_equal(v,state(m)[k]) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 jdump(B105/'MATRIX_SEALED.json',{'n':n,'directions':seals,'arms':ARMS105,'primary':'TTA_jointABC','no_GT':True});print('RUN105_COMPLETE',n,flush=True)
if __name__=='__main__':main()
