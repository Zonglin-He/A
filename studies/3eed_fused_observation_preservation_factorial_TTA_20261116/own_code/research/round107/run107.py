"""R106 fixed evidence: supervision/readout compatibility factorial, no task labels."""
from core107 import *
REF107=[
 'original_source','prepared_frozen','frozen_four','direct_FC','joint3_FC',
 'real_frozen_B','real_frozen_C','real_frozen_ensemble','real_frozen_fusedABC','real_TTA_C','real_TTA_fusedABC',
 'randomA_frozen_fusedABC','randomA_TTA_fusedABC','wrong_frozen_fusedABC','wrong_TTA_fusedABC','teacher_shuffle_fusedABC']
END107=['C','fusedABC','AB_C']
LABELS107=['real_mean_keep','real_native2','real_native_keep','randomA_mean_keep','wrong_mean_keep','teacher_shuffle_keep','norm_old_direction']
ARMS107=REF107+['R106_teacher_'+k for k in ['direct_all','direct_matched','direct_matched_C']]+[k+'_frozen_'+v for k in FAMS107 for v in ['AB','AB_C','ensemble4']]+['real_'+q+'_direct_'+v for q in ['mean','native'] for v in ['all','matched','matched_C']]+[label+'_'+v for label in LABELS107 for v in END107]
def main():
 assert jread(H107/'ENGINEERING.json')['n']==12
 for p,h in jread(H107/'experiment/CODE_LOCK.json')['files'].items():assert sha(R107/p)==h
 anchors={x['A']['metapath']:x for x in jread(R107/'research/round105/FULL_ANCHORS_PRIVATE.json')};pairs={x['pair_PRIVATE']:x for x in jread(R107/'research/round105/FULL_PAIRS_PRIVATE.json')};seals={};n=0
 for src in ['waymo','drone','quad']:
  setsource107(src);base=base97(src);fp=fingerprint(base);m,S=model97(base,src);head=patch94(m)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;oldroot=B106/name;rows=jread(R107/f'research/round99/input/TARGET_{tgt.upper()}128_PRIVATE.json');O=B107/name;assert not O.exists()
   for sub in ['samples_PRIVATE','features_PRIVATE','records_PRIVATE','states_PRIVATE']:(O/sub).mkdir(exist_ok=True,parents=True)
   for ix,r in enumerate(rows):
    tick=time.perf_counter();sid=r['id'];restore(m,S);old=load(oldroot/'samples_PRIVATE'/(sid+'.npz'));oldp=dict(zip(old['arm_names'],old['scores']));oldrec=jread(oldroot/'records_PRIVATE'/(sid+'.json'));sup=load(B106/'direct_teacher_supplement_PRIVATE'/name/(sid+'.npz'));supp=dict(zip(sup['arm_names'],sup['scores']));scores={k:oldp[k] for k in REF107};scores.update({'R106_teacher_'+k:supp['teacher_'+k] for k in ['direct_all','direct_matched','direct_matched_C']});ar=anchors[str(Path(r['metapath']).resolve())];rec={'protected_alias_fallback':oldrec['protected_alias_fallback'],'C_access_after_all_fits':True,'fits':{},'B_proof':{},'events':[],'updated_output_enabled':False};states={};features={}
    if rec['protected_alias_fallback']:scores.update({k:oldp['prepared_frozen'] for k in ARMS107 if k not in scores})
    else:
     with np.load(oldroot/'features_PRIVATE'/(sid+'.npz')) as cache:
      A=cached107(cache,'real','A');BB={k:cached107(cache,k,'B') for k in FAMS107};rec['events']+=['cache_A','cache_B'];bflags=[oldrec['families'][k]['B_available'] for k in FAMS107];assert len(set(bflags))==1;hasB=bflags[0]
      if hasB:
       with torch.no_grad():ab,proof,ctx=obtain_AB107(base,m,S,r,ar,pairs,oldrec);assert np.array_equal(ctx[0]['boxes'],old['source_boxes']);rec['B_proof']=proof;rec['events'].append('source_native_fusedAB')
       for k in FAMS107:
        keep=['mean_keep','native2','native_keep'] if k=='real' else ['mean_keep']
        for variant in keep:
         label=k+'_'+variant;vv,tt=targets107(A,BB[k],ab[k],variant);st,tr,info=lm105(m,vv,tt,head);states[label]=st;rec['fits'][label]={**info,'trials':tr,'target_log':tt.cpu().numpy().tolist(),'fit_view_labels':['A',k+'_B']+([k+'_AB'] if variant.endswith('keep') else [])};rec['events'].append('fit_'+label)
       qsh=(A['p']+BB['real']['p'][torch.as_tensor(qperm94(r),device='cuda')])/2;tt=torch.stack([qsh,qsh,ab['real']['p']]).log();st,tr,info=lm105(m,[A,BB['real'],ab['real']],tt,head);states['teacher_shuffle_keep']=st;rec['fits']['teacher_shuffle_keep']={**info,'trials':tr,'target_log':tt.cpu().numpy().tolist(),'fit_view_labels':['A','real_B','real_AB']};rec['events'].append('fit_teacher_shuffle_keep');oldst=load(oldroot/'states_PRIVATE'/(sid+'_real.npz'));states['norm_old_direction'],rec['norm_control']=normmatch107(head,oldst,states['real_mean_keep']);rec['events'].append('norm_old_direction_without_C')
      else:ab={k:A for k in FAMS107};rec['events'].append('B_unavailable_no_fit')
      # All C/fusedABC tensor values are first read at this point, after fits.
      C={k:cached107(cache,k,'C') for k in FAMS107};ABC={k:cached107(cache,k,'fusedABC') for k in FAMS107};rec['events'].append('cache_C_and_fusedABC');rec['updated_output_enabled']=oldrec['families']['real']['updated_output_enabled'];views={k:{'A':A,'B':BB[k],'AB':ab[k],'C':C[k],'fusedABC':ABC[k]} for k in FAMS107};frozen={k:draw107(m,v,head) for k,v in views.items()}
      for k in FAMS107:
       p=frozen[k];scores[k+'_frozen_AB']=p['AB'].cpu().numpy();scores[k+'_frozen_AB_C']=((p['AB']+p['C'])/2).cpu().numpy();scores[k+'_frozen_ensemble4']=p['ensemble4'].cpu().numpy()
      for qname,q in [('mean',(A['p']+BB['real']['p'])/2),('native',ab['real']['p'])]:
       on=rec['updated_output_enabled'];scores['real_'+qname+'_direct_all']=q.cpu().numpy();scores['real_'+qname+'_direct_matched']=(q if on else frozen['real']['fusedABC']).cpu().numpy();scores['real_'+qname+'_direct_matched_C']=((q+C['real']['p'])/2 if on else (frozen['real']['AB']+frozen['real']['C'])/2).cpu().numpy()
      for label in LABELS107:
       k='randomA' if label.startswith('randomA') else 'wrong' if label.startswith('wrong') else 'real';on=oldrec['families'][k]['updated_output_enabled'];p=draw107(m,views[k],states[label]) if on and label in states else frozen[k]
       for v in END107:scores[label+'_'+v]=((p['AB']+p['C'])/2 if v=='AB_C' else p[v]).cpu().numpy()
      features=feature_save105({k+'_AB':f for k,f in ab.items()})
    arr=np.stack([scores[k] for k in ARMS107]);assert arr.shape==(len(ARMS107),256) and np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-12);rec['seconds']=time.perf_counter()-tick;np.savez_compressed(O/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS107),scores=arr,source_boxes=old['source_boxes'],queries=arr.argmax(1),anno_index=old['anno_index']);jdump(O/'records_PRIVATE'/(sid+'.json'),rec)
    if features:np.savez_compressed(O/'features_PRIVATE'/(sid+'.npz'),**features)
    for label,st in states.items():np.savez_compressed(O/'states_PRIVATE'/(sid+'_'+label+'.npz'),**st)
    restore(m,S);n+=1
    if (ix+1)%16==0:print('RUN107',name,ix+1,len(rows),round(rec['seconds'],2),flush=True)
   jdump(O/'SEALED.json',{'n':len(rows),'no_GT':True,'primary':PRIMARY107,'files':{str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file()}});seals[name]=sha(O/'SEALED.json')
  end=state(m);assert fingerprint(base)==fp and all(np.array_equal(end[k],v) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 jdump(B107/'MATRIX_SEALED.json',{'n':n,'arms':ARMS107,'directions':seals,'primary':PRIMARY107,'no_GT':True});print('RUN107_COMPLETE',n,len(ARMS107),flush=True)
if __name__=='__main__':main()
