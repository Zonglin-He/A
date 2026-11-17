from target132 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
def main():
 basecheck();reg=rd(R132/'research/round131/INPUT_REGISTRY_PRIVATE.json');anchors={a['A']['frame_PRIVATE']:a for a in rd(R132/'research/round131/ANCHORS_PRIVATE.json')};pair_index={p['pair_PRIVATE']:p for p in rd(R132/'research/round131/PAIRS_PRIVATE.json')}
 for a in rd(H132/'TARGET_B3_CANDIDATES_PRIVATE.json'):
  for c in a['candidates']:pair_index[c['pair']['pair_PRIVATE']]=c['pair']
 for p,h in rd(H132/'experiment/TARGET_EXECUTION_LOCK.json')['files'].items():assert sha(R132/p)==h
 tag='engineering' if args.engineering else 'formal';root=B132/tag;assert not root.exists();root.mkdir();n=0;checks=[];seals={};arms=['zero_F','direct_partial_F','partial_F','wrong_base_F']+[kind+'_'+k for kind in ['B3','LOG'] for k in ['zero','direct','partial','wrong']]
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);finger=fingerprint(base)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;rr=[e for e in reg if e['source']==src and e['target']==tgt];wanted={}
   if args.engineering:
    newB=[e for e in rr if rd(B132/'selections_PRIVATE'/('TARGET_'+e['frame_PRIVATE']+'.json'))['selected_rank'] in [2,3]];newL=[]
    for e in rr:
     with np.load(B132/'log_scopes_PRIVATE'/name/(e['row']['id']+'.npz')) as f:
      if bool(f['active']) and not bool(f['same_labels']):newL.append(e)
    for kind,ee in [('B3',newB),('LOG',newL)]:
     if ee:wanted[(ee[0]['row']['id'],kind)]=True
    rr=[e for e in rr if any((e['row']['id'],k) in wanted for k in ['B3','LOG'])]
   D=root/name
   for folder in ['samples_PRIVATE','records_PRIVATE','features_PRIVATE','states_PRIVATE']:(D/folder).mkdir(parents=True,exist_ok=True)
   for i,e in enumerate(rr):
    tick=time.perf_counter();sid=e['row']['id'];_,old,boxes=original132(e);merged=dict(zip(arms[:4],[old[k] for k in ['zero','direct','partial','wrong']]));meta={}
    for kind in ['B3','LOG']:
     if args.engineering and (sid,kind) not in wanted:continue
     scores,ff,states,rec,bb=target_case132(base,m,S,e,kind,anchors,pair_index,args.engineering);assert np.array_equal(bb,boxes);meta[kind]=rec;merged.update({kind+'_'+k:v for k,v in scores.items()})
     if ff is not None:np.savez_compressed(D/'features_PRIVATE'/(sid+'_'+kind+'.npz'),**ff);np.savez_compressed(D/'states_PRIVATE'/(sid+'_'+kind+'.npz'),**states)
     if args.engineering:checks.append({'source':src,'target':tgt,'kind':kind,'active':rec.get('partial_eligible'),'checks':rec.get('engineering_checks',[]),'old_bridge_errors':rec.get('old_bridge_errors',{})})
    order=[k for k in arms if k in merged];arr=np.stack([merged[k] for k in order]);assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-9);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(order),scores=arr,queries=arr.argmax(1),source_boxes=boxes);meta['seconds']=time.perf_counter()-tick;dump(D/'records_PRIVATE'/(sid+'.json'),meta);n+=1
    if args.engineering or i%32==0:print('RUN132',tag,name,i+1,len(rr),round(meta['seconds'],2),flush=True)
   dump(D/'SEALED.json',{'n':len(rr),'files':{str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file()},'new_GT_used':False});seals[name]=sha(D/'SEALED.json')
  assert fingerprint(base)==finger and all(np.array_equal(v,state(m)[k]) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 if args.engineering:dump(H132/'ENGINEERING.json',{'n_inputs':n,'checks':checks,'status':'PASS_SEPARATE_FACTOR_FULL_READER_GRADIENT_SCOPE_AND_BRIDGES'})
 else:assert n==len(reg);dump(B132/'MATRIX_SEALED.json',{'n':n,'arms':arms,'seals':seals,'no_B3_LOG_combination':True,'new_GT_used':False,'FINAL_access':False})
 print('RUN132_COMPLETE',tag,n,flush=True)
if __name__=='__main__':main()
