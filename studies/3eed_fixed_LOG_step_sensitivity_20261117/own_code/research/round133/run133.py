from core133 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
def main():
 basecheck()
 for p,h in rd(H133/'experiment/MODEL_CODE_LOCK.json')['files'].items():assert sha(R133/p)==h
 reg=rd(H133/'INPUT_REGISTRY_PRIVATE.json');anchors={a['A']['frame_PRIVATE']:a for a in rd(R133/'research/round131/ANCHORS_PRIVATE.json')};pairs={p['pair_PRIVATE']:p for p in rd(R133/'research/round131/PAIRS_PRIVATE.json')};eng={(e['source'],e['target'],e['id_PRIVATE']) for e in rd(H133/'ENGINEERING_INPUTS_PRIVATE.json')};root=B133/('engineering' if args.engineering else 'formal');assert not root.exists();root.mkdir();n=0;checks=[];seals={}
 if not args.engineering:assert rd(H133/'ENGINEERING.json')['status']=='PASS_CACHED_AND_REBUILT_B_ORIGINAL_STEP3_READER_BRIDGES'
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);finger=fingerprint(base)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;rr=[e for e in reg if e['source']==src and e['target']==tgt and (not args.engineering or (src,tgt,e['row']['id']) in eng)];D=root/name
   for sub in ['samples_PRIVATE','features_PRIVATE','states_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
   for i,e in enumerate(rr):
    sid=e['row']['id'];tick=time.perf_counter();scores,ff,states,rec,boxes=process133(base,m,S,e,anchors,pairs,args.engineering);arr=np.stack([scores[k] for k in ARMS133]);assert arr.shape==(9,256) and np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-9);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),scores=arr,arm_names=np.array(ARMS133),queries=arr.argmax(1),source_boxes=boxes)
    if ff is not None:np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff);np.savez_compressed(D/'states_PRIVATE'/(sid+'.npz'),**states)
    rec['seconds_new_joint_work']=time.perf_counter()-tick;dump(D/'records_PRIVATE'/(sid+'.json'),rec);n+=1
    if args.engineering:checks.append({'source':src,'target':tgt,'B_provider':rec['B_provider'],'step3_bridge':rec['engineering_step3_bridge']})
    if args.engineering or i%32==0:print('RUN133',args.engineering,name,i+1,len(rr),round(rec['seconds_new_joint_work'],2),flush=True)
   dump(D/'SEALED.json',{'n':len(rr),'files':{str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file()},'new_GT_used':False});seals[name]=sha(D/'SEALED.json')
  assert fingerprint(base)==finger and all(np.array_equal(v,state(m)[k]) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 if args.engineering:dump(H133/'ENGINEERING.json',{'status':'PASS_CACHED_AND_REBUILT_B_ORIGINAL_STEP3_READER_BRIDGES','n':n,'checks':checks,'new_gradient_connectivity_diagnostic':False})
 else:assert n==len(reg);dump(B133/'MATRIX_SEALED.json',{'n':n,'arms':ARMS133,'seals':seals,'primary_steps':3,'target_GT_selects_steps':False,'FINAL_access':False})
 print('COMPLETE133',args.engineering,n,flush=True)
if __name__=='__main__':main()
