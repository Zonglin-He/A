from core134 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
def main():
 basecheck()
 for p,h in rd(H134/'experiment/MODEL_CODE_LOCK.json')['files'].items():assert sha(R134/p)==h,p
 reg=rd(H134/'INPUT_REGISTRY_PRIVATE.json');eng={(e['source'],e['target'],e['id_PRIVATE']) for e in rd(H134/'ENGINEERING_INPUTS_PRIVATE.json')};anchors={a['A']['frame_PRIVATE']:a for a in rd(R134/'research/round131/ANCHORS_PRIVATE.json')};pairs={p['pair_PRIVATE']:p for p in rd(R134/'research/round131/PAIRS_PRIVATE.json')};root=B134/('engineering' if args.engineering else 'formal');assert not root.exists();root.mkdir(parents=True);n=0;checks=[];seals={}
 if not args.engineering:assert rd(H134/'ENGINEERING.json')['status']=='PASS_NATIVE_ZERO_READER_ORDINAL_GRADIENT_AND_SCOPE'
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);finger=fingerprint(base)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;rr=[e for e in reg if e['source']==src and e['target']==tgt and (not args.engineering or (src,tgt,e['row']['id']) in eng)];D=root/name
   for sub in ['samples_PRIVATE','features_PRIVATE','states_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
   for i,e in enumerate(rr):
    sid=e['row']['id'];scores,ff,ss,rec,boxes=process134(base,m,S,e,anchors,pairs,args.engineering);arr=np.stack([scores[k] for k in ARMS134]);assert arr.shape==(9,256) and np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-9);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),scores=arr,arm_names=np.array(ARMS134),queries=arr.argmax(1),source_boxes=boxes)
    if ff is not None:np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff);np.savez_compressed(D/'states_PRIVATE'/(sid+'.npz'),**ss)
    dump(D/'records_PRIVATE'/(sid+'.json'),rec);n+=1
    if args.engineering:checks.append({'source':src,'target':tgt,'B_provider':rec['B_provider'],'old_partial_active':e['active'],'zero_reader_error':rec['zero_reader_max_error'],'finite_difference':rec['finite_difference']})
    if args.engineering or i%32==0:print('RUN134',args.engineering,name,i+1,len(rr),'pair',e['pair_active'],round(rec.get('seconds_joint',0),3),flush=True)
   dump(D/'SEALED.json',{'n':len(rr),'files':{str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file()},'new_GT_used':False});seals[name]=sha(D/'SEALED.json')
  assert fingerprint(base)==finger and all(np.array_equal(v,state(m)[k]) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 if args.engineering:dump(H134/'ENGINEERING.json',{'status':'PASS_NATIVE_ZERO_READER_ORDINAL_GRADIENT_AND_SCOPE','n':n,'checks':checks,'task_GT_used':False})
 else:assert n==len(reg);dump(B134/'MATRIX_SEALED.json',{'n':n,'arms':ARMS134,'seals':seals,'primary':'partial_pair_F','FINAL_access':False,'target_GT_selects_configuration':False})
 print('COMPLETE134',args.engineering,n,flush=True)
if __name__=='__main__':main()
