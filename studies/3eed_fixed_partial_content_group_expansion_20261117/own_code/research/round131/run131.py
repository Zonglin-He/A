from core131 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
def main():
 reg=jread(H131/'INPUT_REGISTRY_PRIVATE.json');anchors={a['A']['frame_PRIVATE']:a for a in jread(H131/'ANCHORS_PRIVATE.json')};pairs={p['pair_PRIVATE']:p for p in jread(H131/'PAIRS_PRIVATE.json')};n=0;seals={};checks=[];assert jread(B131/'registration_PRIVATE/SEALED.json')['n']==len(pairs)
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);finger=fingerprint(base)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;rr=[e for e in reg if e['source']==src and e['target']==tgt]
   if args.engineering:
    eligible=[e for e in rr if not e['protected_alias_fallback']];supported=[e for e in eligible if anchors[e['frame_PRIVATE']]['B_pair'] is not None and jread(B131/'registration_PRIVATE'/(anchors[e['frame_PRIVATE']]['B_pair']+'.json'))['geometry_cycle_supported']];rr=[(supported or eligible)[0]]
   D=B131/('engineering' if args.engineering else 'formal')/name
   assert not D.exists()
   for sub in ['samples_PRIVATE','features_PRIVATE','states_PRIVATE','records_PRIVATE','prefix_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
   for i,e in enumerate(rr):
    sid=e['row']['id'];tick=time.perf_counter();a=anchors[e['frame_PRIVATE']];ff={};states={}
    if e['protected_alias_fallback']:
     old=load(R131/'crossplatform/round97'/name/'samples_PRIVATE'/(sid+'.npz'));pp=dict(zip(old['arm_names'],old['scores']))['prepared_frozen'];scores={k:pp.copy() for k in ARMS131};boxes=old['source_boxes'];rec={'protected_alias_fallback':True,'partial_eligible':False,'scope_eligible':False,'effective_shuffle':False,'no_features_media_or_GT_indexing':True,'traces':{}}
    else:
     scores,ff,states,rec,z0=process131(base,m,S,e,a,pairs.get(a['B_pair']),args.engineering);boxes=z0['boxes'];np.savez_compressed(D/'prefix_PRIVATE'/(sid+'.npz'),**z0);rec['protected_alias_fallback']=False
    arr=np.stack([scores[k] for k in ARMS131]);assert arr.shape==(4,256) and np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-9);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),arm_names=np.array(ARMS131),scores=arr,queries=arr.argmax(1),source_boxes=boxes)
    if ff:np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff)
    if states:np.savez_compressed(D/'states_PRIVATE'/(sid+'.npz'),**states)
    rec['seconds']=time.perf_counter()-tick;jdump(D/'records_PRIVATE'/(sid+'.json'),rec);n+=1
    if args.engineering:checks.append({'source':src,'target':tgt,'partial_eligible':rec['partial_eligible'],'original_source_probability_error':rec.get('original_source_probability_error'),'checks':rec.get('checks',[]),'seconds':rec['seconds']})
    if args.engineering or i%32==0:print('RUN131',args.engineering,name,i+1,len(rr),round(rec['seconds'],2),flush=True)
   jdump(D/'SEALED.json',{'n':len(rr),'task_GT_used':False,'files':{str(p.relative_to(D)):sha(p) for p in D.rglob('*') if p.is_file() and p.name!='SEALED.json'}});seals[name]=sha(D/'SEALED.json')
  assert fingerprint(base)==finger and all(np.array_equal(v,state(m)[k]) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 if args.engineering:jdump(H131/'ENGINEERING.json',{'n':n,'status':'PASS_NEW_FRAME_SOURCE_CONTEXT_A256_AND_FIXED_PARTIAL_INTERFACE','checks':checks})
 else:assert n==len(reg);jdump(B131/'MATRIX_SEALED.json',{'n':n,'arms':ARMS131,'seals':seals,'primary':'partial_F','FINAL_access':False,'task_GT_used':False})
 print('COMPLETE131',args.engineering,n,flush=True)
if __name__=='__main__':main()
