from core135 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');ap.add_argument('--phase',choices=['prepare','fit'],required=True);args=ap.parse_args()
def main():
 basecheck();reg=rd(H135/'INPUT_REGISTRY_PRIVATE.json');eng=set(rd(H135/'ENGINEERING_KEYS_PRIVATE.json'));root=B135/('engineering' if args.engineering else 'formal');root.mkdir(parents=True,exist_ok=True);anchors={a['A']['frame_PRIVATE']:a for a in rd(R135/'research/round131/ANCHORS_PRIVATE.json')};pairs={p['pair_PRIVATE']:p for p in rd(R135/'research/round131/PAIRS_PRIVATE.json')};n=0;summary=[]
 if not args.engineering:assert rd(H135/'ENGINEERING.json')['status']=='PASS'
 if args.phase=='fit':assert (root/'PREPARED_SEALED.json').exists()
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);finger=fingerprint(base);np.savez_compressed(B135/(src+'_SOURCE_MODEL_PRIVATE.npz'),**S)
  for tgt in ['waymo','drone','quad']:
   if src==tgt:continue
   name=src+'_to_'+tgt;rr=[e for e in reg if e['source']==src and e['target']==tgt and (not args.engineering or name+'|'+e['row']['id'] in eng)];D=root/name
   for sub in ['samples_PRIVATE','features_PRIVATE','fit_features_PRIVATE','route_PRIVATE','evidence_PRIVATE','states_PRIVATE','records_PRIVATE','fit_records_PRIVATE','zero_PRIVATE']:(D/sub).mkdir(exist_ok=True,parents=True)
   for i,e in enumerate(rr):
    tick=time.perf_counter();sid=e['row']['id'];n+=1
    if args.phase=='prepare':
     assert not (D/'zero_PRIVATE'/(sid+'.npz')).exists();ps,ff,rt,ev,rec,boxes=prepare135(base,m,S,e,anchors,pairs);np.savez_compressed(D/'zero_PRIVATE'/(sid+'.npz'),scores=np.stack(list(ps.values())),arm_names=np.array(list(ps)),source_boxes=boxes);np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff);np.savez_compressed(D/'route_PRIVATE'/(sid+'.npz'),**{k:rt[k] for k in ['token_roles','supports','weights_plain','weights_relation','point_perm','rank_pairs','rank_weights','own_support','own_valid','hyp_geo','own_geo','relation_tokens']});np.savez_compressed(D/'evidence_PRIVATE'/(sid+'.npz'),**ev);dump(D/'records_PRIVATE'/(sid+'.json'),rec)
    else:
     assert not (D/'samples_PRIVATE'/(sid+'.npz')).exists();old=load(D/'zero_PRIVATE'/(sid+'.npz'));ps=dict(zip(old['arm_names'],old['scores']));ff=load(D/'features_PRIVATE'/(sid+'.npz'));rt=load(D/'route_PRIVATE'/(sid+'.npz'));ev=load(D/'evidence_PRIVATE'/(sid+'.npz'));rec=rd(D/'records_PRIVATE'/(sid+'.json'));states={};fits={};z0=load(R135/'crossplatform/round131/formal'/name/'prefix_PRIVATE'/(sid+'.npz'));z=tensor_prefix(z0);_,d=routegeom135(e,z0);route=tensor_route89(rt);A=z['vis0'][0];B=torch.as_tensor(ff['B_vis'],device='cuda');valid=torch.as_tensor(ff['valid_B'],device='cuda');FAB=torch.where(valid[:,None],(A+B)/2,A);groups=[np.flatnonzero(ev['assignment']==g) for g in range(int(ev['assignment'].max()+1))]
     for kind in FITS135:
      view=kind[-1];P=A if view=='A' else FAB;wrong=kind.startswith('wrong');reuse=not rec['active'] or (wrong and (len(ev['win'])==0 or (np.array_equal(ev['win'],ev['wrong_win']) and np.array_equal(ev['lose'],ev['wrong_lose']))))
      if reuse:
       ps[kind]=ps['tta_'+view].copy() if wrong else ps['intact_live_'+view].copy();fits[kind]={'reused':True,'reason':'no_effective_pair_permutation' if wrong and rec['active'] else 'no_task_or_B'};continue
      out,a,ss,fit,adapt=fit135(base,m,S,z,d,P,route,groups,ev,wrong,args.engineering);ps[kind]=out;npfeatures(ff,kind,a);ff[kind+'_adapted']=adapt
      for k,v in ss.items():states[kind+'_'+k]=v
      fits[kind]=fit
     restore(m,S);freeze91(m);assert all(np.array_equal(v,state(m)[k]) for k,v in S.items());arr=np.stack([ps[k] for k in ARMS135]);assert arr.shape==(len(ARMS135),256) and np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-9)
     if not rec['has_B']:
      for k in ['intact_live_F','AB_mean','direct_A','direct_F']+FITS135:assert np.array_equal(ps[k],ps['intact_live_A']),k
     np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),scores=arr,arm_names=np.array(ARMS135),queries=arr.argmax(1),source_boxes=old['source_boxes']);np.savez_compressed(D/'fit_features_PRIVATE'/(sid+'.npz'),**{k:v for k,v in ff.items() if k.startswith(tuple(FITS135))});np.savez_compressed(D/'states_PRIVATE'/(sid+'.npz'),**states);rec['fits']=fits;rec['fit_seconds']=time.perf_counter()-tick;dump(D/'fit_records_PRIVATE'/(sid+'.json'),rec)
    summary.append({'source':src,'target':tgt,'has_B':rec['has_B'],'active':rec['active'],'pair_count':rec['pair_count'],'full_A_probability_error':rec['full_A_probability_error'],'legacy_F_bridge_error':rec['legacy_F_bridge_error']})
    if args.engineering or i%32==0:print('RUN135',args.phase,args.engineering,name,i+1,len(rr),'active',rec['active'],'pairs',rec['pair_count'],'seconds',round(time.perf_counter()-tick,3),flush=True)
  assert fingerprint(base)==finger and all(np.array_equal(v,state(m)[k]) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 seal={'n':n,'arms':ARMS135,'primary':'tta_A','GT_read':False,'FINAL_access':False,'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.suffix in ['.npz','.json'] and 'SEALED' not in p.name},'summary':summary};dump(root/('PREPARED_SEALED.json' if args.phase=='prepare' else 'MATRIX_SEALED.json'),seal)
 if args.engineering and args.phase=='fit':dump(H135/'ENGINEERING.json',{'status':'PASS','n':n,'checks':summary,'full_source_A_reproduced':True,'scope_Up4896':True,'source_feature_keep_zero_initially':True,'FINAL_access':False})
 print('COMPLETE135',args.phase,args.engineering,n,flush=True)
if __name__=='__main__':main()
