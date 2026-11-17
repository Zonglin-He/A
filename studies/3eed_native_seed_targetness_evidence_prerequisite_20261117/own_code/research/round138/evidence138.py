"""Frozen trained seed-selector evidence, never a final-box quality label."""
from common138 import *
sys.path.insert(0,str(R/'research/round136'));from optim136 import *
from common138 import R,H,B,rd,sha,dump,check,utc
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
def encode(base,z,P):
 V,L=base.cross_encoder(vis_feats=P[None],pos_feats=z['pos0'],padding_mask=torch.zeros((1,len(P)),device=P.device,dtype=torch.bool),text_feats=z['lang0'],text_padding_mask=z['textpad'],end_points={},detected_feats=None,detected_mask=None);return V[0]
def main():
 check();reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');keys=set(rd(H/'ENGINEERING_KEYS_PRIVATE.json'));root=B/('engineering' if args.engineering else 'direct');root.mkdir(parents=True,exist_ok=False);n=0
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);fp=fingerprint(base);np.savez_compressed(root/(src+'_SEED_HEAD_WEIGHTS_PRIVATE.npz'),**state(base.points_obj_cls))
  for e in [e for e in reg if e['source']==src and (not args.engineering or src+'_to_'+e['target']+'|'+e['row']['id'] in keys)]:
   name=src+'_to_'+e['target'];sid=e['row']['id'];D=root/name
   for sub in ['samples_PRIVATE','features_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
   z0,z,d,route,ev,groups,op=inputs136(e);old=load(R/'crossplatform/round135/formal'/name/'features_PRIVATE'/(sid+'.npz'));hasB=bool(old['valid_B'].any());p0=op['intact_live_A'];scores={'Frozen_A':p0,'Frozen_F':op['intact_live_F'],'AB_mean':op['AB_mean'],'R135_A':op['tta_A'],'R135_F':op['tta_F']};ff={};records={};ids=z['query_points_sample_inds'].long();rng=np.random.default_rng(int(hashlib.sha256(('R138seedwrong|'+src+'|'+sid).encode()).hexdigest()[:16],16));order=rng.permutation(256);perm=np.arange(256);perm[order]=np.roll(order,1);ff['permutation']=perm
   with torch.no_grad():
    rawV=encode(base,z,z['vis0'][0]);rawlogit=base.points_obj_cls(rawV.T[None])[0,0];top=torch.topk(rawlogit.sigmoid(),256).indices;sel=rawlogit.sigmoid()[ids];threshold=rawlogit.sigmoid()[top].min();assert float((threshold-sel).clamp_min(0).max())<1e-6;ff['original_seed_logits']=rawlogit.cpu().numpy();ff['original_seed_bank']=rawV.cpu().numpy();records['original_query_selector_exact']=bool(torch.equal(top,ids));records['original_query_selector_max_threshold_error']=float((threshold-sel).clamp_min(0).max())
    for view in ['A','F']:
     P=z['vis0'][0]
     if view=='F':P=torch.where(torch.as_tensor(old['valid_B'],device='cuda')[:,None],(P+torch.as_tensor(old['B_vis'],device='cuda'))/2,P)
     PP,_,_=m.features(P,d,'bidirectional');V=encode(base,z,PP);seed=base.points_obj_cls(V.T[None])[0,0];anchor=seed[ids].double();baseline=torch.as_tensor(op['intact_live_'+view],device='cuda').clamp_min(1e-300).log()
     for kind,logit in [('seed_only',anchor),('seed_product',baseline+anchor),('wrong_seed_product',baseline+anchor[torch.as_tensor(perm,device='cuda')])]:scores[view+'_'+kind]=logit.softmax(0).cpu().numpy() if hasB else p0.copy()
     ff[view+'_seed_bank']=V.cpu().numpy();ff[view+'_seed_logits']=seed.cpu().numpy();ff[view+'_anchor_logits']=anchor.cpu().numpy()
   arr=np.stack(list(scores.values()));assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1)
   np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),scores=arr,arm_names=np.array(list(scores)),queries=arr.argmax(1),source_boxes=z0['boxes']);np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff);dump(D/'records_PRIVATE'/(sid+'.json'),{'has_B':hasB,**records});n+=1
   if args.engineering or n%64==0:print('R138',n,hasB,records['original_query_selector_exact'],flush=True)
  assert fingerprint(base)==fp and all(np.array_equal(v,state(m)[k]) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 dump(root/'SEALED.json',{'n':n,'GT_used':False,'FINAL_access':False,'source_architecture_unchanged':True,'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}});print('R138_COMPLETE',n,flush=True)
if __name__=='__main__':main()
