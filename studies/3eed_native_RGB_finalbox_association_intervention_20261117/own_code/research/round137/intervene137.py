"""GT-free late RGB association intervention; existing native features unchanged."""
from common137 import *
sys.path.insert(0,str(R/'research/round136'));from optim136 import *
from common137 import R,H,B,rd,sha,dump,check,utc
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
def main():
 check();reg=rd(H/'INPUT_REGISTRY_PRIVATE.json');keys=set(rd(H/'ENGINEERING_KEYS_PRIVATE.json'));root=B/('engineering' if args.engineering else 'direct');root.mkdir(parents=True,exist_ok=False);n=0
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);fp=fingerprint(base)
  for e in [e for e in reg if e['source']==src and (not args.engineering or src+'_to_'+e['target']+'|'+e['row']['id'] in keys)]:
   name=src+'_to_'+e['target'];sid=e['row']['id'];D=root/name
   for sub in ['samples_PRIVATE','features_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
   z0,z,d,route,ev,groups,op=inputs136(e);old=load(R/'crossplatform/round135/formal'/name/'features_PRIVATE'/(sid+'.npz'));p0=op['intact_live_A'];hasB=bool(old['valid_B'].any());scores={'Frozen_A':p0,'Frozen_F':op['intact_live_F'],'AB_mean':op['AB_mean'],'R135_A':op['tta_A'],'R135_F':op['tta_F']};features={};recs={};xyz=z['xyz'][0];boxes=torch.as_tensor(z0['boxes'],device='cuda');inside=((xyz[None]>=boxes[:,None,:3]-boxes[:,None,3:]/2)&(xyz[None]<=boxes[:,None,:3]+boxes[:,None,3:]/2)).all(-1);ids=z['query_points_sample_inds'].long();anchorinside=inside[torch.arange(256,device='cuda'),ids];visible=d['visible'].bool()&(d['p2i']>=0).any(-1);support=inside&visible[None];cnt=support.sum(-1);edited=(~anchorinside|(d['p2i'][ids]<0).all(-1))&(cnt>=4)
   if not hasB:edited=edited&False
   perm=np.arange(256);ix=torch.nonzero(edited,as_tuple=True)[0].cpu().numpy();rng=np.random.default_rng(int(hashlib.sha256(('R137wrong|'+src+'|'+sid).encode()).hexdigest()[:16],16));order=rng.permutation(ix)
   if len(order)>1:perm[order]=np.roll(order,1)
   for view in ['A','F']:
    prefix='intact_'+view+'_';a={k:torch.as_tensor(old[prefix+k],device='cuda') for k in FEATURES};P=z['vis0'][0]
    if view=='F':P=torch.where(torch.as_tensor(old['valid_B'],device='cuda')[:,None],(P+torch.as_tensor(old['B_vis'],device='cuda'))/2,P)
    with torch.no_grad():
     _,I,_=m.features(P,d,'bidirectional');rgb=m.point_image(I,d);late=torch.tanh(m.gate_late)*m.late(rgb[ids].float())*((d['p2i'][ids]>=0).any(-1))[:,None];bridge=float(abs(late-a['late']).max());assert bridge<2e-6,(name,sid,view,bridge)
     pooled=(support.float()@rgb)/cnt.clamp_min(1)[:,None];boxlate=torch.tanh(m.gate_late)*m.late(pooled.float());newlate=torch.where(edited[:,None],boxlate,late);wronglate=torch.where(edited[:,None],newlate[torch.as_tensor(perm,device='cuda')],late)
     for kind,ll in [('rebind',newlate),('wrong_rebind',wronglate)]:
      aa={**a,'late':ll};p,_,_=cached91(m,aa,route,'full_disc');scores[view+'_'+kind]=p.cpu().numpy() if hasB else p0.copy()
     features[view+'_point_RGB']=rgb.cpu().numpy();features[view+'_late']=late.cpu().numpy();features[view+'_rebind']=newlate.cpu().numpy();features[view+'_wrong']=wronglate.cpu().numpy()
    recs[view]={'anchor_late_bridge_max':bridge}
   features.update(support=support.cpu().numpy(),edited=edited.cpu().numpy(),permutation=perm,anchor_inside=anchorinside.cpu().numpy(),visible=visible.cpu().numpy());arr=np.stack(list(scores.values()));assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1)
   np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),scores=arr,arm_names=np.array(list(scores)),queries=arr.argmax(1),source_boxes=z0['boxes']);np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**features);dump(D/'records_PRIVATE'/(sid+'.json'),{'has_B':hasB,'edited_queries':int(edited.sum()),'eligible_visible_box_points_min':4,'views':recs});n+=1
   if args.engineering or n%64==0:print('R137',n,hasB,int(edited.sum()),flush=True)
  assert fingerprint(base)==fp and all(np.array_equal(v,state(m)[k]) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 dump(root/'SEALED.json',{'n':n,'GT_used':False,'FINAL_access':False,'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}});print('R137_COMPLETE',n,flush=True)
if __name__=='__main__':main()
