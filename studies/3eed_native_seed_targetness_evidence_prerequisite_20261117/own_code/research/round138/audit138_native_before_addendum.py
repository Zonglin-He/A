"""Independent CPU conv/BN targetness, source identity and complete score audit."""
from common138 import *
import argparse,numpy as np,torch
from torch.nn import functional as F
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
def cpuhead(V,W):
 x=V.T[None]
 for j in [1,2]:x=F.relu(F.batch_norm(F.conv1d(x,W[f'conv{j}.weight'],W[f'conv{j}.bias']),W[f'bn{j}.running_mean'],W[f'bn{j}.running_var'],W[f'bn{j}.weight'],W[f'bn{j}.bias'],training=False,eps=1e-5))
 return F.conv1d(x,W['conv3.weight'],W['conv3.bias'])[0,0]
def main():
 check();torch.set_num_threads(4);root=B/('engineering' if args.engineering else 'direct');seal=rd(root/'SEALED.json');n=0;noB=0;mxh=0.;mxp=0.;exact=0;weights_checked=0
 for p,h in seal['files'].items():assert sha(root/p)==h,p
 for src in ['waymo','drone','quad']:
  W={k:torch.from_numpy(v.copy()) for k,v in dict(np.load(root/(src+'_SEED_HEAD_WEIGHTS_PRIVATE.npz'))).items()};c=next(c for c in rd(R/'research/round97/experiment/DESIGN_LOCK.json')['sources'] if c['source']==src);ck=torch.load(c['checkpoint_path'],map_location='cpu',weights_only=True)['model']
  for k,v in W.items():assert torch.equal(v,ck['points_obj_cls.'+k]);weights_checked+=1
  del ck
  for D in sorted(root.glob(src+'_to_*')):
   for p in sorted((D/'samples_PRIVATE').glob('*.npz')):
    a=np.load(p);ps=dict(zip(a['arm_names'],a['scores']));ff=np.load(D/'features_PRIVATE'/p.name);rec=rd(D/'records_PRIVATE'/(p.stem+'.json'));z=np.load(R/'crossplatform/round131/formal'/D.name/'prefix_PRIVATE'/p.name);old=np.load(R/'crossplatform/round135/formal'/D.name/'samples_PRIVATE'/p.name);op=dict(zip(old['arm_names'],old['scores']));assert np.array_equal(a['source_boxes'],old['source_boxes']) and np.array_equal(a['queries'],a['scores'].argmax(1));assert np.array_equal(ps['Frozen_A'],op['intact_live_A']) and np.array_equal(ps['Frozen_F'],op['intact_live_F']);ids=z['query_points_sample_inds'].astype(int);perm=ff['permutation'];assert np.array_equal(np.sort(perm),np.arange(256)) and np.all(perm!=np.arange(256));raw=cpuhead(torch.from_numpy(ff['original_seed_bank'].copy()),W);err=float(abs(raw.numpy()-ff['original_seed_logits']).max());mxh=max(mxh,err);assert err<2e-5;threshold=np.partition(ff['original_seed_logits'],-256)[-256];assert (threshold-ff['original_seed_logits'][ids]).max()<1e-5;exact+=rec['original_query_selector_exact']
    for view in ['A','F']:
     logits=cpuhead(torch.from_numpy(ff[view+'_seed_bank'].copy()),W);err=float(abs(logits.numpy()-ff[view+'_seed_logits']).max());mxh=max(mxh,err);assert err<2e-5,(D.name,p.stem,view,err);assert np.array_equal(ff[view+'_anchor_logits'],ff[view+'_seed_logits'][ids].astype(float));anchor=logits[ids].double();lp=torch.from_numpy(np.maximum(op['intact_live_'+view],1e-300)).log()
     for kind,x in [('seed_only',anchor),('seed_product',lp+anchor),('wrong_seed_product',lp+anchor[perm])]:
      expect=x.softmax(0).numpy() if rec['has_B'] else ps['Frozen_A'];err=float(abs(expect-ps[view+'_'+kind]).max());mxp=max(mxp,err);assert err<5e-6,(D.name,p.stem,view,kind,err)
    if not rec['has_B']:
     noB+=1
     for k in ps:
      if k not in ['Frozen_A','Frozen_F','AB_mean','R135_A','R135_F']:assert np.array_equal(ps[k],ps['Frozen_A'])
    n+=1
    if n%256==0:print('AUDIT138',n,flush=True)
 assert n==(12 if args.engineering else 1798);out={'status':'PASS','n':n,'no_B_complete_fallback':noB,'source_head_state_items_verified':weights_checked,'exact_original_query_selector':exact,'CPU_seed_logits_max_error':mxh,'CPU_probability_max_error':mxp,'full_encoder_not_CPU_replayed':True,'GT_used':False,'FINAL_access':False};dump(H/('ENGINEERING_AUDIT.json' if args.engineering else 'AUDIT.json'),out);print(out,flush=True)
if __name__=='__main__':main()
