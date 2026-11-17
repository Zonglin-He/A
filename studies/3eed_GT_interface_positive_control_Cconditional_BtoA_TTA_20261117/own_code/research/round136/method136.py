"""No GT access: C-qualified B group-max distribution supervises complete A."""
from optim136 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
def main():
 check()
 if not args.engineering:assert rd(H136/'ORACLE_DECISION.json')['interface_positive_control_passed']
 reg=rd(H136/'INPUT_REGISTRY_PRIVATE.json');eng=set(rd(H136/'ENGINEERING_KEYS_PRIVATE.json'));root=B136/('engineering_method' if args.engineering else 'method');root.mkdir(exist_ok=False);n=0
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);finger=fingerprint(base)
  for e in [e for e in reg if e['source']==src and (not args.engineering or src+'_to_'+e['target']+'|'+e['row']['id'] in eng)]:
   name=src+'_to_'+e['target'];sid=e['row']['id'];D=root/name
   for sub in ['samples_PRIVATE','features_PRIVATE','states_PRIVATE','records_PRIVATE']:(D/sub).mkdir(exist_ok=True,parents=True)
   z0,z,d,route,ev,groups,old=inputs136(e);eligible=bool(ev['eligible']);p0=old['intact_live_A'];scores={'Frozen_A':p0,'Frozen_F':old['intact_live_F'],'AB_mean':old['AB_mean'],'R135_A':old['tta_A'],'R135_F':old['tta_F'],'R135_wrong_A':old['wrong_A'],'direct_C_B_A':ev['direct'],'direct_wrong_C_A':ev['wrong_direct']};ff={};states={};records={};Sg=torch.as_tensor(ev['S'],device='cuda');Cg=torch.as_tensor(ev['C'],device='cuda')
   for kind in ['C_B_A','wrong_C_B_A']:
    if not eligible or (kind.startswith('wrong') and not bool(ev['effective_wrong'])):
     scores[kind]=scores['C_B_A'].copy() if kind.startswith('wrong') else p0.copy();records[kind]={'reused':True};continue
    target=torch.as_tensor(ev['wrong_target'] if kind.startswith('wrong') else ev['target'],device='cuda')
    def task(lp):
     pt=taskloss(lp,groups,Sg,Cg,None,'partial');gm=torch.stack([lp[torch.as_tensor(g,device='cuda')].max() for g in groups]);ls=gm[Cg].log_softmax(0);rank=(target*(target.clamp_min(1e-300).log()-ls)).sum();return pt+rank,{'partial':pt,'conditional_B_KL':rank}
    zero,p,feat,ss,rec=fit136(base,m,S,z,d,route,task,args.engineering);assert abs(zero-p0).max()<1e-7;scores[kind]=p;records[kind]=rec
    for k,v in feat.items():ff[kind+'_'+k]=v
    for k,v in ss.items():states[kind+'_'+k]=v
   restore(m,S);freeze91(m);arr=np.stack(list(scores.values()));assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-9);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),scores=arr,arm_names=np.array(list(scores)),queries=arr.argmax(1),source_boxes=z0['boxes']);np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff);np.savez_compressed(D/'states_PRIVATE'/(sid+'.npz'),**states);dump(D/'records_PRIVATE'/(sid+'.json'),{'eligible':eligible,'fits':records,'GT_used':False,'source_initialization_only':True});n+=1
   if args.engineering or n%32==0:print('METHOD136',args.engineering,n,'eligible',eligible,flush=True)
  assert fingerprint(base)==finger and all(np.array_equal(v,state(m)[k]) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 dump(root/'SEALED.json',{'n':n,'GT_read':False,'primary':'C_B_A','files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()},'FINAL_access':False});print('METHOD136_COMPLETE',n,flush=True)
if __name__=='__main__':main()
