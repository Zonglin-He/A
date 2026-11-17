"""GT-assisted diagnostic only. This module is never imported by method136."""
from optim136 import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
def main():
 check();assert (B136/'EVIDENCE_SEALED.json').exists();reg=rd(H136/'INPUT_REGISTRY_PRIVATE.json');eng=set(rd(H136/'ENGINEERING_KEYS_PRIVATE.json'));root=B136/('engineering_oracle' if args.engineering else 'oracle');root.mkdir(exist_ok=False);n=0;summary=[]
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);finger=fingerprint(base)
  for e in [e for e in reg if e['source']==src and (not args.engineering or src+'_to_'+e['target']+'|'+e['row']['id'] in eng)]:
   name=src+'_to_'+e['target'];sid=e['row']['id'];D=root/name
   for sub in ['samples_PRIVATE','features_PRIVATE','states_PRIVATE','records_PRIVATE']:(D/sub).mkdir(exist_ok=True,parents=True)
   z0,z,d,route,ev,groups,ps=inputs136(e);eligible=bool(ev['eligible']);p0=ps['intact_live_A'];result=p0.copy();direct=p0.copy();rec={'eligible':eligible,'GT_diagnostic_only':True};ff={};ss={}
   if eligible:
    y=load(R136/'research/round131/scores_attempt001'/(name+'_'+sid+'_DIAGNOSTIC_IOUS_PRIVATE.npz'))['ious'];positive=y>=y.max()-1e-12;target=torch.as_tensor(positive,device='cuda')
    def task(lp):
     ce=-torch.logsumexp(lp[target],0);return ce,{'GT_best_set_NLL':ce}
    zero,result,ff,ss,fit=fit136(base,m,S,z,d,route,task,args.engineering);assert abs(zero-p0).max()<1e-7;direct=np.where(positive,p0,0.);direct/=direct.sum();rec.update(fit=fit,positive_queries=int(positive.sum()),no_unequal_box_quality=bool(positive.all()),initial_loss=float(-np.log(p0[positive].sum())),best_target_query_mask_PRIVATE=positive.tolist());ff['positive']=positive
   arr=np.stack([p0,direct,result]);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),scores=arr,arm_names=np.array(['Frozen_A','GT_direct_diagnostic','GT3_diagnostic']),queries=arr.argmax(1),source_boxes=z0['boxes']);np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff);np.savez_compressed(D/'states_PRIVATE'/(sid+'.npz'),**ss);dump(D/'records_PRIVATE'/(sid+'.json'),rec);n+=1
   summary.append({'source':src,'target':e['target'],'eligible':eligible})
   if args.engineering or n%32==0:print('ORACLE136',args.engineering,n,'eligible',eligible,flush=True)
  assert fingerprint(base)==finger and all(np.array_equal(v,state(m)[k]) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 dump(root/'SEALED.json',{'n':n,'GT_is_training_signal_DIAGNOSTIC_ONLY':True,'not_a_TTA_method':True,'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()},'FINAL_access':False});print('ORACLE136_COMPLETE',n,flush=True)
if __name__=='__main__':main()
