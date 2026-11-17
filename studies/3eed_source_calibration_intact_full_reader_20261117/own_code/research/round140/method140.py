"""Source-only calibrated C; otherwise unchanged R136 complete-A adaptation."""
from common140 import *
sys.path.insert(0,str(R/'research/round136'));from optim136 import *
from common140 import R,H,B,rd,sha,dump,check,utc
from prepare140 import LEVELS
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--engineering',action='store_true');args=ap.parse_args()
def main():
 check();assert all(sha(R/p)==h for p,h in rd(H/'experiment/TARGET_CODE_LOCK.json')['files'].items());reg=rd(H/('ENGINEERING_INPUTS_PRIVATE.json' if args.engineering else 'INPUT_REGISTRY_PRIVATE.json'));root=B/('engineering' if args.engineering else 'formal');root.mkdir(exist_ok=False);n=0;nfit=0;reused=0
 if not args.engineering:assert rd(H/'ENGINEERING_AUDIT.json')['status']=='PASS'
 for src in ['waymo','drone','quad']:
  setsource97(src);base=base97(src);m,S=model97(base,src);fp=fingerprint(base)
  for e in [e for e in reg if e['source']==src]:
   tgt,sid=e['target'],e['row']['id'];name=src+'_to_'+tgt;D=root/name
   for sub in ['samples_PRIVATE','features_PRIVATE','states_PRIVATE','records_PRIVATE']:(D/sub).mkdir(parents=True,exist_ok=True)
   z0,z,d,route,old_ev,groups,old=inputs136(e);ev=load(B/'evidence_PRIVATE'/name/(sid+'.npz'));info=rd(B/'evidence_PRIVATE'/name/(sid+'.json'));prev=load(R/'crossplatform/round136/method'/name/'samples_PRIVATE'/(sid+'.npz'));prevps=dict(zip(prev['arm_names'],prev['scores']));scores={k:prevps[k].copy() for k in ['Frozen_A','Frozen_F','AB_mean','R135_A','R135_F','C_B_A','wrong_C_B_A','direct_C_B_A']};ff={};ss={};records={};Sg=torch.as_tensor(ev['S'],device='cuda');cache={}
   for level in LEVELS:
    Cg=torch.as_tensor(ev[level+'_C'],device='cuda');eligible=bool(ev[level+'_eligible']);scores[level+'_direct']=ev[level+'_direct'];scores[level+'_wrong_direct']=ev[level+'_wrong_direct'];key=ev[level+'_C'].tobytes()
    for kind in ['correct','wrong']:
     tag=level+'_'+kind
     if not eligible:scores[tag]=scores['Frozen_A'].copy();records[tag]={'reused':'completeFrozenA_no_evidence'};continue
     if kind=='wrong' and not bool(ev[level+'_effective_wrong']):scores[tag]=scores[level+'_correct'].copy();records[tag]={'reused':level+'_correct'};continue
     if not args.engineering and info[level]['same_R136_evidence']:
      scores[tag]=prevps['C_B_A' if kind=='correct' else 'wrong_C_B_A'].copy();records[tag]={'reused':'R136_'+kind};reused+=1;continue
     if not args.engineering and (key,kind) in cache:
      prevtag=cache[key,kind];scores[tag]=scores[prevtag].copy();records[tag]={'reused':prevtag};reused+=1;continue
     target=torch.as_tensor(ev[level+('_wrong_target' if kind=='wrong' else '_target')],device='cuda')
     def task(lp):
      pt=taskloss(lp,groups,Sg,Cg,None,'partial');gm=torch.stack([lp[torch.as_tensor(g,device='cuda')].max() for g in groups]);ls=gm[Cg].log_softmax(0);rank=(target*(target.clamp_min(1e-300).log()-ls)).sum();return pt+rank,{'partial':pt,'conditional_B_KL':rank}
     zero,p,feat,states,rec=fit136(base,m,S,z,d,route,task,args.engineering);assert abs(zero-scores['Frozen_A']).max()<1e-7;scores[tag]=p;records[tag]=rec;nfit+=1;cache[key,kind]=tag
     for k,v in feat.items():ff[tag+'_'+k]=v
     for k,v in states.items():ss[tag+'_'+k]=v
   restore(m,S);freeze91(m);arr=np.stack(list(scores.values()));assert np.isfinite(arr).all() and np.allclose(arr.sum(1),1,atol=1e-9);np.savez_compressed(D/'samples_PRIVATE'/(sid+'.npz'),scores=arr,arm_names=np.array(list(scores)),queries=arr.argmax(1),source_boxes=z0['boxes']);np.savez_compressed(D/'features_PRIVATE'/(sid+'.npz'),**ff);np.savez_compressed(D/'states_PRIVATE'/(sid+'.npz'),**ss);dump(D/'records_PRIVATE'/(sid+'.json'),{'fits':records,'task_GT_used':False,'source_initialization_only':True});n+=1
   if args.engineering or n%32==0:print('METHOD140',args.engineering,src,n,'newfits',nfit,'reused',reused,flush=True)
  assert fingerprint(base)==fp and all(np.array_equal(v,state(m)[k]) for k,v in S.items());del base,m;torch.cuda.empty_cache()
 dump(root/'SEALED.json',{'n':n,'new_fits':nfit,'compatible_historical_or_factorial_fits_reused':reused,'GT_read':False,'primary':'fullreader_ownsource_correct','files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()},'FINAL_access':False});print('METHOD140_COMPLETE',n,nfit,reused,flush=True)
if __name__=='__main__':main()
