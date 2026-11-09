"""Two hash-selected real cached inputs; rollback and readout checks, no GT."""
import argparse,time,torch
from common58 import *
import numerics58 as n

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);a=ap.parse_args();p=read(a.plan);pin(p);torch.set_num_threads(4);torch.manual_seed(0);rows=read(ROOT/p['manifest']);initial=load(ROOT/p['AB_root']/'SOURCE_MATCHING_MLP_PRIVATE.npz');out=ROOT/'research/round58/engineering_attempt001';assert not out.exists();out.mkdir(parents=True);start=time.perf_counter();records=[]
 for pl in ['drone','quad']:
  r=min([r for r in rows if r['platform']==pl],key=lambda r:hashlib.sha256(('R58_engineering_v1|'+r['id']).encode()).hexdigest());sid=r['id'];rel='samples_PRIVATE/'+sid+'.npz';v=load(ROOT/p['AB_root']/rel);c=load(ROOT/p['capture_root']/rel);g=load(ROOT/p['association_root']/rel);d=n.prepared(v,c,g);old=load(ROOT/p['r57_prediction_root']/rel);source=n.head(initial).requires_grad_(False)
  with torch.no_grad():pp=n.ensemble(source,d).numpy();baseline=old['scores'][list(old['arm_names']).index('frozen_ABC')];err=float(abs(pp-baseline).max());assert int(pp.argmax())==int(baseline.argmax()) and err<3e-5
  results={}
  for rule in ['original','backtrack']:
   s=n.head(initial).requires_grad_(True);t=n.head(initial).requires_grad_(False);opt=n.optimizer(s);trace=[n.step(s,t,opt,d,'ema',rule) for _ in range(3)];results[rule]=trace
  sh=n.phash(s);th=n.phash(t);oh=n.ohash(opt);original_test=n.permitted
  try:
   n.permitted=lambda before,after:False;reject=n.step(s,t,opt,d,'ema','backtrack');assert not reject['accepted'] and reject['trial_count']==9 and n.phash(s)==sh and n.phash(t)==th and n.ohash(opt)==oh
  finally:n.permitted=original_test
  records.append({'platform':pl,'CPU_frozen_bridge_max_absolute_error':err,'source_top1_bridge_exact':True,'original_first_KL_before':results['original'][0]['KL_before'],'original_first_KL_after':results['original'][0]['KL_after_same_detached_target'],'backtrack_first_scale':results['backtrack'][0]['scale'],'all_accepted_backtrack_losses_decrease':all(not r['accepted'] or r['KL_after_same_detached_target']<r['KL_before'] for r in results['backtrack']),'forced_rejection_nonempty_optimizer_rollback_exact':True,'forced_rejection_teacher_unchanged':True})
 dump(out/'RESULT.json',{'state':'PASS_R58_REAL_CACHE_ENGINEERING_ROLLBACK_AND_SOURCE_BRIDGE','n':2,'seconds':time.perf_counter()-start,'records':records,'GT_access':False});seal(out,'R58_ENGINEERING_COMPLETE_NO_GT');print(read(out/'RESULT.json'),flush=True)
if __name__=='__main__':main()
