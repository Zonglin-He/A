"""Supplementary canonical frozen means; no model or GT access."""
from common142 import *
import numpy as np
def main():
 check();assert rd(H/'TTA_AUDIT.json')['status']=='PASS';root=B/'supplementary_mean';root.mkdir(exist_ok=False);n=0
 for e in rd(H/'INPUT_REGISTRY_PRIVATE.json'):
  src,tgt,sid=e['source'],e['target'],e['row']['id'];name=src+'_to_'+tgt;old=np.load(B/'image'/name/'samples_PRIVATE'/(sid+'.npz'));p=dict(zip(old['arm_names'],old['scores']));score=np.stack([(p['Frozen_F']+p['F_Bimage_only_teacher'])/2,(p['F_RGBB_only']+p['F_Bimage_only_teacher'])/2]);assert np.isfinite(score).all() and np.allclose(score.sum(1),1,atol=1e-12);D=root/name;D.mkdir(exist_ok=True);np.savez_compressed(D/(sid+'.npz'),scores=score,arm_names=np.array(['FA_FB_probability_mean','FAB_FB_probability_mean']),queries=score.argmax(1),source_boxes=old['source_boxes']);n+=1
 dump(root/'SEALED.json',{'n':n,'posthoc_strong_control_not_preregistered_with_original_TTA':True,'canonical_weights_fixed_equal_no_GT_search':True,'no_models_new_observations_or_GT_read':True,'FINAL_access':False,'files':{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}});print('FROZEN_MEAN142_SEALED',n,flush=True)
if __name__=='__main__':main()
