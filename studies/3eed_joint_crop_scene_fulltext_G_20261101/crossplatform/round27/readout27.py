"""All readouts use fixed masks/weights/boxes and exactly one joint image-text cue."""
import numpy as np
from common27 import *
NAMES=['scene_base','scene_hires','scene_repeat','crop_scene_joint','R26_scene','legacy_complex']
def main():
 out=BASE/'scores528_attempt001';out.mkdir(exist_ok=False);(out/'samples_PRIVATE').mkdir();count=0
 for r in rows():
  pp=paths(r);prep=read(pp['prep']/'cases_PRIVATE'/(r['id']+'.json'));a=load_npz(pp['prep']/'arrays_PRIVATE'/(r['id']+'.npz'));b=load_npz(pp['bank']/'samples_PRIVATE'/(r['id']+'.npz'));e=read(BASE/'joint528_attempt001/cases_PRIVATE'/(r['id']+'.json'));z=feature(r);pool=prep['mask_pool_PRIVATE'];w=a['weights'];cues=[]
  for n in NAMES:
   v=b['vision']+b['position']+b['relation'] if n=='legacy_complex' else w@np.array([e['targets_PRIVATE'][str(m)][n]['log_odds_yes_no'] for m in pool]) if pool else np.zeros(256);cues.append(v)
  np.savez_compressed(out/'samples_PRIVATE'/(r['id']+'.npz'),source=z['scores'],boxes=z['boxes'],cue_names=np.array(NAMES),cues=np.stack(cues),mapped=w.sum(1)>0);count+=1
 dump(out/'VARIANTS.json',NAMES);seal(out,'R27_one_evidence_joint_scores_before_GT',n=count,zero_updates=True);print('READOUT27_SEALED',count)
if __name__=='__main__':main()
