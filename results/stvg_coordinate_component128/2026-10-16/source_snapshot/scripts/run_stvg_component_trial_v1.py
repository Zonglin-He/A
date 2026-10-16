"""One matched M2 deletion; uses existing full-model capture and redecode ABI."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_component_ablation_common_v1 import *
def main():
 from scripts import run_stvg_soft_ep_coordinate_trial003 as w
 from scripts import stvg_soft_ep_coordinate_fit002 as dispatch
 from scripts import stvg_component_spatial_fit_v1 as ablation
 final=verify_final();verify_preparation();trial=sys.argv[1];assert trial in {x[0]for x in SPATIAL}
 design=read(BASE/'trials'/trial/'DESIGN_LOCK.json');row=next(x for x in SPATIAL if x[0]==trial)
 assert design['ablation_flags']==row[2]
 for ds in ('hc2','vidstg'):
  assert design['configs'][ds]['M2']==final['configs'][ds]
  assert design['response_settings'][ds]['M2']==final['response_settings'][ds]
 original_loader=w._load_prior_trial_context
 def prior_loader(prior,hashes):
  saved=w.TRIALS
  try:w.TRIALS=SEARCH/'trials';return original_loader(prior,hashes)
  finally:w.TRIALS=saved
 original_fit=dispatch.fit
 def fit(*args,**kwargs):return ablation.fit(*args,**kwargs,flags=dict(row[2]))
 try:
  w.TRIALS=BASE/'trials';w.MODES=('M2',);w.ARM_NAMES=ARMS;w._load_prior_trial_context=prior_loader;dispatch.fit=fit
  w.main()
 finally:dispatch.fit=original_fit;w._load_prior_trial_context=original_loader
if __name__=='__main__':main()
