"""Independent parent reductions and paired difference-in-differences."""
import sys,argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from vg_tta.desta3d_v3_oracle_io import OUT,read,write,sha
from scripts.crosscheck_desta3d_v3_oracle_summary import run as common

def run(name):
 common(name);d=OUT/name/'independent_readback_v1';r=read(d/'REPORT.json');mx=0.;nchecks=0
 for b in ['event','spatial']:
  order=sorted(x['source'] for x in r['arms']['original']['rows']);tab={a:{x['source']:x['metrics'] for x in v['rows']} for a,v in r['arms'].items()}
  exclusion=r['correct_minus_wrong'][b+'_early_correct_minus_'+b+'_early_wrong']['excluded_non_discriminating']
  for mode in ['all_parents','eligible_only']:
   use=[p for p in order if mode=='all_parents' or p not in exclusion];n=len(use)
   samples=np.random.default_rng(20260927).integers(0,n,size=(10000,n))
   for m in ['vIoU','sIoU','tIoU']:
    delta=np.array([tab[b+'_early_correct'][p][m]-tab[b+'_early_wrong'][p][m]-tab[b+'_late_correct'][p][m]+tab[b+'_late_wrong'][p][m] for p in use]);old=r['interaction'][b][mode][m]
    vals=[100*delta.mean(),*np.percentile(100*delta[samples].mean(1),[2.5,97.5])];ref=[old['mean_delta_pp'],*old['bootstrap_ci95_pp']]
    e=float(np.max(np.abs(np.array(vals)-ref)));assert e<1e-10;mx=max(mx,e);nchecks+=3
    assert sum(delta>1e-10)==old['positive_parents'] and sum(delta< -1e-10)==old['negative_parents'] and sum(delta<-.05)==old['severe_loss_below_minus5pp']
 write(d/'ROOT_INTERACTION_CROSSCHECK.json',dict(status='passed',values=nchecks,max_error=mx,report_sha=sha(d/'REPORT.json'),no_new_GPU_or_GT=True));print('interaction passed',mx)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(p.parse_args().name)
