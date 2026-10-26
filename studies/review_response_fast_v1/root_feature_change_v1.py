"""Verify that ROI controls actually reach representations and output intervals."""
from common import *
import numpy as np

def run():
 validate();gt_guard();start=time.perf_counter();proof={}
 for ds in ['hc2','vidstg']:
  assert read(D/ds/'reporting_receipt.json')['all_arrays_rederived_exact'];seal=read(D/ds/'sealed.json');counts={a:dict(feature_changed=0,posterior_changed=0,interval_changed=0,box_trajectory_changed=0)for a in ROI[2:]}
  for e in seal['files'].values():
   p=e['outputs']['roi'];assert sha(D/p['path'])==p['sha256']
   with np.load(D/p['path'])as z:
    for a,cc in counts.items():cc['feature_changed']+=e['diagnostics']['features_different_from_corrected'][a];cc['posterior_changed']+=e['diagnostics']['posterior_different_from_corrected'][a];cc['interval_changed']+=not np.array_equal(z[a+'_interval'],z['CorrectedROI_interval']);cc['box_trajectory_changed']+=not np.array_equal(z[a],z['CorrectedROI'])
  rec=dict(queries=192,counts=counts,proof='Actual independently audited feature/posterior equality flags, NPZ interval/final trajectory equality exactly compared. Not task accuracy or a change of any control.',audit_sha256=sha(D/ds/'reporting_receipt.json'),code_sha256=sha(Path(__file__)),new_GT_scoring=0,new_Source=0,new_model=0);atomic(P/'roi'/ds/'actual_control_change_counts_v1.json',rec);proof[ds]=rec
 atomic(D/'root_feature_change_execution_v1.json',dict(worker=identity(),elapsed_wall_seconds=time.perf_counter()-start,datasets=proof));print(json.dumps(proof))
if __name__=='__main__':run()
