"""Synthetic postscore bridge contract/tiny-target audit; no real data/model."""
from pathlib import Path
import sys,json,numpy as np,hashlib
P=Path(__file__).resolve().parents[3];sys.path.insert(0,str(P/'crossplatform/round9'))
from run_pseudo_exactfit import validate_direct_targets,ARM_NAMES
from score_pseudo_exactfit_bridge import ARMS
assert tuple(ARM_NAMES)==ARMS
s=np.array([1,2,3,2,2,2],np.float32);targets=np.tile(s,(2,1));targets[0,0]+=np.float32(1e-6)
assert validate_direct_targets(s,targets) is False
r=(s-targets[0]).astype(float);shift=np.linalg.norm(targets[0,:3].astype(float)-s[:3]);relative=np.linalg.norm(r[:3])/shift
assert abs(r[:3]).max()<5e-5 and relative==1.0,'Absolute tolerance alone can pass unchanged source for tiny target'
invalid=s.copy();invalid[3]=-1;assert validate_direct_targets(invalid,np.tile(invalid,(2,1))) is True
bad=targets.copy();bad[0,3]=3
try:validate_direct_targets(s,bad)
except AssertionError:pass
else:raise AssertionError('Sizeprior change accepted')
out={'state':'synthetic_CPU_pseudo_exactfit_bridge_contract_pass','three_actual_arm_names_match':True,'tiny_nonzero_target_source_unchanged_can_pass_absolute_tolerance':True,'relative_residual_required_with_defined_denominator':True,'invalid_source_explicit_not_applicable':True,'changed_sizeprior_rejected':True,'no_GT_data_model_GPU':True,'source_sha256':hashlib.sha256((P/'crossplatform/round9/run_pseudo_exactfit.py').read_bytes()).hexdigest()};p=Path(__file__).with_name('CPU_PSEUDO_EXACTFIT_BRIDGE_FIXTURE.json');assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
