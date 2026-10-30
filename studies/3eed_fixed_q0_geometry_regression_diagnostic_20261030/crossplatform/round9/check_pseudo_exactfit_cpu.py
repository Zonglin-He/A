"""Pure synthetic target/schema gate only; no sealed data/model/GPU/GT read."""
from pathlib import Path
import ast,json,hashlib
import numpy as np
from run_pseudo_exactfit import validate_direct_targets,ARM_NAMES,FORBIDDEN_LABEL_OR_SCORE_KEYS

def main():
    D=Path(__file__).resolve().parent;ast.parse((D/'run_pseudo_exactfit.py').read_text())
    source=np.array([1,2,3,4,2,1],np.float32);targets=np.tile(source,(2,1));targets[0,0]+=.01
    assert validate_direct_targets(source,targets) is False
    bad=source.copy();bad[3]=-1;badtargets=np.tile(bad,(2,1));assert validate_direct_targets(bad,badtargets) is True
    altered=targets.copy();altered[0,3]+=.1
    try:validate_direct_targets(source,altered)
    except AssertionError:pass
    else:raise AssertionError('size changes must be rejected')
    assert len(ARM_NAMES)==3 and 'target_index_path' in FORBIDDEN_LABEL_OR_SCORE_KEYS and 'scored_path' in FORBIDDEN_LABEL_OR_SCORE_KEYS
    r={'state':'pseudo_exactfit_bridge_preparation_AST_and_pure_array_fixture_pass','arm_names':list(ARM_NAMES),'source_and_constructed_directG_size_prior_gate':True,'invalid_source_size_explicit_fallback':True,'GT_and_scored_input_keys_rejected_by_contract':True,'no_actual_source_cache_prediction_GT_model_or_GPU_loaded':True,'source_sha256':{n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ('run_pseudo_exactfit.py',Path(__file__).name)}}
    (D/'CPU_PSEUDO_EXACTFIT_BRIDGE_FIXTURE.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
