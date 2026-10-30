"""Fixed-q0 identity/consensus fixture only; no real target data/model/GT."""
from pathlib import Path
import sys,json,hashlib,copy
import numpy as np
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D.parent/'round8'))
from check_association_geometry_cpu import scene
from association_geometry import prepare_candidates,associate_and_consensus
from fixed_q0_geometry import fixed_q0_pseudo

def main():
    d=scene();p=prepare_candidates(**d);sem=np.array([[.1,.1],[.85,.85]])
    changed=associate_and_consensus(p,sem);fixed=fixed_q0_pseudo(p,sem)
    assert changed['associated_q']==2 and fixed['original_source_q']==fixed['associated_q']==0
    assert fixed['diagnostics']['RGB_mismatch_diagnostic_only'] and fixed['r_global']==0 and not fixed['w6'].any()
    assert np.array_equal(fixed['pseudo6'],d['source_boxes'][0])
    sem2=sem[::-1];old=associate_and_consensus(p,sem2);new=fixed_q0_pseudo(p,sem2)
    assert new['associated_q']==0 and np.array_equal(new['pseudo6'],old['pseudo6']) and np.array_equal(new['w6'],old['w6'])
    assert new['r_global']>0 and new['diagnostics']['nonzero_pseudo_supervision'] and not new['w6'][3:].any()
    pc=fixed_q0_pseudo(p,sem,mode='pc_only');assert pc['associated_q']==0
    missing=copy.deepcopy(p);missing['roi_groups']=[g for g in missing['roi_groups'] if g['group_id']!=int(missing['dedup_membership'][0])];m=fixed_q0_pseudo(missing,np.array([[.8,.8]]));assert m['associated_q']==0 and not m['w6'].any() and m['diagnostics']['reason']=='q0_group_not_in_semantic_ROIs'
    outside=copy.deepcopy(p);outside['dedup_membership'][0]=-1;o=fixed_q0_pseudo(outside,sem);assert o['associated_q']==0 and not o['w6'].any()
    negative=copy.deepcopy(p);negative['source_boxes'][0,3]=-1;n=fixed_q0_pseudo(negative,sem);assert n['associated_q']==0 and n['diagnostics']['reason']=='q0_nonpositive_source_size'
    r={'state':'synthetic_fixed_q0_geometry_pass_no_real_model_data_GT','R8_reselect_removed_even_if_RGB_favors_other_group':True,'same_consensus_as_R8_when_q0_already_selected':True,'missing_q0_prepared_or_ROI_or_valid_size_failclosed_q0':True,'normal_geometry_sizeprior_w0_yaw0':True,'RGB_mismatch_diagnostic_not_newloss_or_reselect':True,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'fixed_q0_geometry.py',Path(__file__)]},'no_GPU_or_GT_read':True}
    (D/'CPU_FIXED_Q0_FIXTURE.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
