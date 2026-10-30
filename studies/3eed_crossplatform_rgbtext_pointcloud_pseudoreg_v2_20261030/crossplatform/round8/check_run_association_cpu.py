"""AST + synthetic19arm serialization only. No real model/CLIP/data/GT/GPU."""
from pathlib import Path
import ast,hashlib,json,numpy as np
from run_association import serialize_npz,ARM_NAMES,empty_G,MODES

def main():
    D=Path(__file__).resolve().parent
    for n in ('run_association.py',Path(__file__).name):ast.parse((D/n).read_text())
    source=np.zeros((256,6),np.float32);source[:,3:]=[2,1,1];source[:,0]=np.arange(256,dtype=np.float32)
    source[2,3]=-1;scores=np.arange(256,0,-1,dtype=np.float32);q0=0
    results=[empty_G(source,q0,m,'synthetic') for m in MODES]
    results[0].update(associated_q=1,association_only_box=source[1].copy(),pseudo6=source[1].astype(np.float64)+[.01,0,0,0,0,0],raw_consensus_center=source[1,:3].astype(float)+[.05,0,0],r_global=.2,w6=np.array([1,1,1,0,0,0]))
    arms=[source[0]]
    for g in results:arms.extend([source[g['associated_q']],np.asarray(g['pseudo6'],np.float32)]+[source[g['associated_q']]]*4)
    z=serialize_npz(sample_id='synthetic_only',source=source,scores=scores,top=np.arange(10),q0=q0,G_results=results,selected_arms=arms,statuses=['synthetic']*19,prepared=None,cosines=np.empty((0,2)),point_sha='0'*64,index_sha='1'*64,image_sha='2'*64)
    assert z['arm_boxes_raw6'].shape==(19,6) and len(z['arm_names'])==19
    assert z['q_source']==0 and np.array_equal(z['q_association'],[1,0,0])
    assert np.array_equal(z['arm_query_index'],[0]+[1]*6+[0]*12)
    assert z['source_boxes_eval6'][2,3]==np.float32(1e-6) and not z['pseudo_reliability6'][:,3:].any()
    assert z['raw_consensus_center'].dtype==np.float64 and z['pseudo_target6'].dtype==np.float32 and z['pseudo_target6_float64'].dtype==np.float64
    assert z['roi_xyxy'].shape==(0,2,4)
    r={'state':'AST_and_synthetic19arm_schema_pass_no_real_execution','original_q0_and_associated_q1_not_retagged':True,'actual_target32_and_prequantization64_retained':True,'source_and_allarms_same_evalclamp':True,'arm_names':list(ARM_NAMES),'code_sha256':{n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ('run_association.py',Path(__file__).name)},'no_GPU_model_CLIP_GT_realdata_or_score_execution':True}
    (D/'CPU_R8_RUNNER_FIXTURE.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
