"""Syntax + synthetic sampling/serialization fixture; no model import/GT/GPU."""
import ast,hashlib,json,tempfile
from pathlib import Path
import numpy as np
from runner_helpers import sample_points,array_sha,npz_payload,validate_manifest,ARM_NAMES

def main():
    here=Path(__file__).resolve().parent
    for n in ('run_crossdomain.py','runner_helpers.py','check_runner_cpu.py'):ast.parse((here/n).read_text())
    with tempfile.TemporaryDirectory(prefix='synthetic_runner_',dir=here) as temp:
        t=Path(temp);raw=np.array([[1,2,3,.1],[4,5,6,.2]],np.float32);np.save(t/'synthetic.npy',raw);(t/'synthetic_meta.json').write_text('{}');(t/'synthetic_image').write_bytes(b'fixture-not-decoded')
        row={'id':'synthetic_id','platform':'drone','scene':'synthetic_scene','pointpath':str(t/'synthetic.npy'),'metapath':str(t/'synthetic_meta.json'),'imagepath':str(t/'synthetic_image'),'caption':'the red car','suffix':0,'seed':9}
        validate_manifest([row]);pc,ix,n=sample_points(row);ref=raw[np.random.default_rng(9).choice(2,16384,replace=True),:3].copy();ref[:,2]+=1.8
        assert np.array_equal(pc,ref) and pc.dtype==np.float32 and n==2
        rejected=False
        try:validate_manifest([dict(row,positive_map=[1])])
        except AssertionError:rejected=True
        assert rejected
        source=np.zeros((256,6),np.float32);source[:,3:]=[2,1,1];source[1,3]=-1
        scores=np.arange(256,0,-1,dtype=np.float32);arms=np.tile(source[0],(16,1));targets=np.tile(source[0],(3,1))
        z=npz_payload(sample_id='synthetic_id',source_boxes=source,scores=scores,native_top10=np.arange(10),q=0,selected_arms=arms,statuses=['synthetic_only']*16,pseudo_targets=targets,pseudo_weights=np.zeros((3,6)),point_sha=array_sha(pc),image_sha='0'*64)
        assert len(z['arm_names'])==16 and z['source_boxes_eval6'][1,3]==np.float32(1e-6)
        np.savez_compressed(t/'synthetic.npz',**z)
        with np.load(t/'synthetic.npz',allow_pickle=False) as saved:
            assert all(np.array_equal(v,saved[k]) for k,v in z.items())
    r={'state':'syntax_and_synthetic_runner_helper_pass_no_model_or_GPU_or_real_data','manifest_exact9keys_GT_extra_rejected':True,'R5_float32_Drone_sampling_bridge_bitexact':True,'all16arm_NPZ_safe_roundtrip':True,'raw_to_eval_clamp_checked':True,'actual_model_constructed':False,'GPU_initialized':False,'source_program_sha256':{n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in ('run_crossdomain.py','runner_helpers.py',Path(__file__).name)},'arm_names':list(ARM_NAMES)}
    (here/'CPU_RUNNER_FIXTURE.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
