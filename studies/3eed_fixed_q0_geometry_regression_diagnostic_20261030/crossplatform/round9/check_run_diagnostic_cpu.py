"""AST, normal11arm and private-cache serialization fixture; no real labels."""
from pathlib import Path
from types import SimpleNamespace
import tempfile,hashlib,json,ast
import numpy as np
import torch
from run_diagnostic import serialize_npz,empty_G,save_source_cache,MODES,GT_ARM_NAMES

def main():
    D=Path(__file__).resolve().parent;ast.parse((D/'run_diagnostic.py').read_text())
    assert not torch.cuda.is_initialized();source=np.zeros((256,6),np.float32);source[:,3:]=[2,1,1];scores=np.arange(256,0,-1,dtype=np.float32);q0=0
    gs=[empty_G(source,q0,m,'synthetic') for m in MODES];arms=np.tile(source[0],(11,1))
    z=serialize_npz(sample_id='synthetic',source=source,scores=scores,top=np.arange(10),q0=0,G_results=gs,selected_arms=arms,statuses=['synthetic']*11,prepared=None,cosines=np.empty((0,2)),point_sha='0'*64,index_sha='1'*64,image_sha='2'*64)
    assert z['arm_boxes_raw6'].shape==(11,6) and not z['arm_query_index'].any() and not z['pseudo_reliability6'][:,3:].any()
    assert len(GT_ARM_NAMES)==13
    fake=SimpleNamespace(base_xyz=torch.zeros(1,256,3),center_features=torch.zeros(1,288,256),size_features=torch.ones(1,288,256),boxes_raw6=torch.from_numpy(source)[None],text_scores=torch.from_numpy(scores)[None],query_index=q0,selected_parameter_sha256='3'*64)
    with tempfile.TemporaryDirectory(prefix='synthetic_cache_',dir=D) as temp:
        p=Path(temp)/'source_cache/synthetic.npz';save_source_cache(p,fake,sample_id='synthetic',point_sha='0'*64,index_sha='1'*64,model_text='synthetic model text',manifest_row={'id':'synthetic'},checkpoint_sha='4'*64)
        with np.load(p,allow_pickle=False) as saved:
            assert saved['center_features'].shape==(1,288,256) and saved['size_features'].dtype==np.float32
            assert saved['boxes_raw6'].shape==(1,256,6) and saved['text_scores'].shape==(1,256) and saved['query_index']==0
            assert saved['actual_model_text_sha256'].shape==saved['manifest_row_sha256'].shape==(32,)
            assert str(saved['selected_parameter_sha256'])=='3'*64
    assert not torch.cuda.is_initialized()
    r={'state':'R9_AST_normal11arm_privatecache_and_GT13names_synthetic_pass','normal_all_arm_queries_equal_q0':True,'normal_size_w0':True,'privatecache_safe_npz_no_pickle_roundtrip':True,'GT13_names':list(GT_ARM_NAMES),'no_actual_GT_index_or_model_or_GPU_used':True,'source_sha256':{n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ('run_diagnostic.py',Path(__file__).name)}}
    (D/'CPU_R9_RUNNER_FIXTURE.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
if __name__=='__main__':main()
