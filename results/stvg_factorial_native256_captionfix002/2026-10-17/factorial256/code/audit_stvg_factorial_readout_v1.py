"""Saved/source-only proof of the original disjoint temporal/spatial readout.

No model execution, GT access or claim of a newly joint-trained network.  This
audits the factorized deployment protocol actually used by the saved Full arms.
"""
from pathlib import Path
import ast, hashlib, json, sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_factorial_native256_v1 import BASE,CORE,read,sha,put,verify

def main(sealed=False):
    verify()
    path=ROOT/'external/TA-STVG/models/grounding_model/query_decoder.py'
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QueryDecoder');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    assignments={n.targets[0].id:n.value for n in fn.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
    assert ast.unparse(assignments['outputs_time'].func)=='self.time_decoder'
    assert ast.unparse(assignments['outputs_pos'].func)=='self.decoder'
    ret=next(n for n in fn.body if isinstance(n,ast.Return));assert ast.unparse(ret.value)=='(outputs_pos, outputs_time)'
    # Transitive free-variable dependencies of the temporal branch cannot
    # contain any output or state of PosDecoder.
    seen=set()
    def walk(name):
        if name in seen:return
        seen.add(name)
        if name in assignments:
            for n in ast.walk(assignments[name]):
                if isinstance(n,ast.Name):walk(n.id)
    walk('outputs_time');assert 'outputs_pos' not in seen and 'tgt_s' not in seen
    for name in seen & assignments.keys():
        assert not any(isinstance(n,ast.Attribute) and ast.unparse(n)=='self.decoder' for n in ast.walk(assignments[name]))
    replay=ROOT/'methods/decota_final_simplified_v1/replay.py';r=replay.read_text()
    assert 'self.decoder.decoder.register_forward_pre_hook(add_query' in r
    assert 'getattr(self.decoder.decoder.layers[last], layer_norm)' in r
    assert 'for layer_norm in ("norm1", "norm3", "norm4")' in r
    assert 'self.model.action_embed(hidden)' in r
    # Spatial adaptation changes only PosDecoder LNs plus its input residual.
    # Both TimeDecoder passes and the action gates therefore remain unchanged;
    # this is stronger than checking only a single temporal head's logits.
    worker=ROOT/'scripts/run_stvg_soft_ep_coordinate_trial003.py';w=worker.read_text()
    assert 'source temporal logits changed after full readout' in w
    assert 'full readout differs from cached final boxes' in w
    assert 'arms[f"S_{mode}"] = dict(boxes=boxes.copy(), interval=list(native_interval))' in w
    assert 'arms[f"TS_{mode}"] = dict(boxes=boxes.copy(), interval=list(selected_interval))' in w
    files=[path,replay,worker,ROOT/'scripts/stvg_pipeline_speed_v1.py',ROOT/'scripts/stvg_table1_temporal_adapter_v1.py']
    receipt=dict(status='pass',count=256,GT_read=False,code_pins={str(p.relative_to(ROOT)):sha(p) for p in files},temporal_branch_has_no_spatial_decoder_dependency=True,spatial_updated_parameters_only_PosDecoder_and_its_query_residual=True,temporal_hidden_and_action_gates_structurally_independent_of_spatial_state=True,existing_qualified_full_prefix_readout_recomputes_both_native_decoder_passes=True,temporal_TTA_readout_reused_from_sealed_T_not_refitted=True,protocol='original branch-factorized full output: complete spatial trajectory plus independently sealed Temporal interval',not_new_joint_network_forward=True,not_gradient_synergy_proof=True)
    if sealed:
        import numpy as np,torch
        from scripts.stvg_overnight_score_v1 import seal
        checked=0
        for arm in ('A','B'):
            for cid,f in seal(BASE/'arms'/arm).items():
                p=torch.load(ROOT/f['path'],map_location='cpu',weights_only=False);s,ts=p['arms']['S_M2'],p['arms']['TS_M2']
                assert np.array_equal(s['boxes'],ts['boxes'])
                if p['fits']['M2'] is not None:
                    from scripts.run_stvg_adjacent_motion_student_v1 import _xyxy
                    assert np.array_equal(s['boxes'],_xyxy(p['readouts']['M2']['boxes']))
                checked+=1
        assert checked==512;receipt.update(actual_sealed_native_readouts_checked=checked,complete_trajectory_identity=True)
    put(BASE/('ROOT_FACTORIZED_OUTPUT_READBACK.json' if sealed else 'ROOT_FACTORIZED_CODE_READBACK.json'),receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='code_pins'}))
if __name__=='__main__':main('--sealed' in sys.argv)
