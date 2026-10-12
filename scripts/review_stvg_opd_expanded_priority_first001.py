"""Bounded CPU readback of six accepted C first-formal fits; no GT or model call."""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_opd_expanded_after_p8_priority001 import BASE,EXP,verify,stopped
from scripts.stvg_opd_expanded_state_common_v1 import read,write,sha,load,audit,scientific,stages,QUERY

def run():
    verify();stopped()
    import torch
    torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    from methods.decota_final_simplified_v1.observations import uniform_positions
    name='E_hc2_cross_clean';condition='clean';order='order1';stage=stages()[name]
    qualified=EXP/'qualification'/name
    qb=read(qualified/'PREDICTION_BARRIER.json');assert qb['status']=='sealed' and not qb['GT_read']
    source=load(qualified/'SOURCE_INITIAL.pt');previous={};bindings={};inputs={}
    counts=dict(complete_formal_fits=0,complete_qualified_formal_pairs=0,complete_fit_rounds=0,
                state_coordinates=0,bitwise_pair_coordinates=0,bitwise_scalar_values=0,
                second_prediction_hash_checks=0,second_input_hash_checks=0,prediction_bytes=0)
    for at in range(2):
        for arm in ['full','alpha0','alpha1']:
            path=EXP/'stages'/name/condition/order/arm/f'{at:05}.pt'
            receipt=read(path.with_suffix('.json'));z=load(path)
            assert not receipt['GT_read'] and not z['GT_read'] and not z['fit']['GT_used']
            assert sha(path)==receipt['sha256'] and path.stat().st_size==receipt['bytes']
            assert z['runtime_sha256']==receipt['runtime_sha256']==sha(EXP/'RUNTIME_LOCK.json')
            assert z['config']==stage['variant_configs'][arm] and z['query_ordinal']==stage['orders'][order][at]
            assert z['Adam_reset'] and z['query_reset'] and z['Native_WHEN_fixed']
            assert z['fit']['active_parameters']==1792 and z['fit']['selected_step']==z['config']['steps']
            inp_path=ROOT/z['input']['path'];assert sha(inp_path)==z['input']['sha256']
            inp=normalized(load(inp_path));expert=unpack_expert(inp['expert'])
            assert z['interval']==inp['interval']
            assert expert['positions4']==uniform_positions(inp['frame_ids'],inp['indices'],4)
            assert audit(z['fit'],expert)==z['math_audit']
            assert z['compute']['actual_new_DINO_calls']==0 and z['compute']['nominal_observation_budget']==4
            for key,value in z['fit']['initial'].items():
                expected=source[key] if at==0 else previous[arm]['committed'][key]
                if key==QUERY:expected=torch.zeros_like(expected)
                assert torch.equal(value,expected)
                alpha=z['config']['writeback']
                end=z['fit']['state'][key]
                commit=torch.zeros_like(value) if key==QUERY else end.clone() if alpha==1 else value+(end-value)*alpha
                assert torch.equal(commit,z['committed'][key])
                counts['state_coordinates']+=value.numel()
            prev_path=path.with_name('00000.pt')
            assert z['previous_payload_sha256']==(None if at==0 else sha(prev_path))
            qpath=qualified/condition/order/arm/path.name
            assert sha(qpath)==qb['files'][str(qpath.relative_to(ROOT))]
            q=load(qpath);assert q['input']['sha256']==z['input']['sha256']
            pair=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
            equal(scientific(z['fit']),scientific(q['fit']),pair)
            first=read(EXP/'first_formal'/name/condition/order/arm/f'{at:05}.json')
            assert first['status']=='pass' and first['complete_fit_bitwise']==pair
            assert first['qualified_fit_sha256']==sha(qpath)
            counts['complete_formal_fits']+=1;counts['complete_qualified_formal_pairs']+=1
            counts['complete_fit_rounds']+=len(z['fit']['rounds'])
            counts['bitwise_pair_coordinates']+=pair['tensor_coordinates'];counts['bitwise_scalar_values']+=pair['scalar_values']
            counts['prediction_bytes']+=path.stat().st_size
            bindings[str(path.relative_to(ROOT))]=receipt['sha256'];inputs[str(inp_path.relative_to(ROOT))]=z['input']['sha256']
            previous[arm]=z
    for path,digest in bindings.items():assert sha(ROOT/path)==digest;counts['second_prediction_hash_checks']+=1
    for path,digest in inputs.items():assert sha(ROOT/path)==digest;counts['second_input_hash_checks']+=1
    assert counts['complete_formal_fits']==6 and counts['state_coordinates']==10752
    assert not torch.cuda.is_initialized();verify();stopped()
    write(BASE/'ROOT_FIRST_FORMAL_READBACK.json',dict(status='pass',scope='six actually accepted HC/order1 first-formal fits, original full/alpha0/alpha1, complete qualified-formal equality, real source-reset-writeback/Adam-reset/math/input/native and second SHA readback; not full5430 completion or future OOM safety',counts=counts,original_scientific_runtime_sha256=sha(EXP/'RUNTIME_LOCK.json'),review_source_sha256=sha(Path(__file__)),whole_formal_experiment_complete=False,independent_model_Jacobian_replayed=False,new_GPU_fits=0,new_model_forward_calls=0,new_DINO_calls=0,CPU_only=True,GT_read=False,time=time.time()))
    print('C_BOUNDED_FIRST_FORMAL_ROOT_PASS',counts)

if __name__=='__main__':run()
