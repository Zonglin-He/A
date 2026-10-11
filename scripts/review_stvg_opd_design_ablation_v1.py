"""Actual CPU readback of complete new fits/state chains without label access."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_opd_design_ablation_common_v1 import *


def run(prefix=False):
    verify(); activate_old()
    import torch
    torch.set_num_threads(2)
    from scripts.run_stvg_opd_design_ablation_v1 import validate_lifecycle
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal
    from methods.decota_final_simplified_v1.observations import uniform_positions
    assert read(BASE/'ROOT_QUALIFICATION_READBACK.json')['status']=='pass'
    if not prefix:
        assert read(BASE/'GLOBAL_PREDICTION_BARRIER.json')['status']=='sealed'
        assert read(BASE/'CPU_SCORING_COMPLETION.json')['status']=='all_postseal_math_state_input_dense_scoring_complete'
    dest = BASE/'actual_root'
    runtime_path = dest/'RUNTIME.json'
    pins = {'scripts/review_stvg_opd_design_ablation_v1.py':sha(Path(__file__)),
            'scripts/audit_stvg_opd_design_ablation_public_v1.py':sha(ROOT/'scripts/audit_stvg_opd_design_ablation_public_v1.py')}
    if runtime_path.exists():
        assert read(runtime_path)['pins']==pins
        assert read(runtime_path)['scientific_runtime_sha256']==sha(BASE/'RUNTIME_LOCK.json')
    else:
        write(runtime_path,dict(status='locked_for_actual_CPU_root_readback',pins=pins,
            scientific_runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),
            CPU_projection_runtime_sha256=sha(BASE/'CPU_projection_revision002/REVISION_RUNTIME.json'),
            no_new_science_or_model_calls=True,CPU_only=True,time=time.time()))
    counts = dict(complete_fits=0,fit_rounds=0,state_coordinates=0,matched_inputs=0,
                  qualified_formal_pairs=0,paired_coordinates=0,bytes=0,second_byte_checks=0)
    entries=[]; shared_inputs={}; qualification_inputs={}; tick=time.time()
    for name,stage in stages().items():
        for condition in stage['conditions']:
            for order,sequence in stage['orders'].items():
                for arm in ARMS:
                    previous=None; prevhash=None
                    for at,q in enumerate(sequence):
                        path=BASE/'stages'/name/condition/order/arm/f'{at:05}.pt'
                        if prefix and (not path.exists() or not path.with_suffix('.json').exists()): break
                        receipt=read(path.with_suffix('.json')); digest=sha(path)
                        assert digest==receipt['sha256'] and path.stat().st_size==receipt['bytes']
                        assert not receipt['GT_read'] and receipt['runtime_sha256']==sha(BASE/'RUNTIME_LOCK.json')
                        payload=load(path)
                        assert payload['stage']==name and payload['condition']==condition and payload['order']==order
                        assert payload['arrival']==at and payload['query_ordinal']==q and payload['arm']==arm
                        assert payload['config']==stage['variant_configs'][arm]
                        assert payload['runtime_sha256']==sha(BASE/'RUNTIME_LOCK.json')
                        input_path=ROOT/payload['input']['path']
                        if input_path not in shared_inputs:
                            shared_inputs[input_path]=normalized(load(input_path))
                            assert sha(input_path)==payload['input']['sha256']
                        inp=shared_inputs[input_path]; ex=unpack_expert(inp['expert'])
                        assert inp['interval']==payload['interval']
                        extent=observation_extent(inp['frame_ids'],inp['indices'],arm)
                        assert ex['positions4']==uniform_positions(inp['frame_ids'],extent,4)
                        old_path=ROOT/payload['matched_full_payload_path']
                        assert sha(old_path)==payload['matched_full_payload_sha256']
                        original=load(old_path)
                        assert original['query_ordinal']==q and original['interval']==payload['interval']
                        old_input_path=OLD/original['input']['path']
                        if old_input_path not in shared_inputs:
                            shared_inputs[old_input_path]=normalized(load(old_input_path))
                            assert sha(old_input_path)==original['input']['sha256']
                        old_input=shared_inputs[old_input_path]
                        for field in ['dataset','source','pixel_sha256','corruption_spec','source_model_state_sha256','frame_ids','indices','interval']:
                            assert inp[field]==old_input[field],field
                        assert torch.equal(inp['native_boxes'],old_input['native_boxes'])
                        if arm!='whole_video': assert inp['expert']==old_input['expert']
                        else: assert inp['original_event_input_sha256']==original['input']['sha256']
                        validate_lifecycle(payload,previous,prevhash,original['fit']['initial'] if at==0 else None)
                        assert audit(payload['fit'],ex)==payload['math_audit']
                        if at<2 and order=='order1':
                            qp=BASE/'qualification'/name/condition/order/arm/path.name
                            qualified=load(qp)
                            f=read(BASE/'first_formal'/name/condition/order/arm/f'{at:05}.json')
                            assert f['status']=='pass' and f['qualified_fit_sha256']==sha(qp)
                            assert qualified['input']['sha256']==payload['input']['sha256']
                            compared=dict(tensors=0,tensor_coordinates=0,scalar_values=0)
                            equal(scientific(payload['fit']),scientific(qualified['fit']),compared)
                            assert compared==f['complete_fit_bitwise']
                            counts['qualified_formal_pairs']+=1
                            counts['paired_coordinates']+=compared['tensor_coordinates']
                        counts['complete_fits']+=1; counts['fit_rounds']+=len(payload['fit']['rounds'])
                        counts['state_coordinates']+=1792; counts['matched_inputs']+=1
                        counts['bytes']+=path.stat().st_size
                        previous=payload['committed']; prevhash=digest
                        entries.append(dict(path=str(path.relative_to(ROOT)),sha256=digest,bytes=path.stat().st_size,
                            input_path=str(input_path.relative_to(ROOT)),input_sha256=payload['input']['sha256'],
                            stage=name,condition=condition,order=order,arm=arm,arrival=at,query_ordinal=q))
                        if counts['complete_fits']%128==0:
                            print('DESIGN_ROOT_READBACK',counts['complete_fits'],name,condition,order,arm,at,flush=True)
                        del payload,original
    for entry in entries:
        p=ROOT/entry['path']; assert sha(p)==entry['sha256'] and p.stat().st_size==entry['bytes']
        assert read(p.with_suffix('.json'))['sha256']==entry['sha256']
        counts['second_byte_checks']+=1
    for p,value in shared_inputs.items():
        receipt=read(p.with_suffix('.json')); assert sha(p)==receipt['sha256']
    verify()
    if not prefix:
        assert counts['complete_fits']==5376 and counts['qualified_formal_pairs']==72
        score=read(BASE/'CPU_SCORING_COMPLETION.json')['counts']
        assert counts['fit_rounds']==score['fit_rounds']
        assert counts['state_coordinates']==score['full_state_coordinates']
    target=dest/(f'PREFIX_READBACK_{time.time_ns()}.json' if prefix else 'ROOT_ALL_FITS_READBACK.json')
    write(target,dict(status='pass',scope='bounded formal prefix only' if prefix else 'all new formal complete math/input/1792 state chains and qualified-formal fits',
        counts=counts,unique_input_files=len(shared_inputs),entries=entries,
        actual_new_model_calls=0,GT_read=False,CPU_only=True,
        scientific_runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),root_runtime_sha256=sha(runtime_path),
        independent_complete_decoder_Jacobian=False,elapsed_CPU_seconds=time.time()-tick,time=time.time()))
    print('DESIGN_ROOT_COMPLETE',counts['complete_fits'],str(target),flush=True)


if __name__=='__main__': run(prefix='--prefix' in sys.argv)
