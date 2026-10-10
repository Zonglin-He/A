"""Portable P4 arithmetic, added mechanism/cost and anonymous evidence bindings.

Private sample/GT or complete model Jacobian checks are certified by the saved
root evidence, not re-created by this public anonymous package.
"""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
try:
    from scripts.audit_stvg_opd_p4_public_v2 import audit,aggregate,compare,read
except ModuleNotFoundError:
    from audit_stvg_opd_p4_public_v2 import audit,aggregate,compare,read


def run(directory):
    directory=Path(directory);base,data,pool=audit(directory);count=[base['scalar_comparisons']]
    def evidence(name):
        p=directory/name
        if p.exists():return read(p)
        with gzip.open(str(p)+'.gz','rt') as f:return json.load(f)
    binding=read(directory/'CODE_BINDING.json')
    for record in binding['exact_metadata_projection'].values():
        rel=record['public_path'];public=directory/rel.split('results/stvg_opd_p4_complete/2026-10-10/',1)[1]
        raw=public.read_bytes()
        assert len(raw)==record['public_bytes'] and hashlib.sha256(raw).hexdigest()==record['public_sha256']
        original=gzip.decompress(raw) if record['encoding']=='gzip_exact_original_bytes' else raw
        assert len(original)==record['bytes'] and hashlib.sha256(original).hexdigest()==record['sha256']
        count[0]+=4
    mech={}
    for stage,rr in data.items():
        ds=rr[0]['dataset'];mech[ds]={}
        for condition in read(directory/'ROOT_STATISTICS.json')['by_stage'][stage]:
            rows=[r for r in rr if r['condition']==condition]
            metrics=aggregate(rows,['delta_current_v','delta_inherited_v'])[0]['metrics']
            observed=[r['observed_iou_delta'] for r in rows if r['observed_iou_delta'] is not None]
            unobserved=[r['unobserved_iou_delta'] for r in rows if r['unobserved_iou_delta'] is not None]
            mech[ds][condition]=dict(current_v=metrics['delta_current_v'],inherited_v=metrics['delta_inherited_v'],
                observed_positions_iou_mean=math.fsum(observed)/len(observed) if observed else None,
                observed_valid_query_count=len(observed),
                unobserved_positions_iou_mean=math.fsum(unobserved)/len(unobserved) if unobserved else None,
                unobserved_valid_query_count=len(unobserved),
                actual_fit_wall_seconds_per_arrival=math.fsum(r['compute']['fit_GPU_seconds'] for r in rows)/len(rows),
                actual_shared_capture_seconds_per_arrival=math.fsum(r['compute']['shared_capture_seconds'] for r in rows)/len(rows))
    extra=read(directory/'ACTUAL_ROOT_ALL_CONDITION_MECHANISM_COST.json')
    compare(mech,extra['datasets'],count,'all-condition mechanism/cost')
    assert extra['status']=='pass' and extra['inherited_component_is_not_an_alpha0_causal_contrast']
    assert extra['observed_and_unobserved_are_different_frame_populations'] and not extra['cold_deployment_latency']
    rows=evidence('ACTUAL_ROOT_FEEDBACK_ROWS.json')['rows'];assert len(rows)==15504
    lookup={(r['stage'],r['condition'],r['order'],r['arm'],r['arrival']):r for rr in data.values() for r in rr}
    assert len(lookup)==15504
    for r in rows:
        compare({k:v for k,v in r.items() if k!='feedback'},lookup[r['stage'],r['condition'],r['order'],r['arm'],r['arrival']],count,'feedback original row')
    selected=[]
    for ds in ['hc2','vidstg']:
        eligible=[r for r in rows if r['dataset']==ds and r['feedback']];used=set()
        for kind,group in [('success',sorted(eligible,key=lambda r:(-r['delta_current_v'],r['query_ordinal'],r['condition'],r['order']))),
            ('current_harm',sorted(eligible,key=lambda r:(r['delta_current_v'],r['query_ordinal'],r['condition'],r['order']))),
            ('expert_reward_task_mismatch',sorted([r for r in eligible if r['delta_current_v']<0 and r['feedback']['central_expert_reward_delta']>0],
                key=lambda r:(r['delta_current_v'],r['query_ordinal'],r['condition'],r['order'])))]:
            r=next(v for v in group if v['query_ordinal'] not in used);used.add(r['query_ordinal'])
            selected.append(dict(dataset=ds,stage=r['stage'],condition=r['condition'],arm=r['arm'],kind=kind,
                query_ordinal=r['query_ordinal'],source_id=r['source_id'],order=r['order'],arrival=r['arrival'],
                delta_current_v=r['delta_current_v'],delta_inherited_v=r['delta_inherited_v'],
                expert_reward_delta=r['feedback']['central_expert_reward_delta']))
    compare(selected,read(directory/'ACTUAL_ROOT_CASE_SELECTION.json')['records'],count,'deterministic posthoc cases')
    state=evidence('ACTUAL_ROOT_MATH_STATE_DENSE_READBACK.json');n=state['counts']
    assert state['status']=='pass' and state['GPU_model_optimizer_calls']==0 and state['GT_after_global_seal']
    assert n['logical_arrivals']==n['complete_math_dictionary_exact']==n['input_bindings_checked']==15504
    assert n['state_coordinates']==15504*1792 and n['rounds']==3792*40+11712*10
    assert n['actual_qualified_formal_pairs_bitwise']==64 and n['exact_stream_reuse_rows']==0
    assert n['second_opaque_prediction_input_receipt_checks']==4*15504
    records=state['records'];assert len(records)==15504;identities=set()
    barriers={stage:evidence('receipts/stages/'+stage+'/PREDICTION_BARRIER.json') for stage in data}
    for r in records:
        key=r['stage'],r['condition'],r['order'],r['arm'],r['arrival'];assert key not in identities;identities.add(key)
        old=lookup[key];assert r['query_ordinal']==old['query_ordinal']
        barrier=barriers[r['stage']]
        binding=barrier['logical_records'][f"{r['condition']}/{r['order']}/{r['arm']}/{r['arrival']:05}"]
        assert barrier['status']=='sealed' and barrier['GT_read'] is False
        assert r['path']==binding['path'] and r['sha256']==binding['sha256']==barrier['files'][r['path']]
        assert r['input_sha256']==binding['input']['sha256']==barrier['inputs'][r['input_path']]
        count[0]+=5
    return dict(base,scope='all original P4 anonymous arithmetic plus complete added mechanism/cost, original feedback rows, deterministic cases and opaque bindings; no private fit/GT certification',
        scalar_comparisons=count[0],original_population_scalar_comparisons=base['scalar_comparisons'],
        original_and_public_compressed_byte_bindings_exact=True,
        added_mechanism_cost_arithmetic=True,complete_feedback_row_bindings=True,all_15504_opaque_metadata_bindings=True,
        private_sample_GT_and_decoder_Jacobian_independently_recomputed=False)


if __name__=='__main__':print(json.dumps(run(sys.argv[1]),sort_keys=True))
