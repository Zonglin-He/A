"""Portable P5 arithmetic, added mechanism/cost and anonymous evidence bindings.

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
    from scripts.audit_stvg_opd_p5_public_v2 import audit,aggregate,compare,read,budget_analysis
except ModuleNotFoundError:
    from audit_stvg_opd_p5_public_v2 import audit,aggregate,compare,read,budget_analysis


def run(directory):
    directory=Path(directory);base,data,pool=audit(directory);count=[base['scalar_comparisons']]
    def evidence(name):
        p=directory/name
        if p.exists():return read(p)
        with gzip.open(str(p)+'.gz','rt') as f:return json.load(f)
    binding=read(directory/'CODE_BINDING.json')
    for record in binding['exact_metadata_projection'].values():
        rel=record['public_path'];public=directory/rel.split('results/stvg_opd_p5_complete/2026-10-10/',1)[1]
        raw=public.read_bytes()
        assert len(raw)==record['public_bytes'] and hashlib.sha256(raw).hexdigest()==record['public_sha256']
        original=gzip.decompress(raw) if record['encoding']=='gzip_exact_original_bytes' else raw
        assert len(original)==record['bytes'] and hashlib.sha256(original).hexdigest()==record['sha256']
        count[0]+=4
    compare(budget_analysis(data),read(directory/'ACTUAL_ROOT_BUDGET_CONTRASTS.json'),count,'complete budget/source/order/cost contrasts')
    rows=evidence('ACTUAL_ROOT_FEEDBACK_ROWS.json')['rows'];assert len(rows)==2560
    lookup={(r['stage'],r['condition'],r['order'],r['arm'],r['arrival']):r for rr in data.values() for r in rr}
    assert len(lookup)==2560
    for r in rows:
        compare({k:v for k,v in r.items() if k!='feedback'},lookup[r['stage'],r['condition'],r['order'],r['arm'],r['arrival']],count,'feedback original row')
    selected=[]
    for ds in ['hc2','vidstg']:
        eligible=[r for r in rows if r['dataset']==ds and r['feedback'] and not r['stage'].startswith('P5_unified_')];used=set()
        for kind,group in [('success',sorted(eligible,key=lambda r:(-r['delta_current_v'],r['query_ordinal'],r['condition'],r['order'],r['stage']))),
            ('current_harm',sorted(eligible,key=lambda r:(r['delta_current_v'],r['query_ordinal'],r['condition'],r['order'],r['stage']))),
            ('expert_reward_task_mismatch',sorted([r for r in eligible if r['delta_current_v']<0 and r['feedback']['central_expert_reward_delta']>0],
                key=lambda r:(r['delta_current_v'],r['query_ordinal'],r['condition'],r['order'],r['stage'])))]:
            r=next(v for v in group if v['query_ordinal'] not in used);used.add(r['query_ordinal'])
            selected.append(dict(dataset=ds,stage=r['stage'],condition=r['condition'],arm=r['arm'],kind=kind,
                query_ordinal=r['query_ordinal'],source_id=r['source_id'],order=r['order'],arrival=r['arrival'],
                delta_current_v=r['delta_current_v'],delta_inherited_v=r['delta_inherited_v'],
                expert_reward_delta=r['feedback']['central_expert_reward_delta']))
    compare(selected,read(directory/'ACTUAL_ROOT_CASE_SELECTION.json')['records'],count,'deterministic posthoc cases')
    state=evidence('ACTUAL_ROOT_MATH_STATE_DENSE_READBACK.json');n=state['counts']
    assert state['status']=='pass' and state['GPU_model_optimizer_calls']==0 and state['GT_after_global_seal']
    assert n['logical_arrivals']==n['complete_math_dictionary_exact']==n['input_bindings_checked']==2560
    assert n['state_coordinates']==2560*1792 and n['rounds']==1024*40+1536*10
    assert n['actual_qualified_formal_pairs_bitwise']==20 and n['qualified_new_formal_pairs_bitwise']==16 and n['qualified_original_alias_pairs_bitwise']==4 and n['exact_stream_reuse_rows']==512 and n['new_formal_fits']==2048
    assert n['second_opaque_prediction_input_receipt_checks']==4*2560
    assert n['strict_complete_input_source_checks']==2560 and n['qualified_alias_complete_input_comparisons']==4 and n['strict_original_pre_revision_runtime_receipt_readbacks']==256 and n['strict_original_P0_input_receipt_readbacks']==512
    records=state['records'];assert len(records)==2560;identities=set()
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
    return dict(base,scope='all original P5 anonymous arithmetic plus complete original budget/source/order/cost contrasts, original feedback rows, deterministic cases and opaque bindings; no private fit/GT certification',
        scalar_comparisons=count[0],original_population_scalar_comparisons=base['scalar_comparisons'],
        original_and_public_compressed_byte_bindings_exact=True,
        complete_budget_order_cost_arithmetic=True,complete_feedback_row_bindings=True,all_2560_opaque_metadata_bindings=True,
        private_sample_GT_and_decoder_Jacobian_independently_recomputed=False)


if __name__=='__main__':print(json.dumps(run(sys.argv[1]),sort_keys=True))
