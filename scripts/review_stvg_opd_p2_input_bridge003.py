"""Bounded CPU readback of an actually saved formal Direct prefix; no GT."""
import time
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_stvg_opd_p2_input_bridge003 import REC, verify
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE, activate, read, write, sha
from scripts.decota_matrix_common_v1 import load
from scripts.run_stvg_opd_p1_softmax_recovery_002 import equal


def run():
    import torch
    verify()
    activate()
    from scripts.run_decota_paper_main_v1 import unpack_expert
    from scripts.stvg_opd_paper_later_common_v1 import committed
    from scripts.stvg_opd_legacy_input_bridge003 import normalized
    from vg_tta.stvg_opd_paper_component_audit_v1 import audit
    stage = 'P2_hc2_cross_clean'
    folder = BASE / 'stages' / stage
    # Capture one bounded immutable snapshot. Files still being written are not
    # included; their immutable receipt is the acceptance boundary.
    files = sorted(f for f in folder.glob('clean/order*/direct_L1_GIoU/*.pt')
                   if f.with_suffix('.json').exists())
    assert files, 'No formal prediction has actually been accepted yet'
    histories = {}
    records = []
    first_counts = dict(tensors=0, tensor_coordinates=0, scalar_values=0)
    for f in files:
        rc = read(f.with_suffix('.json'))
        assert sha(f) == rc['sha256'] and f.stat().st_size == rc['bytes']
        assert rc['GT_read'] is False
        z = load(f)
        assert z['stage'] == stage and z['arm'] == 'direct_L1_GIoU'
        assert z['GT_read'] is False and z['query_reset'] and z['Adam_reset']
        assert z['Native_WHEN_fixed'] and z['fit']['active_parameters'] == 1792
        assert z['fit']['selected_step'] == z['config']['steps']
        assert z['input_schema_runtime_sha256'] == sha(REC / 'REVISION_RUNTIME.json')
        inp_path = BASE / z['input']['path']
        assert sha(inp_path) == z['input']['sha256']
        inp = normalized(load(inp_path))
        assert inp['GT_read'] is False and z['interval'] == inp['interval']
        assert audit(z['fit'], unpack_expert(inp['expert'])) == z['math_audit']
        initial = z['fit']['initial']
        assert torch.count_nonzero(initial['spatial.query_residual']) == 0
        expected = committed(initial, z['fit']['state'], z['config']['writeback'])
        assert all(torch.equal(expected[n], z['committed'][n]) for n in expected)
        order = z['order']
        if z['arrival'] == 0:
            assert z['previous_payload_sha256'] is None and order not in histories
            if order == 'order1':
                qualified = load(REC / 'qualification' / stage / 'QUALIFIED_FIRST_FORMAL.pt')
                equal(z['fit'], qualified['fit'], first_counts)
                assert z['math_audit'] == qualified['math_audit']
                assert z['input']['sha256'] == qualified['input']['sha256']
        else:
            previous = histories[order]
            assert z['arrival'] == previous['arrival'] + 1
            assert z['previous_payload_sha256'] == previous['sha256']
            for n in initial:
                if n != 'spatial.query_residual':
                    assert torch.equal(initial[n], previous['committed'][n])
        histories[order] = dict(arrival=z['arrival'], sha256=rc['sha256'], committed=z['committed'])
        records.append(dict(path=str(f.relative_to(BASE)), sha256=rc['sha256'],
                            bytes=rc['bytes'], receipt_sha256=sha(f.with_suffix('.json')),
                            order=order, arrival=z['arrival'], query_ordinal=z['query_ordinal'],
                            input_sha256=z['input']['sha256']))
    write(REC / 'ROOT_RESUME_READBACK.json', dict(
        status='pass', scope='bounded actual formal Direct prefix readback; not P2 scores or full-phase closing',
        stage=stage, accepted_snapshot_predictions=len(records),
        opaque_bytes=sum(r['bytes'] for r in records), records=records,
        first_formal_full_fit_bitwise=first_counts, reset_inheritance_writeback=True,
        original_direct_math_dictionaries_exact=True, Native_WHEN_fixed=True,
        root_qualification_sha256=sha(REC / 'ROOT_QUALIFICATION_READBACK.json'),
        input_runtime_sha256=sha(REC / 'REVISION_RUNTIME.json'),
        CPU_readback_script_sha256=sha(Path(__file__)),
        no_GT_scoring=True, GT_read=False, new_model_fits=0,
        P2_global_seal=False, paper_suite_complete=False, time=time.time()))
    print(dict(status='pass', accepted_snapshot_predictions=len(records),
               opaque_bytes=sum(r['bytes'] for r in records), GT_read=False))


if __name__ == '__main__':
    run()
