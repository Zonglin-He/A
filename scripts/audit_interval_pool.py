"""Read-only audit of saved physical distributions/call budgets; never read GT."""
import json
import os
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.interval_pool.confirmation import validate, BASE


def main():
    domains = {}; worst = 0.; views = 0
    for ds in ('hc2', 'vidstg'):
        receipts = validate(ds); tr = BASE / ('recheck_' + ds)
        selection = json.loads((BASE / 'confirmation_selection.json').read_text())
        order = [r['query_id'] for r in selection['development_recheck_selection'][ds]]
        first = receipts[order[0]]
        old = [r['old_formal_compatibility'] for r in receipts.values()
               if r['old_formal_compatibility'] is not None]
        assert all(r['Source_bit_exact'] and r['positions_bit_exact'] for r in old)
        for r in receipts.values():
            pack = torch.load(tr / r['posterior_path'], map_location='cpu', weights_only=False)
            assert pack['GT_read'] is False
            U = list(map(int, pack['ids'])); lookup = {x: i for i, x in enumerate(U)}
            q = np.zeros((len(U), len(U)), np.float64)
            for layer in pack['views']:
                for view in layer:
                    p = np.asarray(view['p']); ij = view['ij']; ids = view['ids']
                    assert np.isfinite(p).all() and (p >= 0).all() and abs(p.sum()-1) < 1e-12
                    for k in range(len(p)):
                        i, j = lookup[int(ids[ij[0, k]])], lookup[int(ids[ij[1, k]])]
                        assert i < j and U[j]+1 > U[i]
                        q[i, j] += p[k]/12
                    views += 1
            error = float(np.max(abs(q-pack['mix']))); worst = max(worst, error)
            assert error < 1e-12 and abs(q.sum()-1) < 1e-12
            assert np.array_equal(pack['hull'], np.mean(pack['hull_layers'], axis=0))
        domains[ds] = dict(queries=len(receipts), previous_formal_Source_and_positions_bitexact=len(old),
            new_Source224_captures=sum(r['Source224_captures'] for r in receipts.values()),
            new_eight_frame_expert_forwards=sum(r['ReferDINO_query_forwards'] for r in receipts.values()),
            Source448_calls=sum(r['new_Source448_calls'] for r in receipts.values()),
            extra_decoder_calls=sum(r['extra_temporal_decoder_calls'] for r in receipts.values()),
            updates=sum(r['updates'] for r in receipts.values()),
            final_layer_head_recoveries=sum(r['head_final_layer_exact'] for r in receipts.values()),
            time_decoder_calls=sum(r['capture_cost']['native_time_decoder_calls'] for r in receipts.values()),
            position_decoder_calls=sum(r['capture_cost']['native_position_decoder_calls'] for r in receipts.values()),
            measured_first_query=dict(query_wall_seconds=first['query_wall_seconds'],
                capture_seconds=first['capture_cost']['capture_wall_seconds'],
                Source_load_seconds=first['capture_cost']['Source_load_seconds'],
                expert_load_seconds=first['capture_cost']['ReferDINO_load_seconds']),
            remaining31_mean_query_wall_seconds=float(np.mean([receipts[k]['query_wall_seconds'] for k in order[1:]])),
            remaining31_mean_capture_seconds=float(np.mean([receipts[k]['capture_cost']['capture_wall_seconds'] for k in order[1:]])),
            cold_scope='First query includes lazy Source/expert load. Python/import/runtime asset hashing and process setup precede timing; not cold application end-to-end.')
    assert not torch.cuda.is_initialized()
    result = dict(status='pass', GT_read=False, new_paths_or_model_calls=0,
        engineering_tests=105, domains=domains, mapped_normalized_views=views,
        independent_scatter_max_abs_error=worst, all_source_config_formal_hashes_unchanged=True,
        scope='Hash-validated sealed files, all12 views independently mapped with physical exclusive endpoints; old Source/positions bitwise comparisons where historical output exists. No new GT, model, DP, or MBR decisions.')
    (ROOT / 'results/interval_pool/engineering.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
