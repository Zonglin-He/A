"""Independent numpy aggregation of every sealed v2 source query moment."""
from pathlib import Path
import collections
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.desta3d_v2_p0 import read, sha
from scripts.score_desta3d_v2_aux_recovery import save_once
from scripts.desta3d_tta_run_v1 import scan_nested_gpu_receipts


def main():
    torch.set_num_threads(2)
    out = ROOT / 'artifacts/desta3d_v2/tta_v2/source_moments_B1_v1'
    seal = read(out / 'COMPLETE.json')
    config = read(out / 'CONFIG.json')
    assert seal['status'] == 'completed_source_moments'
    assert seal['optimizer_steps'] == 0 and not seal['GT_read'] and not seal['target_data_read']
    for path, h in {**read(out / 'LOCK.json')['pins'], **seal['pins']}.items():
        assert sha(Path(path)) == h, path
    assert len(seal['pins']) == 619
    inputs = read(out / 'INPUTS.json')
    expected = sorted([r for r in read(ROOT / 'artifacts/desta3d_v2/source_fit/INPUTS.json')
                       if r['split'] == 'train'], key=lambda r: r['key'])
    assert inputs == expected and len(inputs) == 618
    records = collections.defaultdict(lambda: collections.defaultdict(list))
    hashes = []
    for i, row in enumerate(inputs):
        p = out / 'queries' / f'{i:03}.pt'
        assert str(p) in seal['pins']
        x = torch.load(p, map_location='cpu', weights_only=False)
        assert x['key'] == row['key'] and x['source'] == str(row['source'])
        assert x['frame_ids'] == row['input']['frame_ids'] and x['video_sha'] == row['input']['video_sha256']
        assert x['GT_read'] is False and x['preprocess']['grid'][0][0] == len(x['frame_ids'])
        hs = [x['preprocess']['pixel_sha'], x['query_tokens_sha'], x['visual_grid_sha'], x['frame_times_sha']]
        assert all(len(s) == 64 and all(c in '0123456789abcdef' for c in s) for s in hs)
        hashes.append({'key': row['key'], 'pixel_sha': hs[0], 'query_sha': hs[1],
                       'visual_grid_sha': hs[2], 'physical_times_sha': hs[3]})
        for branch in ('referent', 'event'):
            m = x['moments'][branch]
            a, b = (np.asarray(m[k], dtype=np.float64) for k in ('mean', 'second_moment'))
            assert a.shape == b.shape == (128,) and np.isfinite(a).all() and np.isfinite(b).all()
            assert (b >= 0).all() and np.min(b-a*a) >= -1e-5
            records[branch][x['source']].append((a, b))
    saved = read(out / 'MOMENTS.json')
    assert saved['feature_definition'] == 'v2_query_conditioned_readers'
    assert saved['adapter_sha'] == config['checkpoint']['adapter_sha256'] == seal['adapter_unchanged']
    assert saved['checkpoint_sha'] == config['checkpoint']['sha256']
    assert saved['queries'] == 618 and saved['parents'] == 95
    checks = {}
    for branch, dest in [('referent', 'spatial'), ('event', 'event')]:
        parents = records[branch]
        assert len(parents) == 95
        means, seconds, errors = [], [], []
        for parent, items in sorted(parents.items()):
            m = np.stack([x[0] for x in items]).mean(0)
            s = np.stack([x[1] for x in items]).mean(0)
            old = saved['parent_moments'][branch][parent]
            assert old['query_count'] == len(items)
            errors.extend([float(np.max(np.abs(m-old['mean']))), float(np.max(np.abs(s-old['second_moment'])))])
            means.append(m); seconds.append(s)
        mean = np.stack(means).mean(0)
        second = np.stack(seconds).mean(0)
        var = second-mean**2
        assert np.min(var) >= 0
        std = np.sqrt(np.maximum(var, 1e-12))
        global_saved = saved['parent_moments'][branch]['__global__']
        for name, value in [('mean', mean), ('second_moment', second), ('variance', var), ('std', std)]:
            errors.append(float(np.max(np.abs(value-global_saved[name]))))
            if name != 'variance': errors.append(float(np.max(np.abs(value-saved[dest][name]))))
        assert max(errors) < 1e-10, (branch, max(errors))
        checks[dest] = {'parent_count': 95, 'query_count': 618, 'channels': 128,
                       'max_abs_error': max(errors), 'min_variance': float(var.min()),
                       'queries_per_parent': [min(map(len, parents.values())), max(map(len, parents.values()))]}
    total = sum(scan_nested_gpu_receipts(ROOT / 'artifacts' / n)[0] for n in ['desta3d_v1', 'desta3d_v2'])
    result = {'status': 'passed', 'time': time.time(), 'script_sha': sha(Path(__file__)),
        'complete_sha': sha(out / 'COMPLETE.json'), 'moments_sha': sha(out / 'MOMENTS.json'),
        'queries': 618, 'parents': 95, 'branches': checks, 'raw_hashes': hashes,
        'input_and_roster_identity_checked': True,
        'pixels_scope': 'sealed captured pixel and feature hashes verified; no independent video decoding in CPU aggregation',
        'freeze_scope': 'runtime assertions and unchanged adapter digest; not an independent all-backbone rehash',
        'cumulative_actual_GPU_seconds': total, 'cumulative_cap_seconds': None,
        'GT_read': False, 'target_data_read': False, 'GPU_started_by_audit': False}
    save_once(out / 'ROOT_FULL_READBACK.json', result)
    print({k: v for k, v in result.items() if k != 'raw_hashes'})


if __name__ == '__main__':
    main()
