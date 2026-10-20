"""Aggregate costs only from a complete, hash-sealed TubeDETR population."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stvg.runtime import sha
from stvg.tube_runner import write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = Path(args.predictions).resolve()
    seal = json.loads((root/'sealed.json').read_text())
    if seal['queries'] != len(seal['files']):
        raise ValueError('Incomplete seal')
    costs = []
    for entry in seal['files'].values():
        for kind in ['path', 'receipt_path']:
            path = (root/entry[kind]).resolve()
            expected = entry['sha256' if kind == 'path' else 'receipt_sha256']
            if not path.is_relative_to(root) or sha(path) != expected:
                raise ValueError('Sealed output/receipt mismatch')
        costs.append(json.loads((root/entry['receipt_path']).read_text())['cost'])
    def average(field):
        return float(np.mean([c[field] for c in costs]))
    result = dict(status='complete_population_efficiency', queries=len(costs),
        Source=dict(mean_total_seconds=average('source_total_seconds'), mean_adaptation_seconds=0.,
                    mean_peak_allocated_bytes=average('source_peak_bytes'),
                    max_peak_allocated_bytes=max(c['source_peak_bytes'] for c in costs)),
        Ours=dict(mean_total_seconds=average('total_seconds'), mean_adaptation_seconds=average('adaptation_seconds'),
                  mean_peak_allocated_bytes=average('peak_combined_allocated_bytes'),
                  max_peak_allocated_bytes=max(c['peak_combined_allocated_bytes'] for c in costs)),
        mean_decode_seconds=average('decode_seconds'), mean_source_forward_seconds=average('source_seconds'),
        mean_strong_seconds=average('strong_seconds'),
        mean_expert_forward_and_postprocess_seconds=float(np.mean([c['teacher']['expert_seconds'] for c in costs])),
        teacher_startup_seconds_charged=sum(c['teacher_startup_seconds'] for c in costs),
        source_startup_seconds_charged=sum(c['source_load_seconds'] for c in costs),
        total_includes_ReferDINO=True, total_includes_decoding=True,
        memory_definition='torch allocated; max over serial parent stages and teacher peak plus idle parent allocation; not NVML reserved',
        Source_timing_scope='native inference measured during paired capture, including its capture instrumentation',
        manifest_sha256=seal['manifest_sha256'])
    write(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
