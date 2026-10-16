"""Conditional sensitivity from sealed saved rows; no inference or new labels."""
import hashlib
import json
import math
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEARCH = ROOT / 'artifacts/stvg_soft_ep_coordinate128_v1'
OUT = ROOT / 'artifacts/stvg_component_ablation128_v1/analysis'


def main():
    source_hashes = {}

    def read(path):
        data = path.read_bytes()
        source_hashes[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    blocks = {
        'mu': [f'D{i:03}' for i in range(1, 6)],
        'lr': [f'D{i:03}' for i in range(6, 12)] + ['E012'],
        'sigma': [f'F{i:03}' for i in range(22, 27)],
        'tau': [f'F{i:03}' for i in range(27, 32)],
        'steps': [f'F{i:03}' for i in range(32, 37)],
        'response_ratio': [f'F{i:03}' for i in range(37, 42)],
    }
    results = []
    for coordinate, trials in blocks.items():
        for dataset in ['hc2', 'vidstg']:
            selected = list(trials)
            if coordinate == 'lr' and dataset == 'vidstg':
                selected += ['E013', 'E014'] + [f'F{i:03}' for i in range(15, 22)]
            points = {}
            fingerprints = set()
            for trial in selected:
                base = SEARCH / 'trials' / trial
                design = read(base / 'DESIGN_LOCK.json')
                assert read(base / 'ROOT_READBACK.json')['status'] == 'pass'
                assert read(base / 'CPU_COMPLETION.json')['status'] == 'complete'
                cfg = dict(design['configs'][dataset]['M2'])
                settings = dict(design['response_settings'][dataset]['M2'])
                value = (cfg if coordinate in cfg else settings).pop(coordinate)
                fingerprints.add(json.dumps([cfg, settings], sort_keys=True))
                rows = read(base / 'POSTSEAL_SCORE_ROWS.json')['rows']
                rows = [r for r in rows if r['dataset'] == dataset and r['arm'] == 'TS_M2']
                assert len(rows) == 64 and len({r['case_id'] for r in rows}) == 64
                mean = math.fsum(r['metrics']['m_vIoU'] for r in rows) * 100 / 64
                if value in points:
                    assert abs(points[value]['viou_percent'] - mean) < 1e-10
                    points[value]['trials'].append(trial)
                else:
                    points[value] = dict(value=value, viou_percent=mean, trials=[trial])
            assert len(fingerprints) == 1, (coordinate, dataset, fingerprints)
            ordered = [points[k] for k in sorted(points)]
            best = max(ordered, key=lambda x: x['viou_percent'])
            worst = min(ordered, key=lambda x: x['viou_percent'])
            results.append(dict(coordinate=coordinate, dataset=dataset,
                                other_parameters_fixed=json.loads(next(iter(fingerprints))),
                                points=ordered, best=best, worst=worst,
                                spread_pp=best['viou_percent']-worst['viou_percent']))
    epoch = int(time.time())
    result = dict(status='pass', scope='conditional M2 T+S sensitivity on all 128 development sources',
                  time=epoch, results=results, source_hashes=source_hashes,
                  reader_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  new_GT_read=False, new_inference=False, bootstrap_executed=False,
                  global_importance_claim=False, method_changed=False)
    target = OUT / f'PARAMETER_SENSITIVITY_{epoch}.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print('saved', target.relative_to(ROOT))
    for r in results:
        print(r['coordinate'], r['dataset'], 'spread_pp', round(r['spread_pp'], 9),
              'curve', [(p['value'], round(p['viou_percent'], 6)) for p in r['points']])


if __name__ == '__main__':
    main()
