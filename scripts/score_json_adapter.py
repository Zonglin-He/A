"""CPU-only output serialization adapter; frozen scorer and predictions unchanged."""
import argparse, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path.cwd().resolve()
assert (ROOT / 'stvg/evaluation.py').is_file(), 'Run this adapter from the project root'
sys.path.insert(0, str(ROOT))

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda: f.read(8 * 2**20), b''): h.update(part)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['hc2', 'vidstg'], required=True)
    parser.add_argument('--work-dir', default='outputs/maintenance/scoring_json_adapter')
    args = parser.parse_args()
    work_dir = (ROOT / args.work_dir).resolve()
    assert work_dir.is_relative_to((ROOT / 'outputs').resolve()), 'Scoring scratch must remain private under outputs'
    work_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('This adapter requires CUDA_VISIBLE_DEVICES to be empty')
    import numpy as np
    from tools import run_formal as formal
    from stvg import evaluation
    plan_path = ROOT / 'outputs/formal/plan.json'
    plan = json.loads(plan_path.read_text())
    state = json.loads((ROOT / 'outputs/formal/state.json').read_text())
    plan_sha = hashlib.sha256(formal.canonical(plan)).hexdigest()
    assert state['plan_sha256'] == plan_sha
    assert plan == formal.make_plan(ROOT), 'Frozen source/configuration/labels changed'
    job = next(j for j in plan['jobs'] if j['dataset'] == args.dataset)
    output = ROOT / job['output']; destination_metrics = output / 'metrics.json'
    metrics = work_dir / (args.dataset+'_metrics.pending.json')
    assert not destination_metrics.exists() and not metrics.exists(), 'Refusing to replace any prior score'
    resume_hash = formal.validate_worker(ROOT, job, require_seal=True)
    assert digest(ROOT / job['labels']) == job['labels_sha256']
    seal_sha = digest(output / 'sealed.json')
    scorer_sha = digest(ROOT / 'stvg/evaluation.py')
    started = datetime.now(timezone.utc).isoformat()
    converted = []
    def scalar_default(value):
        if isinstance(value, np.generic):
            native = value.item()
            if type(native) not in (bool, int, float):
                raise TypeError('Unsupported NumPy scalar type: '+type(value).__name__)
            converted.append(type(value).__name__)
            return native
        raise TypeError('Unsupported JSON value: '+type(value).__name__)
    def dumps(value, *a, **kw):
        assert 'default' not in kw
        return json.dumps(value, *a, default=scalar_default, **kw)
    original = evaluation.json
    evaluation.json = SimpleNamespace(loads=json.loads, dumps=dumps)
    try:
        result = evaluation.score(ROOT / job['manifest'], ROOT / job['labels'], output, metrics)
    finally:
        evaluation.json = original
    def normalize(value):
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, dict): return {k: normalize(v) for k,v in value.items()}
        if isinstance(value, list): return [normalize(v) for v in value]
        return value
    saved = json.loads(metrics.read_text())
    assert saved == normalize(result), 'Serialization changed numerical results'
    assert saved['status'] == 'scored_sealed_predictions'
    assert saved['datasets'][args.dataset]['queries'] == job['queries']
    assert saved['official_dense_checks'] == 5 * job['queries']
    records = evaluation.read_rows(ROOT / job['manifest'])
    assert len(saved['rows']) == len(records) == len({r['id'] for r in saved['rows']})
    assert {r['id'] for r in saved['rows']} == {r['id'] for r in records}
    assert all(set(r['values']) == {'Source','Spatial','Temporal','WithoutOPD','Full'} for r in saved['rows'])
    assert digest(output / 'sealed.json') == seal_sha
    assert digest(ROOT / 'stvg/evaluation.py') == scorer_sha
    assert formal.validate_worker(ROOT, job, require_seal=True) == resume_hash
    assert plan == formal.make_plan(ROOT)
    from collections import Counter
    os.replace(metrics, destination_metrics)
    metrics = destination_metrics
    receipt = dict(schema=1, status='passed', dataset=args.dataset,
        started_utc=started, finished_utc=datetime.now(timezone.utc).isoformat(),
        adapter_sha256=digest(__file__), frozen_scorer_sha256=scorer_sha,
        frozen_plan_sha256=plan_sha, manifest_sha256=job['manifest_sha256'],
        labels_sha256=job['labels_sha256'], seal_sha256=seal_sha,
        resume_hash=resume_hash, metrics_sha256=digest(metrics),
        converted_scalar_types=dict(Counter(converted)),
        original_scorer_used=True, serialization_only=True,
        all_original_prediction_receipts_verified_before_and_after=True,
        serialized_values_equal_original_scorer_values=True,
        metric_or_prediction_changes=False, new_model_or_expert_calls=0,
        original_dense_checks=saved['official_dense_checks'])
    destination=work_dir / (args.dataset+'_receipt.json')
    destination.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt), flush=True)

if __name__ == '__main__': main()
