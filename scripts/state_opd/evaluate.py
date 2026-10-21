"""CPU-only scoring of a complete, separately sealed HC state-OPD scope.

Importing this module does not read inputs or labels. Scoring requires the
explicit --run-after-seal flag and all prediction/receipt/reference checks.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(os.environ.get('STVG_PROJECT_ROOT', Path(__file__).resolve().parents[2])).expanduser().resolve()
sys.path.insert(0, str(ROOT))

BASELINES = {'SourceStrong': 'Source', 'ERefStrong': 'E_Ref', 'GeometryOPD': 'GeometryOPD'}
OLD_NAMES = {'SourceStrong': 'Source', 'ERefStrong': 'WithoutOPD', 'GeometryOPD': 'GeometryOPD'}
BOOTSTRAP_SEED = 20261021
BOOTSTRAP_DRAWS = 10000
DELTA_EPS_PP = 1e-8
LOCKED_ARMS = ['StateOPD', 'Direct', 'DirectMC', 'DirectFullRef']
EVALUATION_ALLOWANCE = 1321


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    def native(x):
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=native) + '\n')
    temporary.replace(path)


def within(parent, relative):
    path = (Path(parent) / relative).resolve()
    if not path.is_relative_to(Path(parent).resolve()):
        raise ValueError('Artifact path escapes its declared trial directory')
    return path


def arm_names(config):
    arms = config['arms']
    names = list(arms) if isinstance(arms, dict) else list(arms)
    if not names or not all(isinstance(a, str) and a.isidentifier() for a in names):
        raise ValueError('config.arms must contain explicit NPZ array names')
    if len(names) != len(set(names)) or set(names) & set(BASELINES):
        raise ValueError('Arm names must be unique and distinct from saved baselines')
    return names


def validate_before_labels(trial, scope='hc128'):
    """Validate the entire cohort before opening any GT or scored-result file."""
    trial = Path(trial).resolve()
    assert trial == (ROOT / 'outputs/state_opd_hc128').resolve(), 'Do not score into an older or unrelated trial'
    config_path = ROOT / 'scripts/state_opd/config.json'
    config = read(config_path)
    assert scope in ('hc128', 'hc64')
    evaluation_root = trial if scope == 'hc128' else trial / 'scope_hc64'
    seal_path = evaluation_root / 'sealed.json'
    seal = read(seal_path)
    lock_path = trial / 'lock.json'
    assert seal['lock_sha256'] == sha(lock_path)
    execution_lock = read(lock_path)
    assert execution_lock['fingerprint'] == seal['fingerprint']
    assert execution_lock['config_sha256'] == seal['config_sha256']
    expected_count = 128 if scope == 'hc128' else 64
    assert seal['queries'] == expected_count and seal['dataset'] == 'hc2' and seal['GT_read'] is False
    assert isinstance(seal['fingerprint'], str) and seal['fingerprint']
    if 'fingerprint' in config:
        assert config['fingerprint'] == seal['fingerprint']
    assert seal['config_sha256'] == sha(config_path)
    names = arm_names(config)
    assert names == LOCKED_ARMS, 'This scorer is bound to the locked four-arm HC trial'
    assert config['queries'] == 128, 'The original planned configuration is retained'
    if (trial / 'config.json').exists():
        assert sha(trial / 'config.json') == sha(config_path)
    manifest = ROOT / 'data/evaluation/development.jsonl'
    assert sha(manifest) == seal['manifest_sha256']
    records = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    records = [record for record in records if record['dataset'] == 'hc2']
    assert len(records) == 128 and len({r['id'] for r in records}) == 128
    planned_ids = {r['id'] for r in records}
    scope_sha = None
    if scope == 'hc64':
        scope_path = trial / 'user_scope64.json'
        user_scope = read(scope_path)
        scope_sha = sha(scope_path)
        assert seal['scope_sha256'] == scope_sha
        assert user_scope['ids'] == [r['id'] for r in records[:64]], 'Scope must be the first 64 manifest parents'
        bound_fingerprints = [user_scope[k] for k in ('source_fingerprint', 'sourcefingerprint', 'fingerprint') if k in user_scope]
        assert bound_fingerprints and all(v == execution_lock['fingerprint'] for v in bound_fingerprints)
        bound_configs = [user_scope[k] for k in ('config_sha256', 'configsha') if k in user_scope]
        assert bound_configs and all(v == seal['config_sha256'] for v in bound_configs)
        selection_rules = [user_scope[k] for k in ('selection_rule', 'parentselectionrule', 'parent_selection_rule') if k in user_scope]
        assert selection_rules and all(isinstance(v, str) and v for v in selection_rules)
        records = records[:64]
    assert set(seal['files']) == {r['id'] for r in records}
    reference_root = ROOT / 'outputs/layout_reward_dev'
    reference_seal_path = reference_root / 'scope_hc2/sealed.json'
    reference_seal = read(reference_seal_path)
    assert reference_seal['queries'] == 128 and reference_seal['GT_read'] is False
    assert reference_seal['manifest_sha256'] == seal['manifest_sha256']
    assert set(reference_seal['files']) == planned_ids
    receipts = {}
    for record in records:
        key = record['id']
        entry = seal['files'][key]
        path = within(trial, entry['path'])
        receipt_path = path.with_suffix('.json')
        assert sha(path) == entry['sha256']
        assert sha(receipt_path) == entry['receipt_sha256']
        receipt = read(receipt_path)
        assert receipt['GT_read'] is False and receipt['fingerprint'] == seal['fingerprint']
        assert receipt['sha256'] == entry['sha256']
        assert sha(within(trial, receipt.get('trace_path', 'traces/' + key + '.pt'))) == receipt['trace_sha256']
        reference_entry = reference_seal['files'][key]
        reference = within(reference_root, reference_entry['path'])
        assert sha(reference) == reference_entry['sha256'] == receipt['reference_sha256']
        with np.load(path, allow_pickle=False) as current, np.load(reference, allow_pickle=False) as old:
            for field in ['frame_ids', 'positions8', 'Strong_interval']:
                assert np.array_equal(current[field], old[field]), (key, field)
            ids = current['frame_ids']
            assert ids.ndim == 1 and ids.tolist() == record['metadata']['frame_ids']
            positions = current['positions8']
            assert positions.shape == (8,) and len(set(positions.tolist())) == 8
            assert positions.dtype.kind in 'iu' and positions.min() >= 0 and positions.max() < len(ids)
            assert current['Strong_interval'].shape == (2,)
            for name in names:
                boxes = current[name]
                assert boxes.shape == (len(ids), 4) and np.isfinite(boxes).all()
                assert (boxes[:, 2:] > 0).all(), (key, name, 'nonpositive size')
            # Reused methods remain separate baselines, never overwritten by a new array.
            for name, field in BASELINES.items():
                if name in current.files:
                    assert np.array_equal(current[name], old[field]), (key, name)
        assert set(receipt['costs']) == set(names)
        for name in names:
            costs = receipt['costs'][name]
            assert np.isfinite(costs['seconds']) and costs['seconds'] >= 0
            assert isinstance(costs['updates'], int) and costs['updates'] >= 0
            assert isinstance(costs['accepted_updates'], int) and 0 <= costs['accepted_updates'] <= costs['updates']
            for field in ['full_trajectory_evals', 'native_offset_decoder_calls', 'decoder_backwards', 'reward_evals', 'peak_GPU_bytes']:
                assert isinstance(costs[field], (int, float)) and np.isfinite(costs[field]) and costs[field] >= 0
            assert 1 <= costs['full_trajectory_evals'] <= EVALUATION_ALLOWANCE
            assert costs['native_offset_decoder_calls'] == 2 * costs['full_trajectory_evals']
        receipts[key] = receipt
    proof = dict(all_scope_predictions_and_receipts_before_labels=True,
                 evaluated_queries=expected_count, original_planned_queries=128, scope=scope,
                 source_reference_predictions_exact=True,
                 config_sha256=sha(config_path), seal_sha256=sha(seal_path),
                 seal_relative_path=str(seal_path.relative_to(trial)), scope_sha256=scope_sha,
                 reference_seal_sha256=sha(reference_seal_path),
                 manifest_sha256=sha(manifest), fingerprint=seal['fingerprint'], lock_sha256=sha(lock_path))
    if scope == 'hc128':
        proof['all_128_predictions_and_receipts_before_labels'] = True
    return config, names, records, receipts, reference_root, reference_seal, proof


def score_boxes(boxes, record, truth, interval, span, positions):
    from stvg.metrics import DenseTube, official, box_iou
    from stvg.metrics_base import xyxy
    from stvg.evaluation import add_reporting_metrics
    metadata = record['metadata']
    ids = metadata['frame_ids']
    row = dict(input=metadata, frame_ids=ids)
    values = official(boxes, row, truth, span, interval, 'hc2')
    independent = DenseTube(boxes, row, truth, span, clip=True).score(interval)
    error = max(abs(values[k] - independent[k]) for k in ['v', 't', 's'])
    assert error < 2e-10
    pixels = np.maximum(xyxy(boxes, metadata['width'], metadata['height']), 0)
    observed = {ids[int(position)] for position in positions}
    fixed = [f for f in ids if f in truth]
    obs = [f for f in fixed if f in observed]
    unobs = [f for f in fixed if f not in observed]
    lookup = {f: i for i, f in enumerate(ids)}
    for field, frames in [('fixed_actual_sIoU', fixed), ('observed_actual_sIoU', obs),
                          ('unobserved_actual_sIoU', unobs)]:
        values[field] = float(np.mean([box_iou(pixels[lookup[f]], truth[f]) for f in frames])) if frames else None
    values['boundary_MAE'] = (abs(interval[0] - span[0]) + abs(interval[1] - span[1])) / 2
    return add_reporting_metrics(values, metadata), error, dict(actual=len(fixed), observed=len(obs), unobserved=len(unobs))


def mean_table(rows, names):
    from stvg.evaluation import finite_mean, main_table
    fields = list(rows[0]['values'][names[0]])
    means = {name: {field: finite_mean([r['values'][name][field] for r in rows])
                    for field in fields} for name in names}
    counts = {name: {field: sum(r['values'][name][field] is not None and
                               np.isfinite(r['values'][name][field]) for r in rows)
                     for field in fields} for name in names}
    return main_table(means), counts


def paired_stats(rows, names):
    """One query per parent is enforced; bootstrap parents, not frames/candidates."""
    assert len(rows) == len({r['parent'] for r in rows}) and len(rows) in (64, 128)
    pairs = {}
    field_names = ['v', 't', 's', 'observed_actual_sIoU', 'unobserved_actual_sIoU']
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = {}
    for first, second in itertools.combinations(names, 2):
        # Both directions make a downstream report independent of arm order.
        for a, b in [(first, second), (second, first)]:
            fields = {}
            for field in field_names:
                valid = [r for r in rows if r['values'][a][field] is not None and r['values'][b][field] is not None]
                n = len(valid)
                if not n:
                    fields[field] = dict(valid_parents=0, mean_pp=None, CI95_pp=None)
                    continue
                difference = 100 * np.asarray([r['values'][a][field] - r['values'][b][field] for r in valid])
                if n not in draws:
                    draws[n] = rng.integers(0, n, (BOOTSTRAP_DRAWS, n))
                fields[field] = dict(valid_parents=n, mean_pp=float(difference.mean()),
                    CI95_pp=np.quantile(difference[draws[n]].mean(1), [.025, .975]).tolist(),
                    improved=int((difference > DELTA_EPS_PP).sum()),
                    declined=int((difference < -DELTA_EPS_PP).sum()),
                    unchanged=int((abs(difference) <= DELTA_EPS_PP).sum()),
                    declined_over5pp=int((difference < -5).sum()),
                    positive_mean_contribution_pp=float(difference[difference > 0].sum() / n),
                    negative_mean_contribution_pp=float(difference[difference < 0].sum() / n))
            pairs[a + '-' + b] = fields
    return pairs


def cost_summary(receipts, names):
    fields = sorted({field for receipt in receipts.values() for name in names for field in receipt['costs'][name]})
    out = {}
    for name in names:
        statistics = {}
        for field in fields:
            values = [r['costs'][name].get(field) for r in receipts.values()]
            values = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)]
            if values:
                statistics[field] = dict(valid_queries=len(values), mean=float(np.mean(values)),
                    median=float(np.median(values)), p90=float(np.quantile(values, .9)), maximum=float(np.max(values)))
                if field != 'peak_GPU_bytes':
                    statistics[field]['total'] = float(np.sum(values))
        out[name] = statistics
    return dict(scope='Observed incremental trial costs with existing Source/teacher caches; not standalone end-to-end inference latency', arms=out)


def render_report(public):
    names = public['arm_order']
    means = public['means']
    count = public['queries']
    lines = [f'# HC{count} state-OPD comparison', '',
        f'Exposed development panel: VidSTG to HC2, {count} parent videos, one query each. '
        'All methods use the exact saved Strong interval. All queries are retained; stopping follows '
        'the predeclared reward-based acceptance rules and evaluation allowance, without GT selection.', '',
        'SourceStrong uses the saved frozen Source spatial trajectory with Strong temporal inference; '
        'it is not the original Source temporal readout. ERefStrong uses the saved single-ReferDINO '
        'initialization with the same Strong interval.', '',
        '| Method | Full vIoU (%) | tIoU (%) | Observed-node sIoU (%) | Unobserved-node sIoU (%) |',
        '|---|---:|---:|---:|---:|']
    if count == 64:
        lines[2:2] = ['The original 128-query plan was reduced by the user to the first 64 parent videos '
            'in the fixed development manifest, one query per parent. The original method configuration '
            'and per-query budgets remain unchanged. All baseline means below are recomputed on these '
            'same 64 queries; no old 128-query aggregate is used as a comparison.', '']
    def percent(value):
        return 'NA' if value is None else '%.4f' % (100 * value)
    for name in names:
        m = means[name]
        lines.append('| %s | %s | %s | %s | %s |' % (name, percent(m['m_vIoU']), percent(m['m_tIoU']),
            percent(m['m_sIoU_observed_actual']), percent(m['m_sIoU_unobserved_actual'])))
    lines += ['', 'Observed and unobserved sIoU use actual sampled nodes with GT; all eight expert positions '
        'are excluded from unobserved sIoU. These metrics are independent of temporal truncation. '
        'Interpolation is used only by the existing complete-video metric, not counted as new observations.', '',
        '| Comparison | vIoU delta pp [95% CI] | Improved / declined / unchanged | vIoU drops >5 pp | '
        'Unobserved sIoU delta pp [95% CI] | Unobserved drops >5 pp |',
        '|---|---:|---:|---:|---:|---:|']
    new_names = public['new_arms']
    focus = [(a, b) for a in new_names for b in ['ERefStrong', 'GeometryOPD']]
    focus += list(itertools.combinations(new_names, 2))
    for a, b in focus:
        p = public['paired'][a + '-' + b]
        v, s = p['v'], p['unobserved_actual_sIoU']
        def ci(z):
            return 'NA' if z['mean_pp'] is None else '%+.4f [%.4f, %.4f]' % (z['mean_pp'], *z['CI95_pp'])
        lines.append('| %s-%s | %s | %d / %d / %d | %d | %s | %d |' %
            (a, b, ci(v), v['improved'], v['declined'], v['unchanged'], v['declined_over5pp'], ci(s), s.get('declined_over5pp', 0)))
    lines += ['', 'Intervals use 10,000 paired parent-video bootstrap draws. All positive, negative, '
        'zero-overlap, and unchanged queries remain in the anonymous result JSON. Valid-query counts '
        'are reported per metric; queries without GT at an expert observation are retained in complete vIoU.', '',
        'The four new methods share the locked 8 x 4 temporary state space. GeometryOPD uses the older '
        '8 x 256 state and is a saved method reference, not a matched optimizer control. This experiment '
        'does not establish superiority over every Direct update in the older full state space. '
        'Direct and DirectFullRef allow up to 660 point attempts; StateOPD and DirectMC allow up to '
        '40 rounds. All share an allowance of at most 1,321 full-trajectory evaluations, including '
        'initialization and every backtracking validation. They need not consume the same allowance. '
        'The fixed backtracking scales are 1, 0.5, 0.25, 0.125, and 0.0625; the first utility increase '
        'above 1e-8 is accepted, otherwise the state and Adam buffers are restored. Point methods stop '
        'after all scales fail; OPD/MC may use the next prespecified sampling seed until their round '
        'or evaluation allowance is exhausted. Backward counts and actual wall times differ.', '',
        '## Declining queries', '',
        'The JSON retains every query. This table shows the union of the ten largest vIoU declines '
        'for each new method relative to ERef+Strong; values are percentage points.', '',
        '| Query | ' + ' | '.join(new_names) + ' |', '|---|' + '---:|' * len(new_names)]
    worst = set()
    for name in new_names:
        ordered = sorted(public['queries_anonymous'], key=lambda q: q['values'][name]['v'] - q['values']['ERefStrong']['v'])
        worst.update(q['query'] for q in ordered[:10] if q['values'][name]['v'] < q['values']['ERefStrong']['v'])
    for q in public['queries_anonymous']:
        if q['query'] in worst:
            values = ['%+.4f' % (100 * (q['values'][name]['v'] - q['values']['ERefStrong']['v'])) for name in new_names]
            lines.append('| ' + q['query'] + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '## Recorded costs', '', public['costs']['scope'], '',
        '| Method | Seconds/query | Attempts | Accepted updates | Full trajectory evaluations | Native offset decoder calls | Decoder backwards | Reward evaluations | Mean per-query peak GPU MiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in new_names:
        fields = public['costs']['arms'][name]
        values = ['%.4f' % fields[k]['mean'] if k in fields else 'not recorded' for k in ['seconds', 'updates', 'accepted_updates', 'full_trajectory_evals', 'native_offset_decoder_calls', 'decoder_backwards', 'reward_evals']]
        values.append('%.2f' % (fields['peak_GPU_bytes']['mean'] / 2**20) if 'peak_GPU_bytes' in fields else 'not recorded')
        lines.append('| ' + name + ' | ' + ' | '.join(values) + ' |')
    lines += ['', 'The geometric baseline is reused from exact sealed outputs; its old runtime is not '
        'presented as a new timing. Equal evaluation allowances do not imply equal consumed work, wall time '
        'or equal backward computation. Configuration and code are bound by the recorded hashes. '
        'The original formal VidSTG task and prior reward results are separate and are not modified.']
    return '\n'.join(lines) + '\n'


def evaluate(trial, public_output=None, scope='hc128'):
    trial = Path(trial).resolve()
    config, new_names, records, receipts, reference_root, reference_seal, proof = validate_before_labels(trial, scope)
    count = len(records)
    selected_ids = {r['id'] for r in records}
    evaluation_root = trial if scope == 'hc128' else trial / 'scope_hc64'
    seal_path = within(trial, proof['seal_relative_path'])
    prediction_seal = read(seal_path)
    # No GT or old scored values are accessed until every new prediction passes the gate.
    old_metrics_path = reference_root / 'scope_hc2/metrics.json'
    old_metrics = read(old_metrics_path)
    labels_path = ROOT / 'data/evaluation/development_labels.jsonl'
    labels_sha = sha(labels_path)
    assert labels_sha == old_metrics['labels_sha256']
    truth = {r['id']: r for r in [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
             if r['dataset'] == 'hc2' and r['id'] in selected_ids}
    old_rows = {r['id']: r for r in old_metrics['rows'] if r['id'] in selected_ids}
    assert set(truth) == set(old_rows) == selected_ids
    names = list(BASELINES) + new_names
    rows = []
    maximum_error = 0.
    baseline_error = 0.
    checks = 0
    for record in records:
        key = record['id']
        g = truth[key]
        assert g['dataset'] == 'hc2'
        boxes_gt = {int(frame): box for frame, box in g['boxes'].items()}
        entry = prediction_seal['files'][key]
        with np.load(within(trial, entry['path']), allow_pickle=False) as current, np.load(within(reference_root, reference_seal['files'][key]['path']), allow_pickle=False) as old:
            interval = current['Strong_interval'].tolist()
            positions = current['positions8'].tolist()
            values = {}
            for name in names:
                boxes = old[BASELINES[name]] if name in BASELINES else current[name]
                v, error, support = score_boxes(boxes, record, boxes_gt, interval, g['interval'], positions)
                maximum_error = max(maximum_error, error)
                checks += 1
                values[name] = v
                if name in BASELINES:
                    prior = old_rows[key]['values'][OLD_NAMES[name]]
                    for field in ['v', 't', 's', 'fixed_actual_sIoU', 'observed_actual_sIoU', 'unobserved_actual_sIoU']:
                        if prior[field] is None:
                            assert v[field] is None
                        else:
                            baseline_error = max(baseline_error, abs(v[field] - prior[field]))
            rows.append(dict(id=key, parent=g['parent'], dataset='hc2', values=values, support=support,
                             development_exposure='exposed_development'))
    assert len(rows) == len({r['parent'] for r in rows}) == count
    assert baseline_error < 2e-10 and maximum_error < 2e-10
    assert sha(labels_path) == labels_sha
    assert sha(seal_path) == proof['seal_sha256']
    assert sha(trial / 'lock.json') == proof['lock_sha256']
    if scope == 'hc64':
        assert sha(trial / 'user_scope64.json') == proof['scope_sha256']
    for record in records:
        key = record['id']
        entry = prediction_seal['files'][key]
        current = within(trial, entry['path'])
        assert sha(current) == entry['sha256'] and sha(current.with_suffix('.json')) == entry['receipt_sha256']
        receipt = receipts[key]
        assert sha(within(trial, receipt.get('trace_path', 'traces/' + key + '.pt'))) == receipt['trace_sha256']
        assert sha(within(reference_root, reference_seal['files'][key]['path'])) == reference_seal['files'][key]['sha256']
    proof.update(labels_sha256=labels_sha, old_development_metrics_sha256=sha(old_metrics_path),
                 official_dense_checks=checks, official_dense_max_absolute_error=maximum_error,
                 reused_baseline_max_absolute_error=baseline_error, prediction_hashes_rechecked_after_scoring=True)
    torch_module = sys.modules.get('torch')
    assert torch_module is None or not torch_module.cuda.is_initialized()
    proof['CUDA_initialized'] = False
    means, counts = mean_table(rows, names)
    pairs = paired_stats(rows, names)
    costs = cost_summary(receipts, new_names)
    private = dict(status=f'scored_complete_sealed_HC{count}_development', queries=count, parents=count,
        scope=scope, original_planned_queries=128,
        new_arms=new_names, arm_order=names, means=means, valid_query_counts=counts,
        paired=pairs, costs=costs, proof=proof, rows=rows, config=config)
    write_json(evaluation_root / 'metrics.json', private)
    public = dict(status=f'complete_HC{count}_exposed_development', direction='VidSTG_to_HC2',
        queries=count, parents=count, scope=scope, original_planned_queries=128,
        selection='first_64_in_fixed_HC128_manifest_by_user_request' if scope == 'hc64' else 'all_fixed_HC128_manifest',
        seed=config.get('seed'), new_arms=new_names, arm_order=names,
        baseline_definitions={'SourceStrong': 'saved frozen Source spatial trajectory + saved Strong interval',
                              'ERefStrong': 'saved single-ReferDINO E_Ref trajectory + saved Strong interval',
                              'GeometryOPD': 'saved geometric OPD spatial trajectory + saved Strong interval'},
        all_arms_same_Strong_interval=True, means=means, valid_query_counts=counts, paired=pairs,
        costs=costs, proof=proof, queries_anonymous=[dict(query='H%03d' % (i + 1), values=row['values']) for i, row in enumerate(rows)])
    if public_output is not None:
        destination = Path(public_output).resolve()
        assert destination != reference_root.resolve() and destination != trial.resolve()
        public_path = destination / f'hc{count}.json'
        if public_path.exists():
            prior = read(public_path)
            assert prior['proof']['fingerprint'] == proof['fingerprint'] and prior['proof']['seal_sha256'] == proof['seal_sha256'], 'Do not replace an older result'
        destination.mkdir(parents=True, exist_ok=True)
        write_json(public_path, public)
        (destination / 'REPORT.md').write_text(render_report(public))
    write_json(evaluation_root / 'score_receipt.json', dict(status='scored', metrics_sha256=sha(evaluation_root / 'metrics.json'),
        queries=count, scope=scope, proof=proof, no_model_or_decoder_calls=True))
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--public-output', type=Path)
    parser.add_argument('--scope', choices=['hc128', 'hc64'], default='hc128')
    parser.add_argument('--run-after-seal', action='store_true')
    args = parser.parse_args()
    if not args.run_after_seal:
        parser.error('Scoring requires explicit --run-after-seal after every output in the selected scope is sealed')
    result = evaluate(args.trial, args.public_output, args.scope)
    print(json.dumps(dict(status=result['status'], queries=result['queries'], means=result['means'], costs=result['costs']), indent=2))


if __name__ == '__main__':
    main()
