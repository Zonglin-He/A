"""Replay official evaluators on sealed outputs and retain per-question errors."""
import argparse
import base64
from collections import Counter, defaultdict
import csv
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import types
from unittest.mock import patch

from PIL import Image

from download import DATA, OUTPUT, PROTOCOL, read_json, sha256, write_json
from prepare import cv_rows, emb_prompt, sr_prompt, tsv_rows


DUMMY_IMAGE = Image.new('RGB', (2, 2))
GENERATION_LETTER = re.compile(r'^\s*\(?([A-F])(?:\b|\))', re.IGNORECASE)


def response_letter(response):
    match = GENERATION_LETTER.match(response) if response else None
    return match.group(1).upper() if match else None


def import_official(benchmark, source):
    config = read_json(PROTOCOL)
    actual = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    assert actual == config['hispatial_commit']
    assert not subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True)
    script = source / config['datasets'][benchmark]['official_entry']
    spec = spec_from_file_location(f'official_{benchmark}', script)
    module = module_from_spec(spec)
    inference = types.ModuleType('hispatial.inference')
    inference.MoGeProcessor = inference.HiSpatialPredictor = object
    stubs = {'hispatial.inference': inference, 'cv2': types.ModuleType('cv2')}
    original = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        spec.loader.exec_module(module)
    finally:
        for name, old in original.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return module, sha256(script)


class Replay:
    def __init__(self, cases):
        self.cases = iter(cases)
        self.count = 0

    def query(self, image, prompt, xyz_values):
        expected, output = next(self.cases)
        assert prompt == expected and isinstance(output, str)
        self.count += 1
        return output

    def assert_consumed(self, n):
        assert self.count == n and next(self.cases, None) is None


def patched_predictor(module, replay):
    return patch.multiple(module, HiSpatialPredictor=lambda **kwargs: replay,
                          MoGeProcessor=lambda **kwargs: types.SimpleNamespace(apply_transform=lambda image: None))


def score_cv(module, work, sealed, prompts):
    rows = []
    splits = defaultdict(list)
    for subset, index, record in cv_rows(('idx', 'prompt', 'answer', 'source')):
        qid = f'{subset}/{index}'
        prompt, answer = record['prompt'], record['answer']
        output = sealed[qid]['response']
        assert prompts[qid] == prompt and isinstance(output, str)
        correct = answer.replace('(', '').replace(')', '').replace(' ', '') in (
            output.replace('<eos>', '').replace(' ', '').replace(')', '').replace('(', ''))
        rows.append(dict(id=qid, dataset_idx=record['idx'], subset=subset, task=record['task'], source=record['source'],
            prompt=prompt, response=output, official_answer=answer, parsed_leading_option=response_letter(output),
            correct=bool(correct)))
        splits[subset].append((prompt, answer, output))
    replay_metrics = {}
    prefix = str(work / 'official_replay' / 'cv_bench')
    for subset, loader, call, suffix in [('3d', 'load_cv_bench_3d', module.test_cv_bench_3d, '3d'),
                                          ('2d', 'load_cv_bench_2d_relation', module.test_cv_bench_2d_relation, 'relation')]:
        records = splits[subset]
        replay = Replay((prompt, output) for prompt, _, output in records)
        images = [dict(prompt=prompt, answer=answer, image=DUMMY_IMAGE) for prompt, answer, _ in records]
        with patched_predictor(module, replay), patch.object(module, loader, return_value=images):
            call('replay', prefix)
        replay.assert_consumed(len(records))
        observed = read_json(f'{prefix}_{suffix}_accuracy.json')
        local = [x for x in rows if x['subset'] == subset]
        assert observed['correct_answers'] == sum(x['correct'] for x in local)
        assert observed['total_questions'] == len(local)
        assert observed['accuracy'] == sum(x['correct'] for x in local)/len(local)
        replay_metrics[subset] = observed
    return rows, [], replay_metrics


def score_sr(module, work, sealed, prompts):
    rows = []
    grouped = {}
    fixture = work / 'official_replay' / '3dsr_rows.tsv'
    with fixture.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t')
        writer.writerow(['index', 'qid', 'question', 'A', 'B', 'C', 'D', 'answer', 'category', 'image', 'image_source', 'image_url'])
        for index, record in tsv_rows():
            if len(record) < 12:
                continue
            qid = f'tsv/{index}'
            variant = record[1].strip()
            group, topic = module._id_to_group(variant), record[8].strip()
            answer = record[7].strip()
            options = dict(zip('ABCD', (value.strip() for value in record[3:7])))
            prompt = module._build_prompt(record[2].strip(), *options.values())
            output = sealed[qid]['response']
            assert prompt == sr_prompt(record) == prompts[qid]
            trunc = bool(output is not None and module._is_correct_truncate(output, answer))
            regex = bool(output is not None and module._is_correct_regex(output, answer, options))
            row = dict(id=qid, variant_id=variant, original_group=group, subtype=topic, flip=module._needs_flip(variant),
                prompt=prompt, response=output, official_answer=answer, parsed_regex_option=module._normalize_to_letter(output, options),
                correct_truncate=trunc, correct_regex=regex)
            rows.append(row)
            g = grouped.setdefault(group, dict(original_group=group, subtype=topic, variants=0, correct_truncate=True, correct_regex=True))
            g['variants'] += 1
            g['correct_truncate'] &= trunc
            g['correct_regex'] &= regex
            writer.writerow(record[:9] + [qid, '', ''])

    replay = Replay((r['prompt'], r['response']) for r in rows if r['response'] is not None)

    def replay_image(key, url):
        if sealed[key]['response'] is None:
            raise ValueError('Previously sealed image failure')
        return DUMMY_IMAGE, True

    prefix = str(work / 'official_replay' / '3dsrbench.txt')
    with patched_predictor(module, replay), patch.object(module, '_load_image_from_row', side_effect=replay_image):
        module.test('replay', str(fixture), prefix)
    replay.assert_consumed(sum(r['response'] is not None for r in rows))
    observed = read_json(f'{prefix}_accuracy.json')
    groups = list(grouped.values())

    def agreement(value, matched, key):
        return math.isnan(value) if not matched else math.isclose(value, sum(r[key] for r in matched)/len(matched), abs_tol=1e-12)

    for label in ('truncate', 'regex'):
        results = observed[label]
        k = 'correct_' + label
        assert agreement(results['accuracy'], groups, k)
        for category, reported in results['by_category'].items():
            assert agreement(reported, [g for g in groups if g['subtype'] in module.MAPPING[category]], k)
        for subtype, reported in results['by_subcategory'].items():
            assert agreement(reported, [g for g in groups if g['subtype'] == subtype], k)
    return rows, groups, observed


def score_emb(module, work, sealed, prompts):
    rows, stub = [], []
    with BytesIO() as buffer:
        DUMMY_IMAGE.save(buffer, format='PNG')
        pixel = base64.b64encode(buffer.getvalue()).decode('ascii')
    official_dataset = read_json(DATA / 'embspatial' / 'embspatial_bench.json')
    for index, record in enumerate(official_dataset):
        qid = f'json/{index}'
        answer = 'ABCDEF'[record['answer']]
        prompt = emb_prompt(record)
        output = sealed[qid]['response']
        assert prompts[qid] == prompt and isinstance(output, str)
        rows.append(dict(id=qid, question_id=record['question_id'], relation=record['relation'], source=record['data_source'],
            prompt=prompt, response=output, official_answer=answer, parsed_leading_option=response_letter(output),
            correct=answer in output[:2]))
        stub.append(dict(image=pixel, question=record['question'], answer_options=record['answer_options'], answer=record['answer']))
    del official_dataset
    fixture = work / 'official_replay' / 'emb_questions.json'
    write_json(fixture, stub)
    replay = Replay((r['prompt'], r['response']) for r in rows)
    prefix = str(work / 'official_replay' / 'embspatial')
    with patched_predictor(module, replay):
        module.test('replay', prefix, str(fixture))
    replay.assert_consumed(len(rows))
    observed = read_json(f'{prefix}_accuracy.json')
    assert observed['total_questions'] == len(rows)
    assert observed['correct_answers'] == sum(r['correct'] for r in rows)
    assert observed['accuracy'] == sum(r['correct'] for r in rows)/len(rows)
    return rows, [], observed


def breakdowns(benchmark, rows, grouped, official):
    table = []

    def add(level, kind, records, metric):
        if records:
            n = len(records)
            correct = sum(r[metric] for r in records)
            table.append(dict(benchmark=benchmark, level=level, category=kind, metric=metric,
                questions=n, correct=correct, errors=n-correct, accuracy=correct/n))

    if benchmark == 'cv_bench':
        for subset in ('3d', '2d'):
            add('split', subset, [r for r in rows if r['subset'] == subset], 'correct')
        for kind in sorted({r['task'] for r in rows}):
            add('task', kind, [r for r in rows if r['task'] == kind], 'correct')
        for kind in sorted({r['source'] for r in rows}):
            add('source', kind, [r for r in rows if r['source'] == kind], 'correct')
    elif benchmark == '3dsrbench':
        for metric in ('correct_truncate', 'correct_regex'):
            add('official_group', 'overall', grouped, metric)
            for name, topics in official.MAPPING.items():
                add('official_category', name, [r for r in grouped if r['subtype'] in topics], metric)
            for name in official.SUBTYPES:
                add('official_subcategory', name, [r for r in grouped if r['subtype'] == name], metric)
            for flipped in (False, True):
                add('variant_diagnostic', 'flip' if flipped else 'original', [r for r in rows if r['flip'] == flipped], metric)
    else:
        add('overall', 'All', rows, 'correct')
        for kind in sorted({str(r['relation']) for r in rows}):
            add('relation', kind, [r for r in rows if str(r['relation']) == kind], 'correct')
        for kind in sorted({r['source'] for r in rows}):
            add('source', kind, [r for r in rows if r['source'] == kind], 'correct')
    return table


def sanitize(data):
    if isinstance(data, dict):
        return {k: sanitize(v) for k, v in data.items()}
    if isinstance(data, float) and math.isnan(data):
        return None
    return data


def jsonl(path, rows):
    partial = path.with_suffix('.jsonl.tmp')
    with partial.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    partial.replace(path)


def run(benchmark, official_source):
    work = OUTPUT / benchmark
    seal = read_json(work / 'predictions_seal.json')
    pred = work / 'predictions.jsonl'
    manifest = work / 'inputs.jsonl'
    assert sha256(pred) == seal['predictions_sha256'] and sha256(manifest) == seal['inputs_sha256']
    assert sha256(work / 'runtime.json') == seal['runtime_sha256'] and seal['answers_loaded_by_gpu_runner'] == 0
    sealed = {r['id']: r for r in (json.loads(line) for line in pred.read_text().splitlines())}
    prompts = {r['id']: r['prompt'] for r in (json.loads(line) for line in manifest.read_text().splitlines())}
    assert set(prompts) == set(sealed) and len(sealed) == len(prompts) == seal['rows']
    previous = read_json(work / 'inputs_seal.json')
    source = DATA / benchmark / 'source.json'
    assert sha256(source) == previous['source_sha256'] and sha256(PROTOCOL) == previous['protocol_sha256']
    for asset in read_json(source)['files']:
        assert sha256(source.parent / asset['file']) == asset['sha256']
    module, script_hash = import_official(benchmark, official_source)
    (work / 'official_replay').mkdir(exist_ok=True, mode=0o700)
    scorer = {'cv_bench': score_cv, '3dsrbench': score_sr, 'embspatial': score_emb}[benchmark]
    rows, groups, observed = scorer(module, work, sealed, prompts)
    assert {r['id'] for r in rows} == set(sealed) and len(rows) == len(sealed)
    errors = [r for r in rows if not (r['correct_truncate'] and r['correct_regex'] if groups else r['correct'])]
    jsonl(work / 'scored_rows.jsonl', rows)
    jsonl(work / 'errors.jsonl', errors)
    if groups:
        jsonl(work / 'original_groups.jsonl', groups)
    breakdown = breakdowns(benchmark, rows, groups, module)
    with (work / 'breakdown.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, breakdown[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(breakdown)
    score_paths = ('scored_rows.jsonl', 'errors.jsonl', 'breakdown.csv') + (('original_groups.jsonl',) if groups else ())
    report = dict(benchmark=benchmark, task_rows=len(rows), original_groups=len(groups) or None,
        official_metrics=sanitize(observed), upstream_scoring_script_sha256=script_hash,
        prediction_seal_sha256=sha256(work / 'predictions_seal.json'), scorer_sha256=sha256(__file__),
        official_replay_checks_passed=True, input_assets=read_json(source),
        ungenerated_rows=sum(r['response'] is None for r in rows), private_error_rows=len(errors),
        decode_modes=dict(Counter(r['mode'] for r in sealed.values())),
        derived_artifacts={f: sha256(work / f) for f in score_paths})
    write_json(work / 'score_seal.json', report)
    print(benchmark, len(rows), len(groups), sanitize(observed), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('benchmark', choices=read_json(PROTOCOL)['order'])
    parser.add_argument('--official-source', type=Path, required=True)
    args = parser.parse_args()
    run(args.benchmark, args.official_source)
