"""Prepare a label-free matched cohort and separate diagnostic questions."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

from download import DATA, OUTPUT as BASELINE, ROOT, STUDY, read_json, sha256, write_json
from prepare import emb_prompt, sr_prompt, tsv_rows
from score import jsonl


RUN = ROOT / 'outputs' / 'geometry_diagnosis_20261121'
CONFIG = STUDY / 'GEOMETRY_DIAGNOSIS.json'
TOPIC = 'multi_object_viewpoint_towards_object'
ROLE_RE = re.compile(r'which side of (.+?) is facing (.+?)\?', re.I)
GRID = ('top left', 'top middle', 'top right', 'middle left', 'center',
        'middle right', 'bottom left', 'bottom middle', 'bottom right')


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream]


def stable(value):
    return hashlib.sha256(value.encode()).hexdigest()


def verified_baseline(benchmark):
    directory = BASELINE / benchmark
    input_seal = read_json(directory / 'inputs_seal.json')
    prediction_seal = read_json(directory / 'predictions_seal.json')
    assert input_seal['inputs_sha256'] == prediction_seal['inputs_sha256'] == sha256(directory / 'inputs.jsonl')
    assert prediction_seal['predictions_sha256'] == sha256(directory / 'predictions.jsonl')
    assert prediction_seal['runtime_sha256'] == sha256(directory / 'runtime.json')
    assert prediction_seal['answers_loaded_by_gpu_runner'] == 0
    return directory, input_seal, prediction_seal


def benchmark_rows(benchmark):
    if benchmark == '3dsrbench':
        for i, row in tsv_rows():
            if len(row) < 12 or row[8].strip() != TOPIC:
                continue
            variant = row[1].strip()
            yield f'tsv/{i}', dict(topic=TOPIC, source=row[10].strip(), scene=row[11].split('?')[0],
                                  group=variant.split('-')[0], flip='-flip' in variant, variant=variant,
                                  options=[v.strip() for v in row[3:7]], prompt=sr_prompt(row))
    else:
        for i, row in enumerate(read_json(DATA / benchmark / 'embspatial_bench.json')):
            if row['relation'] in {'far', 'close'}:
                yield f'json/{i}', dict(topic=row['relation'], source=row['data_source'], scene=row['scene_id'],
                                       group=row['question_id'], flip=False, options=row['answer_options'],
                                       prompt=emb_prompt(row))


def assign_donors(frames):
    strata = defaultdict(list)
    for key, frame in frames.items():
        strata[(frame['benchmark'], frame['source'], frame['flip'])].append(key)
    for keys in strata.values():
        keys.sort(key=lambda k: (stable(frames[k]['scene']), k))
        recipients = len(keys)
        donor_keys = keys[recipients // 2:] + keys[:recipients // 2]
        for key, donor in zip(keys, donor_keys):
            assert frames[key]['scene'] != frames[donor]['scene']
            assert frames[key]['image_sha256'] != frames[donor]['image_sha256']
            frames[key]['donor'] = donor


def grid_prompt(name):
    return (f'Divide the image into a 3 by 3 grid. Which cell contains the center of the {name}? '
            'Answer with one option letter.' + ''.join(f'\n{letter}. {cell}' for letter, cell in zip('ABCDEFGHI', GRID)))


def make_probes(cases, metadata, config):
    by_id = {row['id']: row for row in cases}
    original = [row for row in metadata if row['benchmark'] == '3dsrbench' and row['variant'] == row['group']]
    ordered = sorted(original, key=lambda r: stable('sr-role-grid:' + r['group']))
    sr_grid = {row['id'] for row in ordered[:config['sr_role_probe_groups']]}
    probes, details = [], []

    def add(row, task, prompt, **specific):
        case_id = row['id']
        probe_id = f'{task}/{case_id}/{len(probes)}'
        probes.append({'id': probe_id, 'key': by_id[case_id]['key'], 'prompt': prompt})
        details.append({'id': probe_id, 'case_id': case_id, 'task': task, **specific})

    sides = ('front', 'left', 'back', 'right')
    for row in original:
        match = ROLE_RE.search(by_id[row['id']]['prompt'])
        assert match, row['id']
        subject, reference = match.groups()
        add(row, 'sr_camera_side', f'Consider the real-world 3D orientation. Which side of {subject} is facing the camera?' +
            ''.join(f'\n{letter}. {side}' for letter, side in zip('ABCD', sides)), options=list(sides), subject=subject, reference=reference)
        if row['id'] in sr_grid:
            for role, name in (('subject', subject), ('reference', reference)):
                add(row, 'sr_role_grid', grid_prompt(re.sub('^the ', '', name, flags=re.I)), role=role, name=name)

    strata = defaultdict(list)
    for row in metadata:
        if row['benchmark'] == 'embspatial':
            strata[(row['source'], row['topic'])].append(row)
    for stratum in sorted(strata):
        candidates = sorted(strata[stratum], key=lambda r: stable('emb-probes:' + r['id']))
        seen = set()
        panel = []
        for row in candidates:
            key = by_id[row['id']]['key']
            if key not in seen:
                seen.add(key)
                panel.append(row)
        for row in panel[:config['embspatial_probe_per_source_relation']]:
            count = len(row['options'])
            alphabet = 'ABCDEF'[:count]
            for index, name in enumerate(row['options']):
                add(row, 'emb_object_grid', grid_prompt(name), option=index)
            prompt = ('Order the listed objects from nearest to farthest from the camera by depth. '
                      'Write each option letter once, separated by spaces.'
                      + ''.join(f'\n{letter}. {name}' for letter, name in zip(alphabet, row['options'])))
            add(row, 'emb_order', prompt)
    return probes, details


def prepare():
    if (RUN / 'inputs_seal.json').exists():
        raise FileExistsError(RUN / 'inputs_seal.json')
    RUN.mkdir(parents=True, exist_ok=True, mode=0o700)
    config = read_json(CONFIG)
    cases, metadata, frames, baseline_hashes = [], [], {}, {}
    for benchmark in config['scope']:
        directory, _, _ = verified_baseline(benchmark)
        manifest = {row['id']: row for row in read_jsonl(directory / 'inputs.jsonl')}
        source_file = DATA / benchmark / 'source.json'
        source = read_json(source_file)
        for file in source['files']:
            assert sha256(source_file.parent / file['file']) == file['sha256']
        baseline_hashes[benchmark] = {'prediction_seal': sha256(directory / 'predictions_seal.json'),
                                      'score_seal': sha256(directory / 'score_seal.json'),
                                      'dataset_source': sha256(source_file)}
        for dataset_id, row in benchmark_rows(benchmark):
            image = manifest[dataset_id]
            assert row.pop('prompt') == image['prompt']
            key = stable(benchmark + '/' + image['image_sha256'] + '/' + image['transform'])
            frame = {'benchmark': benchmark, 'image_sha256': image['image_sha256'], 'transform': image['transform'],
                     'source': row['source'], 'scene': row['scene'], 'flip': row['flip']}
            assert frame == frames.setdefault(key, frame), dataset_id
            qid = benchmark + '/' + dataset_id
            cases.append(dict(id=qid, key=key, prompt=image['prompt']))
            metadata.append(dict(id=qid, dataset_id=dataset_id, benchmark=benchmark, **row))
    focus = config['focus_counts']
    assert sum(r['benchmark'] == '3dsrbench' for r in metadata) == focus['3dsrbench_rows']
    assert len({r['group'] for r in metadata if r['benchmark'] == '3dsrbench'}) == focus['3dsrbench_groups']
    assert sum(r['benchmark'] == 'embspatial' for r in metadata) == focus['embspatial_rows']
    assign_donors(frames)
    probes, details = make_probes(cases, metadata, config)
    for name, table in (('cases', cases), ('metadata', metadata), ('probes', probes), ('probe_metadata', details)):
        jsonl(RUN / f'{name}.jsonl', table)
    write_json(RUN / 'frames.json', frames)
    seal = dict(config_sha256=sha256(CONFIG), baseline=baseline_hashes,
                files={f: sha256(RUN / f) for f in ('cases.jsonl', 'metadata.jsonl', 'frames.json', 'probes.jsonl', 'probe_metadata.jsonl')},
                task_rows=len(cases), probe_rows=len(probes), frame_rows=len(frames),
                labels_in_inference_manifest=False)
    write_json(RUN / 'inputs_seal.json', seal)
    print({k: seal[k] for k in ('task_rows', 'probe_rows', 'frame_rows')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    prepare()
