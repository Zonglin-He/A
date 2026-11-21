"""Bind EmbSpatial option nouns to physical regions using one frozen HiSpatial."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from download import read_json, sha256, write_json
from geometry_audit import select_letter
from geometry_cohort import stable
from object_region_plan import CONFIG, DIAG, MARKS, RUN, index_rows
from predict import BATCH_N, batch_query, cached_single, feature, release
from score import jsonl


def concrete(name):
    return name.strip().casefold() != 'unknown'


def mark_prompt(name, count):
    choices = ''.join(f'\n{letter}. Marked region {letter}' for letter in MARKS[:count])
    return ('Which marked region contains the object named ' + json.dumps(name, ensure_ascii=False)
            + '? Read the letter drawn inside that region. Answer with only the option letter.'
            + choices + f'\n{MARKS[count]}. No marked region contains that object')


def task_layout(roster, regions):
    tasks = {}
    references = {}
    for item in roster.values():
        q = item['id']
        key = item['key']
        count = len(regions[key]['marked'])
        whole_id = 'whole/' + q
        prompt = ('The colored overlay shows modeled camera depth. Warm red tones mark closer surfaces and cool blue tones mark farther surfaces. '
                  'Choose the correct option letter.\n' + item['prompt'])
        tasks[whole_id] = dict(id=whole_id,key=key,view='whole',prompt=prompt)
        sides = {}
        for view in (('marks','reversed') if item['order_control'] else ('marks',)):
            ids = []
            for name in item['options']:
                probe = view+'/'+key+'/'+stable(name)
                if not concrete(name) or count == 0:
                    ids.append(None)
                    continue
                tasks.setdefault(probe, dict(id=probe,key=key,view=view,prompt=mark_prompt(name,count)))
                ids.append(probe)
            sides[view] = ids
        references[q] = (whole_id, sides)
    return tasks, references


def unique_extreme(depth, topic):
    sequence = sorted(depth, key=lambda i: depth[i], reverse=(topic=='far'))
    if len(sequence)<2 or abs(depth[sequence[0]]-depth[sequence[1]])<=1e-6:
        return None
    return 'ABCD'[sequence[0]]


def seal_readouts(roster, regions, references, captured):
    previous = read_json(DIAG / 'predictions_seal.json')
    baseline_path = DIAG / 'condition_predictions.jsonl'
    assert sha256(baseline_path) == previous['files'][baseline_path.name]
    baseline = index_rows(baseline_path)
    outputs = []
    for q, item in roster.items():
        whole, views = references[q]
        candidates = regions[item['key']]['marked']
        model_bindings = {}
        for mode, calls in views.items():
            ids = [r['id'] for r in candidates]
            if mode == 'reversed':
                ids.reverse()
            letters = dict(zip(MARKS, ids))
            bind = []
            for name, call in zip(item['options'], calls, strict=True):
                response = captured[call]['response'] if call is not None else None
                letter = select_letter(response, MARKS[:len(ids)+1]) if response is not None else None
                region = letters.get(letter)
                if not concrete(name):
                    reason = 'abstract_option'
                elif call is None:
                    reason = 'empty_pool'
                elif region is not None:
                    reason = 'bound'
                elif letter == MARKS[len(ids)]:
                    reason = 'abstained'
                else:
                    reason = 'unparsed'
                bind.append(dict(option=name,region=region,response=response,letter=letter,status=reason))
            model_bindings[mode] = bind

        data = {r['id']: r for r in candidates}
        selected = {}
        for j,b in enumerate(model_bindings['marks']):
            if b['region'] is not None:
                selected[j] = data[b['region']]
                b['region_z'] = data[b['region']]['region_z']
                b['window_z'] = data[b['region']]['window_z']
                b['visible_centroid'] = data[b['region']]['visible_centroid']
        initial = baseline[q]['aligned']
        both = dict(original_hispatial=initial,whole_map_depth=captured[whole]['response'],
                    region_depth=initial,window_depth=initial)
        count = sum(concrete(name) for name in item['options'])
        if len(selected) != count:
            decision = 'incomplete'
        elif len({r['id'] for r in selected.values()}) != count:
            decision = 'collision'
        else:
            region = unique_extreme({j:r['region_z'] for j,r in selected.items()},item['topic'])
            window = unique_extreme({j:r['window_z'] for j,r in selected.items()},item['topic'])
            if region and window:
                decision = 'complete'
                both.update(region_depth=region,window_depth=window)
            else:
                decision = 'tie'
        outputs.append(dict(id=q, key=item['key'], bindings=model_bindings, decision=decision, answers=both))
    return outputs


def run(model_cache):
    if (RUN / 'prediction_seal.json').exists():
        raise FileExistsError(RUN / 'prediction_seal.json')
    torch.manual_seed(20261121)
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    cv2.setNumThreads(1)
    seal = read_json(RUN / 'input_seal.json')
    assert seal['protocol']==sha256(CONFIG) and seal['label_access'] is False
    for k, digest in seal['files'].items():
        assert sha256(RUN / k)==digest
    assert seal['diagnosis_predictions']==sha256(DIAG / 'predictions_seal.json')
    roster = index_rows(RUN / 'roster.jsonl')
    regions = read_json(RUN / 'regions.json')
    views = read_json(RUN / 'views_sha256.json')
    tasks, names = task_layout(roster, regions)
    rows = sorted(tasks.values(),key=lambda r:(r['key'],r['view'],r['id']))
    assert len(roster)==seal['rows']==1206
    assert len(regions)==len(views)==seal['frames']
    model, _, runtime = release(model_cache)
    assert runtime==read_json(DIAG / 'runtime.json')['runtime']

    @lru_cache(maxsize=10)
    def prepare(key, kind):
        path = RUN / 'views' / f'{key}.{kind}.png'
        assert sha256(path)==views[key][kind]
        img = cv2.imread(str(path),cv2.IMREAD_COLOR)[...,::-1].copy()
        assert img.shape==(448,448,3)
        xyz_path = DIAG / 'pointmaps' / f'{key}.npy'
        assert sha256(xyz_path)==regions[key]['xyz_sha256']
        xyz = torch.from_numpy(np.load(xyz_path,allow_pickle=False))
        return img,xyz,feature(model,img,xyz)

    def infer(group, batch):
        prepared = [prepare(r['key'],r['view']) for r in group]
        if batch and len(group)>1:
            return batch_query(model,prepared,group)
        return [cached_single(model,p,r['prompt']) for p,r in zip(prepared,group,strict=True)]

    exact = {}
    for kind in ('marks','reversed','whole'):
        selected = sorted((r for r in rows if r['view']==kind),key=lambda r: stable('self-grounding-anchor/'+r['id']))[:16]
        matches = 0
        for start in range(0,len(selected),BATCH_N):
            sample = selected[start:start+BATCH_N]
            a, b = infer(sample, False), infer(sample, True)
            matches += sum(x==y for x,y in zip(a,b,strict=True))
        exact[kind] = dict(queries=len(selected),matches=matches)
    batch = BATCH_N if all(v['queries']==v['matches'] for v in exact.values()) else 1
    write_json(RUN / 'runtime.json',dict(runtime=runtime,batch_size=batch,parity=exact,total_queries=len(rows)))
    print('batch match',exact,'batch',batch,'unique requests',len(rows),flush=True)

    began = time.monotonic()
    partial = RUN / 'query_responses.jsonl.tmp'
    capture = {}
    next_log = 100
    with partial.open('w') as stream:
        for j in range(0,len(rows),batch):
            group = rows[j:j+batch]
            responses = infer(group,batch>1)
            for r,text in zip(group,responses,strict=True):
                record = dict(id=r['id'],view=r['view'],response=text)
                capture[r['id']] = record
                stream.write(json.dumps(record,ensure_ascii=False)+'\n')
            stream.flush()
            if len(capture)>=next_log or len(capture)==len(rows):
                print('queries',len(capture),'/',len(rows),round(time.monotonic()-began,1),'s',flush=True)
                next_log += 100
    assert set(capture)==set(tasks)
    partial.replace(RUN / 'query_responses.jsonl')
    results = seal_readouts(roster,regions,names,capture)
    jsonl(RUN / 'predictions.jsonl',results)
    assert len(results)==1206 and not any(p.requires_grad for p in model.model.parameters())
    write_json(RUN / 'prediction_seal.json',dict(
        rows=len(results),queries=len(capture),training_steps=0,label_access=False,
        inputs=sha256(RUN / 'input_seal.json'),model=sha256(RUN / 'runtime.json'),
        files={n:sha256(RUN/n) for n in ('predictions.jsonl','query_responses.jsonl','runtime.json')},
        gpu_seconds=time.monotonic()-began,peak_cuda_bytes=torch.cuda.max_memory_allocated()))
    print('sealed',len(results),len(capture),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-cache',type=Path,required=True)
    run(parser.parse_args().model_cache)
