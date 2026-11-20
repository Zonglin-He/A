"""Freeze two caption scores on the same physical four-box slate, then seal."""
import argparse
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

from calibrate import HERE, ORD, OUT, PLATFORMS, ROOT, SOURCE, SUP, WORK, digest, read, save, source_rows
from factor import ARMS, horizontal, image_word, scores

PAIR = HERE.parent/'3eed_v1_paired_description_subject_readout_20261120'
sys.path[:0] = [str(PAIR), str(SUP), str(ORD)]
from paired import archive_npz, fixed_prior
from scope import evaluation_rows
from scope183 import check_input, ordinary
from crossplatform.candidate_frontend import WAYMO_VIEWS, sensor_frame_box
from crossplatform.ground import Grounder

BLOCK = 64


def inputs(stage):
    if stage == 'source':
        return {(s,s):source_rows(s,'calibration') for s in PLATFORMS}
    if stage == 'pilot':
        rows = evaluation_rows()
        return {(s,t):[r for r in rows if r['platform']==t] for s in PLATFORMS for t in PLATFORMS if s!=t}
    if stage == 'ordinary':
        rows = ordinary()
        return {(s,t):rows[t] for s in PLATFORMS for t in PLATFORMS if s!=t}
    raise ValueError(stage)


def checked_baseline(stage, source, target):
    if stage != 'pilot':
        return None
    folder = HERE.parent/'3eed_v1_frontend_simplification_20261119'/'predictions_PRIVATE'
    name = f'{source}_to_{target}.npz'
    assert digest(folder/'SEALED.json') == read(SUP/'protocol.json')['baseline_predictions_sha256']
    assert digest(folder/name) == read(folder/'SEALED.json')['files'][name]['sha256']
    with np.load(folder/name,allow_pickle=False) as prior:
        return prior['boxes'].copy()


def ordinary_baseline(direction, first, count):
    folder = ORD/'predictions_PRIVATE'
    block = first//128*128
    name = f'{block:05d}.npz'
    file = folder/direction/name
    assert digest(file) == read(folder/'SEALED.json')['directions'][direction]['files'][name]['sha256']
    with np.load(file, allow_pickle=False) as z:
        chosen = z['choices'][first-block:first-block+count,0]
        boxes = z['four'][first-block:first-block+count]
        return boxes, chosen


def frame(row):
    return dict(target=row['platform'],points=row['pointpath'],rgb=row['imagepath'],
                calibration=row['metapath'],text=row['caption'],
                seed=int(row.get('seed',0)),waymo_view=WAYMO_VIEWS[int(row['suffix'] or 0)])


def intervention_order(caption):
    order = [hashlib.sha256(f'{slot}|camera-word|{caption}'.encode()).digest() for slot in range(4)]
    return np.argsort(np.argsort(order))


@torch.inference_mode()
def observe(model, inp, slope):
    word = image_word(inp['text'])
    saved = {}
    def remember(module, args, output):
        saved.update({k:output[k] for k in ('fp2_xyz','fp2_features','fp2_inds')})
    with model._frontend.base.register_forward_hook(remember):
        scene, ev = model._read_observation(inp)
    base = model._decoder(ev)['subject_logits'][0].double().cpu().numpy()
    anchors = ev['subject_boxes'][0].cpu().numpy()
    boxes = np.stack([sensor_frame_box(b,scene.target,scene.view_index) for b in anchors])
    valid = np.zeros(4, dtype=bool)
    coordinate = np.zeros(4)
    alternate = base.copy()
    prior_err = 0.
    if word.sign:
        assert image_word(word.flipped).flipped == inp['text']
        coordinate, view = horizontal(anchors[:,:3],scene.target,scene.view_index,inp['calibration'],inp['rgb'])
        old_prior, old_ok = fixed_prior(anchors,scene)
        actual_prior = ev['base_logit'][0].cpu().numpy()
        # Zero-volume native inputs have singleton votes, and cannot be transported.
        if old_ok.any():
            prior_err = float(np.max(np.abs(old_prior[old_ok]-actual_prior[old_ok])))
            if prior_err > 5e-5:
                raise RuntimeError(f'Factual overlap-vote reconstruction differs by {prior_err:g}')
        if np.count_nonzero(view & old_ok) >= 2:
            # Leave language-conditioned query sampling free; transfer by physical intersection.
            with patch.object(model._frontend.base.backbone_net,'forward',return_value=dict(saved)):
                other, alternative_evidence = model._read_observation(dict(inp,text=word.flipped))
            np.testing.assert_array_equal(scene.sampled_points, other.sampled_points)
            prior, overlap = fixed_prior(anchors,other)
            alternative_evidence['subject_boxes'] = ev['subject_boxes']
            alternative_evidence['base_logit'] = torch.as_tensor(prior[None],device=ev['base_logit'].device)
            alternate = model._decoder(alternative_evidence)['subject_logits'][0].double().cpu().numpy()
            valid = old_ok & overlap & view
    order = intervention_order(inp['text'])
    logits = scores(base, alternate, coordinate, valid, word.sign, slope, order)
    assert logits[0].argmax() == base.argmax()
    return dict(boxes=boxes,base=base,other=alternate,coordinate=coordinate,valid=valid,sign=word.sign,
                order=order,logits=logits,choices=logits.argmax(1).astype(np.int8),prior_error=prior_err)


def predict(stage):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    cal = read(OUT/'calibration.json')
    assert cal['matcher'] == digest(HERE/'factor.py') and cal['protocol'] == digest(HERE/'PROTOCOL.json')
    if stage == 'pilot':
        assert cal['improves_null'] and read(WORK/'source'/'SEALED.json')['comparable']>0
    if stage == 'ordinary':
        assert read(OUT/'pilot_decision.json')['expand']
    root = WORK/stage
    root.mkdir(exist_ok=True,mode=0o700)
    roster = inputs(stage)
    cfg = dict(stage=stage,code={p:digest(HERE/p) for p in ('predict.py','calibrate.py','factor.py','PROTOCOL.json')},
               calibration=digest(OUT/'calibration.json'),arms=list(ARMS),
               source_assets=digest(ROOT/'crossplatform/single_frame_sources.json'),
               roster={f'{s}_to_{t}':hashlib.sha256('\n'.join(
                   f"{r['pointpath']}|{r['metapath']}|{r['caption']}|{r.get('seed',0)}" for r in rows
               ).encode()).hexdigest() for (s,t), rows in roster.items()})
    assert cfg['source_assets'] == read(HERE/'PROTOCOL.json')['registered_sources']
    if (root/'LOCK.json').exists():
        assert read(root/'LOCK.json') == cfg
    else:
        save(root/'LOCK.json',cfg)
    model = None
    last_source = None
    blocks, start, valid_count = {}, perf_counter(), 0
    for (src,tgt), rows in roster.items():
        direction = f'{src}_to_{tgt}'
        directory = root/direction
        directory.mkdir(exist_ok=True)
        old = checked_baseline(stage,src,tgt)
        for first in range(0,len(rows),BLOCK):
            subset = rows[first:first+BLOCK]
            path = directory/f'{first:05d}.npz'
            receipt = path.with_suffix('.json')
            if not receipt.exists():
                if last_source != src:
                    if model is not None:
                        del model
                        torch.cuda.empty_cache()
                    model = Grounder(src)
                    model._frontend.parser = lru_cache(maxsize=2048)(model._frontend.parser)
                    last_source = src
                result = []
                for row in subset:
                    if stage == 'ordinary':
                        check_input(row)
                    result.append(observe(model,frame(row),cal['slopes'][src]))
                data = {k:np.stack([r[k] for r in result]) for k in result[0]}
                if old is not None:
                    np.testing.assert_array_equal(data['boxes'][np.arange(len(subset)),data['choices'][:,0]],
                                                  old[first:first+len(subset)])
                if stage == 'ordinary':
                    orig_boxes, orig_picks = ordinary_baseline(direction,first,len(subset))
                    np.testing.assert_array_equal(data['boxes'],orig_boxes)
                    np.testing.assert_array_equal(data['choices'][:,0],orig_picks)
                archive_npz(path,**data)
                save(receipt,dict(n=len(subset),sha256=digest(path),
                                  comparable=int((data['valid'].sum(1)>=2).sum())))
            pin = read(receipt)
            assert pin['sha256'] == digest(path) and pin['n'] == len(subset)
            blocks[f'{direction}/{path.name}'] = pin
            valid_count += pin['comparable']
            print(stage,direction,first+len(subset),'/',len(rows),round(perf_counter()-start,1),flush=True)
    n = sum(x['n'] for x in blocks.values())
    assert n == {'source':384,'pilot':3072,'ordinary':18620}[stage]
    save(root/'SEALED.json',dict(n=n,comparable=valid_count,files=blocks,lock=digest(root/'LOCK.json'),
                                elapsed=perf_counter()-start,task_labels_opened=False,FINAL_access=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('source','pilot','ordinary'))
    predict(parser.parse_args().stage)
