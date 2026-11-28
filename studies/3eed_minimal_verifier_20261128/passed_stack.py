"""Evaluate the mechanical stack of individually admitted source simplifications."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import sys

import cv2
import numpy as np
from safetensors.torch import load_file
import torch

from cache_mono import guard
from eval_minimal_verifier import (development_labels,dev_jobs,external_labels,mono_jobs,
                                   receipt,replay,scores)
from fit import fit,load_bank,passes,read_masks,source_panel
from models import Arm,model_for
from provenance import HERE,RUN,SOURCE,V2_SOURCE,digest,read,runtime,source_guard,write

NAME = 'S5-pass'
OUT = RUN/'passed-stack'


def stack(stamp):
    passed = stamp['passed']
    cap = 13 if passed['S1-13'] else 17 if passed['S1'] else 21
    return Arm(cap,passed['S2'],passed['S3'],passed['S4'])


def source():
    source_guard()
    locked = receipt()
    arm = stack(locked)
    assert arm!=Arm(**locked['arms'][locked['selected']]['arm'])
    runtime()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    bank = read(V2_SOURCE/'features-seal.json')
    tag = SOURCE/'membership-seal.json'
    mask = read(tag)
    assert digest(tag)==locked['membership']
    data,_ = load_bank('train',bank)
    validation,meta = load_bank('val',bank)
    membership = tuple(read_masks(k,bank,mask) for k in ('train','val'))
    degraded = meta['modes']!=0
    baseline = validation[-1][:,0].cpu().numpy()
    card,record = fit(NAME,arm,data,validation,membership,degraded)
    card['clean'] = source_panel(record,baseline,~degraded)
    card['degraded'] = source_panel(record,baseline,degraded)
    file = SOURCE/'passed-stack-seal.json'
    write(file,dict(card=card,passed=passes(card,locked['arms']['S0']),source_only=True,
          primary=digest(SOURCE/'selection-seal.json'),code=digest(__file__)))
    print(NAME,card['arm'],card['params'],card['degraded'],flush=True)


def pinned():
    locked = receipt()
    seal = read(SOURCE/'passed-stack-seal.json')
    assert seal['source_only'] and seal['primary']==digest(SOURCE/'selection-seal.json')
    assert seal['code']==digest(__file__) and seal['card']['arm']==asdict(stack(locked))
    return locked,seal


def predict():
    locked,seal = pinned()
    sys.addaudithook(guard)
    runtime()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    cv2.setNumThreads(1)
    OUT.mkdir(exist_ok=True)
    files = ((V2_SOURCE/'verifier.safetensors','S0',locked['arms']['S0']),
             (SOURCE/f'{NAME}.safetensors',NAME,seal['card']))
    tasks = dict(development=dev_jobs(),mono=mono_jobs())
    with ProcessPoolExecutor(8) as workers:
        list(workers.map(len,[[0]]*8))
        models = {}
        for path,name,card in files:
            assert card['checkpoint']==digest(path)
            arm = Arm(**card['arm'])
            model = model_for(arm)
            model.load_state_dict(load_file(path))
            models[name] = (model.cuda().eval(),arm,card['threshold'])
        for name,(jobs,pins) in tasks.items():
            predictions,timing = replay(jobs,models,workers)
            file = OUT/f'{name}-predictions-private.json'
            write(file,predictions)
            write(OUT/f'{name}-seal.json',dict(payload=digest(file),pins=pins,labels_read=False,
                  n=len(predictions),source=digest(SOURCE/'passed-stack-seal.json'),code=digest(__file__),
                  timing=timing))


def score():
    _,seal = pinned()
    from shared import setup
    setup()
    torch.set_num_threads(1)
    blind = {}
    for panel in ('development','mono'):
        stamp = read(OUT/f'{panel}-seal.json')
        file = OUT/f'{panel}-predictions-private.json'
        assert stamp['payload']==digest(file) and stamp['code']==digest(__file__) and not stamp['labels_read']
        assert stamp['source']==digest(SOURCE/'passed-stack-seal.json')
        blind[panel] = (read(file),stamp)
    primary = read(RUN/'results.json')
    summary = {}
    for panel,ref in (('development',development_labels()),('mono',external_labels())):
        preds,stamp = blind[panel]
        paired,rows = scores(preds,ref,('S0',NAME))
        assert paired['overall']['metrics']['S0']==primary['target'][panel]['panel']['overall']['metrics']['S0']
        detail = OUT/f'{panel}-scores-private.json'
        write(detail,rows)
        summary[panel] = dict(panel=paired,score_sha=digest(detail),prediction=stamp)
        print(panel,paired['overall']['metrics'][NAME],flush=True)
    write(OUT/'results.json',dict(source=seal,target=summary,labels_after_both_seals=True,
          primary_choice=primary['source_selection']))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('source','predict','score'))
    dict(source=source,predict=predict,score=score)[parser.parse_args().phase]()
