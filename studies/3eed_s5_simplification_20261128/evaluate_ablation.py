"""Replay the fixed pool/ranker crosses and score after both prediction seals."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import sys
from time import perf_counter_ns

import cv2
import numpy as np
from safetensors.torch import load_file
import torch

from pool_ablation import HERE,POOLS,axes,edge_features,generate_all,tensors
from source_banks import BANKS,RUN
from train_ablation import FILES as TRAIN_FILES,margins
from rankers import CONTROL,DESIGNS as CORE_DESIGNS,model_for,transform
from blend_ablation import BLEND
from eval_minimal_verifier import dev_jobs,development_labels,external_labels,mono_jobs
from cache_mono import guard
from provenance import RUN as OLD_RUN,SOURCE,digest,read,retain,runtime,write
from score_paradigms import ious

FILES = ('evaluate_ablation.py','pool_ablation.py','rankers.py','blend_ablation.py','PROTOCOL_ZH.md')
DESIGNS = {**CORE_DESIGNS,**BLEND}
PHASES = ('original','proposal','blend')
FIXED = {f'{name}-old':pool for name,pool in
         (('No90','no90'),('Extent','extent'),('Geometry','geometry'),('Adaptive','adaptive'))}
NAMES = (*DESIGNS,*FIXED)


def code():
    return {n:digest(HERE/n) for n in FILES}


def evidence(job):
    case,row,_ = job
    frame,replies = row['frame'],row['raw']
    pool,tensor = {},{}
    for collection,view,anchor in generate_all(frame,replies):
        if case is not None and 'base' in collection:
            assert anchor==case['base'],frame['uid']
        pool.update(collection)
        tensor.update(tensors(collection,view))
    return dict(uid=frame['uid'],target=row['target'],pools=pool),tensor


def source_models():
    models,cards,pins = {},{},{}
    for phase in PHASES:
        path = BANKS/f'{phase}-training.json'
        value = read(path)
        assert value['source_only']
        assert all(value['code'][n]==digest(HERE/n) for n in TRAIN_FILES)
        if CONTROL in cards and 'reference' in value:
            assert cards[CONTROL]==value['reference']
        if 'reference' in value:
            cards[CONTROL] = value['reference']
        cards.update(value['cards'])
        pins[phase] = digest(path)
    for name,design in DESIGNS.items():
        model = model_for(design)
        path = (SOURCE if name==CONTROL else BANKS)/f'{name}.safetensors'
        assert cards[name]['checkpoint']==digest(path)
        model.load_state_dict(load_file(path))
        models[name] = model.cuda().eval()
    return models,cards,pins


def choices(batch,model,design,threshold):
    data = tuple(torch.as_tensor(np.stack([d[design.pool][j] for d in batch]),device='cuda') for j in range(3))
    x,e,valid = transform(data,design)
    scored = margins(model,(x,e,valid,None))
    use = scored[:,1]>threshold if threshold is not None else np.zeros(len(scored),bool)
    return np.where(use,scored[:,0],0).astype(int).tolist()


def predict():
    sys.addaudithook(guard)
    runtime()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    RUN.mkdir(exist_ok=True)
    with ProcessPoolExecutor(8) as workers:
        list(workers.map(len,[[0]]*8))
        models,cards,source = source_models()
        for scope,(jobs,pins) in (('development',dev_jobs()),('mono',mono_jobs())):
            old = OLD_RUN/'passed-stack'/f'{scope}-predictions-private.json'
            locked = read(OLD_RUN/'passed-stack'/f'{scope}-seal.json')
            assert locked['payload']==digest(old) and not locked['labels_read']
            expected = {r['uid']:r for r in read(old)}
            output = []
            for i in range(0,len(jobs),64):
                rows,batch = zip(*workers.map(evidence,jobs[i:i+64],chunksize=3),strict=True)
                selected = {n:choices(batch,models[n],d,cards[n]['threshold']) for n,d in DESIGNS.items()}
                selected.update({n:choices(batch,models[CONTROL],replace(DESIGNS[CONTROL],pool=pool),
                                           cards[CONTROL]['threshold']) for n,pool in FIXED.items()})
                for j,row in enumerate(rows):
                    row['choices'] = {n:seq[j] for n,seq in selected.items()}
                    baseline = expected[row['uid']]
                    keep = np.flatnonzero(retain(baseline['pool'])[:,1])
                    assert row['pools']['base']==[baseline['pool'][k]['box'] for k in keep]
                    chosen = row['pools']['base'][row['choices'][CONTROL]]
                    assert chosen==baseline['pool'][baseline['choice'][CONTROL]]['box'],row['uid']
                    output.append(row)
                print('Predictions',scope,len(output),'/',len(jobs),flush=True)
            file = RUN/f'{scope}-predictions-private.json'
            write(file,output)
            write(RUN/f'{scope}-seal.json',dict(payload=digest(file),n=len(output),inputs=pins,source=source,
                                               original=locked['payload'],code=code(),labels_read=False))


def summarize(rows):
    from evaluate_structure import stat_panel

    names = (*NAMES,'Minimal-D')
    out = stat_panel(rows,names,CONTROL)
    out['pool_oracle'] = stat_panel([dict(r,iou=r['oracle']) for r in rows],POOLS,'base')['metrics']
    out['pool_anchor'] = stat_panel([dict(r,iou=r['anchor']) for r in rows],POOLS,'base')['metrics']
    out['avg_pool'] = {p:float(np.mean([r['counts'][p] for r in rows])) for p in POOLS}
    out['replaced'] = {n:sum(r['replaced'][n] for r in rows) for n in NAMES}
    return out


def query(row,truth):
    values = {n:ious(pool,truth['box']) for n,pool in row['pools'].items()}
    pools = {**{n:d.pool for n,d in DESIGNS.items()},**FIXED}
    return dict(recording=truth['recording'],target=row['target'],
                iou={**{n:float(values[p][row['choices'][n]]) for n,p in pools.items()},
                     'Minimal-D':float(values['base'][0])},
                oracle={n:float(v.max()) for n,v in values.items()},
                anchor={n:float(v[0]) for n,v in values.items()},
                counts={n:len(v) for n,v in values.items()},
                replaced={n:row['choices'][n]!=0 for n in NAMES})


def benchmark(rows):
    # Recompute both actual pair layouts on the same saved boxes.
    matched = [r['pools']['base'] for r in rows if len(r['pools']['base'])>1]
    times = {False:[],True:[]}
    relations = {False:0,True:0}
    for i,pool in enumerate(matched):
        boxes = np.asarray(pool)
        frames = np.asarray([axes(b[6]) for b in boxes])
        if i==0:
            edge_features(boxes,frames)
            edge_features(boxes,frames,True)
        for anchor in ((False,True) if i%2==0 else (True,False)):
            start = perf_counter_ns()
            edge = edge_features(boxes,frames,anchor)
            times[anchor].append(perf_counter_ns()-start)
            relations[anchor] += int(np.prod(edge.shape[:-1]))
            if anchor:
                linear = edge
            else:
                dense = edge[0]
        np.testing.assert_allclose(linear,dense,rtol=1.e-6,atol=1.e-6)
    return dict(n=len(matched),pairs_full=relations[False],pairs_anchor=relations[True],
                median_full_ms=float(np.median(times[False])/1.e6),
                median_anchor_ms=float(np.median(times[True])/1.e6))


def score():
    from shared import setup
    setup()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    target,seen = {},{}
    source = {p:read(BANKS/f'{p}-training.json') for p in PHASES}
    pins = {p:digest(BANKS/f'{p}-training.json') for p in source}
    for scope in ('development','mono'):
        seal = read(RUN/f'{scope}-seal.json')
        path = RUN/f'{scope}-predictions-private.json'
        assert seal['source']==pins and seal['code']==code()
        assert seal['payload']==digest(path) and not seal['labels_read']
        seen[scope] = read(path),seal
    previous = read(OLD_RUN/'passed-stack/results.json')
    for scope,truth in (('development',development_labels()),('mono',external_labels())):
        rows,seal = seen[scope]
        pairs = [query(r,truth[r['uid']]) for r in rows if r['uid'] in truth]
        assert len(pairs)==len(truth)==(665 if scope=='development' else 148)
        panels = {'overall':summarize(pairs)}
        for key in sorted({r['target'] for r in pairs}):
            panels[key] = summarize([r for r in pairs if r['target']==key])
        out = panels['overall']['metrics']
        old = previous['target'][scope]['panel']['overall']['metrics'][CONTROL]
        assert all(abs(out[CONTROL][k]-old[k])<1.e-10 for k in ('miou','acc50'))
        timing = benchmark([r for r in rows if r['uid'] in truth])
        target[scope] = dict(panels=panels,prediction=seal,relation_timing=timing)
        print(scope,[(n,round(r['miou'],3),round(r['acc50'],3)) for n,r in out.items()],flush=True)
    write(RUN/'results.json',dict(source=source,target=target,labels_after_both_seals=True))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('predict','score'))
    dict(predict=predict,score=score)[parser.parse_args().phase]()
