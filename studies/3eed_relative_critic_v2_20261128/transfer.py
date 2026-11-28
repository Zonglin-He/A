"""Run frozen zero-shot transfer only after the Waymo validation gate."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter

import cv2
import numpy as np
from safetensors.torch import load_file
import torch

from compact_pool import HERE, PRIOR, RUN, digest, read, sealed_inputs, write
from score_paradigms import ious
from source import SOURCE, gate
from source_features import pool_features
from train import decision
from verifier import RelativeVerifier, top_nonanchor

TARGET = RUN/'target'


def training():
    gate()
    receipt = read(SOURCE/'training-seal.json')
    assert receipt['stage'] == 'go' and receipt['source_only'], 'Waymo validation gate not met'
    assert receipt['features'] == digest(SOURCE/'features-seal.json')
    assert all(digest(HERE/file)==sha for file,sha in receipt['code'].items())
    checkpoint = SOURCE/'verifier.safetensors'
    assert receipt['checkpoint'] == digest(checkpoint)
    return receipt, checkpoint


def features(task):
    case,row,expected = task
    pack, (nodes,edges,valid) = pool_features(case,row['frame'],row['raw'])
    if expected is not None:
        assert pack == expected, case['uid']
    return pack,nodes,edges,valid


@torch.inference_mode()
def infer(tasks):
    receipt, checkpoint = training()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    cv2.setNumThreads(1)
    model = RelativeVerifier()
    model.load_state_dict(load_file(checkpoint))
    model.eval()
    rows = []
    with ProcessPoolExecutor(8) as pool:
        for i,batch in enumerate(tasks):
            pack,x,e,v = zip(*pool.map(features,batch,chunksize=4),strict=True)
            if i == 0:
                model.to('cuda')
            nodes,edges,valid = (torch.as_tensor(np.stack(a),device='cuda') for a in (x,e,v))
            data = model(nodes,edges,valid)
            val = torch.stack(top_nonanchor(*data,valid),1).cpu().numpy()
            apply = decision(val,receipt['thresholds'])
            aabb = nodes[:,:,20].masked_fill(~valid,-torch.inf).argmax(1).tolist()
            for p,z,replace,vis in zip(pack,val,apply,aabb,strict=True):
                raw = int(z[0])
                p['selection'] = dict(minimal=0,aabb=vis,rank=raw if z[1]>0 else 0,verifier=raw if replace else 0)
                rows.append(p)
    return rows


def batches(jobs, width=64):
    for i in range(0,len(jobs),width):
        yield jobs[i:i+width]


def predict():
    training()
    source,_,_,pins,_ = sealed_inputs()
    case_file = PRIOR/'inputs-private.json'
    assert read(PRIOR/'input-seal.json')['evidence'] == digest(case_file)
    cases = [c for c in read(case_file) if c['target'] != 'mono']
    frozen = RUN/'candidates-private.json'
    assert read(RUN/'candidate-seal.json')['payload'] == digest(frozen)
    pools = {r['uid']:r for r in read(frozen)}
    assert len(cases) == len(pools) == 702
    TARGET.mkdir(exist_ok=True)
    start = perf_counter()
    jobs = [(c,source[c['uid']],pools[c['uid']]) for c in cases]
    rows = infer(batches(jobs))
    file = TARGET/'development-predictions-private.json'
    write(file,rows)
    write(TARGET/'development-prediction-seal.json',dict(
        payload=digest(file),n=len(rows),input=pins,train=digest(SOURCE/'training-seal.json'),
        source_pool=digest(frozen),labels_read=False,code=digest(__file__),seconds=perf_counter()-start))


def paired_summary(predictions, labels, roster):
    from evaluate_structure import stat_panel

    result = []
    for row in predictions:
        uid = row['uid']
        if uid not in roster:
            continue
        yt = ious([v['box'] for v in row['pool']],labels[uid]['box'])
        m = {k:float(yt[idx]) for k,idx in row['selection'].items()}
        if 'iou' in roster[uid]:
            assert abs(m['minimal']-roster[uid]['iou']['Minimal-D']) < 1.e-9
        reasons = row['selection']['verifier']
        result.append(dict(uid=uid,target=row['target'], recording=roster[uid]['recording'],
                           iou=dict(m,oracle=float(yt.max())),replaced=reasons!=0, branch=row['branch']))
    assert len(result) == len(roster)
    panel = dict(overall=result)
    panel.update({name:[r for r in result if r['target']==name] for name in sorted({r['target'] for r in result})})
    panel.update({name:[r for r in result if r['branch']==name] for name in sorted({r['branch'] for r in result})})
    summary = {}
    for name,subset in panel.items():
        methods = ('minimal','aabb','rank','verifier','oracle')
        summary[name] = stat_panel(subset,methods,'minimal')
        summary[name]['vs_aabb'] = stat_panel(subset,('aabb','verifier'),'aabb')['metrics']['verifier']
        changes = [r['iou']['verifier']-r['iou']['minimal'] for r in subset if r['replaced']]
        summary[name]['replacements'] = dict(total=len(changes), improved=sum(d>1.e-8 for d in changes),
                                              worsened=sum(d< -1.e-8 for d in changes),
                                              tied=sum(abs(d)<=1.e-8 for d in changes))
    return summary,result


def score():
    training()
    file = TARGET/'development-predictions-private.json'
    stamp = read(TARGET/'development-prediction-seal.json')
    assert stamp['payload'] == digest(file) and not stamp['labels_read']
    assert stamp['train'] == digest(SOURCE/'training-seal.json') and stamp['code'] == digest(__file__)
    previous = PRIOR/'scores-private.json'
    assert digest(previous) == read(PRIOR/'results.json')['offline_score_sha']
    refs = {r['uid']:r for r in read(previous)['critic'] if r['target'] != 'mono'}
    assert len(refs) == 665
    from shared import setup
    setup()
    from metric_image_common import inputs
    from truth_score_inputs import truth_data

    frames = inputs()
    actual,_ = truth_data({u:frames[u] for u in refs})
    summary,rows = paired_summary(read(file),actual,refs)
    base = summary['overall']['metrics']['verifier']
    platforms = ('waymo','drone','quad','kitti')
    negative = [name for name in platforms if any(summary[name]['metrics']['verifier'][k][1]<0.
                 for k in ('miou_ci','acc50_ci'))]
    passed = base['delta_miou']>0. and base['delta_acc50']>0. and not negative
    detail = TARGET/'development-scores-private.json'
    write(detail,rows)
    write(TARGET/'development-results.json',dict(panel=summary,decision='go' if passed else 'no_go',
          clear_negative_platforms=negative,predictions=stamp,labels_after_seal=True,scores=digest(detail)))
    print('Stage 3 development',passed,base,'negative',negative,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=('predict','score'))
    (predict if parser.parse_args().stage=='predict' else score)()
