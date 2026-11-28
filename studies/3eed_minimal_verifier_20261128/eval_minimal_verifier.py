"""Freeze all Minimal Verifier transfers before opening the evaluation rosters."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np
from safetensors.torch import load_file
import torch

from provenance import HERE, PLAN, ROOT, RUN, SOURCE, V2_SOURCE, digest, read, retain, runtime, write
from cache_mono import OLD as MONO, OUT as MONO_READS, guard
from compact_pool import PRIOR, RUN as V2_RUN, sealed_inputs
from fit import switches
from models import Arm,model_for,project_tensors,raw_choice
from prepare_blind import prepare
from score_paradigms import ious
from source_features import pool_features
from transfer import batches

OUT = RUN/'target'


def receipt():
    stamp = read(SOURCE/'selection-seal.json')
    assert stamp['source_only'] and stamp['protocol']==digest(PLAN)
    assert all(digest(HERE/k)==v for k,v in stamp['code'].items())
    return stamp


def tensorize(job):
    case,row,expected = job
    frame,raw = row['frame'],row['raw']
    if case is None:
        import minimal
        baseline = minimal.predict(frame,raw)
        case = prepare((frame['uid'],row,dict(outputs=dict(axis_2d=baseline))))
    pack,(x,e,m) = pool_features(case,frame,raw)
    actual = dict(expected)
    old = actual.pop('selection')
    assert actual==pack,case['uid']
    return pack,old,x,e,m,retain(pack['pool'])


@torch.inference_mode()
def replay(jobs,models,workers):
    pred = []
    elapsed = dict(geometry_seconds=0.,project_and_forward_seconds=dict.fromkeys(models,0.))
    for batch in batches(jobs):
        start = perf_counter()
        packs,old,x,e,m,keep = zip(*workers.map(tensorize,batch,chunksize=4),strict=True)
        elapsed['geometry_seconds'] += perf_counter()-start
        data = tuple(torch.as_tensor(np.stack(a),device='cuda') for a in (x,e,m))
        dummy = torch.zeros_like(data[2],dtype=torch.float32)
        masks = torch.as_tensor(np.stack(keep),device='cuda')
        out = [dict(uid=p['uid'],target=p['target'],branch=p['branch'],pool=p['pool'],
                    choice={'Minimal-D':0}) for p in packs]
        for name,(model,arm,threshold) in models.items():
            start = perf_counter()
            (xn,en,mn,_),idx = project_tensors((*data,dummy),masks,arm)
            result = raw_choice(model(xn,en,mn),mn,arm.utility).cpu().numpy()
            idx = idx.cpu().numpy()
            choose = np.where(switches(result,threshold,arm.utility),result[:,0].astype(int),0)
            chosen = idx[np.arange(len(idx)),choose]
            elapsed['project_and_forward_seconds'][name] += perf_counter()-start
            for row,i in zip(out,chosen,strict=True):
                row['choice'][name] = int(i)
        assert all(row['choice']['S0']==ref['verifier'] for row,ref in zip(out,old,strict=True))
        pred.extend(out)
    return pred,elapsed


def dev_jobs():
    raw,_,_,pins,_ = sealed_inputs()
    case_file = PRIOR/'inputs-private.json'
    assert read(PRIOR/'input-seal.json')['evidence']==digest(case_file)
    old = V2_RUN/'target'
    pred = old/'development-predictions-private.json'
    assert read(old/'development-prediction-seal.json')['payload']==digest(pred)
    lookup = {r['uid']:r for r in read(pred)}
    cases = [r for r in read(case_file) if r['target']!='mono']
    assert len(cases)==len(lookup)==702
    return [(c,raw[c['uid']],lookup[c['uid']]) for c in cases],pins


def mono_jobs():
    stamp = read(MONO_READS/'decode-seal.json')
    answer = MONO_READS/'qwen-private.json'
    assert stamp['payload']==digest(answer) and stamp['code']==digest(HERE/'cache_mono.py')
    assert not stamp['labels_read'] and stamp['inputs']==digest(MONO/'input-seal.json')
    replies = read(answer)
    old_pred = MONO/'predictions-private.json'
    locked = read(MONO/'prediction-seal.json')
    assert locked['payload']==digest(old_pred) and not locked['labels_read']
    lookup = {r['uid']:r for r in read(old_pred)}
    frames = read(MONO/'inputs-private.json')
    assert len(frames)==len(lookup)==len(replies)==148
    return [(None,dict(frame=f,raw=replies[f['uid']],target='mono'),lookup[f['uid']]) for f in frames],stamp


def predict():
    locked = receipt()
    sys.addaudithook(guard)
    runtime()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('high')
    cv2.setNumThreads(1)
    tasks = dict(development=dev_jobs(),mono=mono_jobs())
    models = {}
    OUT.mkdir(exist_ok=True,parents=True)
    with ProcessPoolExecutor(8) as workers:
        # Start CPU workers before the first CUDA operation.
        list(workers.map(len,[[0]]*8))
        for name,card in locked['arms'].items():
            arm = Arm(**card['arm'])
            path = (V2_SOURCE/'verifier.safetensors') if name=='S0' else SOURCE/f'{name}.safetensors'
            assert card['checkpoint']==digest(path)
            model = model_for(arm)
            model.load_state_dict(load_file(path))
            models[name] = (model.cuda().eval(),arm,card['threshold'])
        for name,(jobs,pins) in tasks.items():
            rows,times = replay(jobs,models,workers)
            file = OUT/f'{name}-predictions-private.json'
            write(file,rows)
            write(OUT/f'{name}-seal.json',dict(payload=digest(file),source=digest(SOURCE/'selection-seal.json'),
                  pins=pins,labels_read=False,n=len(rows),timing=times,code=digest(__file__)))
            print('Sealed transfer',name,len(rows),times,flush=True)


def external_labels():
    import pyarrow.parquet as pq
    from mono import archive
    from score import canonical_box

    old = read(MONO/'input-seal.json')
    parquet = archive.ASSETS/(old['split']+'.parquet')
    join = MONO/'label-join-private.json'
    assert old['joins']==digest(join) and old['parquet']==digest(parquet)
    entries = pq.read_table(parquet,columns=['info','objects']).to_pylist()
    by_frame = {int(i['info']['img_id'][0]):i for i in entries}
    refs = {}
    for item in read(join):
        entry = by_frame[item['image_id']]
        i = item['slot']
        assert entry['info']['instance_id'][i]==item['instance']
        gt = entry['objects']
        h,w,l,yaw = [float(gt[k][i]) for k in ('h','w','l','ry')]
        inv = np.linalg.inv(item['transform'])
        center = inv[:3,:3]@(np.asarray(gt['pos'][i])-np.array([0,h/2,0]))+inv[:3,3]
        direction = inv[:3,:3]@np.array([np.cos(yaw),0.,-np.sin(yaw)])
        refs[item['uid']] = dict(recording=item['recording'],
            box=canonical_box(np.r_[center,l,w,h,np.arctan2(direction[1],direction[0])]))
    return refs


def development_labels():
    from metric_image_common import inputs
    from truth_score_inputs import truth_data

    previous = PRIOR/'scores-private.json'
    assert digest(previous)==read(PRIOR/'results.json')['offline_score_sha']
    meta = {r['uid']:r for r in read(previous)['critic'] if r['target']!='mono'}
    frames = inputs()
    truth,_ = truth_data({u:frames[u] for u in meta})
    return {k:dict(recording=v['recording'],box=truth[k]['box']) for k,v in meta.items()}


def scores(pred,labels,profiles):
    from evaluate_structure import stat_panel

    rows = []
    for p in pred:
        q = labels.get(p['uid'])
        if q is None:
            continue
        values = ious([k['box'] for k in p['pool']],q['box'])
        keep = retain(p['pool'])[:len(values)]
        named = {k:float(values[i]) for k,i in p['choice'].items()}
        named.update(oracle21=float(values.max()),oracle17=float(values[keep[:,0]].max()),
                     oracle13=float(values[keep[:,1]].max()))
        rows.append(dict(uid=p['uid'],target=p['target'],recording=q['recording'],branch=p['branch'],iou=named,
                         switched={k:i!=0 for k,i in p['choice'].items()},counts=[len(values),*map(int,keep.sum(0))]))
    assert len(rows)==len(labels)
    names = ('Minimal-D',*profiles,'oracle21','oracle17','oracle13')
    groups = {'overall':rows}
    for field in ('target','branch'):
        for tag in sorted({r[field] for r in rows}):
            groups[tag] = [r for r in rows if r[field]==tag]
    report = {}
    for name,group in groups.items():
        stat = stat_panel(group,names,'S0')
        stat['vs_minimal'] = stat_panel(group,names,'Minimal-D')['metrics']
        stat['switches'] = {a:sum(r['switched'][a] for r in group) for a in profiles}
        stat['mean_candidates'] = np.mean([r['counts'] for r in group],axis=0).tolist()
        stat['max_candidates'] = np.max([r['counts'] for r in group],axis=0).tolist()
        report[name] = stat
    return report,rows


def score():
    before = receipt()
    from shared import setup
    setup()
    torch.set_num_threads(1)
    blind = {}
    for name in ('development','mono'):
        stamp = read(OUT/f'{name}-seal.json')
        file = OUT/f'{name}-predictions-private.json'
        assert stamp['payload']==digest(file) and stamp['code']==digest(__file__)
        assert stamp['source']==digest(SOURCE/'selection-seal.json') and not stamp['labels_read']
        blind[name] = (read(file),stamp)
    target = {}
    for name,labels in (('development',development_labels()),('mono',external_labels())):
        pred,stamp = blind[name]
        summary,detail = scores(pred,labels,before['arms'])
        oldfile = V2_RUN/'target/development-results.json' if name=='development' else MONO/'results.json'
        prior = read(oldfile)['panel']['overall']['metrics']['verifier']
        control = summary['overall']['metrics']['S0']
        assert len(detail)==(665 if name=='development' else 148)
        for x in ('miou','acc50'):
            assert abs(control[x]-prior[x])<1.e-10
        out = OUT/f'{name}-scores-private.json'
        write(out,detail)
        target[name] = dict(panel=summary, predictions=stamp,scores=digest(out))
        print(name,[(k,round(v['miou'],3),round(v['acc50'],3)) for k,v in summary['overall']['metrics'].items()],
              flush=True)
    picked = before['selected']
    card = before['arms'][picked]
    result = target['development']['panel']['overall']['metrics'][picked]
    qualifies = (card['arm']['mean'] and card['arm']['lean'] and card['arm']['utility']
                 and card['arm']['cap']<=17 and result['miou']>=36.8
                 and result['delta_miou']>=-1. and result['delta_acc50']>=-1.)
    write(RUN/'results.json',dict(source=before,target=target,source_selection=picked,
          paper_preference='Minimal Verifier' if qualifies else 'Verifier v2',
          fully_simplified_meets_target=qualifies,labels_after_both_seals=True))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=('predict','score'))
    dict(predict=predict,score=score)[parser.parse_args().phase]()
