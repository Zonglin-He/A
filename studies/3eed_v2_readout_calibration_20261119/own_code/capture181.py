"""Seal target readout terms; separately prepare labeled source-only training data."""

import argparse
from time import perf_counter

import numpy as np
import torch

from readout181 import HERE, R180, K, TERMS, ReadoutGrounder, load_retriever
from scope import ROOT, PLATFORMS, evaluation_rows, file_sha, read, save, source_rows
from features import FEATURES, observation
from train import prepare_device, read_features, take
from source180 import calibrated as previous_selection
from evaluate180 import OUT as PREVIOUS, archive_lock, verified_directions
from evaluate import DIRECTIONS, to_sensor, score_boxes

TARGET = HERE/'components_PRIVATE'
SOURCE = HERE/'source_components_PRIVATE'
R180_FIT = R180/'source_PRIVATE/FITTED.json'


def protocol():
    return read(HERE/'PROTOCOL.json')


def manifest():
    paths = [HERE/p for p in ('PROTOCOL.json','capture181.py','readout181.py')]
    paths += [R180/p for p in ('retrieval180.py','evaluate180.py','source180.py')]
    paths += [ROOT/'crossplatform'/p for p in ('ground.py','CURRENT_METHOD.json','v1_decoder.py','v1_evidence.py')]
    paths += [HERE.parent/'3eed_v1_retrieval_binding_refinement_20261119/mechanisms.py']
    return {str(path.relative_to(ROOT)):file_sha(path) for path in paths}


def load_r180(rows):
    cfg = protocol()['ordinary_evaluation']
    assert file_sha(PREVIOUS/'SEALED.json')==cfg['r180_prediction_seal_sha256']
    assert file_sha(ROOT/'crossplatform/round159/RECORDING_INPUTS_PRIVATE.json')==cfg['registry_sha256']
    sealed = read(PREVIOUS/'SEALED.json')
    assert sealed['n']==cfg['direction_records'] and not sealed['FINAL_access'] and not sealed['target_labels_read']
    assert sealed['lock_sha256']==file_sha(PREVIOUS/'LOCK.json')
    return verified_directions(rows,archive_lock(),sealed,previous_selection())


def write_npz(path,values):
    with path.with_suffix('.tmp').open('wb') as f:
        np.savez_compressed(f,**values)
    path.with_suffix('.tmp').replace(path)


def block_output(samples):
    fields = ('terms','scores','incumbent','chosen','has_reference','can_win')
    return {key:np.stack([x[key] for x in samples]) for key in fields}


def target(smoke):
    rows = evaluation_rows()
    fixed = load_r180(rows)
    prepare_device()
    out = HERE/'smoke_PRIVATE' if smoke else TARGET
    out.mkdir(mode=0o700)
    start = perf_counter()
    files = {}
    error = 0.
    point_checks = set()
    for source in PLATFORMS:
        model = ReadoutGrounder(source)
        for target_name in PLATFORMS:
            if source==target_name:
                continue
            old = fixed[source,target_name]
            assert int(old['caps'][6])==K
            count = 2 if smoke else len(old['ordinals'])
            gathered = []
            for j,i in enumerate(old['ordinals'][:count]):
                row = rows[int(i)]
                pin=(row['pointpath'],row['point_sha256'])
                if pin not in point_checks:
                    assert file_sha(pin[0])==pin[1]
                    point_checks.add(pin)
                indices = old['slate'][j,6,:K]
                observed = model.collect(observation(row),fixed=indices)
                np.testing.assert_array_equal(observed['base_slots'],old['slate'][j,0,:4])
                np.testing.assert_array_equal(observed['can_win'],old['can_win'][j,indices])
                boxes = to_sensor(observed['boxes'],target_name,observed['view'])
                np.testing.assert_array_equal(boxes,old['pool'][j,indices])
                assert indices[observed['incumbent']]==old['chosen'][j,0]
                assert indices[observed['chosen'][0]]==old['chosen'][j,6]
                error=max(error,observed['reconstruction_error'])
                gathered.append(observed)
                if (j+1)%64==0 or j+1==count:
                    print(source,target_name,j+1,count,round(perf_counter()-start,1),flush=True)
            path=out/f'{source}_to_{target_name}.npz'
            write_npz(path,dict(ordinals=old['ordinals'][:count],**block_output(gathered)))
            files[path.name]=dict(n=count,sha256=file_sha(path))
        del model
        torch.cuda.empty_cache()
    n=sum(v['n'] for v in files.values())
    assert n==(12 if smoke else 3072)
    save(out/'SEALED.json',dict(n=n,files=files,implementation=manifest(),components=list(TERMS),
         identical_R180_frozen_winners=n,identical_raw_slates=n,reconstructed_score_error=error,
         earlier_seal_sha256=file_sha(PREVIOUS/'SEALED.json'),
         target_labels_read=False,target_updates=0,FINAL_access=False,elapsed_seconds=perf_counter()-start))


@torch.inference_mode()
def source():
    cfg=protocol()
    assert file_sha(R180_FIT)==cfg['ordinary_evaluation']['r180_source_fit_sha256']
    prepare_device()
    SOURCE.mkdir(mode=0o700)
    registered=cfg['source_calibration']
    files={}
    start=perf_counter()
    largest_error=0.
    iou_error=0.
    for platform in PLATFORMS:
        grounder=ReadoutGrounder(platform)
        retriever=load_retriever(platform)
        for split in ('fit','calibration'):
            rows=source_rows(platform,split)
            cached=read_features(FEATURES,platform,split)
            natural=torch.where(cached['origin']<2)[0]
            expected=(registered['fit_per_source'] if split=='fit' else registered['heldout_source_counts'][platform])
            assert len(rows)==len(natural)==expected
            q={k:cached[k][natural] for k in ('anchor_query','token_proj','anchors','prior')}
            rank=retriever(q).cpu().numpy()
            pool=q['anchors'].cpu().numpy()
            truth=cached['gt'][natural,0].cpu().numpy()
            checks=cached['v1_box'][natural].cpu().numpy()
            gathered, ious=[],[]
            for j,row in enumerate(rows):
                assert file_sha(row['pointpath'])==row['point_sha256']
                observed=grounder.collect(observation(row),ranking=rank[j])
                boxes=observed['boxes']
                np.testing.assert_array_equal(boxes,pool[j,observed['slate']])
                np.testing.assert_array_equal(boxes[observed['incumbent']],checks[j])
                val,e=score_boxes(boxes.astype(np.float64),truth[j].astype(np.float64))
                gathered.append(observed)
                ious.append(val)
                iou_error=max(iou_error,e)
                largest_error=max(largest_error,observed['reconstruction_error'])
                if (j+1)%80==0 or j+1==expected:
                    print(platform,split,j+1,expected,round(perf_counter()-start,1),flush=True)
            name=f'{platform}_{split}.npz'
            path=SOURCE/name
            write_npz(path,dict(ious=np.stack(ious),**block_output(gathered)))
            files[name]=dict(n=expected,recordings=len({r['recording'] for r in rows}),sha256=file_sha(path))
            del q,cached
        del grounder,retriever
        torch.cuda.empty_cache()
    save(SOURCE/'SEALED.json',dict(n=sum(row['n'] for row in files.values()),files=files,
         implementation=manifest(),source_only=True,components=list(TERMS),
         reconstructed_score_error=largest_error,oriented_iou_error=iou_error,
         target_labels_used=False,FINAL_access=False,elapsed_seconds=perf_counter()-start))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('smoke','target','source'))
    phase=parser.parse_args().phase
    source() if phase=='source' else target(smoke=phase=='smoke')
