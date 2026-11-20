"""Predict without labels on a pinned source panel and conditional ordinary panel."""
import argparse
from contextlib import ExitStack
from functools import lru_cache
import json
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch
from scipy.spatial import cKDTree

from rays import ARMS, decode, derange, key, selection

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R183 = HERE.parent / '3eed_v2_ordinary_val_subject_ranking_20261119'
SUP = HERE.parent / '3eed_v1_supervised_instance_grounding_20261119'
PAIR = HERE.parent / '3eed_v1_disagreement_comparator_20261120/run_PRIVATE/source'
R181 = HERE.parent / '3eed_v2_readout_calibration_20261119/source_components_PRIVATE'
PRIVATE = HERE / 'run_PRIVATE'
sys.path[:0] = [str(ROOT), str(R183)]
import predict183
from scope183 import atomic, digest, read, check_input, frame, ordinary
from features import observation
from roles import CANONICAL, nominal
from train import prepare_device
from evaluate import to_sensor
from retrieval180 import load_retriever
from readout181 import ReadoutGrounder

CFG = read(HERE / 'PROTOCOL.json')
PREFIXES = ('proposal_', '0head_', '1head_', '2head_', '3head_', '4head_', 'last_')
PLATFORMS = tuple(CFG['source_sizes'])


def checked(path, expected):
    if digest(path) != expected:
        raise ValueError(f'Pinned input changed: {path.name}')
    return read(path)


def source_inputs(platform):
    folder = SUP / 'inputs_PRIVATE'
    files = checked(folder / 'SEAL.json', CFG['pins']['source_roster'])['files']
    name = f'{platform}_calibration.json'
    return checked(folder / name, files[name])


def sealed_arrays(folder, seal, pin, platform, keys):
    name = f'{platform}_calibration.npz'
    item = checked(folder / seal, CFG['pins'][pin])['files'][name]
    expected = item if isinstance(item, str) else item['sha256']
    assert digest(folder / name) == expected
    with np.load(folder / name, allow_pickle=False) as z:
        return {k: z[k] for k in keys}


def source_ranks(platform):
    # The original retriever applied a single batch to natural calibration rows.
    root = SUP / 'features_PRIVATE'
    lock = read(root / 'SEALED.json')
    part = root / f'{platform}_calibration'
    seal = checked(part / 'SEAL.json', lock['splits'][part.name]['seal_sha256'])
    fields = {k: [] for k in ('anchor_query', 'token_proj', 'anchors', 'prior')}
    for name, record in sorted(seal['files'].items()):
        assert digest(part / name) == record['sha256']
        with np.load(part / name, allow_pickle=False) as z:
            natural = z['origin'] < 2
            for k, values in fields.items():
                values.append(z[k][natural])
    batch = {k: torch.as_tensor(np.concatenate(values), device='cuda') for k, values in fields.items()}
    with torch.inference_mode():
        out = load_retriever(platform)(batch).cpu().numpy()
    assert out.shape == (CFG['source_sizes'][platform], 256)
    return out


def as_np(tensor):
    return tensor.detach().cpu().numpy()


def class_query(front, caption):
    from parse97 import dependency_document, parse_subject
    document = dependency_document(front.parser(caption))
    subject = parse_subject(caption, document)
    if subject['status'] != 'FOUND':
        return None, 'unknown_subject'
    label = nominal(subject['head_text'])
    if label is None:
        return None, 'unknown_category'
    return 'a ' + CANONICAL[label], 'ok'


def trajectory(output):
    return torch.stack([torch.cat((output[p + 'center'], output[p + 'pred_size']), -1)[0]
                        for p in PREFIXES])


def replay(model, inputs, full, caption):
    from model51 import SENTINEL
    ids = full['query_points_sample_inds']
    cached = {k: full[k] for k in ('fp2_features', 'fp2_xyz', 'fp2_inds')}
    positions = []
    def use_original(_mod, args):
        assert args[0].data_ptr() == full['fp2_xyz'].data_ptr()
        return args[0], args[1], ids
    def track_position(_mod, args):
        positions.append(args[3][0])
    with ExitStack() as stack:
        stack.enter_context(patch.object(model.backbone_net, 'forward', return_value=dict(cached)))
        stack.enter_context(model.gsample_module.register_forward_pre_hook(use_original))
        for layer in model.decoder:
            stack.enter_context(layer.register_forward_pre_hook(track_position))
        out = model(dict(point_clouds=inputs['point_clouds'], text=[caption + SENTINEL]))
    assert len(positions) == 6
    assert torch.equal(out['query_points_sample_inds'], ids)
    assert torch.equal(out['query_points_xyz'], full['query_points_xyz'])
    path = trajectory(out)
    assert torch.equal(torch.stack(positions), path[:-1])
    return path


@torch.inference_mode()
def predict(grounder, input_frame, rowkey, same_prompt_check, ranking=None):
    captured = []
    front = grounder._frontend
    model = front.base
    with model.register_forward_hook(lambda _model, args, output: captured.append((args[0], output))):
        result = grounder.collect(input_frame, ranking=ranking)
    (arguments, raw), = captured
    text = input_frame['text'].strip()
    ids = raw['query_points_sample_inds'][0].long()
    fpcoords = raw['fp2_xyz'][0, ids]
    assert len(ids.unique()) == 256 and torch.equal(fpcoords, raw['query_points_xyz'][0])
    # Check physical membership in the actual input cloud, independently of fp2_inds.
    cloud = as_np(arguments['point_clouds'][0, :, :3])
    assert cKDTree(cloud).query(as_np(fpcoords))[0].max() == 0
    full_path = trajectory(raw)
    assert full_path.shape == (7, 256, 6)
    q0, reason = class_query(front, text)
    if same_prompt_check:
        assert torch.equal(replay(model, arguments, raw, text), full_path)
    torch.cuda.synchronize()
    start = perf_counter()
    reduced_path = full_path if q0 is None else replay(model, arguments, raw, q0)
    torch.cuda.synchronize()
    extra_seconds = perf_counter() - start
    slate = result['slate']
    boxes = result['boxes']
    joint = result['scores'][:, 0]
    full, reduced = as_np(full_path), as_np(reduced_path)
    np.testing.assert_array_equal(boxes[:, :3], full[-1, slate, :3])
    seed = as_np(raw['seeds_obj_cls_logits'][0, 0, ids])
    permutation = derange(256, rowkey)
    choices, energy = decode(boxes, joint, result['incumbent'], seed[slate], reduced[-1,:,:3],
                             full[-1,:,:3], permutation, q0 is not None)
    assert choices[1] == result['chosen'][0]
    offsets = front.base.tokenizer(arguments['text'], return_offsets_mapping=True,
                                  return_tensors='np')['offset_mapping'][0]
    return dict(boxes=boxes, slate=slate, joint=joint, choices=choices, energy=energy,
                seed=seed, ids=as_np(ids), anchors=as_np(fpcoords), full=full, reduced=reduced,
                permutation=permutation, valid=q0 is not None, reason=reason, parity=same_prompt_check,
                extra_seconds=extra_seconds, sem_logits=as_np(raw['last_sem_cls_scores'][0]),
                offsets=offsets, view=result['view'])


def run(stage):
    if stage not in ('source', 'ordinary'):
        raise ValueError(stage)
    assert tuple(CFG['arms']) == ARMS and digest(ROOT / 'crossplatform/single_frame_sources.json') == CFG['pins']['registry']
    if stage == 'ordinary':
        decision = read(HERE / 'results/source_gate.json')
        assert decision['expand'] and decision['source_seal'] == digest(PRIVATE / 'source/SEALED.json')
    prepare_device()
    dest = PRIVATE / stage
    dest.mkdir(mode=0o700, parents=True, exist_ok=True)
    paths = [HERE / f for f in ('PROTOCOL.json','capture.py','rays.py')]
    paths += [ROOT/'crossplatform'/f for f in ('CURRENT_METHOD.json','single_frame_sources.json','ground.py','candidate_frontend.py','round3/p0/upstream/models/bdetr.py')]
    roster = {p: selection(stage, p, CFG['source_sizes' if stage=='source' else 'ordinary_sizes'][p]) for p in PLATFORMS}
    lock = dict(code={str(p.relative_to(ROOT)): digest(p) for p in paths}, roster=roster, arms=ARMS)
    if (dest / 'LOCK.json').exists():
        assert read(dest / 'LOCK.json') == json.loads(json.dumps(lock))
    else:
        atomic(dest / 'LOCK.json',lock)
    assert not (dest / 'SEALED.json').exists()
    start_all = perf_counter()
    expected = 192 if stage == 'source' else 384
    ordinary_rows = ordinary() if stage == 'ordinary' else None
    count = 0
    files, weights, pilot = {}, {}, None
    source_time = read(PRIVATE/'source/SEALED.json')['seconds'] if stage == 'ordinary' else 0.
    with patch.object(predict183.readout181, 'encode', predict183.ranking_fields):
        for source in PLATFORMS:
            grounder = ReadoutGrounder(source)
            from model51 import fingerprint
            grounder._frontend.parser = lru_cache(maxsize=64)(grounder._frontend.parser)
            base = grounder._frontend.base
            assert all(not m.training for m in base.modules()) and not any(p.requires_grad for p in base.parameters())
            before = fingerprint(base)
            rankings = None
            if stage == 'source':
                rankings = source_ranks(source)
                boxes = sealed_arrays(PAIR,'seal.json','source_k16',source,['boxes'])['boxes']
                archived = sealed_arrays(R181,'SEALED.json','source_scores',source,['scores','chosen','incumbent'])
            targets = [source] if stage == 'source' else [t for t in PLATFORMS if source != t]
            for target in targets:
                rows = source_inputs(source) if stage == 'source' else ordinary_rows[target]
                assert len(rows) == CFG['source_sizes' if stage=='source' else 'ordinary_sizes'][target]
                d = source if stage == 'source' else f'{source}_to_{target}'
                folder = dest / d
                folder.mkdir(mode=0o700, exist_ok=True)
                tick = perf_counter()
                extra = []
                for j, i in enumerate(roster[target]):
                    row = rows[i]
                    if stage == 'source':
                        assert digest(Path(row['pointpath'])) == row['point_sha256']
                        example = observation(row)
                    else:
                        check_input(row)
                        example = frame(row)
                    val = predict(grounder, example, key('derange', stage, source,target,i),
                                  stage == 'source' and source=='waymo' and j<32,
                                  rankings[i] if rankings is not None else None)
                    if stage == 'source':
                        np.testing.assert_array_equal(boxes[i], val['boxes'])
                        np.testing.assert_array_equal(archived['scores'][i,:,0], val['joint'])
                        assert (val['choices'][:2] == [archived['incumbent'][i], archived['chosen'][i,0]]).all()
                    else:
                        old = R183/'predictions_PRIVATE'
                        meta = checked(old/'SEALED.json',CFG['pins']['ordinary_boxes'])['directions'][d]['files']
                        name = f'{128*(i//128):05}.npz'
                        assert digest(old/d/name) == meta[name]['sha256']
                        with np.load(old/d/name,allow_pickle=False) as z:
                            np.testing.assert_array_equal(z['choices'][i%128,:2], val['choices'][:2])
                            np.testing.assert_array_equal(z['boxes'][i%128], to_sensor(val['boxes'], target, val['view']))
                    path = folder / f'{i:05}.npz'
                    with path.with_suffix('.tmp').open('wb') as f:
                        np.savez_compressed(f,**val)
                    path.with_suffix('.tmp').replace(path)
                    files[str(path.relative_to(dest))] = digest(path)
                    extra.append(float(val['extra_seconds']))
                    count += 1
                    if stage == 'source' and count == 32:
                        elapsed = perf_counter() - start_all
                        projected = elapsed + (perf_counter()-tick)/32 * (576-32) + 5 * (tick-start_all)
                        pilot = dict(rows=32, wall_seconds=elapsed, row_seconds=(perf_counter()-tick)/32,
                                     mean_extra_seconds=float(np.mean(extra)), projected_seconds=projected)
                        atomic(dest/'PILOT.json',pilot)
                        print('pilot',pilot,flush=True)
                        if projected > 3600:
                            raise RuntimeError('GPU-hour budget exceeded by pilot projection')
                    if count % 16 == 0:
                        print(stage,count,expected,d,round(perf_counter()-start_all,1),flush=True)
                    if perf_counter() - start_all + source_time >= 3600:
                        raise RuntimeError('GPU-hour budget exhausted')
            assert fingerprint(base) == before
            weights[source] = before
            del grounder,base
            torch.cuda.empty_cache()
    assert count == len(files) == expected
    atomic(dest/'SEALED.json',dict(stage=stage,n=count,files=files,lock=digest(dest/'LOCK.json'),
           seconds=perf_counter()-start_all,pilot=pilot,native_state_hashes=weights,
           annotations_used=False,FINAL_access=False))
    print('SEALED',stage,count,flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('source','ordinary'))
    run(p.parse_args().stage)
