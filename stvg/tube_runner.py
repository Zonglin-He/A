"""Persistent TubeDETR paired evaluation; immutable per-query receipts and seal."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def population_hash(rows):
    # Local filesystem locations do not define the scientific population.
    return digest(sorted([{**r, 'metadata': {k: v for k, v in r['metadata'].items() if k != 'video_path'}}
                          for r in rows], key=lambda r: r['id']))


def implementation_hash():
    from .runtime import ROOT, sha
    files = [p for folder in ['stvg', 'external/TubeDETR', 'external/ReferDINO']
             for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in {'.py', '.cpp', '.cu', '.cuh', '.h'}]
    return digest({str(p.relative_to(ROOT)): sha(p) for p in sorted(files)})


def runtime_identity():
    from .runtime import ROOT, sha
    import importlib.metadata
    files = ['runtime/referdino/cuda_build/MultiScaleDeformableAttention.so']
    for folder in ['runtime/referdino/bert-from-frozen-teacher', 'runtime/tubedetr/roberta-base']:
        files.extend(str(p.relative_to(ROOT)) for p in (ROOT/folder).glob('*') if p.is_file())
    if not (ROOT/'runtime/referdino/bert-from-frozen-teacher/model.safetensors').is_file():
        raise ValueError('Frozen-teacher BERT assets have not been prepared')
    packages = ['torch', 'torchvision', 'transformers', 'numpy', 'scipy', 'timm', 'einops', 'Pillow']
    return dict(files={p: sha(ROOT/p) for p in sorted(files)},
                packages={p: importlib.metadata.version(p) for p in packages},
                python=sys.version)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def save_prediction(path, value, receipt):
    from .runtime import sha
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix('.json').exists():
        raise ValueError('Never overwrite an existing prediction or receipt')
    keys = ['Source', 'E_Ref', 'Live_raw', 'Live', 'frame_ids', 'positions8', 'Native_interval', 'Strong_interval']
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        np.savez(f, **{k: value[k] for k in keys})
    tmp.replace(path)
    write(path.with_suffix('.json'), {**receipt, 'prediction_sha256': sha(path),
                                    'cost': value['cost'], 'checkpoint': value['checkpoint'],
                                    'initialization': value['initialization'], 'labels_read': False})


def evaluate(args):
    import torch
    from .runtime import ROOT, setup, lease, sha
    from .tube_decode import decode, asset_identity
    from .inputs import validate_metadata, validate_input
    from .tubedetr import TubePredictor
    from .teacher_service import PersistentRefer
    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    if not rows or len({r['id'] for r in rows}) != len(rows) or len({r['dataset'] for r in rows}) != 1:
        raise ValueError('One nonempty, unique-ID, single-direction manifest is required')
    for row in rows:
        if set(row) != {'id', 'dataset', 'key', 'metadata'} or not row['id'] or not all(c.isalnum() or c in '_-' for c in row['id']):
            raise ValueError('Invalid input-only evaluation row')
        validate_metadata(row['metadata'])
    ds = rows[0]['dataset']
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'sealed.json').exists():
        raise ValueError('Evaluation is already sealed')
    if not args.protocol or not args.qualification:
        raise ValueError('Formal TubeDETR evaluation requires frozen protocol and real-checkpoint qualification receipts')
    protocol = json.loads(Path(args.protocol).read_text())
    qualification = json.loads(Path(args.qualification).read_text())
    if protocol['status'] != 'frozen' or protocol['populations'][ds] != population_hash(rows):
        raise ValueError('Frozen canonical population differs from evaluation inputs')
    setup()
    code_hash = implementation_hash()
    predictor = TubePredictor(ds)
    if (qualification.get('status') != 'pass' or qualification.get('implementation_sha256') != code_hash
            or qualification['datasets'][ds]['checkpoint_sha256'] != sha(ROOT/predictor.config['source_checkpoint'])):
        raise ValueError('Qualification is missing, stale, or for a different checkpoint')
    lock = lease()
    teacher = None
    files = {}
    start = time.perf_counter()
    identity = dict(manifest_sha256=sha(args.manifest), population_sha256=population_hash(rows),
                    implementation_sha256=code_hash, configuration=predictor.config,
                    qualification_sha256=sha(args.qualification), protocol_sha256=sha(args.protocol),
                    decoder_asset=asset_identity(), runtime_asset=runtime_identity())
    if qualification.get('decoder_asset') != identity['decoder_asset']:
        raise ValueError('Decoder build changed after qualification')
    if qualification.get('runtime_asset') != identity['runtime_asset']:
        raise ValueError('Tokenizer, runtime library, or compiled operator changed after qualification')
    if (output/'run.json').exists():
        if json.loads((output/'run.json').read_text()) != identity:
            raise ValueError('Resume configuration or implementation changed')
    else:
        write(output/'run.json', identity)
    try:
        for i, row in enumerate(rows):
            path = output/'predictions'/f"{row['id']}.npz"
            expected = dict(identity_sha256=digest(identity), request_sha256=digest(row))
            receipt = path.with_suffix('.json')
            if path.exists() or receipt.exists():
                if not path.exists() or not receipt.exists():
                    raise ValueError('Incomplete query write; retain and inspect before resuming')
                saved = json.loads(receipt.read_text())
                if any(saved.get(k) != v for k, v in expected.items()) or saved['prediction_sha256'] != sha(path):
                    raise ValueError('Existing per-query receipt differs from frozen inputs')
            else:
                query_tick = time.perf_counter()
                teacher_startup = 0.
                if teacher is None:
                    startup_tick = time.perf_counter()
                    teacher = PersistentRefer(output/'teacher.log')
                    predictor.teacher = teacher
                    teacher_startup = time.perf_counter()-startup_tick
                decode_tick = time.perf_counter()
                frames, ids = decode(row['metadata'], ds)
                decode_seconds = time.perf_counter()-decode_tick
                validate_input(frames, ids, row['metadata'])
                value = predictor.predict(frames, ids, row['metadata'], row['key'])
                value['cost'].update(decode_seconds=decode_seconds, teacher_startup_seconds=teacher_startup,
                    total_seconds=time.perf_counter()-query_tick,
                    source_total_seconds=decode_seconds+value['cost']['source_seconds']+value['cost']['source_load_seconds'])
                save_prediction(path, value, expected)
                del frames, value
            files[row['id']] = dict(path=str(path.relative_to(output)), sha256=sha(path),
                                   receipt_path=str(receipt.relative_to(output)), receipt_sha256=sha(receipt))
            write(output/'progress.json', dict(status='running', complete=i+1, total=len(rows),
                                               elapsed_seconds=time.perf_counter()-start, labels_read=False))
            print(f'TubeDETR {ds}: {i+1}/{len(rows)} sealed query receipts', flush=True)
        write(output/'sealed.json', dict(status='sealed_before_offline_scoring', queries=len(rows),
                                        manifest_sha256=identity['manifest_sha256'], files=files,
                                        identity=identity, labels_read=False))
        write(output/'progress.json', dict(status='sealed', complete=len(rows), total=len(rows),
                                           elapsed_seconds=time.perf_counter()-start, labels_read=False))
    finally:
        if teacher is not None:
            teacher.close()
        lock.close()


def predict(args):
    from .runtime import setup, lease
    from .tube_decode import decode
    from .inputs import validate_input, validate_metadata
    from .tubedetr import TubePredictor
    request = json.loads(Path(args.request).read_text())
    if set(request) != {'dataset', 'key', 'metadata'}:
        raise ValueError('Expected only dataset/key/metadata')
    validate_metadata(request['metadata'])
    setup()
    lock = lease()
    try:
        frames, ids = decode(request['metadata'], request['dataset'])
        validate_input(frames, ids, request['metadata'])
        predictor = TubePredictor(request['dataset'])
        value = predictor.predict(frames, ids, request['metadata'], request['key'])
        save_prediction(Path(args.output), value, dict(request_sha256=digest(request),
                        scope='single-query engineering inference; no formal-population claim'))
    finally:
        lock.close()
