"""Replay four fitting/ranking arms without reading evaluation labels."""
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
from pathlib import Path
import sys
from time import perf_counter_ns

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
STUDIES = HERE.parent
REPO = STUDIES.parent
sys.path.insert(0, str(STUDIES/'3eed_nullspace_solver_20261128'))
from run_experiment import ROOT, check, digest, load_blind, publish, read

RUN = ROOT/'runs/extent-2d-factorial-20261128'
MONO = ROOT/'runs/2d-pose-finalization-20261128'
RELOC = ROOT/'runs/2d-relocalization-20261128'
REGISTRATION = REPO/'QWEN_CURRENT_METHOD.json'
ARMS = ('height_ray', 'axis_ray', 'height_2d', 'axis_2d')
TOL = 1.e-8


def code_stamp():
    files = ('axiswise_2d.py', 'replay_factorial.py', 'PROTOCOL_ZH.md')
    return {f:digest(HERE/f) for f in files}


def registry():
    spec = read(REGISTRATION)
    assert spec['method'] == 'ray_equivalent_height_only_2d_rank'
    for k in ('prediction_entry', 'method_lock'):
        ref = spec[k]
        assert digest(REPO/ref['file']) == ref['sha256']
    history = read(REPO/spec['method_lock']['file'])['config']['historical_geometry']
    assert all(digest(REPO/f) == h for f,h in history.items())
    return spec


def source_inputs():
    internal, o0, pool, choices, receipt = load_blind()
    registry()
    from run_qwen_3eed import input_frames

    manifest = MONO/'mono-inputs.json'
    archive = MONO/'mono-raw-predictions-private.json'
    strict = MONO/'mono-strict-roster-private.json'
    sensor_bank = MONO/'mono-sensor-hashes.json'
    input_seal = check(manifest, 'manifest', MONO/'mono-input-seal.json')
    assert not input_seal['target_labels_read'] and input_seal['sensor_bank'] == digest(sensor_bank)
    raw_seal = check(archive, 'payload', MONO/'mono-prediction-seal.json')
    assert raw_seal['input_seal'] == digest(MONO/'mono-input-seal.json')
    assert raw_seal['registered_method'] == digest(REGISTRATION) and not raw_seal['target_labels_read']
    strict_seal = check(strict, 'file', MONO/'mono-strict-roster-seal.json')
    assert strict_seal['source_input_seal'] == digest(MONO/'mono-input-seal.json')
    members = set(read(strict))
    assert len(members) == strict_seal['n'] == 118
    frames = input_frames(manifest)
    refs = {r['uid']:r for r in read(archive)['predictions']}
    assert 171 == len(frames) == len(refs) and frames.keys() == refs.keys() and members <= frames.keys()
    for raw_path, expected in read(sensor_bank).items():
        assert digest(Path(raw_path)) == expected
    external = {uid:dict(frame=f, raw=refs[uid]['replies'], target='mono', recording=None) for uid,f in frames.items()}
    tasks = [(uid, r, o0[uid], pool[uid]['pool']['cray'], choices[uid]['selected']['cray']['cached'], None)
             for uid,r in internal.items()]
    tasks.extend((uid, r, None, None, None, refs[uid]['outputs']) for uid,r in external.items())
    assert len({r[0] for r in tasks}) == 873
    return tasks, dict(**receipt, mono_inputs=input_seal['manifest'], mono_replies=raw_seal['payload'],
                       mono_strict=strict_seal['file'])


def compare_boxes(a, b, *, tolerance=0.):
    if a is None or b is None:
        assert a is None and b is None
    else:
        np.testing.assert_allclose(a, b, rtol=0., atol=tolerance)


def ray_changes(reference, modified):
    hit = reference > 0.
    valid = modified > 0.
    common = hit & valid
    delta = abs(reference[common]-modified[common])
    changed = (hit != valid).sum() + (delta > TOL).sum()
    return dict(n=len(reference), ref_hits=int(hit.sum()), changed=int(changed), gained=int((valid & ~hit).sum()),
                lost=int((hit & ~valid).sum()), maximum_common_depth_m=float(delta.max()) if len(delta) else 0.)


def audit_box(ctx, box, modified, pack, old_depth, arrays):
    import axiswise_2d as trial
    from center_sets import local

    offset = local(modified[None,:3], box)[0]
    assert modified[6] == box[6] and np.all(modified[3:6] > 0.)
    assert np.all(abs(offset)+modified[3:6]/2 <= box[3:6]/2 + TOL)
    _, measured, entries = pack
    lost_measured = int((abs(measured-offset) > modified[3:6]/2 + TOL).any(1).sum())
    lost_entries = int((abs(entries-offset) > modified[3:6]/2 + TOL).any(1).sum())
    assert lost_measured == lost_entries == 0
    depths = [trial.live.first_depth(modified, points, ctx.transform) for points in arrays]
    return dict(trim=(box[3:6]-modified[3:6]).tolist(), center_shift=float(np.linalg.norm(modified[:3]-box[:3])),
                protected_points=len(measured), protected_entries=len(entries), lost_measured=lost_measured,
                lost_entries=lost_entries,
                rays={name:ray_changes(before,after) for name,before,after in zip(('selection','audit'),old_depth,depths,strict=True)})


def audit_pair(ctx, audits, outputs, options, packs):
    import axiswise_2d as trial

    arrays = (ctx.selection, audits)
    first = lambda box,rays:trial.live.first_depth(box, rays, ctx.transform)
    rows = {}
    for name, boxes in options.items():
        rows[name] = []
        for box,modified,pack in zip(ctx.boxes,boxes,packs,strict=True):
            before = [first(box,a) for a in arrays]
            rows[name].append(audit_box(ctx,box,modified,pack,before,arrays))
    between = {}
    for new, prior in (('axis_2d','height_2d'),('height_2d','height_ray'),('axis_2d','axis_ray')):
        a,b = (np.asarray(outputs[k]['box']) for k in (new,prior))
        between[new+'_vs_'+prior] = {k:ray_changes(first(b,r),first(a,r))
                                     for k,r in zip(('selection','audit'),arrays,strict=True)}
    return rows, between


def armset(ctx):
    import axiswise_2d as trial

    packs = tuple(trial.witnesses(ctx,b) for b in ctx.boxes)
    options = {k:tuple(trial.fit(ctx,b,w,height_only=(k=='height')) for b,w in zip(ctx.boxes,packs,strict=True))
               for k in ('height','axis')}
    ranked = {k:trial.rank(ctx,box) for k,box in options.items()}
    winner = {k+'_ray':ctx.ray_winner for k in options} | {k+'_2d':r[0] for k,r in ranked.items()}
    outputs = {name:trial.output(ctx,options[name.split('_')[0]],winner[name]) for name in ARMS}
    return outputs,winner,options,packs,{k:list(r[1]) for k,r in ranked.items()}


def one(job):
    from shared import canonical
    import axiswise_2d as trial

    uid, row, old, pool, cached, mono = job
    frame, raw = row['frame'], row['raw']
    ctx = trial.prepare(frame, raw)
    registered = trial.official.paired(frame, raw)
    axis_reference = trial.live.predict(frame,raw,height_only=False)
    payload = dict(uid=uid, target=row['target'], recording=row['recording'], active=ctx is not None)
    if ctx is None:
        fallback = trial.fallback(frame,raw)
        outputs = dict.fromkeys(ARMS,fallback)
        payload.update(winner={},candidate_audit={}, between={},projection_scores={})
    else:
        outputs,winner,options,packs,scores = armset(ctx)
        trace, audit = trial.live.incumbent(frame,raw,collect_audit=True)
        assert audit is not None and trace['heading'] == trial.HEADINGS[ctx.ray_winner]
        untrimmed, _, _, support, selection, withheld, transform = audit
        compare_boxes(untrimmed,ctx.boxes[ctx.ray_winner])
        np.testing.assert_array_equal(transform,ctx.transform)
        np.testing.assert_array_equal(support,ctx.support)
        np.testing.assert_array_equal(selection,ctx.selection)
        if old:
            np.testing.assert_array_equal(np.asarray(old['selection']).reshape(-1,3),selection)
            np.testing.assert_array_equal(np.asarray(old['audit']).reshape(-1,3),withheld)
        diagnostics, between = audit_pair(ctx,withheld,outputs,options,packs)
        payload.update(winner=winner,candidate_audit=diagnostics,between=between,projection_scores=scores)
    for name, value in (('height_ray',registered[0]),('height_2d',registered[1]),('axis_ray',axis_reference)):
        assert outputs[name] == value,(uid,name)
    if pool is not None:
        for k,i in (('height_ray',0),('height_2d',cached)):
            value = None if outputs[k]['box'] is None else canonical(outputs[k]['box'])
            expected = pool['boxes'][i] if pool['boxes'] else None
            compare_boxes(value,expected,tolerance=2e-9)
    if mono is not None:
        compare_boxes(outputs['height_ray']['box'],mono['old']['box'])
        compare_boxes(outputs['height_2d']['box'],mono['promoted']['box'])
    return dict(**payload, outputs=outputs)


def bench(jobs, rows):
    import axiswise_2d as trial

    by_id = {job[0]:job[1] for job in jobs}
    order = lambda r:sha256(('extent-timing-20261128|'+r['uid']).encode()).hexdigest()
    out = []
    for name, domains in (('3eed',{'waymo','drone','quad'}),('kitti',{'kitti'}),('mono',{'mono'})):
        for active,count in ((True,12),(False,4)):
            subset = sorted((r for r in rows if r['target'] in domains and r['active']==active),key=order)[:count]
            assert len(subset)==count
            for index,r in enumerate(subset):
                row = by_id[r['uid']]
                inputs = row['frame'],row['raw']
                compare_boxes(trial.official.predict(*inputs)['box'],r['outputs']['height_2d']['box'])
                compare_boxes(trial.predict(*inputs)['box'],r['outputs']['axis_2d']['box'])
                for replicate in range(3):
                    kernels = (('height_2d',trial.official.predict),('axis_2d',trial.predict))
                    if (replicate+index)%2:
                        kernels = kernels[::-1]
                    durations = {}
                    for arm,run in kernels:
                        begin = perf_counter_ns()
                        result = run(*inputs)
                        durations[arm] = (perf_counter_ns()-begin)/1_000_000.
                        assert result==r['outputs'][arm]
                    out.append(dict(group=name,active=active,round=replicate,ms=durations))
    return out


def main():
    def block(event,args):
        if event=='open' and isinstance(args[0],(str,bytes,Path)):
            from os import fsdecode
            if Path(fsdecode(args[0])).name in {'test.parquet','mono-paired-index-private.json','mono-scores-private.json',
                                               'data_object_label_2.zip','scored.json','scores-private.json'}:
                raise PermissionError('Offline labels or scoring artifacts are outside prediction.')
    sys.addaudithook(block)
    cv2.setNumThreads(1)
    tasks,pins = source_inputs()
    RUN.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(10) as pool:
        result = []
        for row in pool.map(one,tasks,chunksize=4):
            result.append(row)
            if len(result)%100==0 or len(result)==len(tasks):
                print('Factorial replay',len(result),'/',len(tasks),flush=True)
    path = RUN/'predictions-private.json'
    publish(path,result)
    sample = RUN/'timing.json'
    publish(sample,bench(tasks,result))
    publish(RUN/'prediction-seal.json',dict(payload=digest(path),benchmark=digest(sample),code=code_stamp(),sources=pins,
        n=len(result),active=sum(r['active'] for r in result),labels_read=False,model_calls=0,parameter_updates=0,
        registration=digest(REGISTRATION),historical_height_and_axis_ray_exact=len(result)))
    print('Sealed predictions, candidate ray audit, and paired timings.',flush=True)


if __name__=='__main__':
    main()
