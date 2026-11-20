"""Seal all diagnosis-panel feature reads and frozen, zero-training choices."""
from pathlib import Path
from time import perf_counter

from replay import (CONFIG, FEATURES, HERE, PUBLIC, ROOT, SCHEDULES, WORK, CandidateFrontend,
                    archive, atomic, digest, frame_inputs, load_arrays, measure, np, read, save_arrays, torch)


def run():
    archive.prepare_device()
    folder = WORK/'probe'
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    PUBLIC.mkdir(exist_ok=True)
    prior = archive.PRIVATE/'source'
    assert digest(prior/'SEALED.json') == CONFIG['probe_parent_seal_sha256']
    assert digest(prior/'LOCK.json') == CONFIG['probe_parent_lock_sha256']
    assert digest(ROOT/'crossplatform/single_frame_sources.json') == CONFIG['source_model_registration_sha256']
    parent, roster = read(prior/'SEALED.json'), read(prior/'LOCK.json')['roster']
    lock = dict(code={p.name: digest(p) for p in (HERE/'replay.py', HERE/'probe.py', HERE/'PROTOCOL.json')},
                source=CONFIG['probe_parent_seal_sha256'], schedules=SCHEDULES, fields=FEATURES)
    if (folder/'LOCK.json').exists():
        assert read(folder/'LOCK.json') == dict(lock, schedules=list(SCHEDULES), fields=list(FEATURES))
    else:
        atomic(folder/'LOCK.json', lock)
    files, models, totals = {}, {}, {}
    for src, indices in roster.items():
        assert len(indices) == 64
        rows = archive.source_inputs(src)
        part = folder/src
        part.mkdir(exist_ok=True, mode=0o700)
        frontend = CandidateFrontend(src)
        model = frontend.base
        from model51 import fingerprint
        before = fingerprint(model)
        assert before == parent['native_state_hashes'][src]
        assert not any(p.requires_grad for p in model.parameters())
        tick = perf_counter()
        for i in indices:
            key = f'{src}/{i:05}.npz'
            path = folder/key
            done = path.with_suffix('.json')
            if done.exists():
                record = read(done)
                assert record['sha256'] == digest(path) and record['parent'] == parent['files'][key]
            else:
                assert digest(prior/key) == parent['files'][key]
                frozen = load_arrays(prior/key)
                example = archive.observation(rows[i])
                assert digest(Path(example['points'])) == rows[i]['point_sha256']
                values = measure(model, frame_inputs(src, example), frozen['slate'], frozen['boxes'],
                                 f'{src}|probe|{i}', verify_trajectory=frozen['full'])
                np.testing.assert_array_equal(values['seed'], frozen['ids'])
                joint = frozen['joint']
                assert joint.argmax() == frozen['choices'][1]
                scores = joint[None].astype(float)+(values['odds'].astype(float)-values['odds'][0].astype(float))
                picks = scores.argmax(-1)
                assert picks[0] == frozen['choices'][1]
                save_arrays(path, **values, joint=joint, incremental=scores, picks=picks, v1=frozen['choices'][0])
                record = dict(sha256=digest(path), parent=parent['files'][key])
                atomic(done, record)
            files[key] = record
            if len(files)%32 == 0:
                print('probe',len(files),'of 192',src,round(perf_counter()-tick,1),flush=True)
        assert fingerprint(model) == before
        models[src] = before
        totals[src] = perf_counter()-tick
        del model, frontend
        torch.cuda.empty_cache()
    assert len(files) == 192
    atomic(folder/'SEALED.json', dict(files=files, lock=digest(folder/'LOCK.json'), weights=models,
                                      seconds=totals, labels_read=False, FINAL_access=False))


if __name__ == '__main__':
    run()
