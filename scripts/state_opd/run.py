"""HC128 state-candidate experiment. Explicit --run; never resume formal jobs."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import argparse, contextlib, gc, hashlib, json, sys, time, traceback
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
TRIAL = ROOT / 'outputs/state_opd_hc128'
OLD = ROOT / 'outputs/layout_reward_dev'

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def atomic(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, p)

def savept(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    torch.save(x, tmp); os.replace(tmp, p)

def prepare():
    cfg = read(HERE / 'config.json')
    assert (TRIAL / 'config.json').read_bytes() == (HERE / 'config.json').read_bytes()
    assert cfg['state']['coordinates'] == [8, 4] and cfg['state']['radius'] == 4.
    assert cfg['guard_weight'] == .1 and cfg['guard_deadzone'] == .25
    assert cfg['guard_mean_weight'] == cfg['guard_max_weight'] == .5
    assert cfg['backtracking'] == [1., .5, .25, .125, .0625]
    assert cfg['accept_epsilon'] == 1e-8 and cfg['mask_mass_epsilon'] == 1e-6
    formal = ROOT / 'outputs/formal/state.json'
    assert read(formal)['status'] == 'paused_by_user'
    manifest = ROOT / 'data/evaluation/development.jsonl'
    assert sha(manifest) == cfg['manifest_sha256']
    records = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    records = [r for r in records if r['dataset'] == 'hc2']
    assert len(records) == 128
    oldseal = read(OLD / 'scope_hc2/sealed.json')
    native = read(OLD / 'research/native_masks/sealed.json')
    assert oldseal['queries'] == native['queries'] == 128
    assert oldseal['GT_read'] is native['GT_read'] is False
    assert oldseal['fingerprint'] == cfg['source_inference_fingerprint']
    assert set(native['files']) == {r['id'] for r in records}
    inputs = []
    for r in records:
        key = r['id']; cp = OLD / 'student_cache' / (key + '.pt')
        cc = read(cp.with_suffix('.json'))
        rp = OLD / oldseal['files'][key]['path']
        mp = OLD / 'research/native_masks' / native['files'][key]['path']
        mr = read(mp.with_suffix('.json'))
        assert sha(cp) == cc['sha256'] and cc['GT_read'] is False
        assert sha(rp) == oldseal['files'][key]['sha256']
        assert sha(mp) == native['files'][key]['sha256']
        assert mr['GT_read'] is False and mr['native_selected_boxes_bit_exact'] is True
        assert mr['source_cache_sha256'] == cc['sha256']
        inputs.append(dict(id=key, key=r['key'], cache_sha256=cc['sha256'],
                           native_mask_sha256=sha(mp), reference_sha256=sha(rp),
                           reference_path=oldseal['files'][key]['path'], mask_path=native['files'][key]['path']))
    sources = [HERE / 'run.py', HERE / 'core.py', HERE / 'config.json']
    sources += sorted((ROOT / 'stvg').glob('*.py'))
    sources += sorted((ROOT / 'external/TA-STVG').rglob('*.py'))
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    source_cfg = read(OLD / 'config.json')['datasets']['hc2']
    checkpoint = ROOT / source_cfg['source_checkpoint']
    checkpoint_sha = sha(checkpoint)
    # Calibration model_hash is a model-content identifier, not a checkpoint-file hash.
    assert checkpoint_sha == read(ROOT / 'docs/assets.json')['required_files'][source_cfg['source_checkpoint']]
    loader_identity = dict(source_configuration_sha256=sha(OLD / 'config.json'),
                           checkpoint_sha256=checkpoint_sha,
                           E_calibration_model_identifier=source_cfg['E_calibration']['model_hash'],
                           upstream_yaml_sha256=sha(ROOT / 'external/TA-STVG/experiments/vidstg.yaml'),
                           runtime_manifest_sha256=sha(ROOT / 'runtime/source/vidstg/manifest.json'))
    identity = dict(config_sha256=sha(HERE / 'config.json'), sources=source_hashes,
                    loader_identity=loader_identity,
                    manifest_sha256=sha(manifest), reference_seal_sha256=sha(OLD / 'scope_hc2/sealed.json'),
                    native_mask_seal_sha256=sha(OLD / 'research/native_masks/sealed.json'), inputs=inputs)
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    lock = dict(fingerprint=fingerprint, **identity, GT_read=False, queries=128,
                formal_state_sha256=sha(formal), registered_config_sha256=sha(ROOT / 'configs/default.json'))
    if (TRIAL / 'lock.json').exists():
        assert read(TRIAL / 'lock.json') == lock, 'Locked execution identity changed'
    else:
        atomic(TRIAL / 'lock.json', lock)
        for p in sources[:3]:
            dst = TRIAL / 'source_snapshot' / p.name
            dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(p.read_bytes())
    return cfg, records, inputs, lock

def main(run=False, resume=False):
    cfg, records, inputs, lock = prepare()
    if not run:
        print(json.dumps(dict(status='prepared_locked_not_run', fingerprint=lock['fingerprint'], queries=128,
                              arms=cfg['arms'], forward_budget_per_arm=cfg['max_trajectory_evals']), indent=2))
        return
    from stvg.runtime import setup, lease, load_source
    from core import fit
    setup(); torch.set_num_threads(4)
    leasefile = lease(); start = time.perf_counter(); model = None
    if (TRIAL / 'sealed.json').exists():
        raise RuntimeError('This experiment is already sealed; refuse overwrite')
    if (TRIAL / 'run.json').exists() and not resume:
        raise RuntimeError('Existing run requires --resume with the identical lock')
    atomic(TRIAL / 'run.json', dict(status='running', PID=os.getpid(), fingerprint=lock['fingerprint'],
                                   started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), GT_read=False))
    source_cfg = read(OLD / 'config.json')['datasets']['hc2']
    fitcfg = dict(seed=cfg['seed'], sigma=cfg['sigma'], samples=cfg['candidates'], tau=cfg['tau'], lr=cfg['lr'],
                  opd_rounds=cfg['max_rounds'], direct_steps=cfg['max_point_updates'],
                  forward_budget=cfg['max_trajectory_evals'])
    completed = 0; files = {}
    try:
        for record, item in zip(records, inputs):
            key = record['id']; dest = TRIAL / 'predictions' / (key + '.npz'); receipt = dest.with_suffix('.json')
            if receipt.exists():
                oldr = read(receipt)
                assert oldr['fingerprint'] == lock['fingerprint'] and oldr['sha256'] == sha(dest)
                assert oldr['reference_sha256'] == item['reference_sha256'] and oldr['GT_read'] is False
                assert oldr['trace_sha256'] == sha(TRIAL / oldr['trace_path'])
                files[key] = dict(path=str(dest.relative_to(TRIAL)), sha256=sha(dest), receipt_sha256=sha(receipt))
                completed += 1; continue
            begin = time.perf_counter()
            assert sha(ROOT / 'outputs/formal/state.json') == lock['formal_state_sha256']
            cp = OLD / 'student_cache' / (key + '.pt')
            mp = OLD / 'research/native_masks' / item['mask_path']
            assert sha(cp) == item['cache_sha256'] and sha(mp) == item['native_mask_sha256']
            cache = torch.load(cp, map_location='cpu', weights_only=False)
            mask = torch.load(mp, map_location='cpu', weights_only=False)
            with np.load(OLD / item['reference_path'], allow_pickle=False) as z:
                gold = {k: z[k].copy() for k in z.files}
            assert np.array_equal(np.asarray(cache['Source']), gold['Source'])
            assert np.array_equal(np.asarray(cache['E']), gold['E_Ref'])
            assert cache['planned_positions'] == gold['positions8'].tolist()
            assert list(cache['frame_ids']) == gold['frame_ids'].tolist()
            assert np.array_equal(np.asarray(cache['interval']), gold['Strong_interval'])
            assert mask['GT_read'] is False and mask['source_cache_sha256'] == item['cache_sha256']
            assert torch.equal(mask['teacher']['boxes'], cache['teacher']['boxes'])
            assert mask['selected_query'] == cache['teacher']['selected_query']
            if model is None:
                load_start = time.perf_counter(); model = load_source(source_cfg); model.cpu()
                gc.collect(); torch.cuda.empty_cache()
                model.ground_decoder.decoder.cuda().eval().requires_grad_(False)
                atomic(TRIAL / 'load.json', dict(seconds=time.perf_counter()-load_start, GT_read=False,
                                                encoder_forwards=0, expert_forwards=0))
            result = {k: gold[k] for k in ['frame_ids', 'positions8', 'Strong_interval']}
            traces = {}; costs = {}
            for arm in cfg['arms']:
                arm_start = time.perf_counter()
                out = fit(model, cache, mask, item['key'], arm, fitcfg)
                assert torch.isfinite(out['final']).all()
                result[arm] = out['final'].detach().cpu().numpy()
                costs[arm] = out['costs']; traces[arm] = out
                atomic(TRIAL / 'progress.json', dict(completed=completed, total=128, id=key, arm=arm,
                                                     arm_seconds=time.perf_counter()-arm_start,
                                                     costs=costs, GT_read=False))
                print(json.dumps(dict(completed=completed, id=key, arm=arm, costs=costs[arm])), flush=True)
            tracepath = TRIAL / 'traces' / (key + '.pt'); savept(tracepath, traces)
            dest.parent.mkdir(parents=True, exist_ok=True); temp = dest.with_suffix('.npz.tmp')
            with temp.open('wb') as h:
                np.savez(h, **result)
            os.replace(temp, dest)
            atomic(receipt, dict(fingerprint=lock['fingerprint'], sha256=sha(dest),
                                  reference_sha256=item['reference_sha256'], cache_sha256=item['cache_sha256'],
                                  native_mask_sha256=item['native_mask_sha256'], trace_sha256=sha(tracepath),
                                  trace_path=str(tracepath.relative_to(TRIAL)), costs=costs,
                                  total_seconds=time.perf_counter()-begin, GT_read=False))
            files[key] = dict(path=str(dest.relative_to(TRIAL)), sha256=sha(dest), receipt_sha256=sha(receipt))
            completed += 1
            atomic(TRIAL / 'progress.json', dict(completed=completed, total=128, id=key,
                                                 seconds=time.perf_counter()-begin, GT_read=False))
            print(json.dumps(dict(completed=completed, total=128, id=key, seconds=time.perf_counter()-begin)), flush=True)
            del cache, mask, gold, result, traces, out; gc.collect(); torch.cuda.empty_cache()
        assert completed == len(files) == 128
        assert sha(ROOT / 'outputs/formal/state.json') == lock['formal_state_sha256']
        atomic(TRIAL / 'sealed.json', dict(status='sealed_before_offline_scoring', GT_read=False, queries=128,
                                          dataset='hc2', fingerprint=lock['fingerprint'], files=files,
                                          manifest_sha256=lock['manifest_sha256'], config_sha256=lock['config_sha256'],
                                          lock_sha256=sha(TRIAL / 'lock.json')))
        atomic(TRIAL / 'run.json', dict(status='completed_sealed', fingerprint=lock['fingerprint'],
                                       queries=128, seconds=time.perf_counter()-start, GT_read=False))
    except BaseException:
        atomic(TRIAL / 'failure.json', dict(status='failed_preserved_no_skip', completed=completed,
                                            fingerprint=lock['fingerprint'], traceback=traceback.format_exc(), GT_read=False))
        raise
    finally:
        del model
        gc.collect(); torch.cuda.empty_cache(); leasefile.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run', action='store_true'); p.add_argument('--resume', action='store_true')
    a = p.parse_args(); main(a.run, a.resume)
