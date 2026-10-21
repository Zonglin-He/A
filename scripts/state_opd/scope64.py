"""User-authorized deterministic HC64 scope; preserve the original worker identity.

Watch the current finite worker, stop only that owned process after query64 is
fully written, seal the selected prefix and optionally invoke CPU scoring.
No parameter changes, GT selection, new model job or formal evaluation resume.
"""
import argparse, contextlib, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRIAL = ROOT / 'outputs/state_opd_hc128'
OUT = TRIAL / 'scope_hc64'

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
def write(p, x):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(json.dumps(x, indent=2) + '\n'); os.replace(tmp, p)
def proc(pid):
    p = Path('/proc') / str(pid)
    try:
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, state=fields[0], startticks=fields[19],
                    cwd=str((p / 'cwd').resolve()), cmdline=(p / 'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except (FileNotFoundError, ProcessLookupError): return None
def owned(pid, ticks):
    x = proc(pid)
    assert x and x['startticks'] == ticks and x['cwd'] == str(ROOT.resolve())
    assert 'scripts/state_opd/run.py' in x['cmdline'] and '--run' in x['cmdline']
    return x

def scope():
    manifest = ROOT / 'data/evaluation/development.jsonl'
    rows = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    rows = [r for r in rows if r['dataset'] == 'hc2']
    assert len(rows) == 128
    selected = rows[:64]
    assert len({r['metadata']['original_video_id'] for r in selected}) == 64
    lock = read(TRIAL / 'lock.json')
    x = dict(status='locked_by_user_scope_reduction_before_GT', queries=64, parents=64,
             dataset='hc2', ids=[r['id'] for r in selected], original_planned_queries=128,
             source_fingerprint=lock['fingerprint'], config_sha256=lock['config_sha256'],
             manifest_sha256=sha(manifest), lock_sha256=sha(TRIAL / 'lock.json'),
             selection_rule='First64 rows in original HC128 manifest order, one query per unique parent; no GT or new score selection',
             method_parameters_and_per_query_allowance_unchanged=True, GT_read=False)
    p = TRIAL / 'user_scope64.json'
    if p.exists(): assert read(p) == x
    else: write(p, x)
    return x

def seal_selected(s):
    files = {}
    for key in s['ids']:
        p = TRIAL / 'predictions' / (key + '.npz'); q = p.with_suffix('.json'); r = read(q)
        assert r['GT_read'] is False and r['fingerprint'] == s['source_fingerprint']
        assert sha(p) == r['sha256'] and sha(TRIAL / r['trace_path']) == r['trace_sha256']
        assert r['reference_sha256'] == next(i['reference_sha256'] for i in read(TRIAL / 'lock.json')['inputs'] if i['id'] == key)
        files[key] = dict(path=str(p.relative_to(TRIAL)), sha256=sha(p), receipt_sha256=sha(q))
    assert len(files) == 64
    return dict(status='sealed_before_offline_scoring', queries=64, dataset='hc2', GT_read=False,
                fingerprint=s['source_fingerprint'], manifest_sha256=s['manifest_sha256'],
                config_sha256=s['config_sha256'], lock_sha256=s['lock_sha256'],
                scope_sha256=sha(TRIAL / 'user_scope64.json'), files=files,
                original_planned_queries=128, original_scientific_configuration_unchanged=True)

def main(watch=False, score=False):
    s = scope()
    if not watch:
        print(json.dumps(dict(status='locked_HC64_prefix_not_running_controller', queries=64))); return
    OUT.mkdir(parents=True, exist_ok=True)
    if not (OUT / 'sealed.json').exists():
        run = read(TRIAL / 'run.json'); assert run['status'] == 'running'
        pid = int(run['PID']); identity = proc(pid); assert identity
        ticks = identity['startticks']; owned(pid, ticks)
        write(OUT / 'controller.json', dict(status='watching64_boundary', worker=identity, controller_pid=os.getpid(),
                                          helper_sha256=sha(__file__), source_fingerprint=s['source_fingerprint'], GT_read=False))
        target = TRIAL / 'predictions' / (s['ids'][-1] + '.json')
        while not target.exists():
            if (TRIAL / 'failure.json').exists(): raise RuntimeError('Worker failed; preserve outputs and report')
            x = owned(pid, ticks)
            if x['state'] == 'Z': raise RuntimeError('Worker exited before HC64 was complete')
            time.sleep(.05)
        # Stop only the proven owned worker at the fully written selected-query boundary.
        owned(pid, ticks); os.kill(pid, signal.SIGSTOP)
        try:
            start = time.monotonic()
            while owned(pid, ticks)['state'] not in ('T', 't'):
                assert time.monotonic()-start < 5; time.sleep(.01)
            write(OUT / 'pre_stop_run.json', read(TRIAL / 'run.json'))
            write(OUT / 'pre_stop_progress.json', read(TRIAL / 'progress.json'))
            sealed = seal_selected(s)
            assert sha(TRIAL / 'lock.json') == s['lock_sha256']
            assert sha(ROOT / 'outputs/formal/state.json') == read(TRIAL / 'lock.json')['formal_state_sha256']
            snapshot = dict(worker_before_stop=identity, stopped_worker=owned(pid, ticks),
                            user_scope='64 parents', preserved_complete_selected=64,
                            complete_outputs_outside_scope=[p.stem for p in (TRIAL / 'predictions').glob('*.json') if p.stem not in s['ids']],
                            source_fingerprint=s['source_fingerprint'], helper_sha256=sha(__file__), GT_read=False)
            write(OUT / 'boundary_receipt.json', snapshot)
        except BaseException:
            # Do not strand a stopped owned job if sealing itself fails.
            owned(pid, ticks); os.kill(pid, signal.SIGCONT); raise
        owned(pid, ticks); os.kill(pid, signal.SIGTERM)
        with contextlib.suppress(ProcessLookupError): os.kill(pid, signal.SIGCONT)
        start = time.monotonic()
        while True:
            x = proc(pid)
            if x is None or x['state'] == 'Z': break
            assert x['startticks'] == ticks
            assert time.monotonic()-start < 30, 'Owned worker did not exit; no replacement starts'
            time.sleep(.05)
        snapshot.update(owned_worker_exited=True, no_formal_or_unrelated_process_signalled=True)
        write(OUT / 'boundary_receipt.json', snapshot)
        write(OUT / 'sealed.json', sealed)
        write(TRIAL / 'run.json', dict(status='completed_scope64_sealed_by_user', queries=64,
                                      original_planned_queries=128, fingerprint=s['source_fingerprint'],
                                      source_worker_exited=True, GT_read=False))
        write(OUT / 'controller.json', dict(status='completed_sealed64', queries=64, GT_read=False))
        print('Sealed64, original planned128 worker exited, no GT read', flush=True)
    if score:
        plan = read(TRIAL / 'score_plan.json')
        assert sha(ROOT / 'scripts/state_opd/evaluate.py') == plan['scorer_sha256']
        command = [sys.executable, '-B', str(ROOT / 'scripts/state_opd/evaluate.py'),
                   '--trial', str(TRIAL), '--scope', 'hc64', '--run-after-seal',
                   '--public-output', str(ROOT / 'results/state_opd_hc64')]
        subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--watch', action='store_true'); p.add_argument('--score', action='store_true')
    a = p.parse_args(); main(a.watch, a.score)
