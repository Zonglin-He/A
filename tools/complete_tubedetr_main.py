"""Finite continuation of the authorized two-direction TubeDETR main study.

Attach to the already-running complete VidSTG job; wait for complete HC media;
run its paired evaluation; score only sealed populations. Never skip a failed
query, tune a parameter, overwrite a prediction, or start another baseline.
"""
import fcntl
import hashlib
import json
import shlex
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stvg.tube_runner import write, population_hash
from stvg.runtime import sha


def process_exists(needle):
    for proc in Path('/proc').iterdir():
        if proc.name.isdecimal():
            try:
                argv = (proc/'cmdline').read_bytes().split(b'\0')
                if any(needle.encode() == a for a in argv):
                    return True
            except (OSError, ProcessLookupError):
                pass
    return False


def run(*args):
    command = [sys.executable, '-B', *map(str, args)]
    process = subprocess.Popen(command, cwd=ROOT)
    if args[:3] == ('-m', 'stvg', 'evaluate'):
        registry = ROOT/'runtime/tubedetr_active_jobs.json'
        value = json.loads(registry.read_text())
        value['jobs']['hc2_formal'] = dict(pid=process.pid, command=shlex.join(command),
            log='logs/complete_tubedetr_main_complete_decoder.log')
        write(registry, value)
    code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def score(ds, direction):
    out = ROOT/'outputs/tubedetr'/direction
    cached = [out/'metrics.json', out/'efficiency.json', out/'qualitative/selection.json']
    if any(p.exists() for p in cached):
        if not all(p.exists() for p in cached):
            raise ValueError('Incomplete offline results; preserve and inspect before reuse')
        reuse_completed_score(ds, out)
        return
    run('-m', 'stvg', 'score', '--aggregation', 'query', '--arms', 'Source,Full',
        '--manifest', ROOT/f'data/evaluation/{ds}.jsonl', '--labels', ROOT/f'data/evaluation/{ds}_labels.jsonl',
        '--predictions', out, '--output', out/'metrics.json')
    run('tools/report_tubedetr_efficiency.py', '--predictions', out, '--output', out/'efficiency.json')
    run('tools/render_tubedetr_qualitative.py', '--manifest', ROOT/f'data/evaluation/{ds}.jsonl',
        '--labels', ROOT/f'data/evaluation/{ds}_labels.jsonl', '--metrics', out/'metrics.json',
        '--predictions', out, '--output', out/'qualitative')


def reuse_completed_score(ds, out):
    """Validate complete prior artifacts and keep their bytes and mtimes intact."""
    manifest = ROOT/f'data/evaluation/{ds}.jsonl'
    labels = ROOT/f'data/evaluation/{ds}_labels.jsonl'
    records = [json.loads(s) for s in manifest.read_text().splitlines()]
    ids = {r['id'] for r in records}
    seal = json.loads((out/'sealed.json').read_text())
    metrics = json.loads((out/'metrics.json').read_text())
    efficiency = json.loads((out/'efficiency.json').read_text())
    if (len(ids) != len(records) or seal['queries'] != len(records) or set(seal['files']) != ids
            or seal['manifest_sha256'] != sha(manifest)
            or metrics['manifest_sha256'] != sha(manifest) or metrics['labels_sha256'] != sha(labels)
            or metrics['status'] != 'scored_sealed_predictions'
            or set(metrics['datasets']) != {ds} or metrics['datasets'][ds]['queries'] != len(records)
            or len(metrics['rows']) != len(records) or {r['id'] for r in metrics['rows']} != ids
            or set(metrics['datasets'][ds]['query_means']) != {'Source', 'Full'}
            or efficiency['manifest_sha256'] != sha(manifest) or efficiency['queries'] != len(records)
            or efficiency['status'] != 'complete_population_efficiency'):
        raise ValueError('Cached complete scoring population or identity changed')
    for entry in seal['files'].values():
        for key, hash_key in [('path', 'sha256'), ('receipt_path', 'receipt_sha256')]:
            path = (out/entry[key]).resolve()
            if not path.is_relative_to(out.resolve()) or sha(path) != entry[hash_key]:
                raise ValueError('Sealed prediction/receipt changed before cached score reuse')
    selections = json.loads((out/'qualitative/selection.json').read_text())
    artifacts = ['metrics.json', 'efficiency.json', 'qualitative/selection.json']
    for selection in selections['selections'].values():
        if 'file' in selection:
            path = (out/'qualitative'/selection['file']).resolve()
            if not path.is_relative_to((out/'qualitative').resolve()) or not path.is_file():
                raise ValueError('Cached qualitative artifact missing or outside output directory')
            artifacts.append(str(path.relative_to(out.resolve())))
    receipt = dict(status='reused_complete_offline_results', prediction_seal_sha256=sha(out/'sealed.json'),
                   manifest_sha256=sha(manifest), labels_sha256=sha(labels),
                   files={p: sha(out/p) for p in sorted(artifacts)}, predictions_or_scores_rewritten=False)
    path = out/'score_reuse_receipt.json'
    if path.exists():
        if json.loads(path.read_text()) != receipt:
            raise ValueError('Cached offline result hashes changed')
    else:
        write(path, receipt)
    print('Reused verified complete scoring and qualitative artifacts: '+str(out), flush=True)


def main():
    lock = (ROOT/'runtime/tubedetr_main_queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = ROOT/'outputs/tubedetr/MAIN_QUEUE_STATUS.json'
    vid = ROOT/'outputs/tubedetr/hc2_to_vid'
    while not (vid/'sealed.json').exists():
        if not process_exists('--backbone') or not process_exists(str(ROOT/'data/evaluation/vidstg.jsonl')) and not process_exists('data/evaluation/vidstg.jsonl'):
            raise RuntimeError('VidSTG inference stopped before full seal; inspect its log, never score partial outputs')
        progress = json.loads((vid/'progress.json').read_text()) if (vid/'progress.json').exists() else {}
        write(status, dict(status='waiting_for_running_complete_VidSTG_inference', progress=progress,
                           task_scores_computed=False))
        time.sleep(30)
    score('vidstg', 'hc2_to_vid')
    hc_manifest = ROOT/'data/evaluation/hc2.jsonl'
    while not hc_manifest.exists():
        acquisition = ROOT/'runtime/hc2_population_download.json'
        progress = json.loads(acquisition.read_text()) if acquisition.exists() else {}
        if progress.get('status') == 'failed' or not process_exists('tools/fetch_hc2_population.py'):
            raise RuntimeError('Complete HC1902 acquisition stopped; never substitute a partial population')
        write(status, dict(status='waiting_for_complete_HC1902_media', progress=progress,
                           sealed_scored_directions=['hc2_to_vid']))
        time.sleep(30)
    rows = [json.loads(s) for s in hc_manifest.read_text().splitlines()]
    if len(rows) != 1902:
        raise ValueError('Confirmed complete HC1902 population changed')
    from stvg.tube_decode import asset_identity
    asset = asset_identity()
    probe = Path(asset['ffmpeg_path']).with_name('ffprobe')
    preflight_path = ROOT/'results/tubedetr/hc2_media_preflight.json'
    write(status, dict(status='preflighting_complete_HC1902_media', sealed_scored_directions=['hc2_to_vid']))
    if preflight_path.exists():
        preflight = json.loads(preflight_path.read_text())
        if (preflight['status'] != 'pass' or preflight['queries'] != 1902 or preflight['videos'] != 1902
                or preflight['manifest_sha256'] != sha(hc_manifest) or preflight['ffprobe_sha256'] != sha(probe)
                or preflight.get('ffmpeg_sha256') != asset['ffmpeg_sha256']
                or preflight.get('frame_index_domain') != 'HC default FFmpeg RGB output'
                or preflight.get('no_queries_filtered') is not True):
            raise ValueError('Cached HC media preflight does not match the complete canonical run')
        print('Reused complete HC default-output media preflight', flush=True)
    else:
        run('tools/preflight_tubedetr_media.py', '--manifest', hc_manifest, '--ffprobe', probe,
            '--output', preflight_path)
    protocol = json.loads((ROOT/'data/evaluation/protocol_vidstg.json').read_text())
    protocol.update(populations={'hc2': population_hash(rows)}, dataset='user-confirmed author HC2 validation release',
                    queries=1902, parents=227, annotation_sha256='61f7a2a74d39c13dd380540d4ac819edb7abcde1826d44229e539d128bf3260c',
                    media_preflight_sha256=sha(ROOT/'results/tubedetr/hc2_media_preflight.json'),
                    input_grid='original HC default FFmpeg output indices; fixed6.4fps; final supported node appended')
    protocol.pop('supersedes_minimal_decoder_protocol_sha256', None)
    protocol_path = ROOT/'data/evaluation/protocol_hc2.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError('Existing frozen HC protocol differs')
    else:
        write(protocol_path, protocol)
    write(ROOT/'results/tubedetr/protocol_hc2.json', protocol)
    write(status, dict(status='evaluating_complete_HC1902', sealed_scored_directions=['hc2_to_vid']))
    hc = ROOT/'outputs/tubedetr/vid_to_hc2'
    if not (hc/'sealed.json').exists():
        run('-m', 'stvg', 'evaluate', '--backbone', 'tubedetr', '--manifest', hc_manifest,
            '--output', hc, '--protocol', protocol_path, '--qualification', ROOT/'results/tubedetr/qualification.json')
    score('hc2', 'vid_to_hc2')
    public, efficiencies = {}, {}
    lines = ['# TubeDETR complete-population paired main results', '',
             'Main estimand: equal queries. Source and Ours share checkpoint, inputs and metrics. Scores below are percentages.', '',
             '| Direction | Queries | Method | m_vIoU | vIoU@0.3 | vIoU@0.5 |',
             '|---|---:|---|---:|---:|---:|']
    for ds, direction in [('hc2', 'vid_to_hc2'), ('vidstg', 'hc2_to_vid')]:
        out = ROOT/'outputs/tubedetr'/direction
        metrics = json.loads((out/'metrics.json').read_text())
        # Public statistics retain cluster structure without media identifiers.
        for group in metrics['datasets'].values():
            for cluster in group['clusters']:
                cluster['parent'] = hashlib.sha256(str(cluster['parent']).encode()).hexdigest()
        for row in metrics['rows']:
            row['parent'] = hashlib.sha256(str(row['parent']).encode()).hexdigest()
        public[direction] = metrics
        efficiencies[direction] = json.loads((out/'efficiency.json').read_text())
        for arm in ['Source', 'Full']:
            values = metrics['datasets'][ds]['query_means'][arm]
            scores = ' | '.join(f'{100*values[k]:.4f}' for k in ['m_vIoU', 'vIoU@0.3', 'vIoU@0.5'])
            lines.append(f'| {direction} | {metrics["datasets"][ds]["queries"]} | {arm} | {scores} |')
    write(ROOT/'results/tubedetr/evaluation.json', dict(status='complete_sealed_offline_scoring',
        backbone='TubeDETR', directions=public, legacy_results_modified=False, other_backbones_evaluated=False))
    write(ROOT/'results/tubedetr/efficiency.json', efficiencies)
    (ROOT/'results/tubedetr/report.md').write_text('\n'.join(lines)+'\n')
    files = ['results/tubedetr/evaluation.json', 'results/tubedetr/efficiency.json', 'results/tubedetr/report.md',
             'results/tubedetr/qualification.json', 'results/tubedetr/protocol_vidstg.json', 'results/tubedetr/protocol_hc2.json']
    write(ROOT/'results/tubedetr/final_seal.json', dict(status='complete', formal_combinations_completed=4,
        files={p: sha(ROOT/p) for p in files}, prediction_seals={d: sha(ROOT/'outputs/tubedetr'/d/'sealed.json')
            for d in ['vid_to_hc2', 'hc2_to_vid']}, publication_pending=True))
    write(status, dict(status='complete', formal_combinations_completed=4,
                       final_results='results/tubedetr/report.md', publication_pending=True))
    state_path = ROOT/'results/tubedetr/STATUS.json'
    state = json.loads(state_path.read_text())
    state.update(status='four_formal_combinations_complete_local_results_sealed',
                 formal_combinations_completed=4, task_accuracy_claims=True, blockers=[],
                 final_results='results/tubedetr/report.md', publication_pending=True)
    write(state_path, state)
    intake = json.loads((ROOT/'results/tubedetr/data_intake.json').read_text())
    intake.update(hc2=json.loads((ROOT/'data/evaluation/hc2_intake.json').read_text()),
                  formal_combinations_completed=4, task_scores_computed=True)
    write(ROOT/'results/tubedetr/data_intake.json', intake)
    run('scripts/research_archive.py', 'check')
    run('scripts/research_archive.py', 'snapshot')
    run('scripts/research_archive.py', 'check')
    print('COMPLETE four TubeDETR evaluation combinations; hash-sealed local results await publication review', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'outputs/tubedetr/MAIN_QUEUE_FAILURE.json', dict(status='failed', error_type=type(error).__name__,
            message=str(error), partial_population_scored=False))
        write(ROOT/'outputs/tubedetr/MAIN_QUEUE_STATUS.json', dict(status='failed', error_type=type(error).__name__,
            message=str(error), partial_population_scored=False))
        raise
