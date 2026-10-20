"""Finite continuation of the authorized two-direction TubeDETR main study.

Attach to the already-running complete VidSTG job; wait for complete HC media;
run its paired evaluation; score only sealed populations. Never skip a failed
query, tune a parameter, overwrite a prediction, or start another baseline.
"""
import fcntl
import hashlib
import json
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
    subprocess.run([sys.executable, '-B', *map(str, args)], cwd=ROOT, check=True)


def score(ds, direction):
    out = ROOT/'outputs/tubedetr'/direction
    run('-m', 'stvg', 'score', '--aggregation', 'query', '--arms', 'Source,Full',
        '--manifest', ROOT/f'data/evaluation/{ds}.jsonl', '--labels', ROOT/f'data/evaluation/{ds}_labels.jsonl',
        '--predictions', out, '--output', out/'metrics.json')
    run('tools/report_tubedetr_efficiency.py', '--predictions', out, '--output', out/'efficiency.json')
    run('tools/render_tubedetr_qualitative.py', '--manifest', ROOT/f'data/evaluation/{ds}.jsonl',
        '--labels', ROOT/f'data/evaluation/{ds}_labels.jsonl', '--metrics', out/'metrics.json',
        '--predictions', out, '--output', out/'qualitative')


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
    protocol = json.loads((ROOT/'data/evaluation/protocol_vidstg.json').read_text())
    protocol.update(populations={'hc2': population_hash(rows)}, dataset='user-confirmed author HC2 validation release',
                    queries=1902, parents=227, annotation_sha256='61f7a2a74d39c13dd380540d4ac819edb7abcde1826d44229e539d128bf3260c',
                    input_grid='original HC default FFmpeg output indices; fixed6.4fps; final supported node appended')
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
        raise
