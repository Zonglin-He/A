"""Immediate authorized bootstrap: verify downloads, prepare assets, qualify.

This is not a recurring monitor. It never scores partial predictions, changes
method parameters, replaces the HC split, or invents a cross-machine agreement.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stvg.runtime import sha
from stvg.tube_runner import write


ASSETS = {
    'checkpoints/referdino/ryt_mevis_swinb.pth': '87e5356c6c463efec4f736c3348d1fbfba9bfaaa8181d51d15a8404295a5a9e3',
    'checkpoints/tubedetr/vidstg_k2_res224.pth': '2802c66049b2e7986a826b4bbd291eb8521446cfba6fee03b56729438bf44213',
    'checkpoints/tubedetr/hc2_k2_res224.pth': 'c646a8b7131bb28806c350e16479cd8bb31d4425ccb15129b26a6e4d38dd4222',
    'data/vidor/validation-video.zip': '40dd102b43ecd2ed70f2dc38ee08258315e25919d2f1fb8f43bb235740a83a25',
}


def run(*args):
    subprocess.run([sys.executable, '-B', *args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-for-assets', action='store_true')
    args = parser.parse_args()
    status = ROOT/'outputs/tubedetr/BOOTSTRAP_STATUS.json'
    verified = set()
    while True:
        missing = []
        for rel, expected in ASSETS.items():
            if rel in verified:
                continue
            path = ROOT/rel
            if not path.exists():
                missing.append(rel)
                continue
            if sha(path) != expected:
                raise RuntimeError('Downloaded asset fingerprint mismatch: '+rel)
            verified.add(rel)
            print('VERIFIED', rel, flush=True)
        if not missing:
            break
        write(status, dict(status='waiting_for_asset_downloads', verified=sorted(verified), missing=missing,
                           formal_evaluation_started=False, task_metrics_computed=False))
        if not args.wait_for_assets:
            print('Assets still incomplete; status saved', flush=True)
            return
        acquisition = ROOT/'runtime/acquisition_status.json'
        if acquisition.exists() and json.loads(acquisition.read_text()).get('status') == 'failed':
            raise RuntimeError('Bounded asset acquisition failed; inspect runtime/acquisition_status.json')
        time.sleep(30)
    run('tools/setup.py', '--prepare', '--tokenizer', 'runtime/tokenizers/bert-base-uncased')
    write(status, dict(status='qualifying_real_frozen_models', formal_evaluation_started=False))
    from stvg.tube_runner import implementation_hash, runtime_identity
    from stvg.tube_decode import asset_identity
    qualification = ROOT/'results/tubedetr/qualification.json'
    previous = json.loads(qualification.read_text()) if qualification.exists() else {}
    if (previous.get('status') != 'pass' or previous.get('implementation_sha256') != implementation_hash()
            or previous.get('decoder_asset') != asset_identity() or previous.get('runtime_asset') != runtime_identity()):
        run('tools/qualify_tubedetr.py', '--hc2-request', 'data/evaluation/engineering_hc2.json',
            '--vidstg-request', 'data/evaluation/engineering_vidstg.json')
    # Extract only the original test-video population, preserving ZIP CRC checks.
    from tools.build_manifests import vidor_records, build
    from tools.remote_archive import extract_member
    annotations = ROOT/'data/annotations/vidstg/test_annotations.json'
    raw = json.loads(annotations.read_text())
    original = vidor_records(ROOT/'data/vidor/validation-annotation.zip')
    names = {Path(original[v['vid']]['video_path']).name for v in raw}
    video_root = ROOT/'data/vidor/videos'
    with zipfile.ZipFile(ROOT/'data/vidor/validation-video.zip') as z:
        mapping = {}
        for name in z.namelist():
            if Path(name).name in names:
                if Path(name).name in mapping:
                    raise RuntimeError('Ambiguous official video member')
                mapping[Path(name).name] = name
        if set(mapping) != names:
            raise RuntimeError('Required VidSTG test media missing from official archive')
        for name in sorted(names):
            path = video_root/name
            if path.exists():
                # Existing extraction cannot silently replace official media.
                import zlib
                crc = 0
                with path.open('rb') as f:
                    for block in iter(lambda: f.read(8 << 20), b''):
                        crc = zlib.crc32(block, crc)
                info = z.getinfo(mapping[name])
                if path.stat().st_size != info.file_size or crc != info.CRC:
                    raise RuntimeError('Existing video differs from official member')
            else:
                extract_member(z, mapping[name], path)
    if not (ROOT/'data/evaluation/vidstg.jsonl').exists():
        build('vidstg', annotations, video_root, ROOT/'data/evaluation', ROOT/'data/vidor/validation-annotation.zip')
    write(status, dict(status='qualification_complete_waiting_for_shared_formal_protocol',
        complete_assets=sorted(verified), qualification='results/tubedetr/qualification.json',
        vidstg_full_manifest='data/evaluation/vidstg.jsonl',
        HC_required_annotation='data/annotations/hc2/val_v2.json',
        HC_required_sha256='2e22d9370e311ca6c420b90867f5128507f5ab145e4fc120495b7a6336e41156',
        missing_requirements=['canonical HC3482 annotations and complete media', 'cross-machine manifest and RGB agreement'],
        formal_evaluation_started=False, task_metrics_computed=False))
    print('Real-model qualification and VidSTG intake complete. Formal evaluation awaits shared HC/protocol assets.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        write(ROOT/'outputs/tubedetr/BOOTSTRAP_FAILURE.json', dict(status='failed', error_type=type(error).__name__,
             message=str(error), formal_evaluation_started=False, task_metrics_computed=False))
        raise
