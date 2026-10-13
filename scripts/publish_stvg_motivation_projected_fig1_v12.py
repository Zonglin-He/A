"""Publish only Fig. 1 layout code and redacted evidence, never case media."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'artifacts/stvg_motivation_projected_fig1_v12'
PUBLIC = ROOT / 'results/stvg_motivation_projected_fig1/2026-10-13'
CHECKOUT = ROOT.parent / 'visual-grounding-public-A'
BRANCH = 'research/stvg-opd-paper-hc2-revision-v2'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(data, handle, indent=2)
        handle.write('\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=CHECKOUT).decode().strip()


def stage():
    review = read(BASE / 'ROOT_VISUAL_REVIEW.json')
    assert review['status'] == 'pass'
    assert review['actual_PNG_view'] and review['actual_PDF_page_pixel_view']
    assert review['Projected_intervals_drawn'] == review['DINO_boxes_drawn'] == 0
    assert not git('diff', '--name-only') and not git('diff', '--cached', '--name-only')
    assert git('rev-parse', 'HEAD') == git('rev-parse', 'origin/main')
    assert git('rev-parse', 'HEAD') == git('rev-parse', 'origin/' + BRANCH)
    assert not PUBLIC.exists()
    PUBLIC.mkdir(parents=True)
    for name in ['DESIGN_LOCK.json', 'LAYOUT_RUNTIME002.json', 'LAYOUT_RUNTIME003.json']:
        shutil.copy2(BASE / name, PUBLIC / name)
    public_review = {k: v for k, v in review.items() if k != 'files'}
    public_review['media_integrity'] = {k: v for k, v in review['files'].items()
                                       if k.endswith(('.png', '.pdf', '.svg'))}
    write(PUBLIC / 'VISUAL_REVIEW.json', public_review)
    write(PUBLIC / 'AVAILABILITY.json', dict(
        status='bounded_saved_evidence_readback_complete',
        scope='original shared case and three measured Native references',
        model_count=3, displayed_frames=5, image_panels=15,
        available_Native_intervals=3, available_GT_intervals=1,
        available_Projected_intervals=0, actual_DINO_observations=0,
        proposed_expert_positions_are_not_actual_observations=True,
        other_case_expert_records_rejected=True,
        completed_old_P7A_roster_exact_RGB_matches=0,
        data_missing_is_not_a_method_failure=True,
        measured_extension_complete=False, layout_preview_only=True,
        source_availability_readback_sha256=sha(BASE / 'SOURCE_AVAILABILITY_READBACK.json'),
        expert_empty_readback_sha256=sha(BASE / 'EXPERT_EMPTY_READBACK.json'),
        new_model_calls=0, new_GPU_calls=0, new_DINO_calls=0,
        new_GT_scoring=0, active_P7A_payload_access=False, time=time.time()))
    code = ['scripts/render_stvg_motivation_projected_fig1_v12.py',
            'scripts/publish_stvg_motivation_projected_fig1_v12.py',
            'docs/STVG_MOTIVATION_PROJECTED_FIG1_V12.md']
    write(PUBLIC / 'SOURCE_BINDING.json', dict(
        status='bound', scope='rendering and publication only',
        code={name: sha(ROOT / name) for name in code},
        private_media_query_GT_and_prediction_values_exported=False,
        layout_preview_only=True, measured_extension_complete=False,
        whole_P7A_paper_complete=False, time=time.time()))
    owned = code + [str(path.relative_to(ROOT)) for path in sorted(PUBLIC.iterdir())]
    for name in owned:
        assert not name.endswith(('.png', '.pdf', '.svg', '.pt', '.safetensors'))
        raw = (ROOT / name).read_bytes()
        if name.endswith('.json'):
            for field in [b'"query":', b'"GT_box":', b'"boxes":',
                          b'"interval":', b'"caption":', b'"predictions":']:
                assert field not in raw, (name, field)
        dst = CHECKOUT / name
        assert not dst.exists(), name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dst)
    subprocess.run(['git', 'add', '--', *owned], cwd=CHECKOUT, check=True)
    assert set(git('diff', '--cached', '--name-only').splitlines()) == set(owned)
    files = [dict(path=name, bytes=(CHECKOUT / name).stat().st_size,
                  sha256=sha(CHECKOUT / name), blob_sha=git('hash-object', name),
                  mode='100644') for name in sorted(owned)]
    receipt = dict(status='private_media_excluded_layout_preview_staged',
                   scope='finite layout only; missing measurements remain missing',
                   repository='Zonglin-He/A', branch=BRANCH,
                   base_commit=git('rev-parse', 'HEAD'), expected_tree=git('write-tree'),
                   files=files, file_count=len(files), bytes=sum(f['bytes'] for f in files),
                   private_media_exported=False, time=time.time())
    write(BASE / 'PUBLIC_STAGE.json', receipt)
    print(json.dumps({k: receipt[k] for k in ['status', 'file_count', 'bytes', 'expected_tree']}))


def verify(commit):
    receipt = read(BASE / 'PUBLIC_STAGE.json')
    subprocess.run(['git', 'fetch', 'origin', 'main', BRANCH], cwd=CHECKOUT, check=True)
    assert git('rev-parse', 'HEAD') == commit
    assert git('rev-parse', 'origin/main') == git('rev-parse', 'origin/' + BRANCH) == commit
    assert git('rev-parse', commit + '^{tree}') == receipt['expected_tree']

    def check(item):
        name = item['path']
        url = 'https://raw.githubusercontent.com/Zonglin-He/A/' + commit + '/' + urllib.parse.quote(name)
        req = urllib.request.Request(url, headers={'User-Agent': 'STVG-Fig1-layout-audit'})
        with urllib.request.urlopen(req, timeout=40) as response:
            raw = response.read()
        assert len(raw) == item['bytes']
        assert hashlib.sha256(raw).hexdigest() == item['sha256']
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        assert blob == item['blob_sha'] == git('rev-parse', commit + ':' + name)
        entry = git('ls-tree', commit, '--', name).split()
        assert entry[0] == item['mode'] and entry[2] == blob
        return item

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(check, receipt['files']))
    result = dict(status='pass', scope='all remote bytes/SHA256/Gitblob/mode/tree and both refs',
                  repository='Zonglin-He/A', commit=commit,
                  tree=git('rev-parse', commit + '^{tree}'),
                  main_sha=commit, research_sha=commit, files=files,
                  file_count=len(files), bytes=sum(f['bytes'] for f in files),
                  private_media_exported=False, layout_preview_only=True,
                  measured_extension_complete=False, whole_P7A_paper_complete=False,
                  time=time.time())
    write(BASE / 'FINAL_GITHUB_RECEIPT.json', result)
    print(json.dumps({k: result[k] for k in ['status', 'commit', 'file_count', 'bytes']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['stage', 'verify'])
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.mode == 'stage':
        stage()
    else:
        assert args.commit
        verify(args.commit)
