"""Explicit anonymous P0/protocol export; P1 remains an unfinished experiment."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.stvg_ecld_common_v1 import BASE, read, write, sha
CHECKOUT = ROOT.parent/'visual-grounding-public-A'
BRANCH = 'research/stvg-opd-paper-hc2-revision-v2'


def git(*args):
    return subprocess.check_output(['git', *args], cwd=CHECKOUT, text=True).strip()


def stage():
    assert read(BASE/'p0/ROOT_SUPPORT_READBACK004.json')['status'] == 'pass'
    assert git('remote', 'get-url', 'origin') == 'https://github.com/Zonglin-He/A.git'
    assert not git('diff', '--name-only') and not git('diff', '--cached', '--name-only')
    subprocess.run(['git', 'fetch', '--quiet', 'origin', 'main', BRANCH], cwd=CHECKOUT, check=True)
    assert git('rev-parse', 'HEAD') == git('rev-parse', 'origin/main') == git('rev-parse', 'origin/'+BRANCH)
    paths = {str(p.relative_to(ROOT)) for p in (ROOT/'results/stvg_evidence_conditioned/2026-10-15/p0').iterdir() if p.is_file()}
    paths |= {str(p.relative_to(ROOT)) for p in (ROOT/'scripts').glob('*ecld*.py')}
    paths |= {'docs/STVG_EVIDENCE_CONDITIONED_V1.md', 'protocols/stvg_evidence_conditioned_v1.md'}
    rows = []
    for rel in sorted(paths):
        path = ROOT/rel
        raw = path.read_bytes()
        assert path.suffix in {'.py', '.md', '.json', '.png'}
        if path.suffix == '.json':
            for token in [b'"query_text"', b'"caption"', b'"video_path"', b'"boxes"', b'"masks"', b'"optimizer_state"', b'"weights"']:
                assert token not in raw, (rel, token)
        assert b'/home/'+b'wwww/.codex/'+b'attachments/' not in raw
        dest = CHECKOUT/rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        rows.append(dict(path=rel, sha256=sha(path), bytes=len(raw), mode='100644',
                         blob_sha=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()))
    subprocess.run(['git', 'add', '--', *sorted(paths)], cwd=CHECKOUT, check=True)
    assert set(git('diff', '--cached', '--name-only').splitlines()) == paths
    write(BASE/'P0_PUBLIC_STAGE.json', dict(status='reviewed_staged', base_commit=git('rev-parse', 'HEAD'),
        expected_tree=git('write-tree'), files=rows, branch=BRANCH,
        scope='completed P0 saved-state audit; P1 implementation/protocol only, no P1 performance or full Table1 claim', time=time.time()))
    print('staged', len(rows), sum(r['bytes'] for r in rows))


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent':'STVG-ecld-byte-audit'}), timeout=50) as response:
        return response.read()


def verify(commit):
    plan = read(BASE/'P0_PUBLIC_STAGE.json')
    for branch in ['main', BRANCH]:
        ref = json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/ref/heads/'+urllib.parse.quote(branch, safe='/')))
        assert ref['object']['sha'] == commit
    record = json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/commits/'+commit))
    assert record['tree']['sha'] == plan['expected_tree'] and record['parents'][0]['sha'] == plan['base_commit']
    tree = json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/trees/'+record['tree']['sha']+'?recursive=1'))
    assert not tree.get('truncated')
    lookup = {r['path']: r for r in tree['tree']}
    def one(row):
        entry = lookup[row['path']]
        assert entry['mode'] == row['mode'] and entry['sha'] == row['blob_sha']
        raw = fetch('https://raw.githubusercontent.com/Zonglin-He/A/'+commit+'/'+urllib.parse.quote(row['path']))
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        assert raw == (ROOT/row['path']).read_bytes() == (CHECKOUT/row['path']).read_bytes()
        return dict(row, remote_bytes_identical=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(one, plan['files']))
    write(BASE/'P0_FINAL_GITHUB_RECEIPT.json', dict(status='pass', commit=commit, tree=record['tree']['sha'],
        branches=['main', BRANCH], files=rows, file_count=len(rows), bytes=sum(r['bytes'] for r in rows),
        scope=plan['scope'], time=time.time()))
    print('verified', commit, len(rows))


if __name__ == '__main__':
    stage() if sys.argv[1] == 'stage' else verify(sys.argv[2])
