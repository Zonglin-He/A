"""Use the locked R163 records and explicit output scopes in R164."""
from datetime import datetime
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT/'crossplatform/round164'
PROTOCOL = HERE/'experiment/PROTOCOL.json'
R163 = ROOT/'research/round163'
W163 = ROOT/'crossplatform/round163'
sys.path.insert(0, str(R163))
from scope163 import (PARENT as R162, PARENT_WORK as W162, PLATFORMS, SETTINGS,
                      atomic_json, pinned_parent, read, sensor_row, sha, source_scope, target_rows)


def source_index():
    seal = read(R163/'SOURCE_SEAL.json')
    audit = read(R163/'SOURCE_AUDIT.json')
    path = W163/'SOURCE_FILES_PRIVATE.json'
    assert seal['index_sha256'] == sha(path) and seal['source_config_sha256'] == sha(SETTINGS)
    assert audit['source_seal_sha256'] == sha(R163/'SOURCE_SEAL.json') and seal['n'] == audit['rows'] == 192
    assert read(R163/'STATE.json')['public_commit'] == read(PROTOCOL)['parent']
    return read(path)


def role_panel():
    path = W162/'AUDIT_PANEL_PRIVATE.json'
    manifest = W162/'DEV_FILES_PRIVATE.json'
    assert sha(path) == read(R162/'PANEL_LOCK.json')['input_sha256']
    seal = read(R162/'DEV_SEAL.json')
    assert sha(manifest) == seal['file_index_sha256']
    return read(path), read(manifest)


def review_rows():
    pair = W163/'ROLE_REVIEW_PRIVATE.json'
    adjudication = W163/'ROLE_ADJUDICATION_PRIVATE.json'
    assert sha(pair) == read(R163/'ROLE_SCOPE.json')['private_review_sha256']
    assert sha(adjudication) == read(R163/'results/ROLE_REVIEW_SUMMARY.json')['review_sha256']
    a, b = read(pair), read(adjudication)
    assert [r['case'] for r in a] == [r['case'] for r in b] == list(range(1,34))
    return zip(a, b)


def write_seal(name, folder, identifiers, **fields):
    digests = {p.name:sha(p) for p in sorted(folder.glob('*.npz'))}
    assert set(digests) == {f'{i:05d}.npz' for i in identifiers}
    index = WORK/(name+'_FILES_PRIVATE.json')
    atomic_json(index, digests)
    atomic_json(HERE/(name+'_SEAL.json'),
                dict(n=len(digests), index_sha256=sha(index), protocol_sha256=sha(PROTOCOL),
                     FINAL_access=False, **fields))


def verify_seal(name, folder):
    seal = read(HERE/(name+'_SEAL.json'))
    index = WORK/(name+'_FILES_PRIVATE.json')
    assert seal['protocol_sha256'] == sha(PROTOCOL) and seal['index_sha256'] == sha(index)
    files = read(index)
    assert seal['n'] == len(files) and not seal['FINAL_access']
    assert all(sha(folder/name) == digest for name, digest in files.items())
    return files


def begin():
    source_scope()
    source_index()
    role_panel()
    pinned_parent()
    local = HERE/'STATE.json'
    assert not local.exists()
    overall_file = ROOT/'research/STATE.json'
    overall = read(overall_file)
    assert overall['R161']['state'] == 'paused_by_user'
    stamp = datetime.now().astimezone().isoformat()
    task = dict(round='R164', state='running', parent=read(PROTOCOL)['parent'], geometry_rows=192,
                audited_language_edges=33, protocol='round164/experiment/PROTOCOL.json',
                baseline_queue='paused', FINAL_access=False, updated=stamp)
    atomic_json(local, task)
    overall['R164'] = task
    overall.update(current_round='R164', current_stage='R164_FIXED_WINNER_GEOMETRY_AND_ROLE_RANKS',
                   manual_active_state='round164/STATE.json', run_state='R164_RUNNING', updated_local=stamp)
    overall['directions']['crossplatform'].update(active_round='R164', execution_open=True)
    atomic_json(overall_file, overall)


if __name__ == '__main__':
    begin()
