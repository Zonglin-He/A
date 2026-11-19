"""Build an anonymous public export and record source-verified publication."""
import argparse
from datetime import datetime
from pathlib import Path
import re
import shutil

from scope177 import HERE,N,PRIVATE,RESULTS,ROOT,SPEC,json_atomic,read,sha,update_state

REPO = 'Zonglin-He/A'
BRANCH = 'research/3eed-r177-r174-fine-ablation-20261119'
EXPORT = HERE/'publication/public_export_attempt001'
CODE = ('scope177','model177','predict177','score177','release177')
CAP = 42000
SECRET = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{30,})')


def shard_csv(source,path):
    header,*rows = source.read_bytes().splitlines(keepends=True)
    path.mkdir(parents=True)
    current = bytearray(header)
    part = 0
    for line in rows:
        if len(current)+len(line)>CAP:
            (path/f'{part:03d}.csv').write_bytes(current)
            part += 1
            current = bytearray(header)
        current.extend(line)
    (path/f'{part:03d}.csv').write_bytes(current)
    assert sum(len(p.read_bytes().splitlines())-1 for p in path.glob('*.csv'))==len(rows)


def stage():
    score = read(PRIVATE/'SCORING_SEAL.json')
    decision = read(RESULTS/'DECISION.json')
    assert score['n']==decision['n']==N and not (score['FINAL_access'] or decision['FINAL_access'])
    assert decision['score_audit']==sha(RESULTS/'independent_audit.json')
    assert all(sha(RESULTS/p)==h for p,h in score['results'].items())
    path = EXPORT/'MANIFEST.json'
    if path.exists():
        proof = read(path)
        assert all(sha(EXPORT/x['path'])==x['sha256'] for x in proof['files'])
        return proof
    EXPORT.mkdir(parents=True,mode=0o700)
    (EXPORT/'own_code').mkdir()
    (EXPORT/'results').mkdir()
    for name in CODE:
        shutil.copyfile(HERE/f'{name}.py',EXPORT/'own_code'/f'{name}.py')
    shutil.copyfile(SPEC, EXPORT/'PROTOCOL.json')
    report = (HERE/'REPORT_ZH.md').read_text(encoding='utf-8')
    for p in sorted(RESULTS.iterdir()):
        assert p.suffix in ('.json','.csv')
        if p.suffix=='.csv' and p.stat().st_size>CAP:
            shard_csv(p,EXPORT/'results'/p.stem)
            report = report.replace(f'results/{p.name}',f'results/{p.stem}')
        else:
            shutil.copyfile(p,EXPORT/'results'/p.name)
    (EXPORT/'REPORT_ZH.md').write_text(report,encoding='utf-8')
    (EXPORT/'README.md').write_text(
        '# R177 R174 component ablations\n\n'
        'The registered full R174 decoder and three single-source checkpoints are fixed. '
        'Fifteen component-level inference changes are paired against the complete method '
        'on 18,620 ordinary Validation direction records.\n\n'
        'Run predict177.py and then score177.py in the registered workspace. '
        'The R174 private features, matching candidates, model weights and audited IoUs are prerequisites. '
        'The released anonymous scalar deltas preserve every improvement and degradation.\n',
        encoding='utf-8')
    digests = {k:sha(PRIVATE/f'{k}.json') for k in ('EXECUTION_LOCK','SCORING_SEAL')}
    digests['PREDICTION_SEAL'] = sha(PRIVATE/'predictions/PREDICTION_SEAL.json')
    json_atomic(EXPORT/'EVIDENCE_DIGESTS.json',digests)
    files = []
    for p in sorted(EXPORT.rglob('*')):
        if p.is_file():
            content = p.read_bytes()
            assert len(content)<=CAP and p.suffix in ('.json','.csv','.md','.py')
            assert not SECRET.search(content.decode('utf-8'))
            files.append(dict(path=p.relative_to(EXPORT).as_posix(),bytes=len(content),sha256=sha(p)))
    proof = dict(round='R177',n=N,parent=read(SPEC)['parent_commit'],
                 audit=decision['score_audit'],files=files)
    json_atomic(path,proof)
    return proof


def finish(receipt):
    manifest = read(EXPORT/'MANIFEST.json')
    assert receipt['repository']==REPO and receipt['branch']==BRANCH
    assert receipt['parent']==manifest['parent'] and re.fullmatch('[a-f0-9]{40}',receipt['commit'])
    assert receipt['mainUnchanged'] and receipt['priorResearchUnchanged'] and receipt['nonForceFastForward']
    assert receipt['files']==receipt['verifiedFiles']==len(manifest['files'])+1
    assert receipt['bytes']==receipt['verifiedBytes']==sum(
        f['bytes'] for f in manifest['files'])+(EXPORT/'MANIFEST.json').stat().st_size
    path = HERE/'publication/PUBLICATION_RECEIPT.json'
    value = dict(receipt,manifest_sha256=sha(EXPORT/'MANIFEST.json'),audit_sha256=manifest['audit'],
                 recorded_local=datetime.now().astimezone().isoformat(),FINAL_access=False)
    if path.exists():
        assert read(path)['commit']==value['commit']
    else:
        json_atomic(path,value)
    archive = ROOT/'research/ARCHIVE.md'
    if value['commit'] not in archive.read_text(encoding='utf-8'):
        decision = read(RESULTS/'DECISION.json')
        with archive.open('a',encoding='utf-8') as f:
            f.write(
                '\n\n## 2026-11-19 — R177 fine-grained ablations of the registered R174 method\n\n'
                'Registered complete R174 as the main method by user selection. '
                'Fifteen fixed-checkpoint alternatives on 18,620 ordinary paired directions. '
                f"Exact inference simplifications {', '.join(decision['exact_inference_simplifications'])}. "
                'All paired changes retained. Target and source updates zero. FINAL access zero.\n\n'
                f'[Immutable report](https://github.com/{REPO}/blob/{value["commit"]}/{value["prefix"]}/REPORT_ZH.md) · '
                f'[Publication receipt](../{path.relative_to(ROOT).as_posix()})\n')
    update_state('completed_published_verified','Audited frozen R174 component ablations published.',value)
    return value


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt',type=Path)
    args = parser.parse_args()
    if args.receipt:
        print(finish(read(args.receipt)))
    else:
        manifest = stage()
        update_state('publication_pending','Audited private results and public code staged for publication.')
        print(dict(path=str(EXPORT),files=len(manifest['files'])+1))
