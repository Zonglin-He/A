"""Build a scalar and code export and accept a byte-verified GitHub receipt."""
from datetime import datetime
from pathlib import Path
import re
import shutil

from scope175 import HERE,N,ORDER,PRIVATE,RESULTS,ROOT,SPEC,json_atomic,read,sha,state

REPO='Zonglin-He/A'
BRANCH='research/3eed-r175-relational-support-20261119'
EXPORT=HERE/'publication/public_export_attempt001'
CODE=('scope175','roles175','source175','features175','model175','train175','predict175',
      'score175','audit175','mechanism175','report175','release175')
MAX_BYTES=42000


def check_piece(p):
    payload=p.read_bytes()
    assert len(payload)<=MAX_BYTES
    assert p.suffix in ('.json','.csv','.md','.py')
    assert not re.search(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{30,})',
                         payload.decode())
    return dict(path=p.relative_to(EXPORT).as_posix(),bytes=len(payload),sha256=sha(p))


def split_csv(src,dst):
    raw=src.read_bytes().splitlines(keepends=True)
    header,rows=raw[0],raw[1:]
    dst.mkdir(exist_ok=True,parents=True)
    block=bytearray(header)
    k=0
    for line in rows:
        if len(block)+len(line)>MAX_BYTES:
            (dst/f'{k:03}.csv').write_bytes(block)
            block=bytearray(header)
            k+=1
        block.extend(line)
    (dst/f'{k:03}.csv').write_bytes(block)
    assert sum(len(f.read_bytes().splitlines())-1 for f in dst.glob('*.csv'))==len(rows)


def stage():
    decision=read(RESULTS/'DECISION.json')
    audit=read(RESULTS/'independent_audit.json')
    for item in (decision,audit):
        assert item['n']==N and not item['FINAL_access']
    seal=read(PRIVATE/'SCORING_SEAL.json')
    for file,digest in seal['result_hashes'].items():
        assert sha(RESULTS/file)==digest
    assert decision['audit_sha256']==sha(RESULTS/'independent_audit.json')
    if EXPORT.exists():
        m=read(EXPORT/'MANIFEST.json')
        assert m['audit']==decision['audit_sha256']
        assert all(sha(EXPORT/x['path'])==x['sha256'] for x in m['files'])
        return dict(path=str(EXPORT),files=len(m['files'])+1)
    EXPORT.mkdir(parents=True,mode=0o700)
    (EXPORT/'own_code').mkdir()
    (EXPORT/'results').mkdir()
    md=(HERE/'REPORT_ZH.md').read_text(encoding='utf-8')
    for name in CODE:
        shutil.copyfile(HERE/f'{name}.py',EXPORT/'own_code'/f'{name}.py')
    shutil.copyfile(SPEC,EXPORT/'PROTOCOL.json')
    for p in sorted(RESULTS.iterdir()):
        if p.suffix=='.csv':
            if p.name.startswith('paired_') and '_to_' in p.name:
                split_csv(p,EXPORT/'results'/'anonymous_pair_scores'/p.stem.removeprefix('paired_'))
            elif p.stat().st_size>MAX_BYTES:
                split_csv(p,EXPORT/'results'/p.stem)
                md=md.replace(f'results/{p.name}',f'results/{p.stem}')
            else:
                shutil.copyfile(p,EXPORT/'results'/p.name)
        elif p.suffix=='.json':
            shutil.copyfile(p,EXPORT/'results'/p.name)
    (EXPORT/'REPORT_ZH.md').write_text(md,encoding='utf-8')
    (EXPORT/'README.md').write_text(
        '# Source-Supervised Relational Support Feedback\n\n'
        'R175 uses each of three frozen single-platform 3EED models and same-source RGB features. '
        'Source-only labelled role pairs train the candidate decoder. Target checkpoints are locked before '
        'paired prediction on the same 18,620 ordinary Validation direction records.\n\n'
        'The locked implementation and controls are under own_code. The workspace commands are '
        'source175.py, features175.py, train175.py, predict175.py, score175.py, audit175.py, '
        'mechanism175.py and report175.py. The preparation and training commands take the three source names. '
        'Frozen feature caches, raw annotations, checkpoint weights and coordinates are private prerequisites. '
        'The public scalar files retain both signs of all observed differences. See REPORT_ZH.md for details.\n',
        encoding='utf-8')
    if (HERE/'ATTEMPTS.md').exists():
        shutil.copyfile(HERE/'ATTEMPTS.md',EXPORT/'ATTEMPTS.md')
    evidence={k:sha(PRIVATE/f'{k}.json') for k in ('INPUT_LOCK','EVALUATION_LOCK','SCORING_SEAL')}
    evidence['PREDICTION_SEAL']=sha(PRIVATE/'predictions/PREDICTION_SEAL.json')
    json_atomic(EXPORT/'EVIDENCE_DIGESTS.json',evidence)
    manifest=[check_piece(p) for p in sorted(EXPORT.rglob('*')) if p.is_file()]
    json_atomic(EXPORT/'MANIFEST.json',dict(round='R175',n=N,parent=read(SPEC)['parent_commit'],
                            audit=decision['audit_sha256'],files=manifest))
    return dict(path=str(EXPORT),files=len(manifest)+1)


def finish(receipt):
    manifest=read(EXPORT/'MANIFEST.json')
    assert receipt['repository']==REPO and receipt['branch']==BRANCH
    assert receipt['parent']==manifest['parent'] and receipt['mainUnchanged'] and receipt['priorResearchUnchanged']
    assert receipt['nonForceFastForward'] and re.fullmatch('[a-f0-9]{40}',receipt['commit'])
    assert receipt['files']==receipt['verifiedFiles']==len(manifest['files'])+1
    assert receipt['verifiedBytes']==receipt['bytes']==sum(x['bytes'] for x in manifest['files'])+(
        EXPORT/'MANIFEST.json').stat().st_size
    path=HERE/'publication/PUBLICATION_RECEIPT.json'
    out=dict(receipt,manifest_sha256=sha(EXPORT/'MANIFEST.json'),audit_sha256=manifest['audit'],
             recorded_local=datetime.now().astimezone().isoformat(),FINAL_access=False)
    if path.exists():
        assert read(path)['commit']==out['commit']
    else:
        json_atomic(path,out)
    d=read(RESULTS/'DECISION.json')
    a=d['mean_deltas_pp']['self_geometry']
    b=d['mean_deltas_pp']['pair_only']
    archive=ROOT/'research/ARCHIVE.md'
    text=(f'\n\n## 2026-11-19 — R175 source-supervised relational support feedback\n\n'
          f'18,620 ordinary paired directions. Full minus self geometry Acc25, Acc50, mIoU '
          f'{a["Acc25"]:+.4f}/{a["Acc50"]:+.4f}/{a["mIoU"]:+.4f} pp. '
          f'Full minus pair-only {b["Acc25"]:+.4f}/{b["Acc50"]:+.4f}/{b["mIoU"]:+.4f} pp. '
          f'Source-only checkpoint selection. Frozen target, preserved negative cases, FINAL access zero. '
          f'Deployment candidate {d["deployment_candidate"]}.\n\n'
          f'[Immutable report](https://github.com/{REPO}/blob/{out["commit"]}/{out["prefix"]}/REPORT_ZH.md) · '
          f'[Publication receipt](../{path.relative_to(ROOT).as_posix()})\n')
    if out['commit'] not in archive.read_text(encoding='utf-8'):
        with archive.open('a',encoding='utf-8') as f:
            f.write(text)
    phase='completed_published_candidate' if d['deployment_candidate'] else 'completed_published_not_adopted'
    state(phase,'Ordinary evaluation and source-trained relation controls published and byte-verified.',out)
    return out


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--receipt',type=Path)
    args=p.parse_args()
    print(finish(read(args.receipt)) if args.receipt else stage())
