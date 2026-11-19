"""Export code and anonymous paired scalars, then record verified publication."""
import argparse
from datetime import datetime
from pathlib import Path
import re
import shutil

from scope176 import HERE, N, PRIVATE, RESULTS, ROOT, SPEC, json_atomic, read, sha, state

REPO = 'Zonglin-He/A'
BRANCH = 'research/3eed-r176-dual-role-coverage-20261119'
EXPORT = HERE/'publication/public_export_attempt001'
CODE = ('scope176','diagnose176','retrieve176','score176','audit176','summarize176','release176')
CAP = 42000
SECRET = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{30,})')


def shards(src,dst):
    header,*rows=src.read_bytes().splitlines(keepends=True)
    dst.mkdir(parents=True)
    group=bytearray(header)
    i=0
    for row in rows:
        if len(group)+len(row)>CAP:
            (dst/f'{i:03}.csv').write_bytes(group)
            i+=1
            group=bytearray(header)
        group.extend(row)
    (dst/f'{i:03}.csv').write_bytes(group)
    assert sum(len(p.read_bytes().splitlines())-1 for p in dst.glob('*.csv'))==len(rows)


def manifest():
    decision=read(RESULTS/'DECISION.json')
    scoring=read(PRIVATE/'RESULTS_SEAL.json')
    assert decision['n']==scoring['n']==N
    assert not (decision['FINAL_access'] or scoring['FINAL_access'])
    assert sha(RESULTS/'independent_audit.json')==decision['audit_sha256']
    assert all(sha(RESULTS/k)==v for k,v in scoring['result_hashes'].items())
    done=EXPORT/'MANIFEST.json'
    if done.exists():
        value=read(done)
        assert all(sha(EXPORT/x['path'])==x['sha256'] for x in value['files'])
        return value
    EXPORT.mkdir(parents=True,mode=0o700)
    (EXPORT/'own_code').mkdir()
    (EXPORT/'results').mkdir()
    (EXPORT/'experiment').mkdir()
    md=(HERE/'REPORT_ZH.md').read_text(encoding='utf-8')
    for name in CODE:
        shutil.copyfile(HERE/f'{name}.py',EXPORT/'own_code'/f'{name}.py')
    for p in (SPEC,HERE/'retrieval_protocol.json'):
        shutil.copyfile(p,EXPORT/'experiment'/p.name)
    shutil.copyfile(HERE/'ATTEMPTS.md',EXPORT/'ATTEMPTS.md')
    for p in sorted(RESULTS.iterdir()):
        assert p.suffix in ('.csv','.json')
        if p.suffix=='.csv' and p.stat().st_size>CAP:
            shards(p,EXPORT/'results'/p.stem)
            md=md.replace(f'results/{p.name}',f'results/{p.stem}')
        else:
            shutil.copyfile(p,EXPORT/'results'/p.name)
    (EXPORT/'REPORT_ZH.md').write_text(md,encoding='utf-8')
    (EXPORT/'README.md').write_text(
        '# R176 Dual-role coverage and frozen four-slot retrieval\n\n'
        'The diagnosis traces missing A256 hypotheses, shortlist misses, and output misses. '
        'Source-held verified boxes and ordinary target reviewed references are scored separately. '
        'The retriever intervention uses the frozen R175 pair-only decoder on the same 18,620 rows.\n\n'
        'In the registered workspace, run diagnose176.py, summarize176.py --gate, retrieve176.py, '
        'score176.py, audit176.py, and summarize176.py in that order. The retrieval process reads no '
        'grounding labels. The diagnostics require the existing private benchmark and R175 caches. '
        'Only anonymous scalar outcomes, protocols, implementation and audit receipts are published.\n',
        encoding='utf-8')
    keys=('INPUT_LOCK','DIAGNOSIS_SEAL','RETRIEVAL_GATE','RETRIEVAL_LOCK',
          'RETRIEVAL_SCORE_SEAL','RESULTS_SEAL')
    evidence={name:sha(PRIVATE/f'{name}.json') for name in keys}
    evidence['PREDICTION_SEAL']=sha(PRIVATE/'retrieval_predictions/PREDICTION_SEAL.json')
    json_atomic(EXPORT/'EVIDENCE_DIGESTS.json',evidence)
    entries=[]
    for p in sorted(EXPORT.rglob('*')):
        if p.is_file():
            body=p.read_bytes()
            assert len(body)<=CAP and p.suffix in ('.csv','.json','.py','.md')
            assert not SECRET.search(body.decode('utf-8'))
            entries.append(dict(path=p.relative_to(EXPORT).as_posix(),bytes=len(body),sha256=sha(p)))
    out=dict(round='R176',n=N,parent=read(SPEC)['parent_commit'],audit=decision['audit_sha256'],files=entries)
    json_atomic(done,out)
    return out


def finish(receipt):
    info=read(EXPORT/'MANIFEST.json')
    assert receipt['repository']==REPO and receipt['branch']==BRANCH and receipt['parent']==info['parent']
    assert receipt['mainUnchanged'] and receipt['priorResearchUnchanged'] and receipt['nonForceFastForward']
    assert receipt['files']==receipt['verifiedFiles']==len(info['files'])+1
    assert receipt['bytes']==receipt['verifiedBytes']==(
        sum(p['bytes'] for p in info['files'])+(EXPORT/'MANIFEST.json').stat().st_size)
    assert re.fullmatch('[a-f0-9]{40}',receipt['commit'])
    p=HERE/'publication/PUBLICATION_RECEIPT.json'
    out=dict(receipt,manifest_sha256=sha(EXPORT/'MANIFEST.json'),audit_sha256=info['audit'],
             recorded_local=datetime.now().astimezone().isoformat(),FINAL_access=False)
    if p.exists():
        assert read(p)['commit']==out['commit']
    else:
        json_atomic(p,out)
    d=read(RESULTS/'DECISION.json')
    gate=d['gate']
    gain=d['primary_deltas_pp']['original_pair']
    status='candidate' if d['eligible_to_replace_main'] else 'not_adopted'
    archive=ROOT/'research/ARCHIVE.md'
    if out['commit'] not in archive.read_text(encoding='utf-8'):
        with archive.open('a',encoding='utf-8') as f:
            f.write(
                '\n\n## 2026-11-19 — R176 dual-role candidate coverage\n\n'
                f"336 recording-held dual labels at IoU > 0.10. No A256 pair {gate['A256_no_pair']}, "
                f"four-slot miss {gate['four_slots_no_pair']}, output miss {gate['output_miss']}, "
                f"success {gate['success']}. Overlap retriever minus original pair-only on the same "
                f"18,620 ordinary records Acc25/Acc50/mIoU {gain['Acc25']:+.4f}/{gain['Acc50']:+.4f}/"
                f"{gain['mIoU']:+.4f} pp. Outcome {status}. All paired outcomes preserved. "
                'Source and target updates zero. FINAL access zero.\n\n'
                f'[Immutable report](https://github.com/{REPO}/blob/{out["commit"]}/{out["prefix"]}/REPORT_ZH.md) · '
                f'[Publication receipt](../{p.relative_to(ROOT).as_posix()})\n')
    state('completed_published_'+status,'Coverage attribution and frozen four-slot retrieval comparison published.',out)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--receipt',type=Path)
    args=parser.parse_args()
    if args.receipt:
        print(finish(read(args.receipt)))
    else:
        result=manifest()
        state('publication_pending','Audited results and code ready for publication')
        print(dict(path=str(EXPORT),files=len(result['files'])+1))
