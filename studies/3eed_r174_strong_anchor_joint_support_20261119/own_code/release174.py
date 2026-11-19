"""Assemble the study's public code and verified scalar export."""
from datetime import datetime
from pathlib import Path
import csv
import re
import shutil

from scope174 import CELLS, HERE, ORDER, PLATFORMS, PRIVATE, RESULTS, ROOT, SPEC, json_atomic, read, sha, state

REPO='Zonglin-He/A'
BRANCH='research/3eed-r174-strong-anchor-20261119'
EXPORT=HERE/'publication/public_export_attempt001'
INVENTORY=('scope174','anchors174','capture_source174','model174','train174','predict174',
           'score174','audit174','run_audit174','identity174','report174','release174')
TABLES=('endpoints','paired_deltas','factorial_effects','candidate_coverage',
        'source_training','source_reference_intervention','target_reference_intervention',
        'source_reference_paired','target_reference_paired')
EXPORT_BYTES=42000


def split_table(path, target):
    target.mkdir(parents=True,exist_ok=True)
    with path.open(encoding='utf-8',newline='') as f:
        lines=f.readlines()
    header, rows = lines[0], lines[1:]
    assert rows
    bundle=[header]
    length=len(header.encode())
    j=0
    for row in rows:
        n=len(row.encode())
        if n+length>EXPORT_BYTES:
            (target/f'part{j:03}.csv').write_text(''.join(bundle),encoding='utf-8')
            j+=1
            bundle=[header]
            length=len(header.encode())
        bundle.append(row)
        length+=n
    (target/f'part{j:03}.csv').write_text(''.join(bundle),encoding='utf-8')
    expected=sum(len(p.read_text().splitlines())-1 for p in target.glob('*.csv'))
    assert len(rows)==expected


def script_sources():
    bundle=EXPORT/'own_code'
    bundle.mkdir(exist_ok=True)
    for p in INVENTORY:
        shutil.copyfile(HERE/f'{p}.py',bundle/f'{p}.py')
    shutil.copyfile(SPEC,EXPORT/'PROTOCOL.json')


def bundle_tables():
    scoring=read(PRIVATE/'SCORING_SEAL.json')
    for p,h in scoring['public_tables'].items():
        assert sha(RESULTS/p)==h
    out=EXPORT/'results'
    out.mkdir(exist_ok=True)
    for f in TABLES:
        p=RESULTS/f'{f}.csv'
        if p.stat().st_size>EXPORT_BYTES:
            split_table(p,out/f)
        else:
            shutil.copyfile(p,out/p.name)
    for source,target in ORDER:
        p=RESULTS/f'paired_{source}_to_{target}.csv'
        # Preserve every score and outcome, without an observation or candidate identifier.
        with p.open(encoding='utf-8',newline='') as src:
            fields=csv.DictReader(src).fieldnames
        assert fields==['reference_mention','direct_left_right','candidate_set_best_iou',
                        'r158_pca',*CELLS]
        split_table(p,out/'anonymous_pair_scores'/f'{source}_to_{target}')
    for f in ('paired_recording_intervals.json','independent_audit.json','DECISION.json'):
        shutil.copyfile(RESULTS/f,out/f)
    sanitized={}
    for p in ('source','target'):
        r=read(RESULTS/f'{p}_reference_intervention_receipt.json')
        sanitized[p]={k:r[k] for k in ('n','status','FINAL_access','descriptive_only')}
    json_atomic(out/'reference_intervention_counts.json',sanitized)
    evidence = {k:sha(PRIVATE/f'{k}.json') for k in
                ('INPUT_LOCK','EVALUATION_LOCK','SCORING_SEAL')}
    for p in ('PREDICTION_SEAL',):
        evidence[p]=sha(PRIVATE/'predictions'/f'{p}.json')
    json_atomic(out/'LOCAL_EVIDENCE_DIGESTS.json',evidence)


def staging():
    result=read(RESULTS/'DECISION.json')
    audit=read(RESULTS/'independent_audit.json')
    assert result['audit']==sha(RESULTS/'independent_audit.json')
    assert not result['FINAL_access'] and audit['n']==18620
    if EXPORT.exists():
        m=read(EXPORT/'MANIFEST.json')
        assert m['audit_sha256']==result['audit']
        assert all(sha(EXPORT/x['path'])==x['sha256'] for x in m['files'])
        return dict(path=str(EXPORT),files=len(m['files'])+1)
    EXPORT.mkdir(parents=True,mode=0o700)
    script_sources()
    bundle_tables()
    md=(HERE/'REPORT_ZH.md').read_text(encoding='utf-8')
    for f in TABLES:
        if (EXPORT/'results'/f).is_dir():
            md=md.replace(f'results/{f}.csv',f'results/{f}')
    (EXPORT/'REPORT_ZH.md').write_text(md,encoding='utf-8')
    (EXPORT/'README.md').write_text(
        '# Strong-Anchor Joint Support Refinement\n\n'
        'The Chinese report gives paired outcomes for the four source-trained heads and '
        'the archived R158+PCA baseline. The ordinary Validation population contains '
        '18,620 directional records. All paired scalar outcomes are included in '
        'results/anonymous_pair_scores, with arbitrary order within direction.\n\n'
        'Reproduction inside the research workspace uses the pinned R173 source feature '
        'and roster seals, the R165 ordinary predictions, and the R97 frozen score cache. '
        'R174 first prepares the source anchors, trains all four heads per source, runs '
        'the source reference intervention and freezes the target method. It then seals '
        'target predictions before independent scoring and the target identity intervention. '
        'Private prerequisites are listed in the code and protocol. This export contains '
        'code, aggregate diagnostics and paired scalar metrics.\n',encoding='utf-8')
    attempts=HERE/'ATTEMPTS.md'
    if attempts.exists():
        shutil.copyfile(attempts,EXPORT/attempts.name)
    public=[]
    for f in sorted(EXPORT.rglob('*')):
        if f.is_dir():
            continue
        k=f.relative_to(EXPORT).as_posix()
        assert re.fullmatch(r'[-_./A-Za-z0-9]+',k) and '..' not in k
        assert f.suffix in ('.csv','.py','.json','.md')
        assert f.stat().st_size<=EXPORT_BYTES
        content=f.read_text(encoding='utf-8')
        assert not re.search(r'(gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-)?[A-Za-z0-9_-]{30,})',
                             content)
        public.append(dict(path=k,sha256=sha(f),bytes=f.stat().st_size))
    json_atomic(EXPORT/'MANIFEST.json',dict(study='R174',audit_sha256=result['audit'],
                parent=read(SPEC)['parent_commit'],ordinary_records=18620, files=public))
    return dict(path=str(EXPORT),files=len(public)+1)


def finish(remote):
    expected=read(EXPORT/'MANIFEST.json')
    assert remote['repository']==REPO and remote['branch']==BRANCH
    assert remote['parent']==read(SPEC)['parent_commit']
    assert remote['mainUnchanged'] and remote['priorResearchUnchanged'] and remote['nonForceFastForward']
    assert remote['files']==remote['verifiedFiles']==len(expected['files'])+1
    assert remote['bytes']==remote['verifiedBytes']==sum(x['bytes'] for x in expected['files']) + (
        EXPORT/'MANIFEST.json').stat().st_size
    assert re.fullmatch('[0-9a-f]{40}',remote['commit'])
    receipt=HERE/'publication/PUBLICATION_RECEIPT.json'
    data=dict(remote,FINAL_access=False,recorded_local=datetime.now().astimezone().isoformat(),
              manifest=sha(EXPORT/'MANIFEST.json'),audit=expected['audit_sha256'])
    if receipt.exists():
        assert read(receipt)['commit']==data['commit']
    else:
        json_atomic(receipt,data)
    result=read(RESULTS/'DECISION.json')
    d=result['full_deltas_pp']
    text=(f'\n\n## 2026-11-19 — R174 R158-anchored support refinement\n\n'
          f'R174 completed on 18,620 ordinary six-direction rows with four matched source-trained heads. '
          f'Full-minus-R158 Acc@25, Acc@50 and mIoU are {d["Acc25"]:+.4f}, {d["Acc50"]:+.4f} '
          f'and {d["mIoU"]:+.4f} pp. Decision {result["active_main"]}. '
          f'FINAL unread and all paired losses retained.\n\n'
          f'[Immutable report](https://github.com/{REPO}/blob/{data["commit"]}/{data["prefix"]}/REPORT_ZH.md) · '
          f'[Publication receipt](../{receipt.relative_to(ROOT).as_posix()})\n')
    archive=ROOT/'research/ARCHIVE.md'
    if data['commit'] not in archive.read_text(encoding='utf-8'):
        with archive.open('a',encoding='utf-8') as f:
            f.write(text)
    phase='completed_published_adoptable_candidate' if result['eligible_for_replacement'] else 'completed_published_not_adopted'
    state(phase,'Published and byte-verified the R174 code, results and paired negatives.',data)
    return data


if __name__=='__main__':
    import argparse

    p=argparse.ArgumentParser()
    p.add_argument('--receipt',type=Path)
    args=p.parse_args()
    print(finish(read(args.receipt)) if args.receipt else staging())
