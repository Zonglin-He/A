"""Archive the selected R174 inference profiles and byte-verified public results."""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import re
import shutil
from statistics import median

from scope178 import HERE,N,PRIVATE,RESULTS,ROOT,SPEC,json_atomic,read,sha,state
from release177 import CAP,SECRET,shard_csv
from model178 import CONFIGS

REPO = 'Zonglin-He/A'
BRANCH = 'research/3eed-r178-r174-combined-simplification-20261119'
EXPORT = HERE/'publication/public_export_attempt001'
CODE = ('scope178','model178','predict178','score178','reference178','inference178','release178')


def choose_profiles(preferred,relationship):
    metrics = ('Acc25','Acc50','mIoU')
    rows = {(x['variant'],x['subset']):x for x in csv.DictReader((RESULTS/'endpoints.csv').open())
            if x['direction']=='six_equal'}
    timing = list(csv.DictReader((RESULTS/'decoder_timing.csv').open()))
    profiles = {}
    for label,name in (('preferred',preferred),('relation_priority',relationship)):
        cfg = CONFIGS[name]
        assert cfg.rounds in (2,3)
        ratios = [float(r['relative_to_full']) for r in timing if r['variant']==name]
        assert len(ratios)==6
        profiles[label] = dict(variant=name,rounds=cfg.rounds,reference_slots=cfg.reference_slots,
                    final_pair_score=cfg.final_pair_score,
                    entrypoint=(HERE/'inference178.py').relative_to(ROOT).as_posix()+':load_profile',
                    overall={m:float(rows[name,'all'][m]) for m in metrics},
                    direct_left_right={m:float(rows[name,'direct_left_right'][m]) for m in metrics},
                    decoder_time_reduction_pct=100*(1-median(ratios)),
                    evidence_scope='ordinary Validation development comparison')
    decision = dict(n=N,main_method='R174',profiles=profiles,
                    score_seal=sha(PRIVATE/'SCORING_SEAL.json'),
                    audit=sha(RESULTS/'independent_audit.json'),FINAL_access=False)
    json_atomic(RESULTS/'DECISION.json',decision)
    return decision


def stage(preferred,relationship):
    score = read(PRIVATE/'SCORING_SEAL.json')
    assert score['n']==N and not score['FINAL_access']
    assert all(sha(RESULTS/p)==h for p,h in score['results'].items())
    decision = choose_profiles(preferred,relationship)
    path = EXPORT/'MANIFEST.json'
    if path.exists():
        proof=read(path)
        assert all(sha(EXPORT/x['path'])==x['sha256'] for x in proof['files'])
        assert sha(RESULTS/'DECISION.json')==sha(EXPORT/'results/DECISION.json')
        return proof
    EXPORT.mkdir(parents=True,mode=0o700)
    (EXPORT/'own_code').mkdir()
    (EXPORT/'results').mkdir()
    for name in CODE:
        shutil.copyfile(HERE/f'{name}.py',EXPORT/'own_code'/f'{name}.py')
    shutil.copyfile(SPEC,EXPORT/'PROTOCOL.json')
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
        '# R178 R174 combined simplification\n\n'
        'Six inherited inference combinations and the complete R174 use the same three '
        'frozen checkpoints and all 18,620 ordinary Validation records. '
        'Read the run definitions in PROTOCOL.json and the paired results in REPORT_ZH.md.\n\n'
        'Within the registered workspace run predict178.py followed by score178.py. '
        'Load a registered recipe with inference178.load_profile(source, device, profile). '
        'The upstream R174 and R177 predictions, weights, features and private review evidence are required. '
        'Public paired deltas are anonymous and include positive, negative and tied outcomes.\n',
        encoding='utf-8')
    json_atomic(EXPORT/'EVIDENCE_DIGESTS.json',dict(execution=sha(PRIVATE/'EXECUTION_LOCK.json'),
                     scoring=decision['score_seal'],predictions=score['predictions']))
    files = []
    for p in sorted(EXPORT.rglob('*')):
        if p.is_file():
            content = p.read_bytes()
            assert len(content)<=CAP and p.suffix in ('.json','.csv','.md','.py')
            assert not SECRET.search(content.decode('utf-8'))
            files.append(dict(path=p.relative_to(EXPORT).as_posix(),bytes=len(content),sha256=sha(p)))
    proof = dict(round='R178',n=N,parent=read(SPEC)['parent_commit'],audit=decision['audit'],files=files)
    json_atomic(path,proof)
    return proof


def finish(receipt):
    manifest = read(EXPORT/'MANIFEST.json')
    outpath = HERE/'publication/PUBLICATION_RECEIPT.json'
    assert receipt['repository']==REPO and receipt['branch']==BRANCH
    assert receipt['parent']==manifest['parent'] and re.fullmatch('[a-f0-9]{40}',receipt['commit'])
    assert receipt['mainUnchanged'] and receipt['priorResearchUnchanged'] and receipt['nonForceFastForward']
    assert receipt['files']==receipt['verifiedFiles']==len(manifest['files'])+1
    assert receipt['bytes']==receipt['verifiedBytes']==sum(f['bytes'] for f in manifest['files'])+(
        EXPORT/'MANIFEST.json').stat().st_size
    value = dict(receipt,manifest_sha256=sha(EXPORT/'MANIFEST.json'),audit_sha256=manifest['audit'],
                 recorded_local=datetime.now().astimezone().isoformat(),FINAL_access=False)
    if outpath.exists():
        assert read(outpath)['commit']==value['commit']
    else:
        json_atomic(outpath,value)
    decision = read(RESULTS/'DECISION.json')
    assert sha(RESULTS/'DECISION.json')==sha(EXPORT/'results/DECISION.json')
    archive = ROOT/'research/ARCHIVE.md'
    if value['commit'] not in archive.read_text(encoding='utf-8'):
        with archive.open('a',encoding='utf-8') as f:
            f.write(
                '\n\n## 2026-11-19 — R178 combined R174 inference simplifications\n\n'
                'Six combined configurations, 18,620 paired direction records. '
                f"Preferred {decision['profiles']['preferred']['variant']}, "
                f"relation-priority profile {decision['profiles']['relation_priority']['variant']}. "
                'All paired negatives retained. Training, target updates and FINAL access zero.\n\n'
                f'[Immutable report](https://github.com/{REPO}/blob/{value["commit"]}/{value["prefix"]}/REPORT_ZH.md) · '
                f'[Publication receipt](../{outpath.relative_to(ROOT).as_posix()})\n')
    state('completed_published_verified','Audited combined R174 inference profiles registered.',
          receipt=value,decision=decision)
    return value


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt',type=Path)
    parser.add_argument('--preferred')
    parser.add_argument('--relation-priority')
    args = parser.parse_args()
    if args.receipt:
        print(finish(read(args.receipt)))
    else:
        assert args.preferred and args.relation_priority
        proof = stage(args.preferred,args.relation_priority)
        state('publication_pending','Combined inference results audited and staged.',
              decision=read(RESULTS/'DECISION.json'))
        print(dict(path=str(EXPORT),files=len(proof['files'])+1))
