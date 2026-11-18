"""Release only owned code, traceable scalar results and verification receipts."""
import argparse
from pathlib import Path
import shutil
import sys
import tempfile

from scope169 import CAL, HERE, OUT, PANEL, ROOT, atomic_json, read, roster, sha, status
from audit169 import audit
from score165 import table

sys.path.insert(0, str(ROOT/'research/round154'))
from package154 import scan_release, transport_csv

PREFIX = 'studies/3eed_reference_as_subject_requery_20261118'
PROGRAMS = ('scope169.py','cache169.py','panel169.py','requery169.py','score169.py','source169.py','audit169.py','release169.py')
RECEIPTS = ('CACHE_LOCK.json','CACHE_PREDICTION_SEAL.json','CACHE_EVALUATION_AUDIT.json',
            'PANEL_INPUT_LOCK.json','PANEL_INPUT_SEAL.json','PANEL_FORWARD_LOCK.json','QUERY_PREDICTION_SEAL.json',
            'PANEL_SCORE_LOCK.json','QUERY_IDENTITY_AUDIT.json','QUERY_JOINT_AUDIT.json',
            'SOURCE_REVIEW_LOCK.json','SOURCE_PANEL_SEAL.json','SOURCE_IMAGE_AUDIT.json','SCALAR_AUDIT.json')


def private_literals():
    tokens=set()
    rows=[r for batch in roster().values() for r in batch]
    rows.extend(r['row'] for r in read(CAL/'ROSTER.json'))
    for row in rows:
        for k in ('id','caption','scene','metapath','imagepath','pointpath'):
            v=row[k]
            if isinstance(v,str) and len(v)>=12:
                tokens.add(v.encode())
        frame=Path(row['metapath']).parent.name
        if len(frame)>=16:
            tokens.add(frame.encode())
    for r in read(PANEL/'INPUT_PRIVATE.json'):
        for variant in ('B','C','D'):
            if variant in r:
                v=r[variant]['text']
                if len(v)>=12:
                    tokens.add(v.encode())
    return tokens


def export():
    proof=audit()
    parent=HERE/'publication'
    parent.mkdir(exist_ok=True)
    dest=parent/'public_export_attempt001'
    assert not dest.exists()
    stage=Path(tempfile.mkdtemp(dir=parent))
    try:
        def copy(src,name):
            p=stage/name
            p.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,p)

        for k in PROGRAMS:
            copy(HERE/k,'own_code/research/round169/'+k)
        for k in (*RECEIPTS,'REPORT.md','REPORT_ZH.md','DECISION.json','experiment/PROTOCOL.json'):
            copy(HERE/k,k)
        tables={}
        for result in sorted(OUT.glob('*.csv')):
            rows=table(result)
            tables[result.name]=transport_csv(stage,'results/'+result.name,rows,tuple(rows[0]))
        atomic_json(stage/'PUBLIC_EXPORT_AUDIT.json',dict(scalar_audit=proof,tables=tables,
                raw_media_coordinates_phrase_text_tensors_instance_ids_weights_or_credentials=False))
        files=[dict(path=p.relative_to(stage).as_posix(),sha256=sha(p),bytes=p.stat().st_size)
               for p in sorted(stage.rglob('*')) if p.is_file()]
        atomic_json(stage/'MANIFEST.json',dict(prefix=PREFIX,files=files))
        scan_release(stage,private_literals())
        stage.rename(dest)
        status('local_audited_pending_publication',report='round169/REPORT_ZH.md',decision='round169/DECISION.json')
        print('R169 public files',len(files)+1,flush=True)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def register():
    receipt=read(HERE/'publication/PUBLICATION_RECEIPT.json')
    assert receipt['repository']=='Zonglin-He/A' and receipt['prefix']==PREFIX
    assert receipt['remoteByteVerified'] and receipt['immutableVerifiedBeforeRefUpdate']
    commit=receipt['commit']
    assert receipt['mainRefObserved']==commit
    remote='https://github.com/Zonglin-He/A/blob/'+commit+'/'+PREFIX+'/REPORT_ZH.md'
    verdict=read(HERE/'DECISION.json')
    path='round169/publication/PUBLICATION_RECEIPT.json'
    status('completed_published_verified',public_report=remote,public_commit=commit,
           report='round169/REPORT_ZH.md',decision='round169/DECISION.json',publication_receipt=path)
    state_path=ROOT/'research/STATE.json'
    s=read(state_path)
    s.update(latest_completed_round='R169', publication_state='R169_REMOTE_BYTES_VERIFIED',
             measurement_state='R169_FULL_CACHE_AND_REQUERY_PANEL_AUDITED',claim_state=verdict['finding'],
             public_study=PREFIX,publication_receipt=path,next_action=verdict['decision'])
    s['directions']['crossplatform'].update(latest_completed_round='R169',latest_completed_publication=commit,
                                           latest_public_report=remote,latest_public_commit=commit)
    atomic_json(state_path,s)
    archive=ROOT/'research/ARCHIVE.md'
    title=f'R169 completed and published {commit}'
    if title not in archive.read_text():
        with archive.open('a',encoding='utf-8') as h:
            h.write(f'\n## R169 — Frozen reference-as-subject requery\n\n{title}. 18,620 cache-paired direction rows and 54 '
                    f'independently adjudicated directional edges. {verdict["finding"]} '
                    '[Report](round169/REPORT_ZH.md) · [Verified receipt](round169/publication/PUBLICATION_RECEIPT.json)\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('export','register'))
    {'export':export,'register':register}[p.parse_args().phase]()
