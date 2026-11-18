"""Publish the frozen experiment and scrub all case evidence down to scalars."""
import argparse
from pathlib import Path
import shutil
import sys
import tempfile

from scope170 import HERE, INPUT, OUT, ROOT, atomic_json, read, sha, status
from audit170 import audit
from score165 import table

sys.path.insert(0,str(ROOT/'research/round154'))
from package154 import scan_release, transport_csv

PREFIX='studies/3eed_fixed_reference_requery_recording_confirmation_20261118'
CODE=('scope170.py','select170.py','predict170.py','review170.py','score170.py','audit170.py','release170.py')
RECEIPTS=('SELECTION_LOCK.json','INPUT_SEAL.json','PREDICTION_LOCK.json','PREDICTION_SEAL.json','VISUAL_REVIEW_LOCK.json',
          'REVIEW_PANEL_SEAL.json','VISUAL_TRUTH_SEAL.json','SCORING_LOCK.json','EVALUATION_AUDIT.json','SCALAR_AUDIT.json')


def private_literals():
    terms=set()
    for x in read(INPUT):
        row=x['row']
        for field in ('id','caption','scene','imagepath','pointpath','metapath'):
            v=row[field]
            if isinstance(v,str) and len(v)>=12:
                terms.add(v.encode())
        name=Path(row['metapath']).parent.name
        if len(name)>=16:
            terms.add(name.encode())
        for arm in ('B','C','D'):
            if arm in x['rewrite']:
                v=x['rewrite'][arm]['text']
                if len(v)>=12:
                    terms.add(v.encode())
    return terms


def export():
    proof=audit()
    base=HERE/'publication'
    base.mkdir(exist_ok=True)
    output=base/'public_export_attempt001'
    assert not output.exists()
    stage=Path(tempfile.mkdtemp(dir=base))
    try:
        def copy(file,rel):
            d=stage/rel
            d.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(file,d)
        for source in CODE:
            copy(HERE/source,'own_code/research/round170/'+source)
        for name in (*RECEIPTS,'REPORT.md','REPORT_ZH.md','DECISION.json','experiment/PROTOCOL.json'):
            copy(HERE/name,name)
        tables={}
        for p in sorted(OUT.glob('*.csv')):
            rows=table(p)
            tables[p.name]=transport_csv(stage,'results/'+p.name,rows,tuple(rows[0]))
        atomic_json(stage/'PUBLIC_EXPORT_AUDIT.json',dict(summary=proof,tables=tables,
                raw_rgb_lidar_original_text_coordinates_features_labels_ids_parameters_or_credentials=False))
        files=[dict(path=p.relative_to(stage).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
               for p in sorted(stage.rglob('*')) if p.is_file()]
        atomic_json(stage/'MANIFEST.json',dict(prefix=PREFIX,files=files))
        scan_release(stage,private_literals())
        stage.rename(output)
        status('local_audited_pending_publication',physical_frames=proof['physical_frames'],direction_records=proof['direction_rows'],
               report='round170/REPORT_ZH.md',decision='round170/DECISION.json')
        print('R170 public files',len(files)+1,flush=True)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def register():
    x=read(HERE/'publication/PUBLICATION_RECEIPT.json')
    d=read(HERE/'DECISION.json')
    assert x['repository']=='Zonglin-He/A' and x['prefix']==PREFIX
    assert x['commit']==x['mainRefObserved'] and x['immutableVerifiedBeforeRefUpdate'] and x['remoteByteVerified']
    commit=x['commit']
    remote='https://github.com/Zonglin-He/A/blob/'+commit+'/'+PREFIX+'/REPORT_ZH.md'
    path='round170/publication/PUBLICATION_RECEIPT.json'
    status('completed_published_verified',physical_frames=48,direction_records=96,public_report=remote,public_commit=commit,
           report='round170/REPORT_ZH.md',decision='round170/DECISION.json',publication_receipt=path)
    state=ROOT/'research/STATE.json'
    s=read(state)
    s.update(latest_completed_round='R170',publication_state='R170_REMOTE_BYTES_VERIFIED',
             measurement_state='R170_RECORDING_SEPARATED_REQUERY_AUDITED',claim_state=d['finding'],
             public_study=PREFIX,publication_receipt=path,next_action=d['decision'])
    s['directions']['crossplatform'].update(latest_completed_round='R170',latest_completed_publication=commit,
                                           latest_public_commit=commit,latest_public_report=remote)
    atomic_json(state,s)
    archive=ROOT/'research/ARCHIVE.md'
    key='R170 completed and published '+commit
    if key not in archive.read_text():
        with archive.open('a',encoding='utf-8') as f:
            f.write('\n## R170 — Recording-separated confirmation of the locked R169 requery\n\n'+key+
                    '. 48 unique physical frames and 96 matched source directions. '+d['finding']+
                    ' [Report](round170/REPORT_ZH.md) · [Verified receipt](round170/publication/PUBLICATION_RECEIPT.json)\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('export','register'))
    {'export':export,'register':register}[p.parse_args().phase]()
