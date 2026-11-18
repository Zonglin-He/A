"""Export the audited LCIB experiment and register a verified immutable publication."""
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import tempfile

from manifest162 import HERE, ROOT, WORK, atomic_json, read, sha, target_rows
from score162 import table

sys.path.insert(0,str(ROOT/'research/round154'))
from package154 import scan_release, transport_csv

SLUG='studies/3eed_language_conditioned_instance_box_inference_20261118'
CODE=(
    'evidence162.py','geometry162.py','manifest162.py','solver162.py','support162.py','run162.py',
    'score162.py','audit162.py','counterfactual162.py','mechanisms162.py','trajectory162.py','report162.py',
    'verify_cli162.py','visual162.py','release162.py',
)


def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,dst)


def validated():
    n={}
    for scope,expected in (('DEV',1798),('EVAL',9014)):
        folder=HERE/'results'/scope
        audit=read(folder/'AUDIT.json')
        assert audit['status']=='verified' and audit['rows']==expected
        assert audit['result_rows_sha256']==sha(folder/'rows.csv')
        assert audit['score_seal_sha256']==sha(folder/'SCORE.json')
        assert read(folder/'SCORE.json')['prediction_seal_sha256']==sha(HERE/(scope+'_SEAL.json'))
        records=table(folder/'rows.csv')
        assert len({(r['cohort'],r['ordinal']) for r in records})==len(records)==expected
        n[scope]=len(records)
    assert read(HERE/'RAW_RUNTIME_AUDIT.json')['before_after_weights_identical']
    assert read(HERE/'DECISION.json')['formal_ranges_complete']
    return n


def private_literals():
    values=set()
    def add(row):
        for key in ('id','caption','scene','metapath','imagepath','pointpath','point_sha','image_sha'):
            data=row.get(key)
            if isinstance(data,str) and len(data)>=8:
                values.add(data.encode())
        for key in ('metapath','imagepath','pointpath'):
            if key in row:
                name=Path(row[key]).parent.name
                if len(name)>=13:
                    values.add(name.encode())
    for e in target_rows():
        add(e)
        add(e['r'])
    for r in read(WORK/'SOURCE_VAL_PRIVATE.json'):
        add(r)
    return values


def scalar_rows(p):
    rows=table(p)
    if p.name=='rows.csv':
        for r in rows:
            origin=int(r.pop('selected_origin'))
            r['scene_initialized']=int(origin==-1)
    return rows


def export():
    n=validated()
    location=HERE/'publication'
    location.mkdir(exist_ok=True)
    dest=location/'public_export_attempt001'
    if dest.exists():
        raise FileExistsError(dest)
    stage=Path(tempfile.mkdtemp(dir=location))
    try:
        base=stage/'own_code/research/round162'
        for name in CODE:
            copy(HERE/name,base/name)
        copy(ROOT/'crossplatform/ground_lcib.py',stage/'own_code/crossplatform/ground_lcib.py')
        specs=('PROTOCOL.json','SOURCE_PARAMETERS.json','TARGET_LOCK.json')
        for name in specs:
            copy(HERE/'experiment'/name,stage/'experiment'/name)
        records=('CALIBRATION_INPUTS','CALIBRATION_SEAL','DEV_SEAL','EVAL_SEAL','PANEL_LOCK','EVIDENCE_SMOKE',
                 'SOLVER_NUMERICS','COUNTERFACTUAL_DESIGN','COUNTERFACTUAL_SEAL','RAW_RUNTIME_AUDIT',
                 'DECISION','ATTEMPTS')
        for name in records:
            copy(HERE/(name+'.json'),stage/(name+'.json'))
        copy(HERE/'VISUAL_REVIEW.md',stage/'VISUAL_REVIEW.md')
        report=(HERE/'REPORT_ZH.md').read_text()
        transport={}
        for scope in ('DEV','EVAL'):
            for p in sorted((HERE/'results'/scope).iterdir()):
                logical=f'results/{scope}/{p.name}'
                if p.suffix=='.json':
                    copy(p,stage/logical)
                elif p.suffix=='.csv':
                    rows=scalar_rows(p)
                    transport[logical]=transport_csv(stage,logical,rows,list(rows[0]))
                    if len(transport[logical]['parts'])>1:
                        report=report.replace('('+logical+')','('+logical.removesuffix('.csv')+')')
        (stage/'REPORT_ZH.md').write_text(report)
        (stage/'REPORT.md').write_text('# Language-conditioned instance–box inference\n\n'
            '[Paired cross-platform results, mechanism controls, and the experimental decision](REPORT_ZH.md).\n')
        atomic_json(stage/'TRANSPORT_SPLITS.json',transport)
        atomic_json(stage/'PUBLIC_EXPORT_AUDIT.json',dict(scored_ordinary=5942, scored_cross_recording=3072,
                    scored_development=n['DEV'], full_tail_records=True,
                    types='First-party code, aggregate metrics, de-identified scalar records, and integrity receipts',
                    private_media_raw_labels_or_weights_included=False, FINAL_access=False))
        inventory=[dict(path=p.relative_to(stage).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
                   for p in sorted(stage.rglob('*')) if p.is_file()]
        atomic_json(stage/'MANIFEST.json',dict(prefix=SLUG,files=inventory))
        scan_release(stage,private_literals())
        stage.rename(dest)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print('R162 public export validated', len(inventory)+1, sum(f['bytes'] for f in inventory), flush=True)


def register():
    audited=validated()
    receipt=read(HERE/'publication/PUBLICATION_RECEIPT.json')
    commit=receipt['commit']
    assert receipt['repository']=='Zonglin-He/A' and receipt['prefix']==SLUG
    assert receipt['remoteByteVerified'] and receipt['immutableVerifiedBeforeRefUpdate']
    assert receipt['mainRefObserved']==commit
    decision=read(HERE/'DECISION.json')
    now=datetime.now().astimezone()
    report=f'https://github.com/Zonglin-He/A/blob/{commit}/{SLUG}/REPORT_ZH.md'
    rec=dict(round='R162',state='completed_published_verified',report='round162/REPORT_ZH.md',
             publication_receipt='round162/publication/PUBLICATION_RECEIPT.json',
             public_commit=commit,public_report=report, rows=audited, working_default='R159 frozen R158 + PCA',
             adopted=decision['adopted'],target_parameter_updates=0, FINAL_access=False,updated=now.isoformat())
    atomic_json(HERE/'STATE.json',rec)
    archive=read(ROOT/'research/STATE.json')
    archive['R162']=rec
    if archive['current_round']=='R162':
        archive.update(latest_completed_round='R162', run_state='R162_COMPLETED_PUBLISHED_MAIN_VERIFIED',
                       current_stage='R162_COMPLETE',publication_state='R162_REMOTE_BYTES_VERIFIED',
                       measurement_state='R162_ALL_DIRECTED_OBSERVATIONS_AND_MATCHED_CONTROLS_AUDITED',
                       claim_state='LCIB supported' if decision['adopted'] else 'LCIB prototype not adopted',
                       next_action='none_queued',updated_local=now.isoformat(), public_study=SLUG,
                       publication_receipt=rec['publication_receipt'])
        archive['directions']['crossplatform'].update(active_round=None,execution_open=False,
            latest_completed_report=rec['report'],latest_completed_publication=commit,
            latest_public_report=report,latest_public_commit=commit)
    atomic_json(ROOT/'research/STATE.json',archive)
    ledger=ROOT/'research/ARCHIVE.md'
    marker='R162 language-conditioned instance–box inference. '+commit
    if marker not in ledger.read_text():
        with ledger.open('a') as f:
            f.write('\n'+marker+'. Sealed development 1798, ordinary 5942 and separate recording 3072. '
                    'Support, continuous boxes, same-budget serial and instance controls. '
                    '[Report](round162/REPORT_ZH.md) · [Decision](round162/DECISION.json) · '
                    '[Receipt](round162/publication/PUBLICATION_RECEIPT.json)\n')


if __name__=='__main__':
    if sys.argv[1:]==['register']:
        register()
    else:
        export()
