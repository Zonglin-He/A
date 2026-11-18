"""Check all R163 scalars, export the allowed evidence, and register a verified release."""
from collections import Counter
from datetime import datetime
from math import fsum
from pathlib import Path
import shutil
import sys
import tempfile

from scope163 import HERE, PLATFORMS, PROTOCOL, ROOT, WORK, atomic_json, read, sha, source_scope, target_rows
from score162 import table

sys.path.insert(0, str(ROOT/'research/round154'))
from package154 import scan_release

SLUG = 'studies/3eed_lcib_role_and_first_cycle_diagnostics_20261118'
CODE = ('scope163.py', 'cycle163.py', 'roles163.py', 'score163.py', 'role_scores163.py', 'release163.py')
RESULTS = ('source_rows.csv', 'source_metrics.csv', 'paired_contrasts.csv', 'signed_axis_summary.csv',
           'role_edges.csv', 'role_stage_summary.csv', 'ROLE_REVIEW_SUMMARY.json', 'SOURCE_STAGE_SUMMARY.json')


def validate():
    results = HERE/'results'
    audit = read(HERE/'SOURCE_AUDIT.json')
    seal = read(HERE/'SOURCE_SEAL.json')
    assert audit['rows']==seal['n']==192 and audit['sensor_frame_endpoint_checks']==2304
    assert audit['source_seal_sha256']==sha(HERE/'SOURCE_SEAL.json') and audit['max_abs_iou_error']<1e-7
    assert audit['result_rows_sha256']==sha(results/'source_rows.csv') and seal['protocol_sha256']==sha(PROTOCOL)
    assert not (audit['FINAL_access'] or seal['FINAL_access'] or seal['ground_truth_in_arms'])
    rows = table(results/'source_rows.csv')
    assert [int(r['ordinal']) for r in rows]==list(range(1,193))
    assert Counter(r['source'] for r in rows)==Counter({source:64 for source in PLATFORMS})
    assert all(not any(w in field for w in ('box','xyz','ref_index','caption','path')) for field in rows[0])

    metrics = table(results/'source_metrics.csv')
    pairs = table(results/'paired_contrasts.csv')
    assert len(metrics)==48 and len(pairs)==60
    for summary in metrics:
        x=[float(r[summary['arm']]) for r in rows if summary['source'] in ('all',r['source'])]
        assert len(x)==int(summary['n']) and all(0<=v<=1 for v in x)
        expected=(100*fsum(x)/len(x), *(100*sum(v>t for v in x)/len(x) for t in (.25,.50)))
        assert all(abs(float(summary[name])-v)<1e-10 for name,v in zip(('mIoU','Acc25','Acc50'),expected))
    for comparison in pairs:
        before,after=zip(*[(float(r[comparison['before']]),float(r[comparison['after']])) for r in rows
                           if comparison['source'] in ('all',r['source'])])
        for name,t in (('mIoU',None),('Acc25',.25),('Acc50',.50)):
            differences=[b-a if t is None else int(b>t)-int(a>t) for a,b in zip(before,after)]
            assert abs(100*fsum(differences)/len(differences)-float(comparison[name+'_pp']))<1e-10
            assert sum(d>1e-12 for d in differences)==int(comparison[name+'_gain'])
            assert sum(d< -1e-12 for d in differences)==int(comparison[name+'_loss'])
    role = table(results/'role_edges.csv')
    summary = read(results/'ROLE_REVIEW_SUMMARY.json')
    assert [int(x['case']) for x in role]==list(range(1,34))
    assert Counter(x['review'] for x in role)==summary['counts']
    assert all(r['r91_init_assignment']=='same_box' for r in role)
    assert sum(x['lcib_final_visible_identity']=='both_endpoints_match' for x in role)==1
    assert read(HERE/'ROLE_SCOPE.json')['rows']==33 and not summary['FINAL_access']
    assert not read(HERE/'DECISION.json')['selected_new_method']


def private_literals():
    values = set()
    selected = read(WORK/'ROLE_REVIEW_PRIVATE.json')
    target = target_rows()
    inspected = [target[x['panel_index']]['r'] for x in selected]
    for r in (*source_scope(), *inspected):
        for key in ('id','caption','scene','metapath','imagepath','pointpath','recording'):
            value=r.get(key)
            if isinstance(value,str) and len(value)>=8:
                values.add(value.encode())
    return values


def export():
    validate()
    parent = HERE/'publication'
    parent.mkdir(exist_ok=True)
    dest = parent/'public_export_attempt001'
    if dest.exists():
        raise FileExistsError(dest)
    stage = Path(tempfile.mkdtemp(dir=parent))
    try:
        def copy(path, relative=None):
            name = stage/(relative or path.name)
            name.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,name)

        for f in CODE:
            copy(HERE/f,'own_code/research/round163/'+f)
        copy(PROTOCOL,'experiment/'+PROTOCOL.name)
        for f in RESULTS:
            copy(HERE/'results'/f,'results/'+f)
        for f in ('REPORT_ZH.md','DECISION.json','ROLE_SCOPE.json','SOURCE_SEAL.json','SOURCE_AUDIT.json'):
            copy(HERE/f)
        (stage/'REPORT.md').write_text('# LCIB role binding and first-cycle geometry\n\n'
            '[Full paired results, role audit and decision](REPORT_ZH.md).\n')
        atomic_json(stage/'PUBLIC_EXPORT_AUDIT.json',dict(source_rows=192,role_edges=33,retains_unknown_and_negative_cases=True,
            raw_images_captions_coordinates_checkpoint_or_cache_included=False, FINAL_access=False,
            types='First-party code, anonymous task and role scalars, aggregate fit displacement, integrity receipts'))
        files=[dict(path=p.relative_to(stage).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
               for p in sorted(stage.rglob('*')) if p.is_file()]
        atomic_json(stage/'MANIFEST.json',dict(prefix=SLUG,files=files))
        scan_release(stage,private_literals())
        stage.rename(dest)
        print('R163 public export validated',len(files)+1,flush=True)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def register():
    validate()
    receipt = read(HERE/'publication/PUBLICATION_RECEIPT.json')
    commit = receipt['commit']
    assert receipt['repository']=='Zonglin-He/A' and receipt['prefix']==SLUG
    assert receipt['remoteByteVerified'] and receipt['immutableVerifiedBeforeRefUpdate']
    assert receipt['mainRefObserved']==commit
    stamp = datetime.now().astimezone().isoformat()
    report = f'https://github.com/Zonglin-He/A/blob/{commit}/{SLUG}/REPORT_ZH.md'
    rec = dict(round='R163',state='completed_published_verified', report='round163/REPORT_ZH.md',
               protocol='round163/experiment/PROTOCOL.json',decision='round163/DECISION.json',
               publication_receipt='round163/publication/PUBLICATION_RECEIPT.json',public_commit=commit,
               public_report=report,role_edges=33,source_rows=192,working_default='R159 frozen R158 + PCA',
               baseline_queue='paused',new_training=0,target_parameter_updates=0,FINAL_access=False,updated=stamp)
    archive = ROOT/'research/STATE.json'
    overall = read(archive)
    assert overall['R161']['state']=='paused_by_user'
    overall['R163'] = rec
    if overall['current_round']=='R163':
        overall.update(latest_completed_round='R163',run_state='R163_COMPLETED_PUBLISHED_MAIN_VERIFIED',
            current_stage='R163_COMPLETE',publication_state='R163_REMOTE_BYTES_VERIFIED',
            measurement_state='R163_192_SOURCE_OUTPUTS_AND_33_ROLE_EDGES_AUDITED',
            claim_state='R162 fitting damage localized; R159 remains default',
            next_action='none_queued',updated_local=stamp,public_study=SLUG,publication_receipt=rec['publication_receipt'])
        overall['directions']['crossplatform'].update(active_round=None,execution_open=False,
            latest_completed_report=rec['report'],latest_completed_publication=commit,
            latest_public_report=report,latest_public_commit=commit)
    atomic_json(archive,overall)
    atomic_json(HERE/'STATE.json',rec)
    ledger = ROOT/'research/ARCHIVE.md'
    marker = 'R163 completed and published '+commit
    if marker not in ledger.read_text():
        with ledger.open('a') as f:
            f.write('\n'+marker+'. Source-own 192; fixed-winner first fit −7.0539 mIoU pp and −20.3125 Acc@50 pp. '
                    'Reselection Acc@50 unchanged. 17 visually verified distinct pairs among 33 selected shared '
                    'role edges, with 14 independently labelled pairs. R159 remains the default. No training, '
                    'target updates, baseline resume or FINAL. '
                    '[Report](round163/REPORT_ZH.md) · [Decision](round163/DECISION.json) · '
                    '[Verified receipt](round163/publication/PUBLICATION_RECEIPT.json)\n')


if __name__=='__main__':
    if sys.argv[1:]==['register']:
        register()
    else:
        export()
