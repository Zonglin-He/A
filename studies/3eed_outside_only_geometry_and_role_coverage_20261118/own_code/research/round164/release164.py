"""Validate paired R164 scalars and package the public first-party artifacts."""
from collections import Counter
from datetime import datetime
from math import fsum
from pathlib import Path
import shutil
import sys
import tempfile

from scope164 import HERE, PLATFORMS, PROTOCOL, ROOT, WORK, atomic_json, read, sha, source_scope, target_rows
from score162 import table

sys.path.insert(0, str(ROOT/'research/round154'))
from package154 import scan_release

SLUG = 'studies/3eed_outside_only_geometry_and_role_coverage_20261118'
CODE = ('scope164.py','fit164.py','score164.py','face164.py','roles164.py','release164.py')
RESULTS = ('geometry_rows.csv','geometry_metrics.csv','factor_contrasts.csv','signed_displacement.csv',
           'heldout_rows.csv','GEOMETRY_SUMMARY.json','face_rows.csv','face_metrics.csv','face_contrasts.csv',
           'role_review.csv','role_panel_availability.csv','ROLE_SUMMARY.json')


def scoped_rows(rows, group):
    if group=='relation_subset':
        return [r for r in rows if r['relation_configured']=='True']
    return [r for r in rows if group in (r['source'],'all')]


def assert_metrics(task, metrics, pair, prefix=None):
    for totals in metrics:
        group = scoped_rows(task,totals['source'])
        p = 'split_' if prefix and totals['mode']=='fit_holdout' else ''
        vals = [float(r[p+totals['arm']]) for r in group]
        assert len(vals)==int(totals['n']) and all(0<=x<=1 for x in vals)
        expected = [100*fsum(vals)/len(vals),*(100*sum(x>t for x in vals)/len(vals) for t in (.25,.50))]
        assert all(abs(float(totals[k])-v)<1e-10 for k,v in zip(('mIoU','Acc25','Acc50'),expected))
    for item in pair:
        group = scoped_rows(task,item['source'])
        p = 'split_' if prefix and item['mode']=='fit_holdout' else ''
        a,b = item['control'],item['arm']
        values = [(float(r[p+a]),float(r[p+b])) for r in group]
        for name,limit in (('mIoU',None),('Acc25',.25),('Acc50',.5)):
            diff = [v-u if limit is None else int(v>limit)-int(u>limit) for u,v in values]
            assert abs(100*fsum(diff)/len(diff)-float(item[name+'_pp']))<1e-10
            assert sum(x>1e-12 for x in diff)==int(item[name+'_gain'])
            assert sum(x< -1e-12 for x in diff)==int(item[name+'_loss'])


def verify_results():
    out = HERE/'results'
    for name,count,checks,scalars in (('GEOMETRY',192,2688,'geometry_rows.csv'),('FACE',192,768,'face_rows.csv')):
        audit = read(HERE/(name+'_AUDIT.json'))
        seal = read(HERE/(name+'_SEAL.json'))
        assert audit['rows']==seal['n']==count and audit['endpoint_checks']==checks
        assert audit[name.lower()+'_seal_sha256']==sha(HERE/(name+'_SEAL.json'))
        assert seal['protocol_sha256']==sha(PROTOCOL) and not (audit['FINAL_access'] or seal['FINAL_access'])
        assert audit['scalars_sha256']==sha(out/scalars)
    metrics,face_metrics = [table(out/f) for f in ('geometry_metrics.csv','face_metrics.csv')]
    pair,face_pair = [table(out/f) for f in ('factor_contrasts.csv','face_contrasts.csv')]
    task,face = [table(out/f) for f in ('geometry_rows.csv','face_rows.csv')]
    for rows in (task,face):
        assert [int(r['row']) for r in rows]==list(range(1,193))
        assert Counter(r['source'] for r in rows)==Counter({source:64 for source in PLATFORMS})
        assert all(not any(key in f for key in ('xyz','caption','raw_index','recording_id','box_coordinates')) for f in rows[0])
    assert_metrics(task,metrics,pair,prefix=True)
    assert_metrics(face,face_metrics,face_pair)
    held = table(out/'heldout_rows.csv')
    assert len(held)==5*192 and Counter(r['row'] for r in held)==Counter({str(i):5 for i in range(1,193)})
    for a,b in zip(task,face):
        assert all(a[key]==b[key] for key in ('row','source','r158','initial','split_outside_s0'))
        if b['accepted']=='False':
            assert b['initial']==b['one_face']
    roles = table(out/'role_review.csv')
    panel = table(out/'role_panel_availability.csv')
    truth = [r for r in roles if r['review']=='verified_boxes']
    summary = read(out/'ROLE_SUMMARY.json')
    assert [int(r['case']) for r in roles]==list(range(1,34)) and len(truth)==summary['complete_labels']==14
    assert [int(r['edge']) for r in panel]==list(range(1,133)) and len(panel)==summary['all_panel_edges']==132
    assert summary['prediction_seal_sha256']==sha(HERE/'ROLES_SEAL.json') and not summary['FINAL_access']
    for arm in ('independent','distinct','independent_seed','distinct_seed','lcib_initial'):
        assert sum(r[arm+'_dual_correct']=='True' for r in truth)==summary['dual_correct'][arm]
    decision = read(HERE/'DECISION.json')
    assert not any(decision[k] for k in ('adopted_new_geometry','adopted_role_exclusion','surface_7d_adopted','FINAL_access'))


def private_literals():
    registry = target_rows()
    targets = [registry[int(Path(name).stem)]['r'] for name in read(WORK/'ROLES_FILES_PRIVATE.json')]
    words = set()
    for record in (*source_scope(),*targets):
        for key in ('id','caption','pointpath','imagepath','metapath','scene','recording'):
            val = record.get(key)
            if isinstance(val,str) and len(val)>=8:
                words.add(val.encode())
    return words


def export():
    verify_results()
    parent = HERE/'publication'
    parent.mkdir(exist_ok=True)
    dest = parent/'public_export_attempt001'
    assert not dest.exists()
    stage = Path(tempfile.mkdtemp(dir=parent))
    try:
        def copy(path, relative=None):
            dst = stage/(relative or path.name)
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,dst)

        for f in CODE:
            copy(HERE/f,'own_code/research/round164/'+f)
        copy(PROTOCOL,'experiment/'+PROTOCOL.name)
        for f in RESULTS:
            copy(HERE/'results'/f,'results/'+f)
        for f in ('REPORT_ZH.md','DECISION.json','GEOMETRY_SEAL.json','GEOMETRY_AUDIT.json',
                  'FACE_SEAL.json','FACE_AUDIT.json','ROLES_SEAL.json'):
            copy(HERE/f)
        (stage/'REPORT.md').write_text('# Outside-only geometry and role candidate coverage\n\n'
            '[Full paired findings and decision](REPORT_ZH.md).\n\n'
            'The first-party entry points reuse the parent R162 and R163 method and synthetic check code. '
            'Model replay additionally requires the licensed local data and weights, calibration and the sealed parent inputs. '
            'The tables include every diagnostic row and can be aggregated directly. '
            'Per-row IoU is in [0, 1], summary accuracy in percent, and contrast values in percentage points.\n')
        atomic_json(stage/'PUBLIC_EXPORT_AUDIT.json',dict(n_geometry=192,n_face=192,n_role_edges=132,
            role_truth_verified=14, full_positives_negatives_and_unknown_retained=True,
            release_contents='First-party source, protocol, integrity receipts, and anonymous metric and eligibility scalars',
            raw_media_labels_raw_coordinates_features_weights_or_secrets_included=False, FINAL_access=False))
        files=[dict(path=p.relative_to(stage).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
               for p in sorted(stage.rglob('*')) if p.is_file()]
        atomic_json(stage/'MANIFEST.json',dict(prefix=SLUG,files=files))
        scan_release(stage,private_literals())
        stage.rename(dest)
        print('R164 public export validated',len(files)+1,flush=True)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def register():
    verify_results()
    receipt = read(HERE/'publication/PUBLICATION_RECEIPT.json')
    commit = receipt['commit']
    assert receipt['repository']=='Zonglin-He/A' and receipt['prefix']==SLUG
    assert receipt['immutableVerifiedBeforeRefUpdate'] and receipt['remoteByteVerified']
    assert receipt['mainRefObserved']==commit
    stamp = datetime.now().astimezone().isoformat()
    report = f'https://github.com/Zonglin-He/A/blob/{commit}/{SLUG}/REPORT_ZH.md'
    status = dict(round='R164',state='completed_published_verified',report='round164/REPORT_ZH.md',
                  protocol='round164/experiment/PROTOCOL.json',decision='round164/DECISION.json',
                  publication_receipt='round164/publication/PUBLICATION_RECEIPT.json',
                  public_commit=commit,public_report=report,geometry_rows=192,eligible_role_edges=132,
                  verified_dual_identity_edges=14,working_default='R159 frozen R158 + PCA',
                  baseline_queue='paused',new_training=0,target_parameter_updates=0,FINAL_access=False,updated=stamp)
    path = ROOT/'research/STATE.json'
    data = read(path)
    assert data['R161']['state']=='paused_by_user'
    data['R164'] = status
    if data['current_round']=='R164':
        data.update(latest_completed_round='R164',current_stage='R164_COMPLETE',run_state='R164_COMPLETED_PUBLISHED_MAIN_VERIFIED',
                    publication_state='R164_REMOTE_BYTES_VERIFIED',measurement_state='R164_FIXED_WINNER_GEOMETRY_AND_ROLE_RANKS_AUDITED',
                    claim_state='Outside-only reduces surface fit damage; no new geometry or role rule adopted',
                    next_action='none_queued',updated_local=stamp,public_study=SLUG,publication_receipt=status['publication_receipt'])
        data['directions']['crossplatform'].update(active_round=None,execution_open=False,latest_completed_report=status['report'],
            latest_completed_publication=commit,latest_public_report=report,latest_public_commit=commit)
    atomic_json(path,data)
    atomic_json(HERE/'STATE.json',status)
    ledger = ROOT/'research/ARCHIVE.md'
    marker = 'R164 completed and published '+commit
    if marker not in ledger.read_text():
        with ledger.open('a') as f:
            f.write('\n'+marker+'. 192 source-own records. Inside attraction removed: S0/S1 Acc@50 harms 42/40 → 10/9. '
                    'Outside S1 remains −1.5625 pp versus its zero step. One-plane 27 changes, +0.2214 mIoU pp / −0.5208 Acc@50 pp. '
                    'Role coverage A256 14/14, R88 2/14, exclusivity 0→1 dual-correct. R159 remains default. '
                    '[Report](round164/REPORT_ZH.md) · [Decision](round164/DECISION.json) · '
                    '[Verified receipt](round164/publication/PUBLICATION_RECEIPT.json)\n')


if __name__=='__main__':
    if sys.argv[1:]==['register']:
        register()
    else:
        export()
