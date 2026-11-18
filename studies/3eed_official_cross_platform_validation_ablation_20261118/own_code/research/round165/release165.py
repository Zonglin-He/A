"""Audit the complete scalar export, package the first-party evidence, and register publication."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from math import fsum
from pathlib import Path
import shutil
import sys
import tempfile

from score165 import ARMS, METRICS, RESULTS, table
from scope165 import HERE, ORDER, PAPER, PROTOCOL, ROOT, atomic_json, check_lock, ordinary, read, sha

sys.path.insert(0,str(ROOT/'research/round154'))
from package154 import scan_release, transport_csv

PREFIX='studies/3eed_official_cross_platform_validation_ablation_20261118'
SCRIPTS=('scope165.py','predict165.py','decode165.py','score165.py','release165.py','report165.py')
DISCREPANCY=HERE/'experiment/SCORER_ADDENDUM.json'


def single_metrics(numbers):
    n=len(numbers)
    return (100*fsum(numbers)/n,100*sum(x>.25 for x in numbers)/n,100*sum(x>.5 for x in numbers)/n)


def selected(rows,conditions):
    cells, recordings, targets = rows
    growth=conditions['aggregation']
    if growth=='recording':
        return [recordings[conditions['source'],conditions['target'],conditions['group']]]
    if growth=='direction':
        return [cells[conditions['source'],conditions['target']]]
    if growth=='incoming_macro':
        return [cells[s,t] for s,t in ORDER if t==conditions['target']]
    if growth=='pooled_micro':
        return [[r for t in targets.values() for r in t]]
    assert growth=='six_direction_macro'
    return [cells[k] for k in ORDER]


def verify_results():
    lock=check_lock()
    audit=read(HERE/'GEOMETRY_AUDIT.json')
    cohort=audit['cohort']
    assert cohort in ('ordinary','full')
    total=audit['official_direction_records']
    assert total==(19388 if cohort=='full' else 18620)
    assert audit['independent_sensor_convex_geometry_endpoint_checks']==total*len(ARMS)
    assert audit['max_independent_sensor_abs_iou_error']<=1e-7
    for f,h in audit['outputs'].items():
        assert sha(RESULTS/f)==h
    cases=table(RESULTS/'all_rows.csv')
    assert len(cases)==total==len({(r['source'],r['target'],r['expression']) for r in cases})
    assert tuple(cases[0])==('source','target','recording','frame','expression','official_vs_native_switch',*ARMS)
    expected=read(PROTOCOL)['official_population']['expected_eligible' if cohort=='full' else 'initial_eligible']
    cells, recordings, targets = defaultdict(list),defaultdict(list),defaultdict(list)
    for r in cases:
        a,b=r['source'],r['target']
        assert (a,b) in ORDER
        vals={name:float(r[name]) for name in ARMS}
        assert all(0<=x<=1 for x in vals.values())
        cells[a,b].append(vals)
        recordings[a,b,r['recording']].append(vals)
        targets[b].append(vals)
    for t in expected:
        assert len(targets[t])==2*expected[t]
        groups=[{(r['frame'],r['expression'],r['recording']) for r in cases if r['target']==t and r['source']==s}
                for s,x in ORDER if x==t]
        assert len(groups[0])==expected[t] and groups[0]==groups[1]
    counts,errors=Counter(),[]
    indexed={}
    for f in ('metrics.csv','recordings.csv'):
        for r in table(RESULTS/f):
            cell=selected((cells,recordings,targets),r)
            assert int(r['n'])==sum(map(len,cell))
            numbers=[single_metrics([v[r['arm']] for v in g]) for g in cell]
            metrics=[fsum(g)/len(cell) for g in zip(*numbers)]
            key=(r['source'],r['target'],r['group'],r['arm'])
            assert key not in indexed
            indexed[key]=metrics
            errors.extend(abs(v-float(r[k])) for v,k in zip(metrics,METRICS))
            counts[f]+=1
    for r in table(RESULTS/'paired.csv'):
        cell=selected((cells,recordings,targets),r)
        key=(r['source'],r['target'],r['group'])
        before,after=r['before'],r['after']
        a,b=indexed[(*key,before)],indexed[(*key,after)]
        assert int(r['n'])==sum(map(len,cell))
        v=[(x[before],x[after]) for q in cell for x in q]
        for m,x,y,t in zip(METRICS,a,b,(None,.25,.50)):
            diffs=[q-p if t is None else int(q>t)-int(p>t) for p,q in v]
            assert sum(d>1e-10 for d in diffs)==int(r[m+'_up'])
            assert sum(d< -1e-10 for d in diffs)==int(r[m+'_down'])
            errors.append(abs((y-x)-float(r[m+'_pp'])))
        if before.removesuffix('_axis')==after.removesuffix('_pca'):
            assert int(r['changed_winner'])==0
        counts['paired.csv']+=1
    assert max(errors)<1e-9
    out=dict(status='PASS',cohort=cohort,all_rows=total,methods=len(ARMS),table_records=dict(counts),
             scalar_metric_elements_recomputed=len(errors),max_scalar_error_pp=max(errors),
             all_direction_recording_frames_anonymized_and_paired=True, both_tails_retained=True,
             protocol_sha256=lock['protocol_sha256'], scorer_addendum_sha256=sha(DISCREPANCY),
             independent_geometry_audit_sha256=sha(HERE/'GEOMETRY_AUDIT.json'), FINAL_access=audit['FINAL_access'])
    atomic_json(HERE/'SCALAR_AUDIT.json',out)
    return out


def private_literals():
    out=set()
    for records in ordinary().values():
        for row in records:
            for key in ('id','caption','pointpath','imagepath','metapath','scene'):
                v=row[key]
                if isinstance(v,str) and len(v)>=12:
                    out.add(v.encode())
            label=Path(row['metapath']).parent.name
            if len(label)>=16:
                out.add(label.encode())
    return out


def export():
    audit=verify_results()
    parent=HERE/'publication'
    parent.mkdir(exist_ok=True)
    dest=parent/'public_export_attempt001'
    assert not dest.exists()
    tmp=Path(tempfile.mkdtemp(dir=parent))
    try:
        def copy(source, path):
            to=tmp/path
            to.parent.mkdir(exist_ok=True,parents=True)
            shutil.copyfile(source,to)

        for script in SCRIPTS:
            copy(HERE/script, f'own_code/research/round165/{script}')
        for file in (PROTOCOL,PAPER,DISCREPANCY):
            copy(file,f'experiment/{file.name}')
        parts={}
        for f in sorted(RESULTS.glob('*.csv')):
            rows=table(f)
            parts[f.name]=transport_csv(tmp,f'results/{f.name}',rows,tuple(rows[0]))
        files=('REPORT_ZH.md','REPORT.md','DECISION.json','GEOMETRY_AUDIT.json','SCALAR_AUDIT.json','PILOT_SEAL.json')
        for name in files:
            copy(HERE/name,name)
        atomic_json(tmp/'PUBLIC_EXPORT_AUDIT.json',dict(records=audit['all_rows'], methods=len(ARMS),
            independent_sensor_IoU_checks=audit['all_rows']*len(ARMS), no_raw_ids_or_locator_paths=True,
            attachments='First-party Python, protocol, metrics, signed and anonymous per-expression IoU scalars',
            raw_media_labels_feature_weights_or_secrets_included=False, FINAL_access=audit['FINAL_access'],
            tabular_parts=parts))
        manifest=dict(prefix=PREFIX,files=[dict(path=f.relative_to(tmp).as_posix(),bytes=f.stat().st_size,
                       sha256=sha(f)) for f in sorted(tmp.rglob('*')) if f.is_file()])
        atomic_json(tmp/'MANIFEST.json',manifest)
        scan_release(tmp,private_literals())
        tmp.rename(dest)
        print('R165 public export verified',len(manifest['files'])+1,flush=True)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def register():
    audit=verify_results()
    result=read(HERE/'DECISION.json')
    receipt_path=HERE/'publication/PUBLICATION_RECEIPT.json'
    receipt=read(receipt_path)
    commit=receipt['commit']
    assert receipt['repository']=='Zonglin-He/A' and receipt['prefix']==PREFIX
    assert receipt['remoteByteVerified'] and receipt['immutableVerifiedBeforeRefUpdate'] and receipt['mainRefObserved']==commit
    stamp=datetime.now().astimezone().isoformat()
    report=f'https://github.com/Zonglin-He/A/blob/{commit}/{PREFIX}/REPORT_ZH.md'
    path=ROOT/'research/STATE.json'
    state=read(path)
    assert state['R161']['state']=='paused_by_user' and state['current_round']=='R165'
    entry=dict(round='R165',state='completed_published_verified',official_cross_platform_full=audit['cohort']=='full',
        official_direction_records=audit['all_rows'],directions=6,FINAL_access=audit['FINAL_access'],
        full_scope_state='complete' if audit['cohort']=='full' else 'pending_previously_protected_scope_confirmation',
        receipt=str(receipt_path.relative_to(ROOT/'research')),public_commit=commit,public_report=report,
        protocol='round165/experiment/PROTOCOL.json',report='round165/REPORT_ZH.md',decision='round165/DECISION.json',
        target_updates=0,new_source_training=0,updated=stamp)
    state['R165']=entry
    state.update(latest_completed_round='R165',current_stage='R165_COMPLETE',run_state='R165_COMPLETED_PUBLISHED_MAIN_VERIFIED',
        publication_state='R165_REMOTE_BYTES_VERIFIED',measurement_state='R165_PAIRED_SIX_DIRECTION_ABLATIONS_AUDITED',
        claim_state=result['finding'],next_action=entry['full_scope_state'],updated_local=stamp,public_study=PREFIX,
        publication_receipt=entry['receipt'])
    state['directions']['crossplatform'].update(active_round=None,execution_open=False,
        latest_completed_report=entry['report'],latest_completed_publication=commit,
        latest_public_report=report,latest_public_commit=commit)
    atomic_json(HERE/'STATE.json',entry)
    atomic_json(path,state)
    p=ROOT/'research/ARCHIVE.md'
    if commit not in p.read_text():
        with p.open('a') as f:
            f.write(f'\nR165 completed and published {commit}. Six official single-source transfer directions, '
                    f'{audit["all_rows"]:,} paired records with native last-layer BBF and 14 ablation arms. '
                    f'{result["finding"]} [Report](round165/REPORT_ZH.md) · '
                    '[Verified receipt](round165/publication/PUBLICATION_RECEIPT.json)\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=('audit','export','register'))
    args=p.parse_args()
    if args.phase=='audit':
        print(verify_results())
    elif args.phase=='export':
        export()
    else:
        register()
