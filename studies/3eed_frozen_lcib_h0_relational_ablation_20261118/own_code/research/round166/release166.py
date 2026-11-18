"""Validate the scalar evidence, package a privacy-safe study, and register its receipt."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime
from math import fsum
from pathlib import Path
import shutil
import sys
import tempfile

from scope166 import ARMS,HERE,ORDER,PROTOCOL,ROOT,atomic_json,inputs,lock,read,sha,status
from score166 import IDENTITY,METRICS,PAIR_ARMS,TRANSFER,OUT,table

sys.path.insert(0,str(ROOT/'research/round154'))
from package154 import scan_release,transport_csv

PREFIX='studies/3eed_frozen_lcib_h0_relational_ablation_20261118'
CODE=('scope166.py','init166.py','predict166.py','labels166.py','score166.py','review166.py','release166.py','report166.py','visual_coverage166.py')


def arm_features(r,a):
    score=float(r[a+'_iou'])
    assert 0<=score<=1
    return (score,score>.25,score>.50,int(r[a+'_subject'])==1,int(r[a+'_subject_box'])==1)


def choose_group(scope,rows,scenes=None):
    sub,cell=scope['subset'],scope['group']
    if 'recording' in scope:
        return [scenes[scope['source'],scope['target'],scope['recording'],sub]]
    if cell=='six_equal':
        return [data for (s,t,d),data in rows.items() if d==sub and data]
    if cell=='all_pooled':
        return [[r for (s,t,d),data in rows.items() if d==sub for r in data]]
    if cell.startswith('target_'):
        return [[r for (s,t,d),data in rows.items() if d==sub and t==cell[7:] for r in data]]
    s,_,t=cell.split('_')
    return [rows[s,t,sub]]


def record_metrics(arm,part):
    m=[]
    for g in part:
        values=zip(*(arm_features(r,arm) for r in g))
        m.append([100*fsum(col)/len(g) for col in values])
    return [fsum(v)/len(m) for v in zip(*m)]


def audit():
    lock()
    eval_lock=read(HERE/'EVALUATION_LOCK.json')
    assert all(sha(HERE/name)==value for name,value in eval_lock['pins'].items())
    proof=read(HERE/'EVALUATION_AUDIT.json')
    assert proof['n']==18620 and proof['geometry_endpoints']==93100
    assert proof['max_r165_baseline_iou_difference']<1e-11 and proof['max_absolute_sensor_iou_error']<=1e-7
    assert not proof['FINAL_access'] and sha(HERE/'PREDICTION_SEAL.json')==proof['prediction_seal']
    for file,expected in proof['files'].items():
        assert sha(OUT/file)==expected
    vision=read(HERE/'VISUAL_REVIEW_AUDIT.json')
    assert sha(OUT/'visual_review.csv')==vision['out_sha256']
    assert read(HERE/'VISUAL_COVERAGE_AUDIT.json')['visible_identity_edges']==vision['verified_direction_edges']
    rows=table(OUT/'all_rows.csv')
    edges=table(OUT/'relation_edges.csv')
    assert len(rows)==18620==len({(r['source'],r['target'],r['expression']) for r in rows})
    pools=defaultdict(list)
    scenes=defaultdict(list)
    lookup={}
    for r in rows:
        s,t,rec=r['source'],r['target'],r['recording']
        assert (s,t) in ORDER
        lookup[s,t,r['expression']]=r
        sub=['all',*(x for x in ('parsed_direct','camera_horizontal','active_factor') if int(r[x]))]
        for x in sub:
            pools[s,t,x].append(r)
            scenes[s,t,rec,x].append(r)
    for t in ('waymo','drone','quad'):
        a,b=[{(r['expression'],r['frame'],r['recording']) for r in pools[s,t,'all']} for s,x in ORDER if x==t]
        assert a==b and len(a) in (3782,2792,2736)
    numbers=[]
    for f in ('metrics.csv','recordings.csv'):
        for r in table(OUT/f):
            compare=dict(r,recording=r['group']) if f=='recordings.csv' else r
            group=choose_group(compare,pools,scenes)
            assert int(r['n'])==sum(map(len,group))
            v=record_metrics(r['arm'],group)
            numbers.extend(abs(x-float(r[k])) for x,k in zip(v,METRICS))
            assert int(r['subject_unknown'])==sum(int(x[r['arm']+'_subject'])==-1 for g in group for x in g)
    for r in table(OUT/'paired.csv'):
        group=choose_group(r,pools)
        assert int(r['n'])==sum(map(len,group))
        a,b=r['main'],r['control']
        v=record_metrics(a,group)
        w=record_metrics(b,group)
        diff=list(zip(*(tuple(f-g for f,g in zip(arm_features(x,a),arm_features(x,b))) for part in group for x in part)))
        for m,u,z,d in zip(METRICS,v,w,diff):
            numbers.append(abs(u-z-float(r[m+'_pp'])))
            assert sum(x>1e-10 for x in d)==int(r[m+'_up'])
            assert sum(x< -1e-10 for x in d)==int(r[m+'_down'])
    assert Counter(r['truth'] for r in edges)==proof['edge_truth_counts']
    assert all(lookup[r['source'],r['target'],r['expression']]['camera_horizontal']=='1' for r in edges)
    for f,axis,arms in (('identity.csv','arm',PAIR_ARMS),('coverage.csv','stage',TRANSFER)):
        for r in table(OUT/f):
            truth=r['truth']
            group=[e for e in edges if r['group'] in ('all',f"{e['source']}_to_{e['target']}",f"target_{e['target']}")
                   and (e['truth']==truth or (truth=='all_supported' and e['truth'] in ('annotation_proxy','visual_verified')))]
            assert len(group)==int(r['n']) and r[axis] in arms
            for k in IDENTITY:
                c=[int(e[r[axis]+'_'+k]) for e in group]
                correct=sum(v==1 for v in c)
                if axis=='stage':
                    assert int(r[k])==correct
                    continue
                den=sum(v>=-1 for v in c)
                assert int(r[k+'_n'])==den and int(r[k+'_correct'])==correct
                assert int(r[k+'_wrong'])==sum(v==0 for v in c) and int(r[k+'_unknown'])==sum(v==-1 for v in c)
                if den:
                    numbers.append(abs(float(r[k])-100*correct/den))
                else:
                    assert r[k]==''
    assert len(numbers)>3000 and max(numbers)<1e-9
    record=dict(n=len(rows),edges=len(edges),checked_values=len(numbers),maximum_error=max(numbers),
                all_negative_and_unknown_rows_retained=True,score_audit_sha256=sha(HERE/'EVALUATION_AUDIT.json'),
                visual_audit_sha256=sha(HERE/'VISUAL_REVIEW_AUDIT.json'),FINAL_access=False)
    atomic_json(HERE/'SCALAR_AUDIT.json',record)
    return record


def private_literals():
    rows,_=inputs()
    hidden=set()
    for rr in rows.values():
        for r in rr:
            for key in ('id','caption','scene','pointpath','imagepath','metapath'):
                val=r[key]
                if isinstance(val,str) and len(val)>=12:
                    hidden.add(val.encode())
            frame=Path(r['metapath']).parent.name
            if len(frame)>=16:
                hidden.add(frame.encode())
    return hidden


def export():
    verify=audit()
    pub=HERE/'publication'
    pub.mkdir(exist_ok=True)
    out=pub/'public_export_attempt001'
    assert not out.exists()
    tmp=Path(tempfile.mkdtemp(dir=pub))
    try:
        def copy(p,rel):
            dst=tmp/rel
            dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,dst)
        for f in CODE:
            copy(HERE/f,'own_code/research/round166/'+f)
        for p in (HERE/'experiment').glob('*.json'):
            copy(p,'experiment/'+p.name)
        for f in ('REPORT_ZH.md','REPORT.md','DECISION.json','EVALUATION_AUDIT.json','SCALAR_AUDIT.json',
                  'PILOT_SEAL.json','PREDICTION_SEAL.json','REVIEW_PANEL_SEAL.json','VISUAL_REVIEW_AUDIT.json',
                  'EVALUATION_LOCK.json','VISUAL_COVERAGE_AUDIT.json'):
            copy(HERE/f,f)
        parts={}
        for f in sorted(OUT.glob('*.csv')):
            rows=table(f)
            parts[f.name]=transport_csv(tmp,'results/'+f.name,rows,tuple(rows[0]))
        atomic_json(tmp/'PUBLIC_EXPORT_AUDIT.json',dict(n=verify['n'],edges=verify['edges'],files=parts,
            raw_media_text_coords_true_instance_ids_embeddings_weights_or_credentials_included=False,FINAL_access=False))
        manifest=dict(prefix=PREFIX,files=[dict(path=p.relative_to(tmp).as_posix(),bytes=p.stat().st_size,
                    sha256=sha(p)) for p in sorted(tmp.rglob('*')) if p.is_file()])
        atomic_json(tmp/'MANIFEST.json',manifest)
        scan_release(tmp,private_literals())
        tmp.rename(out)
        print('R166 verified public files',len(manifest['files'])+1,flush=True)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def register():
    audit()
    p=HERE/'publication/PUBLICATION_RECEIPT.json'
    checked=read(p)
    assert checked['repository']=='Zonglin-He/A' and checked['prefix']==PREFIX
    commit=checked['commit']
    assert checked['immutableVerifiedBeforeRefUpdate'] and checked['remoteByteVerified'] and checked['mainRefObserved']==commit
    path=ROOT/'research/STATE.json'
    state=read(path)
    assert state['R161']['state']=='paused_by_user'
    stamp=datetime.now().astimezone().isoformat()
    remote=f'https://github.com/Zonglin-He/A/blob/{commit}/{PREFIX}/REPORT_ZH.md'
    verdict=read(HERE/'DECISION.json')['finding']
    status('completed_published_verified',report='round166/REPORT_ZH.md',decision='round166/DECISION.json',
           protocol='round166/experiment/PROTOCOL.json',public_report=remote,public_commit=commit,source_training_runs=0,
           target_parameter_updates=0,publication_receipt='round166/publication/PUBLICATION_RECEIPT.json')
    state=read(path)
    state.update(latest_completed_round='R166',publication_state='R166_REMOTE_BYTES_VERIFIED',
                 measurement_state='R166_FROZEN_H0_AND_IDENTITY_ABLATIONS_AUDITED',claim_state=verdict,
                 public_study=PREFIX,publication_receipt='round166/publication/PUBLICATION_RECEIPT.json',
                 current_stage='R166_COMPLETE',next_action='none_queued',updated_local=stamp)
    state['directions']['crossplatform'].update(latest_completed_report='round166/REPORT_ZH.md',
             latest_completed_publication=commit,latest_public_report=remote,latest_public_commit=commit)
    atomic_json(path,state)
    archive=ROOT/'research/ARCHIVE.md'
    line=f'R166 completed and published {commit}'
    if line not in archive.read_text():
        with archive.open('a') as f:
            f.write(f'\n{line}. 18,620 R165-paired records. {verdict} [Report](round166/REPORT_ZH.md) · '
                    '[Verified receipt](round166/publication/PUBLICATION_RECEIPT.json)\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('audit','export','register'))
    phase=parser.parse_args().phase
    if phase=='audit':
        print(audit(),flush=True)
    elif phase=='export':
        export()
    else:
        register()
