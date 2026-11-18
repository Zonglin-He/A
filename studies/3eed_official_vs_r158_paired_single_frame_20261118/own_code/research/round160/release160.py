"""Publish only paired metrics and first-party research code, and register the verified result."""
import csv
import json
import os
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
import tempfile
from common160 import ROOT,HERE,read,sha

sys.path.insert(0,str(ROOT/'research/round154'))
from package154 import transport_csv, scan_release, confidential_literals

SLUG='studies/3eed_official_vs_r158_paired_single_frame_20261118'


def dump(p,value):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp160');t.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(t,p)


def copy(src,path):
    path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,path)


def export():
    checks=read(HERE/'AUDIT.json')
    if checks['status']!='PASS' or checks['paired_rows']!=10812:raise ValueError('Missing paired audit')
    pub=HERE/'publication';pub.mkdir(exist_ok=True)
    dst=pub/'public_export_attempt001'
    if dst.exists():raise FileExistsError(dst)
    temp=Path(tempfile.mkdtemp(dir=pub))
    try:
        names=('common160.py','predict160.py','decode160.py','score160.py','audit160.py','report160.py','eval_lock.py','release160.py')
        for name in names:copy(HERE/name,temp/'own_code/research/round160'/name)
        for name in ('FORWARD_SEAL','PREDICTION_SEAL','EVALUATION','AUDIT','DECISION'):
            copy(HERE/(name+'.json'),temp/(name+'.json'))
        for src in sorted((HERE/'experiment').glob('*.json')):
            copy(src,temp/'experiment'/src.name)
        redacted={}
        for k in ('crossplatform/round158/EXTRA_REGISTRY_PRIVATE.json','crossplatform/round159/RECORDING_INPUTS_PRIVATE.json'):
            for entry in read(ROOT/k):
                for key in ('id','caption','scene','pointpath','imagepath','metapath','image_sha256','point_sha256'):
                    word=entry.get('row',entry).get(key)
                    if isinstance(word,str) and len(word)>=8:redacted[word.encode()]=1
        report=(HERE/'REPORT_ZH.md').read_text().replace('(decode160.py)','(own_code/research/round160/decode160.py)')
        splits={}
        for p in sorted((HERE/'results').glob('*.csv')):
            with p.open(newline='') as f:
                reader=csv.DictReader(f);data=list(reader);fields=reader.fieldnames
            if p.name=='all_records.csv':
                assert len(data)==checks['paired_rows'] and fields==['cohort','ordinal','source','target','recording','point_group','sampling_weight',
                    'native_span_and_proxy_differ','R158_winner_differs_from_official','official_span','native_text_only','prepared_axis','prepared_pca','r158_full']
            logical='results/'+p.name
            splits[logical]=transport_csv(temp,logical,data,fields)
            if len(splits[logical]['parts'])>1:report=report.replace('('+logical+')','('+logical[:-4]+')')
        (temp/'REPORT_ZH.md').write_text(report)
        (temp/'REPORT.md').write_text('# Paired single-frame 3EED and R158 evaluation\n\n[Six-direction results and protocols](REPORT_ZH.md) · [Paired differences](results/paired.csv)\n')
        dump(temp/'TRANSPORT_SPLITS.json',splits)
        dump(temp/'PUBLIC_EXPORT_AUDIT.json',dict(all_paired_records=checks['paired_rows'],all_negative_comparisons=True,
            private_media_labels_identifiers_features_and_checkpoints_included=False,third_party_snapshots_included=False,
            runtime_reference='R159 frozen single-frame deployment, official 3EED commit 843a901f5367a6be5cc484b97c6ecad462961dfa'))
        items=[dict(path=f.relative_to(temp).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(temp.rglob('*')) if f.is_file()]
        dump(temp/'MANIFEST.json',dict(prefix=SLUG,files=items))
        scan_release(temp,confidential_literals()|set(redacted))
        os.rename(temp,dst)
    except BaseException:
        shutil.rmtree(temp,ignore_errors=True);raise
    print('validated export',len(items)+1,flush=True)


def register():
    receipt=read(HERE/'publication/PUBLICATION_RECEIPT.json');result=read(HERE/'DECISION.json')
    commit=receipt['commit'];assert receipt['mainRefObserved']==commit and receipt['remoteByteVerified']
    url='https://github.com/Zonglin-He/A/blob/'+commit+'/'+SLUG+'/REPORT_ZH.md'
    now=datetime.now(timezone.utc)
    record=dict(round='R160',state='COMPLETED_PUBLISHED_MAIN_VERIFIED',recorded=now.astimezone().isoformat(timespec='seconds'),
            report='round160/REPORT_ZH.md',protocol='research/round160/experiment/PROTOCOL.json',public_report=url,
            public_commit=commit,receipt='round160/publication/PUBLICATION_RECEIPT.json',decision='round160/DECISION.json',
            n=10812,primary='ordinary5942',primary_delta_pp=result['direction_macro_difference_pp'],source_training=0,target_updates=0,FINAL_access=False)
    dump(HERE/'STATE.json',record)
    s=read(ROOT/'research/STATE.json')
    s.update(R160=record,current_round='R160',latest_completed_round='R160',run_state='R160_COMPLETED_NO_ACTIVE_JOB',
             publication_state='R160_PUBLISHED_MAIN_VERIFIED',measurement_state='R160_PAIRED_OFFICIAL_BBF_AND_SINGLE_FRAME_R158_AUDITED',
             claim_state='R160 measured a direct same-sample comparison. Paired ordinary delta Acc25 / Acc50: '+result['cohorts']['ordinary5942']['delta']+' pp.',
             public_study=SLUG,publication_receipt=record['receipt'],updated_local=record['recorded'],updated_utc=now.isoformat(),next_action='none_queued')
    d=s['directions']['crossplatform']
    d.update(state='R160_COMPLETED_WORKING_DEFAULT_R159_SINGLE_FRAME',active_round=None,active_inputs=0,active_diagnostic_round=None,
             active_report=record['report'],latest_completed_report=record['report'],latest_completed_publication=commit,
             latest_public_report=url,latest_public_commit=commit,latest_research_report='research/'+record['report'],
             execution_open=False,FINAL_access=False)
    dump(ROOT/'research/STATE.json',s)
    p=ROOT/'research/ARCHIVE.md';marker='R160 paired official 3EED and R158. '+commit
    if marker not in p.read_text():
        txt=(marker+'. All 10812 directed single-frame comparisons verified across three separate cohorts. The official category-assisted native BBF versus R158 on ordinary 5942 has paired Acc25 / Acc50 delta '
             +result['cohorts']['ordinary5942']['delta']+' percentage points. Source checkpoints/point samples/shared metric fixed, target updates and FINAL access zero. [Report](round160/REPORT_ZH.md) · [Receipt](round160/publication/PUBLICATION_RECEIPT.json)\n')
        with p.open('a') as f:f.write('\n'+txt)


if __name__=='__main__':
    if sys.argv[1:]==['register']:register()
    else:export()
