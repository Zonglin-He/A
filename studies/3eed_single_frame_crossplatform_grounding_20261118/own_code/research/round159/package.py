"""Package first-party deployment code and audited anonymous results; keep model assets on site."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/'research/round154'))
from package154 import confidential_literals, scan_release, transport_csv

STUDY = 'studies/3eed_single_frame_crossplatform_grounding_20261118'


def read(path):
    return json.loads(path.read_text())


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,dst)


def result_literals():
    values=confidential_literals()
    manifests=(ROOT/'crossplatform/round158/EXTRA_REGISTRY_PRIVATE.json',ROOT/'crossplatform/round159/RECORDING_INPUTS_PRIVATE.json')
    for path in manifests:
        for item in read(path):
            row=item.get('row',item)
            for key in ('id','caption','scene','metapath','imagepath','pointpath','image_sha256','point_sha256'):
                val=row.get(key)
                if isinstance(val,str) and len(val)>=8:
                    values.add(val.encode())
    return values


def main():
    required=('RAW_REPLAY','CLI_REPLAY','BUNDLE_REPLAY','AUDIT','RECORDING_AUDIT','RECORDING_SCALAR_AUDIT')
    for name in required:
        if read(HERE/(name+'.json'))['status']!='PASS':raise ValueError(name)
    n=read(HERE/'RECORDING_AUDIT.json')['direction_records']
    assert n==read(HERE/'RECORDING_PREDICTION_SEAL.json')['total']
    publication=HERE/'publication';publication.mkdir(exist_ok=True)
    dest=publication/'public_export_attempt001'
    if dest.exists():raise FileExistsError(dest)
    temp=Path(tempfile.mkdtemp(dir=publication))
    try:
        allow=read(HERE/'FIRST_PARTY_RUNTIME.json')
        for item in allow:
            parts=Path(item).parts
            if parts[0] not in ('research','crossplatform') or len(parts)<2 or set(parts)&{'upstream','env','runtime','site-packages'}:
                raise ValueError('Invalid first-party file')
            copy(ROOT/item,temp/'own_code'/item)
        copy(ROOT/'crossplatform/single_frame_sources.json',temp/'own_code/crossplatform/single_frame_sources.json')
        tools=('summarize.py','audit.py','verify_raw.py','verify_cli.py','verify_bundle.py',
               'check_recordings.py','score_recordings.py','audit_recordings.py','register_results.py','package.py')
        for name in tools:copy(HERE/name,temp/'own_code/research/round159'/name)
        for item,pin in read(HERE/'experiment/EVALUATOR_LOCK.json')['sha256'].items():
            if hashlib.sha256((ROOT/item).read_bytes()).hexdigest()!=pin:raise ValueError('Evaluator drift')
            copy(ROOT/item,temp/'own_code'/item)
        copy(ROOT/'research/round154/package154.py',temp/'own_code/research/round154/package154.py')
        for name in (*required,'FIRST_PARTY_RUNTIME','RECORDING_INPUT_SEAL','RECORDING_PREDICTION_SEAL','DECISION'):
            copy(HERE/(name+'.json'),temp/(name+'.json'))
        for p in sorted((HERE/'experiment').glob('*.json')):
            copy(p,temp/'experiment'/p.name)
        guide=(ROOT/'crossplatform/SINGLE_FRAME.md').read_text()
        for a,b in {
            '(single_frame_sources.json)':'(own_code/crossplatform/single_frame_sources.json)',
            '(../research/round158/grounding158.py)':'(own_code/research/round158/grounding158.py)',
            '(replication36_surface/common.py)':'(own_code/crossplatform/replication36_surface/common.py)',
            '(../research/round159/REPORT_ZH.md)':'(REPORT_ZH.md)'}.items():guide=guide.replace(a,b)
        guide+='\n公开包的 own_code/ 保留工作区相对结构。将其中的代码置于配有所需依赖的 3EED 工作区，源权重按清单保存在本地。命令行已在仅挂载该源权重、官方运行库及文本资产的独立目录复核。[运行包复核](BUNDLE_REPLAY.json)\n'
        (temp/'RUN_ZH.md').write_text(guide)
        report=(HERE/'REPORT_ZH.md').read_text().replace('../../crossplatform/SINGLE_FRAME.md','RUN_ZH.md').replace('../../crossplatform/ground_single_frame.py','own_code/crossplatform/ground_single_frame.py')
        (temp/'REPORT.md').write_text('# R159 single-frame cross-platform grounding\n\n[中文报告和六方向结果](REPORT_ZH.md) · [Run the model](RUN_ZH.md)\n')
        splits={}
        for p in sorted((HERE/'results').iterdir()):
            if not p.is_file() or p.suffix not in {'.csv','.json'}:raise ValueError('Unexpected result')
            if p.name=='recording_all_rows.csv':
                with p.open(newline='') as f:
                    reader=csv.DictReader(f);columns=reader.fieldnames;rows=list(reader)
                if columns!=['ordinal','source','target','scene_cluster','point_content_cluster','changed_query','old_axis','old_pca','r158_pca'] or len(rows)!=n:raise ValueError('Incorrect public record schema')
                logical='results/'+p.name
                splits[logical]=transport_csv(temp,logical,rows,columns)
                if len(splits[logical]['parts'])>1:report=report.replace('('+logical+')','('+logical[:-4]+')')
            else:copy(p,temp/'results'/p.name)
        (temp/'REPORT_ZH.md').write_text(report)
        save(temp/'TRANSPORT_SPLITS.json',splits)
        save(temp/'PUBLIC_EXPORT_AUDIT.json',dict(first_party_runtime_files=len(allow),source_checkpoint_files_shipped=0,
             original_media_descriptions_ids_boxes_or_features_shipped=False,baseline_partial_rows_omitted=0,
             new_recording_rows=n,all_negative_comparisons_included=True,FINAL_access=False))
        files=[dict(path=p.relative_to(temp).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
               for p in sorted(temp.rglob('*')) if p.is_file()]
        save(temp/'MANIFEST.json',dict(prefix=STUDY,files=files))
        scan_release(temp,result_literals())
        os.rename(temp,dest)
    except BaseException:
        shutil.rmtree(temp,ignore_errors=True)
        raise
    print('validated',len(files)+1,'files')


if __name__=='__main__':
    main()
