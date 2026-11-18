"""Export anonymous scalar outcomes and first-party implementation after both audits."""
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'research/round154'))
from package154 import scan_release, confidential_literals, transport_csv

STUDY='studies/3eed_language_candidate_overlap_readout_20261118'
FILES=('grounding158.py','context158.py','runtime158.py','reader158.py','evidence158.py','predict158.py','revise158.py',
       'check_working158.py','check_reader158.py','verify_live158.py','transfer_inputs158.py','transfer_predict158.py',
       'transfer_score158.py','score158.py','analyze158.py','package158.py')


def load(path):return json.loads(path.read_text())
def save(path,obj):
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def cp(a,b):
    b.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(a,b)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=HERE/'publication/public_export_attempt001';out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    tmp=Path(tempfile.mkdtemp(dir=out.parent,prefix='.build_'));manifest=[]
    try:
        for source,dest in [('working_results','results'),('transfer_results','extra')]:
            dr=HERE/source
            for path in sorted(dr.glob('*.csv')):
                with path.open(newline='') as stream:
                    reader=csv.DictReader(stream);headers=tuple(reader.fieldnames);rows=list(reader)
                if not 0<len(rows) or any(any(q in h.lower() for q in ('caption','path','coordinates','weights_private','sample_id','query_index')) for h in headers):raise ValueError('public columns')
                for r in rows:
                    for k,v in r.items():
                        if k in ('source','target','scope','aggregation','arm','control','subset','group','weighting'):
                            if not v or len(v)>64 or not all(c.isalnum() or c=='_' for c in v):raise ValueError('table key')
                        elif v not in ('True','False') and not math.isfinite(float(v)):
                            raise ValueError('non scalar content')
                manifest.append(transport_csv(tmp,dest+'/'+path.name,rows,headers))
        for source,destination in [('results','first'),('revised_results','revised')]:
            path=HERE/source/'paired.csv'
            with path.open(newline='') as f:
                rows=csv.DictReader(f);headers=tuple(rows.fieldnames);items=list(rows)
            manifest.append(transport_csv(tmp,destination+'/paired.csv',items,headers))
        for source,tag in [('results','first'),('revised_results','revised'),('working_results','working'),('transfer_results','extra')]:
            aud=load(HERE/source/'AUDIT.json')
            if not aud['status']=='PASS' or aud['FINAL_access']:raise ValueError('audit')
            save(tmp/'results'/(tag+'_AUDIT.json'),aud)
        cp(HERE/'working_results/SCALAR_AUDIT.json',tmp/'results/SCALAR_AUDIT.json')
        cp(HERE/'LIVE_API_AUDIT.json',tmp/'results/LIVE_API_AUDIT.json')
        for path in sorted((HERE/'experiment').glob('*.json')):cp(path,tmp/'config'/path.name)
        for name in FILES:cp(HERE/name,tmp/'own_code/research/round158'/name)
        for name in ('PREDICTION_SEAL','REVISION_SEAL','WORKING_SEAL','EXTRA_INPUT_SEAL','EXTRA_SEAL'):
            cp(ROOT/'crossplatform/round158'/(name+'.json'),tmp/'seals'/(name+'.json'))
        cp(HERE/'DECISION.json',tmp/'DECISION.json');cp(HERE/'REPORT_ZH.md',tmp/'REPORT_ZH.md')
        (tmp/'REPORT.md').write_text('# R158 frozen language-driven candidate selection\n\n[中文报告与全部对照](REPORT_ZH.md)\n')
        save(tmp/'TRANSPORT_SPLITS.json',manifest)
        save(tmp/'PUBLIC_EXPORT_AUDIT.json',dict(ordinary_development_rows=1798,fixed_single_frame_rows=5942,
             all_paired_directions_and_negative_outcomes_included=True,private_media_boxes_captions_ids_features_weights_included=False,
             source_training=False,parameter_TTA=False,FINAL_access=False))
        files=[{'path':p.relative_to(tmp).as_posix(),'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(tmp.rglob('*')) if p.is_file()]
        save(tmp/'MANIFEST.json',dict(prefix=STUDY,files=files))
        scan_release(tmp,confidential_literals())
        for rel in ('results/all_rows','extra/all_rows','extra/macro.csv','extra/scene.csv','results/subset.csv','results/paired.csv'):
            if not (tmp/rel).exists():raise ValueError('broken report link '+rel)
        os.rename(tmp,out)
    except BaseException:
        shutil.rmtree(tmp,ignore_errors=True);raise
    print('public files checked',len(files)+1,flush=True)

if __name__=='__main__':main()
