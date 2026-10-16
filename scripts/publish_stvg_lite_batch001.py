"""Stage and verify only the explicit completed-stage batch; never runs science."""
import argparse
import json
import sys
import hashlib
import shutil
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import publish_stvg_soft_ep_v1 as publisher
BASE=ROOT/'artifacts/stvg_lite_event_support_v1/publication/batch001'
OUT=ROOT/'results/stvg_lite_event_support/2026-10-16/batch001'
def paths():
    assert json.loads((OUT/'ROOT_PUBLIC_SCALAR_AUDIT.json').read_text())['rows_checked']==4864
    result={str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file()}
    result.update(['docs/STVG_LITE_EVENT_SUPPORT_PROGRESS001.md','scripts/export_stvg_lite_batch001.py',
                   'scripts/audit_stvg_lite_batch001.py','scripts/publish_stvg_lite_batch001.py'])
    return result
def refresh():
    p=BASE/'PUBLIC_STAGE.json';old=json.loads(p.read_text());checkout=publisher.DEFAULT_CHECKOUT
    staged=set(publisher.git(checkout,'diff','--cached','--name-only').splitlines())
    assert staged==set(old['staged_paths']) and publisher.git(checkout,'rev-parse','HEAD')==old['base_commit']
    allowed=paths();publisher.verify_privacy(allowed);files=[]
    for rel in sorted(allowed):
        src=ROOT/rel;dst=checkout/rel;raw=src.read_bytes()
        if not dst.exists() or dst.read_bytes()!=raw:dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        files.append(dict(path=rel,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),mode='100644',
                          blob_sha=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest(),staged=True))
    subprocess.run(['git','add','--',*sorted(allowed)],cwd=checkout,check=True)
    old.update(files=files,file_count=len(files),bytes=sum(f['bytes'] for f in files),staged_paths=sorted(allowed),
               staged_file_count=len(allowed),expected_tree=publisher.git(checkout,'write-tree'))
    p.write_text(json.dumps(old,indent=2)+'\n')
    known={line.split()[2] for line in publisher.git(checkout,'ls-tree','-r','HEAD').splitlines()}
    unique={f['blob_sha']:f for f in files if f['blob_sha'] not in known}
    plan=dict(base_commit=old['base_commit'],expected_tree=old['expected_tree'],files=files,
              file_count=len(files),bytes=old['bytes'],new_blobs=list(unique.values()))
    (BASE/'UPLOAD_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(status='refreshed',files=len(files),tree=old['expected_tree'])))
def main():
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['stage','refresh','verify']);args=a.parse_args()
    BASE.mkdir(parents=True,exist_ok=True);publisher.BASE=BASE;publisher.expected_paths=paths
    if args.mode=='stage':publisher.stage(publisher.DEFAULT_CHECKOUT,None,False)
    elif args.mode=='refresh':refresh()
    else:publisher.verify_stage(publisher.DEFAULT_CHECKOUT)
if __name__=='__main__':main()
