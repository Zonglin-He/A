"""Additive public staging of reward/order evidence or finite diagnostic closure."""
import ast
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.publish_stvg_opd_supplement_v1 import safe,private_attachment_path
from scripts.publish_stvg_opd_mechanism_review_v1 import project,put
BASE=ROOT/'artifacts/stvg_opd_mechanism_review_v1'
PUB=ROOT/'results/stvg_opd_mechanism_review/2026-10-11'
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
BRANCH='research/stvg-opd-paper-hc2-revision-v2'


def git(*args,env=None):
    return subprocess.check_output(['git',*args],cwd=CHECKOUT,text=True,env=env).strip()


def run(role):
    assert role in {'round_addendum','closing'}
    assert read(BASE/'DIAGNOSTIC_GITHUB_RECEIPT.json')['status']=='pass'
    if role=='round_addendum':
        assert read(BASE/'ROOT_ORDER_ROUND_READBACK.json')['status']=='pass'
        originals=['CPU_ORDER_ROUND_RUNTIME.json','ROOT_ORDER_ROUND_READBACK.json','DIAGNOSTIC_GITHUB_RECEIPT.json']
        output=['PER_ORDER_DIAGNOSTICS.json.gz','REWARD_SCALAR_MANIFEST.json','ORDER_REWARD_ADDENDUM.md']
        manifest=read(PUB/'REWARD_SCALAR_MANIFEST.json')
        output.extend(r['path'] for r in manifest['chunks'])
        combined=b''.join((PUB/r['path']).read_bytes() for r in manifest['chunks'])
        assert len(combined)==manifest['original_bytes']
        assert hashlib.sha256(combined).hexdigest()==manifest['original_sha256']
        for line in gzip.decompress(combined).splitlines():safe(json.loads(line))
        sources=['scripts/read_stvg_opd_mechanism_orders_and_rounds_v1.py',
                 'scripts/stage_stvg_opd_mechanism_closure_v1.py']
        scope='complete actual reward-round and original-order CPU scalar readback; no new GPU or GT'
    else:
        assert read(BASE/'ROOT_DIAGNOSTIC_CLOSING_RECEIPT.json')['status']=='complete'
        assert read(BASE/'DIAGNOSTIC_FINAL_COMPLETION.json')['reviewer_request_complete'] is False
        originals=['ROOT_DIAGNOSTIC_CLOSING_RECEIPT.json','DIAGNOSTIC_FINAL_COMPLETION.json',
                   'ARCHIVE_VERIFICATION_RECEIPT.json','ARCHIVE_POST_CLOSE_SNAPSHOT_RECEIPT.json',
                   'ROUND_ADDENDUM_GITHUB_RECEIPT.json','HOURLY_MONITOR_FINAL_SCOPE_UPDATE.json']
        output=[];sources=[]
        scope='actual finite CPU diagnosis root/view/public/archive closure; requested GPU transfer and latency remain pending'
    for rel in originals:
        p=BASE/rel;out=PUB/'receipts'/rel;value=dict(original_private_receipt_sha256=sha(p),projection=project(read(p)))
        safe(value);put(out,value);output.append('receipts/'+rel)
    paths=[(ROOT/rel,rel) for rel in sources]+[(PUB/rel,str((PUB/rel).relative_to(ROOT))) for rel in output]
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    base=git('rev-parse','HEAD')
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    assert base==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    if role=='round_addendum':
        pins={}
        for f in ['CPU_RUNTIME.json','cross_parent_LN/RUNTIME_LOCK.json','DINO_RECONSTRUCTION_RUNTIME.json']:
            pins.update(read(BASE/f)['pins'])
        # An interrupted earlier preparation can leave a verified dependency in
        # the worktree without its blob being added to the first public tree.
        # Add only exact new-runtime pins that are actually absent from that tree.
        for rel,h in pins.items():
            if Path(rel).suffix not in {'.py','.md'}:continue
            exists=subprocess.run(['git','cat-file','-e',base+':'+rel],cwd=CHECKOUT,
                                  stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
            if not exists:
                p=ROOT/rel;assert sha(p)==h and not private_attachment_path(p.read_text())
                if rel not in {r for _,r in paths}:paths.append((p,rel))
    untracked=git('ls-files','--others','--exclude-standard').splitlines()
    assert all('stvg_motivation' in p.lower() or p in {rel for _,rel in paths} for p in untracked)
    entries=[]
    for p,rel in paths:
        if p.suffix in {'.py','.md'}:
            text=p.read_text();assert not private_attachment_path(text)
            if p.suffix=='.py':ast.parse(text,filename=rel)
        if p.suffix=='.json':safe(read(p))
        if p.name=='PER_ORDER_DIAGNOSTICS.json.gz':
            with gzip.open(p,'rt') as f:safe(json.load(f))
        dest=CHECKOUT/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest);assert sha(dest)==sha(p)
        data=dest.read_bytes();blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        assert git('hash-object','-w',str(dest))==blob
        entries.append(dict(path=rel,bytes=len(data),sha256=sha(p),blob_sha=blob))
    with tempfile.TemporaryDirectory(prefix='stvg-mechanism-closing-index-') as folder:
        env={**os.environ,'GIT_INDEX_FILE':str(Path(folder)/'index')};git('read-tree',base,env=env)
        for r in entries:git('update-index','--add','--cacheinfo','100644',r['blob_sha'],r['path'],env=env)
        tree=git('write-tree',env=env)
    path=BASE/('PUBLIC_'+role.upper()+'_STAGE.json')
    write(path,dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,scope=scope,
          base_commit=base,expected_tree=tree,files=entries,restricted_assets_excluded=True,
          unrelated_untracked_preserved=[p for p in untracked if 'stvg_motivation' in p.lower()],
          reviewer_request_complete=False,time=time.time()))
    print(json.dumps(dict(status='reviewed_staged',role=role,files=len(entries),bytes=sum(r['bytes'] for r in entries),
                         base_commit=base,tree_sha=tree,path=str(path))))


if __name__=='__main__':run(*sys.argv[1:])
