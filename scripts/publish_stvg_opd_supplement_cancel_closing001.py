"""Separate finite publisher; original cancelled-dependent publisher is immutable."""
import ast
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.decota_matrix_common_v1 import read, write, sha
from scripts.audit_stvg_opd_supplement_cancel_public_v1 import REL
from scripts.publish_stvg_opd_actionability_audit_v1 import git, CHECKOUT, BRANCH, put, safe, private_attachment_path
BASE = ROOT/'artifacts/stvg_opd_supplement_v1'; OUT = ROOT/REL
SOURCES = ['protocols/stvg_opd_supplement_v1.md','protocols/stvg_opd_supplement_after_cancel_closing001.md',
    'scripts/stvg_opd_order_supplement_common_v1.py','scripts/run_stvg_opd_order_supplement_v1.py',
    'scripts/score_stvg_opd_order_supplement_v1.py','scripts/finalize_stvg_opd_order_supplement_v1.py','scripts/continue_stvg_opd_order_supplement_v1.py',
    'scripts/stvg_opd_baseline_supplement_common_v1.py','scripts/run_stvg_opd_baseline_supplement_v1.py',
    'scripts/score_stvg_opd_baseline_supplement_v1.py','scripts/continue_stvg_opd_baseline_supplement_v1.py',
    'scripts/analyze_stvg_opd_supplement_saved_v1.py','scripts/render_stvg_opd_supplement_cases_v1.py',
    'scripts/review_stvg_opd_supplement_v1.py','scripts/continue_stvg_opd_supplement_root_v1.py',
    'scripts/publish_stvg_opd_supplement_v1.py','scripts/audit_stvg_opd_supplement_public_v1.py',
    'scripts/continue_stvg_opd_cost_isolation_v1.py','scripts/audit_stvg_opd_LN_complete_public_v1.py',
    'scripts/review_stvg_opd_supplement_cancel_closing001.py','scripts/audit_stvg_opd_supplement_cancel_public_v1.py',
    'scripts/report_stvg_opd_supplement_cancel_closing001.py','scripts/publish_stvg_opd_supplement_cancel_closing001.py',
    'scripts/close_stvg_opd_supplement_cancel_closing001.py']

def metadata(v):
    if not isinstance(v, dict): return {}
    out = {}
    for k,x in v.items():
        if re.search('records|files|payload|settings|rows|summary|originals|prefix|private|command|process|human_instruction', k, re.I): continue
        if isinstance(x, bool): out[k] = x
        elif isinstance(x, str) and (k in ['status','scope','stage','dataset'] or re.fullmatch('[0-9a-f]{40,64}',x)): out[k] = x
        elif isinstance(x,(int,float)) and (k == 'time' or re.search('count|total|done|cells|fits|rounds|coordinates|pairs|bytes|PDFs|pages|arrivals|calls',k,re.I)): out[k] = x
        elif k in ['counts','checks'] and isinstance(x,dict): out[k] = metadata(x)
    safe(out); return out

def run(closing=False):
    assert read(BASE/'ROOT_ALL_AUTHORIZED_S_SCOPES_READBACK.json')['status'] == 'pass'
    assert read(BASE/'ROOT_VISUAL_REVIEW.json')['status'] == 'pass'
    assert git('remote','get-url','origin') == 'https://github.com/Zonglin-He/A.git'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    base = git('rev-parse','HEAD'); assert base == git('rev-parse','origin/main') == git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    unrelated = git('ls-files','--others','--exclude-standard').splitlines(); owned = set()
    def copy_file(p, relative):
        data = p.read_bytes()
        if p.suffix == '.json': safe(read(p))
        elif p.suffix == '.gz':
            with gzip.open(p,'rt') as f:
                if 'ROWS' in p.name:
                    for line in f: safe(json.loads(line))
                else: safe(json.load(f))
        elif p.suffix in {'.py','.md'}:
            assert not private_attachment_path(data.decode())
            if p.suffix == '.py': ast.parse(data.decode())
        q = CHECKOUT/relative; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,q)
        assert sha(q) == sha(p); owned.add(relative)
    for n in SOURCES: copy_file(ROOT/n,n)
    if not closing:
        names = ['ROOT_ALL_AUTHORIZED_S_SCOPES_READBACK.json','ROOT_ANONYMOUS_SCALAR_READBACK.json','ROOT_VISUAL_REVIEW.json',
            'after_cancel_closing001/ROOT_BASELINE_CUSTODY_READBACK.json','after_cancel_closing001/REVISION_RUNTIME.json',
            'after_cancel_closing001/ROOT_reader_scope002/CAPTURE_RECEIPT.json','after_cancel_closing001/ROOT_reader_scope002/RESOLUTION_RECEIPT.json',
            'orders/ROOT_ALL_FITS_AND_STATISTICS.json','orders/ROOT_QUALIFICATION_READBACK.json','orders/ROOT_VISUAL_REVIEW.json',
            'baselines/ROOT_DEVELOPMENT_SELECTION_READBACK.json','baselines/ROOT_QUALIFICATION_READBACK.json',
            'baselines/USER_CANCEL_REMAINING_FORMAL_RECEIPT.json','saved_evidence/ROOT_VISUAL_REVIEW.json',
            'saved_evidence/CASE_COMPLETION.json','cost/ROOT_CLOSING.json','cost/FINAL_COMPLETION.json','cost/CLOSING_FINAL_GITHUB_RECEIPT.json']
        for n in names:
            p = BASE/n; put(OUT/'receipts'/n,dict(original_private_receipt_sha256=sha(p),projection=metadata(read(p))))
        for n,b in read(BASE/'ROOT_VISUAL_REVIEW.json')['PDF_files'].items():
            if not b['public']: continue
            p = ROOT/n; assert sha(p) == b['sha256'] and p.stat().st_size == b['bytes']
            copy_file(p,str((OUT/'figures'/p.name).relative_to(ROOT)))
    else:
        assert read(BASE/'FINAL_COMPLETION.json')['status'] == 'complete'
        for n in ['ROOT_CLOSING.json','FINAL_COMPLETION.json','FINAL_GITHUB_RECEIPT.json','ARCHIVE_VERIFICATION_RECEIPT.json']:
            p=BASE/n; put(OUT/'closing'/n,dict(original_private_receipt_sha256=sha(p),projection=metadata(read(p))))
        for e in read(BASE/'PUBLIC_STAGE.json')['files']:
            assert sha(CHECKOUT/e['path']) == e['sha256']; owned.add(e['path'])
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p.name not in ['SOURCE_MANIFEST.json','SOURCE_MANIFEST_CLOSING.json']: copy_file(p,str(p.relative_to(ROOT)))
    z = subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(CHECKOUT/'scripts/audit_stvg_opd_supplement_cancel_public_v1.py'),str(CHECKOUT)],capture_output=True,text=True,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
    assert z.returncode == 0,z.stdout+z.stderr; audit=json.loads(z.stdout)
    assert audit == read(OUT/'PUBLIC_SCALAR_AUDIT.json') and audit['status'] == 'pass'
    if not closing: write(BASE/'PUBLIC_SCALAR_AUDIT.json',dict(audit,actual_public_checkout=True,time=time.time()))
    relative = str((OUT/'SOURCE_MANIFEST.json').relative_to(ROOT)); manifest = OUT/('SOURCE_MANIFEST_CLOSING.json' if closing else 'SOURCE_MANIFEST.json')
    write(manifest,dict(status='pass',scope='current S after explicit cancellation: all anonymous orders/dev/saved effects/numerical PDFs/source/custody/actual prior view/cost and current decision; excludes self hash and all private case pixels',
        files={n:dict(sha256=sha(CHECKOUT/n),bytes=(CHECKOUT/n).stat().st_size) for n in sorted(owned) if n != relative}))
    copy_file(manifest,relative)
    entries=[]
    for n in sorted(owned):
        p=CHECKOUT/n;data=p.read_bytes();entries.append(dict(path=n,bytes=len(data),sha256=sha(p),blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
    with tempfile.TemporaryDirectory(prefix='S-cancel-public-index-') as folder:
        env={**os.environ,'GIT_INDEX_FILE':str(Path(folder)/'index')};git('read-tree',base,env=env)
        for e in entries:
            h=git('hash-object','-w',str(CHECKOUT/e['path']));assert h==e['blob_sha'];git('update-index','--add','--cacheinfo','100644',h,e['path'],env=env)
        tree=git('write-tree',env=env)
    write(BASE/('CLOSING_PUBLIC_STAGE.json' if closing else 'PUBLIC_STAGE.json'),dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,
        base_commit=base,expected_tree=tree,files=entries,file_count=len(entries),bytes=sum(e['bytes'] for e in entries),
        unrelated_untracked_preserved=unrelated,scope='genuine current S evidence scope after cancellation; no missing formal scoring, no future main completion',time=time.time()))
    print(json.dumps(dict(status='reviewed_staged',closing=closing,file_count=len(entries),bytes=sum(e['bytes'] for e in entries),base_commit=base,expected_tree=tree)))

if __name__ == '__main__': run(len(sys.argv)>1 and sys.argv[1]=='closing')
