"""Verify public Git refs/tree and every staged whitelist file byte-for-byte.

The commit itself is made by the authenticated GitHub connector. This read-only
network verifier records success only after both remote heads and immutable raw
contents match, then moves the clean local tracking checkout with a CAS guard.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import requests
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,read,write,sha
CHECKOUT=ROOT.parent/'visual-grounding-public-A'


def get(url):
    for i in range(3):
        try:
            r=requests.get(url,timeout=45);r.raise_for_status();return r
        except requests.RequestException:
            if i==2:raise
            time.sleep(2*(i+1))


def run(stage_path,commit,receipt_path):
    stage_path=Path(stage_path);receipt_path=Path(receipt_path);stage=read(stage_path)
    assert stage['repository']=='Zonglin-He/A' and stage['status']=='reviewed_staged'
    api='https://api.github.com/repos/Zonglin-He/A'
    def refs():
        result={b:get(api+'/git/ref/heads/'+b).json()['object']['sha'] for b in ['main',stage['branch']]}
        assert set(result.values())=={commit};return result
    first=refs();obj=get(api+'/git/commits/'+commit).json()
    assert obj['tree']['sha']==stage['expected_tree'] and obj['parents'][0]['sha']==stage['base_commit']
    tree=get(api+'/git/trees/'+stage['expected_tree']+'?recursive=1').json();assert not tree['truncated']
    entries={r['path']:r for r in tree['tree']}
    def check(record):
        p=CHECKOUT/record['path'];original=p.read_bytes()
        assert len(original)==record['bytes'] and hashlib.sha256(original).hexdigest()==record['sha256']
        entry=entries[record['path']];assert entry['mode']=='100644' and entry['type']=='blob' and entry['sha']==record['blob_sha']
        remote=get('https://raw.githubusercontent.com/Zonglin-He/A/'+commit+'/'+record['path']).content
        blob=hashlib.sha1(b'blob '+str(len(remote)).encode()+b'\0'+remote).hexdigest()
        assert remote==original and len(remote)==record['bytes']
        assert hashlib.sha256(remote).hexdigest()==record['sha256'] and blob==record['blob_sha']
        return dict(record,remote_bytes_exact=True,remote_SHA256_exact=True,remote_Git_blob_exact=True,remote_tree_mode_exact=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as workers:checked=list(workers.map(check,stage['files']))
    last=refs()
    def git(*args):return subprocess.check_output(['git',*args],cwd=CHECKOUT,text=True).strip()
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    assert git('rev-parse','HEAD')==stage['base_commit']
    subprocess.run(['git','fetch','--quiet','origin','main',stage['branch']],cwd=CHECKOUT,check=True)
    assert git('rev-parse','origin/main')==git('rev-parse','origin/'+stage['branch'])==commit
    subprocess.run(['git','update-ref','HEAD',commit,stage['base_commit']],cwd=CHECKOUT,check=True)
    subprocess.run(['git','read-tree',commit],cwd=CHECKOUT,check=True)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    result=dict(status='pass',scope=stage['scope'],repository=stage['repository'],branch=stage['branch'],
        commit=commit,tree_sha=stage['expected_tree'],base_commit=stage['base_commit'],
        remote_refs_before=first,remote_refs_after=last,file_count=len(checked),bytes=sum(x['bytes'] for x in checked),
        files=checked,staging_receipt_sha256=sha(stage_path),
        actual_remote_bytes_SHA256_Git_blob_tree_verified=True,paper_suite_complete=False,time=time.time())
    write(receipt_path,result)
    print(json.dumps({k:v for k,v in result.items() if k!='files'}))


if __name__=='__main__':run(*sys.argv[1:])
