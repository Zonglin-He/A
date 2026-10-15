"""Explicit reviewed export and remote byte/tree/ref readback of controlled rounds."""
import sys, shutil, subprocess, hashlib, json, time, urllib.request, urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_evidence_common_v1 import BASE, read, write, sha
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
BRANCH='research/stvg-opd-paper-hc2-revision-v2'
def git(*args): return subprocess.check_output(['git',*args],cwd=CHECKOUT,text=True).strip()
def stage():
    assert read(BASE/'ROUND2_ANONYMOUS_SCALAR_AUDIT012.json')['status']=='pass'
    assert read(BASE/'FROZEN_METHOD_LOCK.json')['search_finished']
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    assert not git('diff','--name-only') and not git('diff','--cached','--name-only')
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    paths={str(p.relative_to(ROOT))for p in (ROOT/'results/stvg_evidence_constrained/2026-10-15/round2').iterdir()if p.is_file()}
    names=['stvg_evidence_spatial_inputs005.py','stvg_evidence_spatial_inputs009.py','review_stvg_evidence_inputs007.py',
           'review_stvg_evidence_inputs009.py','review_stvg_evidence_inputs012.py','stvg_evidence_spatial_phase006.py',
           'stvg_evidence_spatial_phase009.py','stvg_evidence_spatial_preservation_v1.py',
           'prepare_stvg_evidence_prompt009.py','prepare_stvg_evidence_preservation006.py',
           'prepare_stvg_evidence_preservation010.py','run_stvg_evidence_student005.py',
           'run_stvg_evidence_student006.py','run_stvg_evidence_student009.py','run_stvg_evidence_student010.py',
           'select_stvg_evidence_method006.py','select_stvg_evidence_method012.py',
           'stvg_evidence_fresh_custody012.py','run_stvg_evidence_confirmation_temporal012.py',
           'fit_stvg_evidence_confirmation012.py','prepare_stvg_evidence_confirmation012.py',
           'continue_stvg_evidence_confirmation012.py','launch_stvg_evidence006.py',
           'report_stvg_evidence_progress012.py','publish_stvg_evidence_progress013.py']
    paths.update('scripts/'+n for n in names)
    paths.add('docs/STVG_EVIDENCE_CONSTRAINED_PROGRESS.md')
    rows=[]
    for rel in sorted(paths):
        p=ROOT/rel
        assert p.suffix in {'.py','.md','.json','.png'} and p.is_file()
        raw=p.read_bytes()
        if p.suffix=='.json':
            for token in [b'"query_text"',b'"caption"',b'"video_path"',b'"boxes"',b'"masks"',b'"optimizer_state"',b'"weights"']:
                assert token not in raw,(rel,token)
        dest=CHECKOUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        assert dest.read_bytes()==raw
        rows.append(dict(path=rel,sha256=sha(p),bytes=len(raw),blob_sha=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest(),mode='100644'))
    subprocess.run(['git','add','--',*sorted(paths)],cwd=CHECKOUT,check=True)
    changed=git('diff','--cached','--name-only').splitlines()
    assert changed and set(changed)<=paths
    write(BASE/'ROUND2_PUBLIC_STAGE013.json',dict(status='reviewed_staged',base_commit=git('rev-parse','HEAD'),
          expected_tree=git('write-tree'),files=rows,changed_files=changed,
          branch=BRANCH,scope='completed controlled development, frozen method and prepared confirmation; not Table1 completion',time=time.time()))
    print('staged',len(rows),sum(x['bytes']for x in rows))
def fetch(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'STVG-evidence-audit'}),timeout=45)as r:return r.read()
        except Exception:
            if attempt==2: raise
            time.sleep(attempt+1)
def verify(commit):
    stage=read(BASE/'ROUND2_PUBLIC_STAGE013.json')
    for branch in ['main',BRANCH]:
        ref=json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/ref/heads/'+urllib.parse.quote(branch,safe='/')))
        assert ref['object']['sha']==commit
    c=json.loads(fetch(f'https://api.github.com/repos/Zonglin-He/A/git/commits/{commit}'))
    assert c['tree']['sha']==stage['expected_tree'] and c['parents'][0]['sha']==stage['base_commit']
    tree=json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/trees/'+c['tree']['sha']+'?recursive=1'))
    assert not tree.get('truncated',False)
    lookup={x['path']:x for x in tree['tree']}
    def one(row):
        rel=row['path'];entry=lookup[rel]
        assert entry['mode']==row['mode'] and entry['type']=='blob' and entry['sha']==row['blob_sha']
        raw=fetch(f'https://raw.githubusercontent.com/Zonglin-He/A/{commit}/'+urllib.parse.quote(rel))
        assert raw==(CHECKOUT/rel).read_bytes() and hashlib.sha256(raw).hexdigest()==row['sha256'] and len(raw)==row['bytes']
        assert hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()==entry['sha']
        return dict(row,remote_bytes_identical=True)
    with ThreadPoolExecutor(max_workers=4)as pool:rows=list(pool.map(one,stage['files']))
    assert git('rev-parse','HEAD')==commit
    write(BASE/'ROUND2_FINAL_GITHUB_RECEIPT013.json',dict(status='pass',commit=commit,tree=c['tree']['sha'],
          branches=['main',BRANCH],files=rows,file_count=len(rows),bytes=sum(x['bytes']for x in rows),
          scope=stage['scope'],time=time.time()))
    print('verified',commit,len(rows))
if __name__=='__main__': stage() if sys.argv[1]=='stage' else verify(sys.argv[2])
