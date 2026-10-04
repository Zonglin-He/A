"""Reviewed public Git tree plus independent remote byte readback; no credentials."""
import sys, time, shutil, subprocess, hashlib, json, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_privileged_p0_common_v1 import *
CHECKOUT=Path('/home/wwww/visual-grounding-public-A')


def git(*args):return subprocess.check_output(['git',*args],cwd=CHECKOUT).decode().strip()


def blob_sha(content):return hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()


def prepare():
    from scripts.score_audit_tastvg_privileged_p0_v1 import public_check
    verify_seal();assert read(BASE/'ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert read(PUB/'ROOT_AUDIT.json')['status']=='pass';public_check(PUB)
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    assert not git('status','--porcelain'),'Unrelated public checkout edits'
    subprocess.run(['git','fetch','--quiet','origin','main'],cwd=CHECKOUT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')
    parent=git('rev-parse','HEAD');base_tree=git('rev-parse','HEAD^{tree}')
    files=OWN+['scripts/score_audit_tastvg_privileged_p0_v1.py','scripts/report_tastvg_privileged_p0_v1.py',
        'scripts/publish_tastvg_privileged_p0_v1.py','docs/TA_PRIVILEGED_ATTENTION_P0_REVIEW.md',
        'docs/tastvg_privileged_attention_p0_v1/EXECUTION.md']
    source={f:sha(ROOT/f) for f in files};existing={};native={}
    for f,h in verify()['pins'].items():
        if f in files:continue
        if f.startswith('external/'):native[f]=h;continue
        assert (CHECKOUT/f).exists() and sha(CHECKOUT/f)==h,f
        existing[f]=h
    cpu=read(BASE/'CPU_RUNTIME_LOCK.json')['pins']
    for f,h in cpu.items():assert sha(ROOT/f)==h,f
    write(PUB/'CODE_BINDING.json',dict(status='pass',new_code=source,existing_public_dependency_pins=existing,
        original_native_runtime_dependency_pins=native,CPU_pins=cpu,
        no_private_media_GT_weights_boxes_attention_prefix_or_attachment_export=True,
        original_native_code_separately_required=True,time=time.time()))
    files += [str(f.relative_to(ROOT)) for f in sorted(PUB.iterdir()) if f.is_file()]
    assert len(files)==len(set(files))
    assert all(f.startswith(('scripts/','vg_tta/','protocols/','docs/','results/')) for f in files)
    for f in files:
        target=CHECKOUT/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,target)
    subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',
        str(CHECKOUT/'scripts/score_audit_tastvg_privileged_p0_v1.py'),'public',str(CHECKOUT/PUB.relative_to(ROOT))],cwd=CHECKOUT,check=True)
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True)
    assert set(git('diff','--cached','--name-only').splitlines())==set(files)
    tree=git('write-tree')
    entries=[]
    for f in files:
        content=(ROOT/f).read_bytes()
        assert blob_sha(content)==git('hash-object',f)
        entries.append(dict(path=f,bytes=len(content),sha256=hashlib.sha256(content).hexdigest(),blob_sha=blob_sha(content),mode='100644'))
    write(BASE/'PUBLICATION_MANIFEST.json',dict(repository='Zonglin-He/A',parent_sha=parent,
        base_tree_sha=base_tree,expected_tree_sha=tree,files=entries,file_count=len(entries),
        total_bytes=sum(f['bytes'] for f in entries),time=time.time()))
    print('REVIEWED_PUBLIC_TREE',tree,len(entries),sum(f['bytes'] for f in entries),flush=True)


def fetch(url):
    failure=None
    for retry in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'STVG-verified-public-readback'}),timeout=45) as r:return r.read()
        except Exception as exc:failure=exc;time.sleep(min(1+retry,5))
    raise failure


def finish(commit):
    from scripts.score_audit_tastvg_privileged_p0_v1 import public_check
    p=read(BASE/'PUBLICATION_MANIFEST.json');verify_seal()
    ref=json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/ref/heads/main'))
    assert ref['object']['sha']==commit
    c=json.loads(fetch(f'https://api.github.com/repos/Zonglin-He/A/git/commits/{commit}'))
    assert c['tree']['sha']==p['expected_tree_sha'] and c['parents'][0]['sha']==p['parent_sha']
    def check(entry):
        content=fetch(f"https://raw.githubusercontent.com/Zonglin-He/A/{commit}/{entry['path']}")
        assert len(content)==entry['bytes'] and hashlib.sha256(content).hexdigest()==entry['sha256']
        assert blob_sha(content)==entry['blob_sha'] and content==(ROOT/entry['path']).read_bytes()
        return dict(**entry,remote_bytes_identical=True)
    with ThreadPoolExecutor(max_workers=4) as pool:receipts=list(pool.map(check,p['files']))
    subprocess.run(['git','fetch','--quiet','origin','main'],cwd=CHECKOUT,check=True)
    assert git('rev-parse','origin/main')==commit and git('rev-parse','HEAD')==p['parent_sha']
    assert not git('diff','--name-only') and git('write-tree')==p['expected_tree_sha']
    subprocess.run(['git','reset','--soft',commit],cwd=CHECKOUT,check=True)
    assert not git('status','--porcelain') and git('rev-parse','HEAD')==commit
    remote_audit=public_check(CHECKOUT/PUB.relative_to(ROOT))
    write(BASE/'REMOTE_READBACK.json',dict(status='pass',repository='Zonglin-He/A',commit=commit,
        files=receipts,file_count=len(receipts),bytes=sum(r['bytes'] for r in receipts),
        remote_ref_parent_tree_and_bytes_verified=True,public_checkout_clean=True,remote_public_audit=remote_audit,time=time.time()))
    result=read(PUB/'DECISION.json')
    write(BASE/'FINAL_COMPLETION.json',dict(status='completed_verified_publication',cells=192,
        commit=commit,repository='Zonglin-He/A',P0_gate_pass=result['P0_gate_pass'],
        OPD_started=False,LN_consolidation_started=False,CURRENT_METHOD_unchanged=True,
        root_checks=read(PUB/'ROOT_AUDIT.json')['checks'],remote_verified_files=len(receipts),
        remote_verified_bytes=sum(r['bytes'] for r in receipts),time=time.time()))
    status(BASE/'STATUS.json',dict(status='completed_verified_publication',cells=192,commit=commit,
        P0_gate_pass=result['P0_gate_pass'],time=time.time()))
    s=read(PUB/'SUMMARY.json');facts=[]
    for ds in DATASETS:
        for split in ['search','confirm']:
            m=s[ds][split]['corruption']['metrics']['delta_v'];facts.append(f"{ds}/{split} Δv {100*m['mean']:+.4f}pp[{100*m['ci95'][0]:+.4f},{100*m['ci95'][1]:+.4f}]")
    archive('完整192预测/GT评分/独立attention与dense审计/两图目检及GitHub逐远端核验已完成；'
        +'；'.join(facts)+f"；实际DINO600次（早前封存日志手写598为笔误，原barrier一直为600），根32043493检查；GitHub{commit}、{len(receipts)}文件逐bytes/SHA256/Gitblob一致；"
        +'P0未建立双集privileged优势，本固定配置停止，不启动OPD/LN持续，不外推所有attention prior或专家无效')
    write(BASE/'ARCHIVE_COMPLETION.json',dict(status='pass',workflow='check/snapshot/check',
        ledger_sha256=sha(ROOT/'docs/RESEARCH_HISTORY.md'),time=time.time()))
    print('PUBLISHED_VERIFIED',commit,len(receipts),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','verify']);p.add_argument('commit',nargs='?');args=p.parse_args()
    prepare() if args.stage=='prepare' else finish(args.commit)
