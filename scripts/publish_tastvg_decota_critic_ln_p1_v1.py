"""Publish reviewed P1 code and anonymous evidence with remote byte verification."""
import sys,time,shutil,subprocess,hashlib,json,concurrent.futures,zlib,base64
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_decota_critic_ln_common_v1 import *
from scripts.publish_tastvg_decota_critic_p0_v1 import git,fetch,CHECKOUT


def stage():
    verify_seal();assert read(BASE/'ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert read(p0.BASE/'FINAL_COMPLETION.json')['status']=='completed_verified_publication'
    assert read(PUB/'ROOT_AUDIT.json')['status']==read(PUB/'PUBLIC_AUDIT.json')['status']=='pass'
    assert not git('status','--porcelain'),'Unrelated public checkout edits'
    subprocess.run(['git','fetch','--quiet','origin','main'],cwd=CHECKOUT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')
    lock=verify();files=[f for f in lock['pins'] if 'decota_critic_ln' in f]
    files+=['scripts/publish_tastvg_decota_critic_ln_p1_v1.py','docs/TA_DECOTA_CRITIC_LN_P1_REVIEW.md',
        'docs/tastvg_decota_critic_ln_p1_v1/EXECUTION.md']
    pins={f:sha(ROOT/f) for f in files};deps={f:h for f,h in lock['pins'].items() if f not in files}
    for f,h in deps.items():assert sha(CHECKOUT/f)==h,('Dependency not public at locked hash',f)
    write(PUB/'CODE_BINDING.json',dict(status='pass',new_code=pins,existing_dependency_pins=deps,
        runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),private_GT_payload_export=False,time=time.time()))
    files += [str(f.relative_to(ROOT)) for f in PUB.iterdir() if f.is_file()]
    assert len(files)==len(set(files))
    for f in files:
        assert not f.startswith(('artifacts/','data/','checkpoints/'))
        dst=CHECKOUT/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dst)
        if f.endswith(('.json','.md')):
            value=dst.read_bytes();assert b'/home/wwww/.codex/attachments/' not in value
            if f.endswith('.json'):assert b'"video_path"' not in value and b'"caption"' not in value
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True)
    assert set(git('diff','--cached','--name-only').splitlines())==set(files)
    staged=[]
    for f in files:
        b=(CHECKOUT/f).read_bytes();staged.append(dict(path=f,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),blob_sha=git('hash-object',f)))
    metadata=dict(status='reviewed_staged',repository='Zonglin-He/A',base_commit=git('rev-parse','HEAD'),
        base_tree=git('rev-parse','HEAD^{tree}'),expected_tree=git('write-tree'),files=staged,
        file_count=len(staged),bytes=sum(f['bytes'] for f in staged),private_assets_excluded=True,time=time.time())
    write(BASE/'PUBLIC_STAGE.json',metadata)
    write(BASE/'PUBLIC_TRANSPORT.json',dict(metadata=metadata,files=[dict(**f,zlib_base64=base64.b64encode(zlib.compress((CHECKOUT/f['path']).read_bytes(),9)).decode()) for f in staged]))
    print('P1_STAGED',len(staged),metadata['bytes'],flush=True)


def verify_remote(commit):
    staged=read(BASE/'PUBLIC_STAGE.json')
    ref=json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/ref/heads/main'));assert ref['object']['sha']==commit
    meta=json.loads(fetch(f'https://api.github.com/repos/Zonglin-He/A/git/commits/{commit}'))
    assert meta['parents'][0]['sha']==staged['base_commit'] and meta['tree']['sha']==staged['expected_tree']
    def one(f):
        from urllib.parse import quote
        b=fetch(f"https://raw.githubusercontent.com/Zonglin-He/A/{commit}/{quote(f['path'])}")
        assert b==(CHECKOUT/f['path']).read_bytes() and hashlib.sha256(b).hexdigest()==f['sha256'],f['path']
        assert hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()==f['blob_sha']
        return dict(**f,remote_bytes_identical=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:receipts=list(pool.map(one,staged['files']))
    from scripts.score_audit_tastvg_decota_critic_ln_p1_v1 import public_check
    audit=public_check(CHECKOUT/PUB.relative_to(ROOT))
    subprocess.run(['git','fetch','--quiet','origin','main'],cwd=CHECKOUT,check=True)
    assert git('rev-parse','origin/main')==commit
    subprocess.run(['git','update-ref','refs/heads/main',commit,staged['base_commit']],cwd=CHECKOUT,check=True)
    subprocess.run(['git','reset','--mixed',commit],cwd=CHECKOUT,check=True);assert not git('status','--porcelain')
    write(BASE/'GITHUB_REMOTE_VERIFICATION.json',dict(status='pass',commit=commit,files=receipts,
        file_count=len(receipts),bytes=sum(f['bytes'] for f in receipts),remote_ref_tree_verified=True,
        all_remote_bytes_identical=True,public_audit=audit,time=time.time()))
    verify();summary=read(PUB/'SUMMARY.json');estimates=[]
    for ds in DATASETS:
        for name in ['inherited','online_vs_episodic','online_vs_frozen']:
            x=summary[ds]['confirm']['corruption']['metrics'][name+'_v']
            estimates.append(f"{ds}/{name} {100*x['mean']:+.4f}pp[{100*x['ci95'][0]:+.4f},{100*x['ci95'][1]:+.4f}]")
    archive('1152真实online到达与P0复用控制全部封存/独立继承算术与Adam/dense/paired source bootstrap/目检/远端核验完成；'
        +'；'.join(estimates)+'；GitHub '+commit+'，不晋升')
    write(BASE/'ARCHIVE_COMPLETION.json',dict(status='pass',workflow='check/snapshot/check',ledger_sha256=sha(ROOT/'docs/RESEARCH_HISTORY.md'),time=time.time()))
    write(BASE/'FINAL_COMPLETION.json',dict(status='completed_verified_publication',arrivals=1152,commit=commit,
        remote_verified_files=len(receipts),remote_verified_bytes=sum(f['bytes'] for f in receipts),
        root_audit_checks=read(PUB/'ROOT_AUDIT.json')['checks'],public_audit_checks=audit['checks'],
        CURRENT_unchanged=True,method_promoted=False,GT_used_for_optimization=False,time=time.time()))
    status(BASE/'STATUS.json',dict(status='completed_verified_publication',commit=commit,time=time.time()))
    print('P1_PUBLISHED_VERIFIED',commit,len(receipts),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['stage','verify']);p.add_argument('commit',nargs='?');a=p.parse_args()
    if a.stage=='stage':stage()
    else:verify_remote(a.commit)
