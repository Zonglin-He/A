"""R1 publication is stage completion, never whole-route completion."""
import sys,time,shutil,subprocess,hashlib,json,concurrent.futures,zlib,base64
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_optimizer_posterior_common_v1 import *
from scripts.publish_tastvg_decota_critic_p0_v1 import git,fetch,CHECKOUT

def stage():
    verify_seal();assert read(BASE/'ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert read(PUB/'ROOT_AUDIT.json')['status']==read(PUB/'PUBLIC_AUDIT.json')['status']=='pass'
    assert not git('status','--porcelain');subprocess.run(['git','fetch','--quiet','origin','main'],cwd=CHECKOUT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')
    lock=verify();effective=dict(lock['pins'])
    for f in sorted((BASE/'revisions').glob('*.json')):effective.update(read(f)['pin_overrides'])
    files=[f for f in effective if 'optimizer_posterior' in f]
    files += ['scripts/test_decota_optimizer_posterior_v1.py','scripts/score_decota_optimizer_posterior_v1.py',
        'scripts/report_decota_optimizer_posterior_v1.py','scripts/publish_decota_optimizer_posterior_v1.py',
        'scripts/review_decota_optimizer_posterior_v1.py',
        'scripts/decota_public_result_io_v1.py',
        'docs/decota_optimizer_posterior_v1/EXECUTION.md','docs/TA_DECOTA_OPTIMIZER_POSTERIOR_REVIEW.md']
    files=list(dict.fromkeys(files));deps={f:h for f,h in effective.items() if f not in files}
    for f,h in deps.items():assert sha(CHECKOUT/f)==h,('Dependency not public',f)
    write(PUB/'CODE_BINDING.json',dict(status='pass',new_code={f:sha(ROOT/f) for f in files},existing_dependency_pins=deps,
        runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),private_payload_export=False))
    import gzip
    raw=(PUB/'SPATIAL_DIAGNOSTICS.json').read_bytes();gz=PUB/'SPATIAL_DIAGNOSTICS.json.gz';gz.write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))
    assert gzip.decompress(gz.read_bytes())==raw
    write(PUB/'PUBLIC_DATA_FORMAT.json',dict(lossless_compressed_files={'SPATIAL_DIAGNOSTICS.json.gz':dict(logical_name='SPATIAL_DIAGNOSTICS.json',uncompressed_sha256=hashlib.sha256(raw).hexdigest(),rows=11520)},reader='scripts/decota_public_result_io_v1.py',metric_filtering=False))
    files += [str(f.relative_to(ROOT)) for f in PUB.iterdir() if f.is_file() and f.name!='SPATIAL_DIAGNOSTICS.json']
    for f in files:
        assert not f.startswith(('artifacts/','data/','checkpoints/'))
        dst=CHECKOUT/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dst)
        if f.endswith(('.json','.md')):
            raw=dst.read_bytes();assert b'/home/wwww/.codex/attachments/' not in raw
            if f.endswith('.json'):assert b'"video_path"' not in raw and b'"caption"' not in raw
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True);assert set(git('diff','--cached','--name-only').splitlines())==set(files)
    ff=[]
    for f in files:
        raw=(CHECKOUT/f).read_bytes();ff.append(dict(path=f,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),blob_sha=git('hash-object',f)))
    meta=dict(repository='Zonglin-He/A',status='reviewed_staged',base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),
        expected_tree=git('write-tree'),files=ff,file_count=len(ff),bytes=sum(f['bytes'] for f in ff),stage='R1',whole_route_complete=False,time=time.time())
    write(BASE/'PUBLIC_STAGE.json',meta)
    write(BASE/'PUBLIC_TRANSPORT.json',dict(metadata=meta,files=[dict(**f,zlib_base64=base64.b64encode(zlib.compress((CHECKOUT/f['path']).read_bytes(),9)).decode()) for f in ff]))
    print('R1_PUBLIC_STAGED',len(ff),meta['bytes'],flush=True)

def verify_remote(commit):
    meta=read(BASE/'PUBLIC_STAGE.json');ref=json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/ref/heads/main'));assert ref['object']['sha']==commit
    cm=json.loads(fetch(f'https://api.github.com/repos/Zonglin-He/A/git/commits/{commit}'));assert cm['parents'][0]['sha']==meta['base_commit'] and cm['tree']['sha']==meta['expected_tree']
    def one(f):
        from urllib.parse import quote
        raw=fetch(f"https://raw.githubusercontent.com/Zonglin-He/A/{commit}/{quote(f['path'])}")
        assert raw==(CHECKOUT/f['path']).read_bytes() and hashlib.sha256(raw).hexdigest()==f['sha256'];return dict(**f,remote_bytes_identical=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rr=list(pool.map(one,meta['files']))
    from scripts.score_decota_optimizer_posterior_v1 import public_check
    audit=public_check(CHECKOUT/PUB.relative_to(ROOT));subprocess.run(['git','fetch','--quiet','origin','main'],cwd=CHECKOUT,check=True)
    assert git('rev-parse','origin/main')==commit;subprocess.run(['git','update-ref','refs/heads/main',commit,meta['base_commit']],cwd=CHECKOUT,check=True)
    subprocess.run(['git','reset','--mixed',commit],cwd=CHECKOUT,check=True);assert not git('status','--porcelain')
    write(BASE/'GITHUB_REMOTE_VERIFICATION.json',dict(status='pass',commit=commit,files=rr,file_count=len(rr),bytes=sum(f['bytes'] for f in rr),public_audit=audit,time=time.time()))
    archive('R1全部factorial/posterior/数学与dense/source审计/三图目检/代码匿名结果公开逐远端核验完成；GitHub '+commit+'；完整条件路线仍待接续')
    write(BASE/'R1_COMPLETION.json',dict(status='completed_verified_publication',commit=commit,full_route_complete=False,
        remote_verified_files=len(rr),remote_verified_bytes=sum(f['bytes'] for f in rr),root_audit_checks=read(PUB/'ROOT_AUDIT.json')['checks'],time=time.time()))
    status(BASE/'STATUS.json',dict(status='R1_verified_pending_conditional_continuation',commit=commit,full_route_complete=False))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['stage','verify']);p.add_argument('commit',nargs='?');a=p.parse_args()
    if a.stage=='stage':stage()
    else:verify_remote(a.commit)
