"""Publish only reviewed conditional-route code and anonymous results."""
import sys,time,shutil,subprocess,hashlib,json,zlib,base64,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_actuation_scope_common_v1 import *
from scripts.publish_tastvg_decota_critic_p0_v1 import git,fetch,CHECKOUT

def stage():
    verify();assert read(BASE/'ROOT_VISUAL_REVIEW.json')['status']=='pass';route=read(BASE/'ROUTE_DECISION.json');assert route['full_route_complete']
    for folder in ['R3','R3G','R4','online','cross_domain']:
        p=PUB/folder
        if not p.exists():continue
        assert read(p/'ROOT_AUDIT.json')['status']==read(p/'PUBLIC_AUDIT.json')['status']=='pass'
    assert not git('status','--porcelain');subprocess.run(['git','fetch','--quiet','origin','main'],cwd=CHECKOUT,check=True);assert git('rev-parse','HEAD')==git('rev-parse','origin/main')
    lock=verify();pins=dict(lock['pins'])
    for f in sorted((BASE/'revisions').glob('*.json')):pins.update(read(f)['pin_overrides'])
    files=[f for f in pins if any(s in f for s in ['decota_actuation_scope','decota_track_critic','decota_online_qualification'])]+['scripts/report_decota_actuation_scope_v1.py','scripts/review_decota_actuation_scope_v1.py',
        'scripts/publish_decota_actuation_scope_v1.py','docs/TA_DECOTA_ACTUATION_SCOPE_REVIEW.md','docs/decota_actuation_scope_v1/EXECUTION.md']
    cross=read(BASE/'cross_domain'/'RUNTIME_LOCK.json');files+=[f for f in cross['pins'] if 'decota_cross_qualification' in f]
    files=list(dict.fromkeys(files));dependencies={f:h for f,h in pins.items() if f not in files}
    dependencies.update({f:h for f,h in cross['pins'].items() if f not in files})
    for f,h in dependencies.items():assert sha(CHECKOUT/f)==h,('Dependency not public',f)
    write(PUB/'CODE_BINDING.json',dict(new_code={f:sha(ROOT/f) for f in files},existing_dependency_pins=dependencies,R1_commit=read(r1.BASE/'R1_COMPLETION.json')['commit'],private_payload_export=False))
    import gzip
    compressed={}
    for folder in ['R3','R4']:
        f=PUB/folder/'DIAGNOSTICS.json';raw=f.read_bytes();gz=Path(str(f)+'.gz');gz.write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))
        assert gzip.decompress(gz.read_bytes())==raw
        compressed[str(f.relative_to(PUB))]=dict(public_file=str(gz.relative_to(PUB)),uncompressed_bytes=len(raw),uncompressed_sha256=hashlib.sha256(raw).hexdigest(),rows=len(read(f)))
    write(PUB/'PUBLIC_DATA_FORMAT.json',dict(lossless_compressed_files=compressed,reader='scripts/decota_public_result_io_v1.py',metric_filtering=False))
    files+=[str(f.relative_to(ROOT)) for f in PUB.rglob('*') if f.is_file() and str(f.relative_to(PUB)) not in compressed]
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
    meta=dict(repository='Zonglin-He/A',status='reviewed_staged',base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),expected_tree=git('write-tree'),
        files=ff,file_count=len(ff),bytes=sum(f['bytes'] for f in ff),whole_route_complete=True,time=time.time())
    write(BASE/'PUBLIC_STAGE.json',meta);write(BASE/'PUBLIC_TRANSPORT.json',dict(metadata=meta,files=[dict(**f,zlib_base64=base64.b64encode(zlib.compress((CHECKOUT/f['path']).read_bytes(),9)).decode()) for f in ff]))
    print('ROUTE_STAGED',len(ff),meta['bytes'],flush=True)

def verify_remote(commit):
    meta=read(BASE/'PUBLIC_STAGE.json');ref=json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/ref/heads/main'));assert ref['object']['sha']==commit
    cm=json.loads(fetch(f'https://api.github.com/repos/Zonglin-He/A/git/commits/{commit}'));assert cm['parents'][0]['sha']==meta['base_commit'] and cm['tree']['sha']==meta['expected_tree']
    def one(f):
        from urllib.parse import quote
        raw=fetch(f"https://raw.githubusercontent.com/Zonglin-He/A/{commit}/{quote(f['path'])}");assert raw==(CHECKOUT/f['path']).read_bytes() and hashlib.sha256(raw).hexdigest()==f['sha256'];return dict(**f,remote_bytes_identical=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rr=list(pool.map(one,meta['files']))
    from scripts.score_decota_actuation_scope_v1 import public_check
    from scripts.score_decota_online_qualification_v1 import public_check as onlinecheck
    checks={}
    for s in ['R3','R3G','R4']:
        if (PUB/s).exists():checks[s]=public_check(CHECKOUT/(PUB/s).relative_to(ROOT))
    checks['online']=onlinecheck(CHECKOUT/(PUB/'online').relative_to(ROOT))
    from scripts.score_decota_cross_qualification_v1 import public_check as crosscheck
    checks['cross_domain']=crosscheck(CHECKOUT/(PUB/'cross_domain').relative_to(ROOT))
    subprocess.run(['git','fetch','--quiet','origin','main'],cwd=CHECKOUT,check=True);assert git('rev-parse','origin/main')==commit
    subprocess.run(['git','update-ref','refs/heads/main',commit,meta['base_commit']],cwd=CHECKOUT,check=True);subprocess.run(['git','reset','--mixed',commit],cwd=CHECKOUT,check=True);assert not git('status','--porcelain')
    write(BASE/'GITHUB_REMOTE_VERIFICATION.json',dict(status='pass',commit=commit,files=rr,file_count=len(rr),bytes=sum(f['bytes'] for f in rr),public_audits=checks,time=time.time()))
    archive('完整授权条件路线的实际阶段和资格跳过已根核验、报告图目检、代码匿名结果公开并逐远端核验；GitHub '+commit+'；无生产晋升')
    write(BASE/'FINAL_COMPLETION.json',dict(status='completed_verified_publication',commit=commit,R1_commit=read(r1.BASE/'R1_COMPLETION.json')['commit'],
        scope='entire_attachment_conditional_route_original_panels',full_official_query_run=False,route=read(BASE/'ROUTE_DECISION.json'),remote_verified_files=len(rr),remote_verified_bytes=sum(f['bytes'] for f in rr),time=time.time()))
    status(BASE/'STATUS.json',dict(status='completed_verified_publication',commit=commit,full_route_complete=True))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','verify']);p.add_argument('commit',nargs='?');a=p.parse_args();stage() if a.action=='stage' else verify_remote(a.commit)
