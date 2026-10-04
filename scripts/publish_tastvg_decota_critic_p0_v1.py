"""Stage reviewed public files, then verify the published GitHub bytes and archive."""
import sys, time, shutil, subprocess, json, hashlib, urllib.request, concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_decota_critic_common_v1 import *
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
RELEASE=ROOT/'artifacts/decota_c1_online_release_v1/export'


def git(*args):
    return subprocess.check_output(['git',*args],cwd=CHECKOUT).decode().strip()


def stage():
    verify_seal();assert read(BASE/'ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert read(PUB/'ROOT_AUDIT.json')['status']==read(PUB/'PUBLIC_AUDIT.json')['status']=='pass'
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    assert not git('status','--porcelain'),'Unrelated public checkout edits'
    subprocess.run(['git','fetch','--quiet','origin','main'],cwd=CHECKOUT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')
    files=[f for f in read(BASE/'RUNTIME_LOCK.json')['pins'] if 'decota_critic' in f]
    files+=['scripts/publish_tastvg_decota_critic_p0_v1.py','docs/TA_DECOTA_CRITIC_P0_REVIEW.md',
        'docs/tastvg_decota_critic_p0_v1/EXECUTION.md']
    source_pins={f:sha(ROOT/f) for f in files}
    dep_pins={f:sha(ROOT/f) for f in read(BASE/'RUNTIME_LOCK.json')['pins'] if f not in files}
    for f,h in dep_pins.items():assert sha(CHECKOUT/f)==h,('Public dependency mismatch',f)
    write(PUB/'CODE_BINDING.json',dict(status='pass',new_code=source_pins,existing_dependency_pins=dep_pins,
        runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),GT_or_payload_export=False,
        protected_private_registry_hashes=read(BASE/'RUNTIME_LOCK.json')['protected_registries'],time=time.time()))
    files += [str(f.relative_to(ROOT)) for f in PUB.iterdir() if f.is_file()]
    assert len(set(files))==len(files)
    for f in files:
        assert not f.startswith(('artifacts/','data/','checkpoints/'))
        dst=CHECKOUT/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dst)
    release_files=[]
    for f in RELEASE.rglob('*'):
        if f.is_file():
            rel=str(f.relative_to(RELEASE));dst=CHECKOUT/rel;dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(f,dst);release_files.append(rel)
    readme=CHECKOUT/'README.md';text=readme.read_text()
    addition='''\n## DeCoTA C1–Scale06 online：最新锁定研究入口\n\n[正式 online 版本与代码导航](docs/DECOTA_C1_ONLINE_RELEASE.md)；[空间锁定配置](methods/C1_FINAL_RESEARCH_CONFIG.json)；[原 NLL＋hinge 时间状态](methods/C1_TEMPORAL_RESEARCH_STATUS.json)。\n\nC1 online 每个 query 重置残差和 Adam，只以 1/16 写回空间 LN；原时间适应仍逐 query 丢弃。更早的 `decota_final_simplified_v1` 是 episodic 路径，不能替代 online 状态继承。两份历史 CURRENT 登记保持不变。\n\n[同域 corruption online 评估](docs/TA_DECOTA_C1_SAME_DOMAIN_REVIEW.md)与[新增 Spatial-DeCoTA Direct/critic P0](docs/TA_DECOTA_CRITIC_P0_REVIEW.md)分别列出实际配置、正负结果及审计。P0 是无时间适应、无 LN 继承的当前空间纠错对照，不自动晋升正式配置。\n'''
    assert '## DeCoTA C1–Scale06 online：最新锁定研究入口' not in text
    readme.write_text(text+addition);files+=release_files+['README.md']
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True)
    assert set(git('diff','--cached','--name-only').splitlines())==set(files)
    staged=[]
    for f in files:
        value=(CHECKOUT/f).read_bytes()
        if f.endswith(('.json','.md')):
            assert b'/home/wwww/.codex/attachments/' not in value,f
            if f.endswith('.json'):assert b'"video_path"' not in value and b'"caption"' not in value,f
        staged.append(dict(path=f,bytes=len(value),sha256=hashlib.sha256(value).hexdigest(),blob_sha=git('hash-object',f)))
    write(BASE/'PUBLIC_STAGE.json',dict(status='reviewed_staged',repository='Zonglin-He/A',
        base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),
        expected_tree=git('write-tree'),files=staged,file_count=len(staged),bytes=sum(f['bytes'] for f in staged),
        private_media_GT_weights_raw_caches_excluded=True,time=time.time()))
    print('STAGED',len(files),sum(f['bytes'] for f in staged),flush=True)


def fetch(url):
    req=urllib.request.Request(url,headers={'User-Agent':'STVG-public-byte-audit'})
    last=None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req,timeout=60) as r:return r.read()
        except Exception as e:last=e;time.sleep(1+attempt)
    raise last


def verify_remote(commit):
    staged=read(BASE/'PUBLIC_STAGE.json');assert staged['status']=='reviewed_staged'
    ref=json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/ref/heads/main'))
    assert ref['object']['sha']==commit
    meta=json.loads(fetch(f'https://api.github.com/repos/Zonglin-He/A/git/commits/{commit}'))
    assert meta['parents'][0]['sha']==staged['base_commit'] and meta['tree']['sha']==staged['expected_tree']
    def one(f):
        from urllib.parse import quote
        content=fetch(f"https://raw.githubusercontent.com/Zonglin-He/A/{commit}/{quote(f['path'])}")
        assert content==(CHECKOUT/f['path']).read_bytes() and hashlib.sha256(content).hexdigest()==f['sha256'],f['path']
        blob=hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest();assert blob==f['blob_sha']
        return dict(**f,remote_bytes_identical=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:receipts=list(pool.map(one,staged['files']))
    from scripts.score_audit_tastvg_decota_critic_p0_v1 import public_check
    public_audit=public_check(CHECKOUT/PUB.relative_to(ROOT))
    subprocess.run(['git','fetch','--quiet','origin','main'],cwd=CHECKOUT,check=True)
    assert git('rev-parse','origin/main')==commit
    subprocess.run(['git','update-ref','refs/heads/main',commit,staged['base_commit']],cwd=CHECKOUT,check=True)
    subprocess.run(['git','reset','--mixed',commit],cwd=CHECKOUT,check=True)
    assert not git('status','--porcelain')
    write(BASE/'GITHUB_REMOTE_VERIFICATION.json',dict(status='pass',commit=commit,repository='Zonglin-He/A',
        files=receipts,file_count=len(receipts),bytes=sum(f['bytes'] for f in receipts),remote_ref_verified=True,
        remote_tree_identical=True,remote_blob_bytes_verified=True,remote_public_audit=public_audit,time=time.time()))
    verify()
    write(BASE/'FINAL_COMPLETION.json',dict(status='completed_verified_publication',unique_inputs=576,
        logical_arrivals_per_arm=1152,arms=ARMS,commit=commit,repository='Zonglin-He/A',
        root_audit_checks=read(PUB/'ROOT_AUDIT.json')['checks'],public_audit_checks=public_audit['checks'],
        remote_verified_files=len(receipts),remote_verified_bytes=sum(f['bytes'] for f in receipts),
        episodic_current_only=True,GT_used_for_optimization=False,CURRENT_registries_unchanged=True,
        method_promoted=False,global_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),time=time.time()))
    status(BASE/'STATUS.json',dict(status='completed_verified_publication',commit=commit,time=time.time()))
    estimates=[];s=read(PUB/'SUMMARY.json')
    for ds in DATASETS:
        m=s[ds]['confirm']['corruption']['metrics']
        for arm in ARMS:
            a=m[f'{arm}_delta_v'];estimates.append(f"{ds}/{arm} Δv {100*a['mean']:+.4f}pp[{100*a['ci95'][0]:+.4f},{100*a['ci95'][1]:+.4f}]")
    archive('576唯一两臂/1152逻辑到达全部预测封存后GT计分/独立目标与Adam/官方dense与source bootstrap/两图目检/代码结果远端核验完成；'
        +'；'.join(estimates)+'；GitHub '+commit+'，不晋升；同批补齐C1–Scale06正式online配置/方法文档，私有原锁未改')
    write(BASE/'ARCHIVE_COMPLETION.json',dict(status='pass',workflow='check/snapshot/check',
        ledger_sha256=sha(ROOT/'docs/RESEARCH_HISTORY.md'),log_sha256=sha(BASE/'ARCHIVE.log'),time=time.time()))
    release_root=RELEASE.parent
    write(release_root/'FINAL_COMPLETION.json',dict(status='completed_verified_publication',commit=commit,
        published_configs=['methods/C1_FINAL_RESEARCH_CONFIG.json','methods/C1_TEMPORAL_RESEARCH_STATUS.json'],
        publication_only=True,original_scientific_values_preserved=True,protected_registries_unchanged=True,time=time.time()))
    print('PUBLISHED_VERIFIED',commit,len(receipts),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['stage','verify']);p.add_argument('commit',nargs='?');a=p.parse_args()
    if a.stage=='stage':stage()
    else:verify_remote(a.commit)
