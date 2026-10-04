"""Verified public export, invoked only after root visual/result review."""
import sys, time, json, shutil, subprocess, hashlib, base64
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.tastvg_decota_c1_common_v1 import *


def shell(args,cwd,**kwargs):
    return subprocess.check_output(args,cwd=cwd,**kwargs).decode().strip()


def run():
    verify_seal()
    assert read(BASE/'ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert read(PUB/'ROOT_AUDIT.json')['status']=='pass'
    from scripts.score_audit_tastvg_decota_c1_same_domain_v1 import public_check
    audit=public_check(PUB);write(PUB/'PUBLIC_AUDIT.json',audit)
    checkout=Path('/home/wwww/visual-grounding-public-A')
    assert shell(['git','remote','get-url','origin'],checkout)=='https://github.com/Zonglin-He/A.git'
    assert not shell(['git','status','--porcelain'],checkout),'Public checkout has unrelated edits'
    subprocess.run(['git','fetch','--quiet','origin','main'],cwd=checkout,check=True)
    assert shell(['git','rev-parse','HEAD'],checkout)==shell(['git','rev-parse','origin/main'],checkout)
    files=['protocols/tastvg_decota_c1_same_domain_v1.md','vg_tta/tastvg_decota_c1_same_domain_v1.py',
        'scripts/tastvg_decota_c1_common_v1.py','scripts/run_tastvg_decota_c1_same_domain_v1.py',
        'scripts/continue_tastvg_decota_c1_same_domain_v1.py','scripts/score_audit_tastvg_decota_c1_same_domain_v1.py',
        'scripts/report_tastvg_decota_c1_same_domain_v1.py','scripts/publish_tastvg_decota_c1_same_domain_v1.py',
        'docs/TA_DECOTA_C1_SAME_DOMAIN_REVIEW.md']
    dependencies={f:h for f,h in verify()['pins'].items() if f.endswith('.py') and f not in files}
    for f,h in dependencies.items():assert sha(checkout/f)==h,f
    source_pin={f:sha(ROOT/f) for f in files}
    binding=dict(status='pass',new_code=source_pin,existing_dependency_pins=dependencies,
        CPU_pins=read(BASE/'CPU_RUNTIME_LOCK.json')['pins'],scope='Public implementation and anonymous metrics only',
        runtime_lock_sha256=sha(BASE/'RUNTIME_LOCK.json'),science_revisions={f.name:read(f) for f in (BASE/'revisions').glob('*.json')},
        no_media_GT_box_parameter_or_feature_export=True,time=time.time())
    write(PUB/'CODE_BINDING.json',binding)
    files+= [str(f.relative_to(ROOT)) for f in PUB.iterdir() if f.is_file()]
    assert all(not f.startswith(('artifacts/','data/','checkpoints/','.cache/')) for f in files)
    for rel in files:
        target=checkout/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
    subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',
        str(checkout/'scripts/score_audit_tastvg_decota_c1_same_domain_v1.py'),'public',str(checkout/PUB.relative_to(ROOT))],cwd=checkout,check=True)
    subprocess.run(['git','add','--',*files],cwd=checkout,check=True)
    staged=shell(['git','diff','--cached','--name-only'],checkout).splitlines()
    assert set(staged)==set(files),('Unexpected staged paths',staged)
    subprocess.run(['git','commit','-m','Evaluate locked C1 online DeCoTA against Frozen on same-domain corruption'],cwd=checkout,check=True)
    commit=shell(['git','rev-parse','HEAD'],checkout)
    subprocess.run(['git','push','origin','HEAD:main'],cwd=checkout,check=True)
    remote=json.loads(shell(['gh','api','repos/Zonglin-He/A/git/ref/heads/main'],checkout))
    assert remote['object']['sha']==commit
    tree=json.loads(shell(['gh','api',f'repos/Zonglin-He/A/git/trees/{commit}?recursive=1'],checkout))
    entries={x['path']:x for x in tree['tree']};receipts=[]
    for rel in files:
        entry=entries[rel];blob=json.loads(shell(['gh','api',f"repos/Zonglin-He/A/git/blobs/{entry['sha']}"],checkout))
        assert blob['encoding']=='base64'
        content=base64.b64decode(blob['content']);expected=(ROOT/rel).read_bytes();assert content==expected,rel
        receipts.append(dict(path=rel,bytes=len(content),sha256=hashlib.sha256(content).hexdigest(),
            blob_sha=entry['sha'],remote_bytes_identical=True))
    remote_result=public_check(checkout/PUB.relative_to(ROOT))
    write(BASE/'GITHUB_REMOTE_VERIFICATION.json',dict(status='pass',repository='Zonglin-He/A',commit=commit,
        files=receipts,file_count=len(receipts),bytes=sum(r['bytes'] for r in receipts),
        remote_ref_verified=True,remote_blob_bytes_verified=True,remote_public_audit=remote_result,
        public_checkout_clean=not bool(shell(['git','status','--porcelain'],checkout)),time=time.time()))
    status(BASE/'STATUS.json',dict(status='completed_verified_publication',arrivals=1152,commit=commit,time=time.time()))
    write(BASE/'FINAL_COMPLETION.json',dict(status='completed_verified_publication',arrivals=1152,
        datasets=DATASETS,commit=commit,repository='Zonglin-He/A',root_audit_checks=read(PUB/'ROOT_AUDIT.json')['checks'],
        public_audit_checks=audit['checks'],remote_verified_files=len(receipts),
        remote_verified_bytes=sum(r['bytes'] for r in receipts),GT_used_for_optimization=False,
        original_online_C1_Scale06=True,CURRENT_METHOD_unchanged=True,method_promoted=False,
        global_barrier_sha256=sha(BASE/'GLOBAL_PREDICTION_BARRIER.json'),time=time.time()))
    summary=read(PUB/'SUMMARY.json');facts=[]
    for ds in DATASETS:
        for split in ['search','confirm']:
            s=summary[ds][split]['corruption'];m=s['metrics'];delta=m['delta_v']
            facts.append(f"{ds}/{split} Frozen {100*m['frozen_v']['mean']:.4f}%→DeCoTA {100*m['decota_v']['mean']:.4f}%, paired delta {100*delta['mean']:+.4f}pp[{100*delta['ci95'][0]:+.4f},{100*delta['ci95'][1]:+.4f}], >5pp harm {s['harm_gt5pp']}/{s['cells']}")
    archive('1152实际到达/全GT评分/独立状态与算术审计/两图目检/匿名代码结果远端逐字节核验全部完成；corrupt源宏平均/10000 paired-source bootstrap(seed20261004)：'
        +'；'.join(facts)+'；GitHub '+commit+'，不晋升。全报告docs/TA_DECOTA_C1_SAME_DOMAIN_REVIEW.md和results/tastvg_decota_c1_same_domain/2026-10-04已公开')
    write(BASE/'ARCHIVE_COMPLETION.json',dict(status='pass',workflow='check/snapshot/check',
        ledger_sha256=sha(ROOT/'docs/RESEARCH_HISTORY.md'),log_sha256=sha(BASE/'ARCHIVE.log'),time=time.time()))
    print('PUBLISHED_VERIFIED',commit,len(receipts),flush=True)


if __name__=='__main__':run()
