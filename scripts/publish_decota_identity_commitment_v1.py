"""Reviewed code and anonymous complete results only; exact remote verification."""
import sys,time,shutil,subprocess,hashlib,json,zlib,base64,gzip,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_identity_common_v1 import *
from scripts.publish_tastvg_decota_critic_p0_v1 import git,fetch,CHECKOUT
from scripts.audit_decota_identity_public_v1 import run as public_audit

def stage():
    verify();assert read(BASE/'ROOT_VISUAL_REVIEW.json')['status']=='pass'
    for s in ['matched','online']:assert read(PUB/s/'ROOT_AUDIT.json')['status']=='pass' and read(PUB/s/'PUBLIC_AUDIT.json')['status']=='pass'
    assert read(BASE/'duration/COMPLETION.json')['status']=='completed_pending_root_review';public_audit(PUB)
    assert not git('status','--porcelain');subprocess.run(['git','fetch','--quiet','origin','main'],cwd=CHECKOUT,check=True);assert git('rev-parse','HEAD')==git('rev-parse','origin/main')
    own=['vg_tta/decota_identity_commitment_v1.py','scripts/decota_identity_common_v1.py','scripts/run_decota_identity_commitment_v1.py','scripts/test_decota_identity_commitment_v1.py',
        'scripts/continue_decota_identity_commitment_v1.py','scripts/decota_duration_bias_v1.py','scripts/score_decota_identity_commitment_v1.py','scripts/report_decota_identity_commitment_v1.py','scripts/audit_decota_identity_public_v1.py',
        'scripts/publish_decota_identity_commitment_v1.py','scripts/derive_decota_identity_persistence_v1.py','protocols/decota_identity_commitment_v1.md','docs/decota_identity_commitment_v1/EXECUTION.md','docs/TA_DECOTA_IDENTITY_COMMITMENT_REVIEW.md']
    deps={p:h for p,h in read(BASE/'RUNTIME_LOCK.json')['pins'].items() if p not in own}
    for p,h in deps.items():assert sha(CHECKOUT/p)==h,('Public dependency mismatch',p)
    write(PUB/'CODE_BINDING.json',dict(code={p:sha(ROOT/p) for p in own},existing_dependencies=deps,predecessor_commit='03b8830f12ef6dca24e97f09e6b5cbec6b562593',private_payload_export=False))
    formats={}
    for f in PUB.rglob('*.json'):
        if f.name not in ['ROWS.json','DIAGNOSTICS.json','OBSERVATION_QUALITY.json','MATCHED_PERSISTENCE_ROWS.json']:continue
        raw=f.read_bytes();gz=Path(str(f)+'.gz');gz.write_bytes(gzip.compress(raw,compresslevel=9,mtime=0));assert gzip.decompress(gz.read_bytes())==raw
        formats[str(f.relative_to(PUB))]=dict(public_file=str(gz.relative_to(PUB)),uncompressed_bytes=len(raw),uncompressed_sha256=hashlib.sha256(raw).hexdigest())
    write(PUB/'PUBLIC_DATA_FORMAT.json',dict(lossless_files=formats,reader='scripts/decota_public_result_io_v1.py',rows_filtered=False))
    files=own+[str(f.relative_to(ROOT)) for f in PUB.rglob('*') if f.is_file() and str(f.relative_to(PUB)) not in formats]
    for p in files:
        assert not p.startswith(('artifacts/','data/','model_zoo/'))
        dst=CHECKOUT/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dst)
        if p.endswith(('.json','.md')):
            raw=dst.read_bytes();assert b'/home/wwww/.codex/attachments/' not in raw
            if p.endswith('.json'):assert b'"caption"' not in raw and b'"video_path"' not in raw
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True);assert set(git('diff','--cached','--name-only').splitlines())==set(files)
    ff=[]
    for p in files:
        raw=(CHECKOUT/p).read_bytes();ff.append(dict(path=p,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),blob_sha=git('hash-object',p)))
    meta=dict(repository='Zonglin-He/A',base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),expected_tree=git('write-tree'),files=ff,file_count=len(ff),bytes=sum(f['bytes'] for f in ff),time=time.time())
    write(BASE/'PUBLIC_STAGE.json',meta);write(BASE/'PUBLIC_TRANSPORT.json',dict(metadata=meta,files=[dict(**f,zlib_base64=base64.b64encode(zlib.compress((CHECKOUT/f['path']).read_bytes(),9)).decode()) for f in ff]))
    print('IDENTITY_PUBLIC_STAGE',len(ff),meta['bytes'],flush=True)

def verify_remote(commit):
    meta=read(BASE/'PUBLIC_STAGE.json');ref=json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/ref/heads/main'));assert ref['object']['sha']==commit
    cm=json.loads(fetch(f'https://api.github.com/repos/Zonglin-He/A/git/commits/{commit}'));assert cm['parents'][0]['sha']==meta['base_commit'] and cm['tree']['sha']==meta['expected_tree']
    def one(f):
        from urllib.parse import quote
        raw=fetch(f"https://raw.githubusercontent.com/Zonglin-He/A/{commit}/{quote(f['path'])}");assert raw==(CHECKOUT/f['path']).read_bytes() and hashlib.sha256(raw).hexdigest()==f['sha256'];return dict(**f,remote_bytes_identical=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rr=list(pool.map(one,meta['files']))
    subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(CHECKOUT/'scripts/audit_decota_identity_public_v1.py'),str(CHECKOUT/PUB.relative_to(ROOT))],cwd=CHECKOUT,check=True)
    subprocess.run(['git','fetch','--quiet','origin','main'],cwd=CHECKOUT,check=True);assert git('rev-parse','origin/main')==commit
    subprocess.run(['git','update-ref','refs/heads/main',commit,meta['base_commit']],cwd=CHECKOUT,check=True);subprocess.run(['git','reset','--mixed',commit],cwd=CHECKOUT,check=True);assert not git('status','--porcelain')
    write(BASE/'GITHUB_REMOTE_VERIFICATION.json',dict(status='pass',commit=commit,files=rr,file_count=len(rr),bytes=sum(f['bytes'] for f in rr),time=time.time()))
    write(BASE/'FINAL_COMPLETION.json',dict(status='completed_verified_publication',commit=commit,matched_predictions=1152,matched_rows=3456,online_predictions=13824,duration_source_condition_cells=270,duration_unique_source_pixels=250,
        selection=read(BASE/'SEARCH_SELECTION.json'),new_expert=0,new_backbone=0,temporal_adaptation_started=False,whole_official_dataset=False,production_promoted=False,remote_verified_files=len(rr),remote_verified_bytes=sum(f['bytes'] for f in rr),time=time.time()))
    status(BASE/'STATUS.json',dict(status='completed_verified_publication',commit=commit,time=time.time()))
    archive('三臂1152匹配预测/3456指标、13824独立流逻辑预测及270缓存输入CPU时长诊断全部根审计/公开复算/三图目检；研究接口'+read(BASE/'SEARCH_SELECTION.json')['arm']+'，Native时间；GitHub '+commit+'逐文件核验完成，未晋升生产')

if __name__=='__main__':stage() if sys.argv[1]=='stage' else verify_remote(sys.argv[2])
