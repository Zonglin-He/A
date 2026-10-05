"""Exact-byte reviewed GitHub export of two-P0 code and anonymous outcomes."""
import sys,time,shutil,subprocess,hashlib,json,zlib,base64,gzip,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_transform_common_v1 import *
from scripts.publish_tastvg_decota_critic_p0_v1 import git,fetch,CHECKOUT

def stage():
    verify();assert read(PUB/'ROOT_AUDIT.json')['status']==read(PUB/'PUBLIC_AUDIT.json')['status']=='pass'
    assert read(BASE/'ROOT_VISUAL_REVIEW.json')['status']=='pass'
    decision=read(PUB/'DECISION.json');assert 'pending' not in decision['conditional_temporal514'] and 'pending' not in decision['conditional_acceptance']
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git' and not git('status','--porcelain')
    subprocess.run(['git','fetch','--quiet','origin','main'],cwd=CHECKOUT,check=True);assert git('rev-parse','HEAD')==git('rev-parse','origin/main')
    own=OWN+['scripts/continue_decota_transform_cpu_v1.py','scripts/score_decota_transform_p0_v1.py','scripts/audit_decota_transform_p0_v1.py','scripts/report_decota_transform_p0_v1.py',
        'scripts/publish_decota_transform_p0_v1.py','docs/decota_transform_p0_v1/EXECUTION.md','docs/TA_DECOTA_TRANSFORM_P0_REVIEW.md']
    deps=['scripts/decota_matrix_common_v1.py','scripts/decota_identity_common_v1.py','scripts/tastvg_decota_c1_common_v1.py',
        'scripts/decota_public_result_io_v1.py','scripts/publish_tastvg_decota_critic_p0_v1.py','scripts/run_spatial_ssl_gpu_v1.py',
        'scripts/run_tastvg_decota_c1_same_domain_v1.py','scripts/run_tastvg_full_b1_experts_v1.py','scripts/score_decota_optimizer_posterior_v1.py',
        'vg_tta/tastvg_oracle_event5_v1.py','vg_tta/tastvg_decota_c1_same_domain_v1.py','vg_tta/exact_frame_decode_audit_v2.py','vg_tta/tastvg_paper48_hc2_decode_v1.py','vg_tta/tastvg_deployment_corruption_v2.py']
    deps += [str(f.relative_to(ROOT)) for f in (ROOT/'methods/decota_final_simplified_v1').glob('*.py')]
    # Export dependency if earlier public workflow did not include it; never edit it.
    newdeps=[]
    for p in deps:
        if (CHECKOUT/p).exists():assert sha(CHECKOUT/p)==sha(ROOT/p),p
        else:newdeps.append(p)
    own+=newdeps
    write(PUB/'CODE_BINDING.json',dict(code={p:sha(ROOT/p) for p in own},existing_dependencies={p:sha(ROOT/p) for p in deps if p not in newdeps},
        predecessor_commit='53c76f7affd3c6ffed21c435d92007f8c868db89',private_RGB_queries_GT_H_weights_states_exported=False))
    formats={}
    for f in PUB.rglob('*.json'):
        if f.stat().st_size<300000:continue
        raw=f.read_bytes();gz=Path(str(f)+'.gz');gz.write_bytes(gzip.compress(raw,9,mtime=0));assert gzip.decompress(gz.read_bytes())==raw
        formats[str(f.relative_to(PUB))]=dict(public_file=str(gz.relative_to(PUB)),uncompressed_bytes=len(raw),uncompressed_sha256=hashlib.sha256(raw).hexdigest())
    write(PUB/'PUBLIC_DATA_FORMAT.json',dict(lossless_files=formats,reader='scripts/decota_public_result_io_v1.py',rows_filtered=False))
    files=own+[str(f.relative_to(ROOT)) for f in PUB.rglob('*') if f.is_file() and str(f.relative_to(PUB)) not in formats]
    for p in files:
        assert not p.startswith(('artifacts/','data/','checkpoints/','model_zoo/'))
        dst=CHECKOUT/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dst)
        if p.endswith(('.json','.md')):
            raw=dst.read_bytes();assert b'/home/wwww/.codex/attachments/' not in raw
            if p.endswith('.json'):assert b'"caption"' not in raw and b'"video_path"' not in raw
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True);assert set(git('diff','--cached','--name-only').splitlines())==set(files)
    rr=[]
    for p in files:
        raw=(CHECKOUT/p).read_bytes();rr.append(dict(path=p,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),blob_sha=git('hash-object',p)))
    meta=dict(repository='Zonglin-He/A',base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),expected_tree=git('write-tree'),files=rr,file_count=len(rr),bytes=sum(r['bytes'] for r in rr),time=time.time())
    write(BASE/'PUBLIC_STAGE.json',meta);write(BASE/'PUBLIC_TRANSPORT.json',dict(metadata=meta,files=[dict(**r,zlib_base64=base64.b64encode(zlib.compress((CHECKOUT/r['path']).read_bytes(),9)).decode()) for r in rr]))
    print('TRANSFORM_PUBLIC_STAGE',len(rr),meta['bytes'],flush=True)

def verify_remote(commit):
    meta=read(BASE/'PUBLIC_STAGE.json');ref=json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/ref/heads/main'));assert ref['object']['sha']==commit
    cm=json.loads(fetch(f'https://api.github.com/repos/Zonglin-He/A/git/commits/{commit}'));assert cm['parents'][0]['sha']==meta['base_commit'] and cm['tree']['sha']==meta['expected_tree']
    def one(f):
        from urllib.parse import quote
        raw=fetch(f"https://raw.githubusercontent.com/Zonglin-He/A/{commit}/{quote(f['path'])}");assert raw==(CHECKOUT/f['path']).read_bytes() and hashlib.sha256(raw).hexdigest()==f['sha256'];return dict(**f,remote_bytes_identical=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rr=list(pool.map(one,meta['files']))
    subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(CHECKOUT/'scripts/audit_decota_transform_p0_v1.py'),str(CHECKOUT/PUB.relative_to(ROOT))],cwd=CHECKOUT,check=True)
    subprocess.run(['git','fetch','--quiet','origin','main'],cwd=CHECKOUT,check=True);assert git('rev-parse','origin/main')==commit
    subprocess.run(['git','update-ref','refs/heads/main',commit,meta['base_commit']],cwd=CHECKOUT,check=True);subprocess.run(['git','reset','--mixed',commit],cwd=CHECKOUT,check=True);assert not git('status','--porcelain')
    write(BASE/'GITHUB_REMOTE_VERIFICATION.json',dict(status='pass',commit=commit,files=rr,file_count=len(rr),bytes=sum(f['bytes'] for f in rr),time=time.time()))
    write(BASE/'FINAL_COMPLETION.json',dict(status='completed_verified_publication',commit=commit,unique_inputs=576,logical_cells=1152,
        decision=read(PUB/'DECISION.json'),production_promoted=False,remote_verified_files=len(rr),remote_verified_bytes=sum(f['bytes'] for f in rr),time=time.time()))
    status(BASE/'STATUS.json',dict(status='completed_verified_publication',commit=commit,time=time.time()))
    archive('两P0真实前向/全封存评分/独立审计/三图与正负例完成；GitHub '+commit+' '+str(len(rr))+'文件逐字节核验，不晋升CURRENT')

if __name__=='__main__':stage() if sys.argv[1]=='stage' else verify_remote(sys.argv[2])
