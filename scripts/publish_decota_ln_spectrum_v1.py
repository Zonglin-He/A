"""Exact-byte publication of complete anonymous correction-geometry audit."""
import sys,time,shutil,subprocess,hashlib,json,zlib,base64,gzip,concurrent.futures
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_decota_ln_spectrum_v1 import BASE,PUB,OWN,verify,archive
from scripts.decota_matrix_common_v1 import read,write,status,sha
from scripts.publish_tastvg_decota_critic_p0_v1 import git,fetch,CHECKOUT
from scripts.audit_decota_ln_spectrum_v1 import public

def stage():
    verify();assert read(BASE/'ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert read(PUB/'ROOT_AUDIT.json')['status']==read(PUB/'PUBLIC_AUDIT.json')['status']=='pass';public(PUB)
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    assert not git('status','--porcelain');subprocess.run(['git','fetch','--quiet','origin','main'],cwd=CHECKOUT,check=True);assert git('rev-parse','HEAD')==git('rev-parse','origin/main')
    own=OWN+['docs/TA_DECOTA_LN_SPECTRUM_REVIEW.md'];deps=['scripts/decota_matrix_common_v1.py','scripts/decota_public_result_io_v1.py','scripts/publish_tastvg_decota_critic_p0_v1.py']
    for p in deps:assert sha(CHECKOUT/p)==sha(ROOT/p),p
    write(PUB/'CODE_BINDING.json',dict(code={p:sha(ROOT/p) for p in own},existing_dependencies={p:sha(ROOT/p) for p in deps},predecessor_commit='38810f55a09d2aec1a384a8b585981ac4eec0781',private_weights_states_vectors_media_annotations_exported=False))
    formats={}
    for f in PUB.glob('*.json'):
        if f.stat().st_size<300000:continue
        raw=f.read_bytes();gz=Path(str(f)+'.gz');gz.write_bytes(gzip.compress(raw,compresslevel=9,mtime=0));assert gzip.decompress(gz.read_bytes())==raw
        formats[str(f.relative_to(PUB))]=dict(public_file=str(gz.relative_to(PUB)),uncompressed_bytes=len(raw),uncompressed_sha256=hashlib.sha256(raw).hexdigest())
    write(PUB/'PUBLIC_DATA_FORMAT.json',dict(lossless_files=formats,reader='scripts/decota_public_result_io_v1.py',rows_filtered=False,Gram_is_derived_geometry_not_weight_state=True))
    files=own+[str(f.relative_to(ROOT)) for f in PUB.iterdir() if f.is_file() and str(f.relative_to(PUB)) not in formats]
    for p in files:
        assert not p.startswith(('artifacts/','data/','checkpoints/','model_zoo/'))
        dst=CHECKOUT/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dst)
        if p.endswith(('.json','.md')):
            raw=dst.read_bytes();assert b'/home/wwww/.codex/attachments/' not in raw
            if p.endswith('.json'):assert b'"caption"' not in raw and b'"video_path"' not in raw
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True);assert set(git('diff','--cached','--name-only').splitlines())==set(files)
    ff=[]
    for p in files:
        raw=(CHECKOUT/p).read_bytes();ff.append(dict(path=p,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),blob_sha=git('hash-object',p)))
    m=dict(repository='Zonglin-He/A',base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),expected_tree=git('write-tree'),files=ff,file_count=len(ff),bytes=sum(f['bytes'] for f in ff),time=time.time())
    write(BASE/'PUBLIC_STAGE.json',m);write(BASE/'PUBLIC_TRANSPORT.json',dict(metadata=m,files=[dict(**f,zlib_base64=base64.b64encode(zlib.compress((CHECKOUT/f['path']).read_bytes(),9)).decode()) for f in ff]))
    print('LN_PUBLIC_STAGE',len(ff),m['bytes'],flush=True)

def verify_remote(commit):
    m=read(BASE/'PUBLIC_STAGE.json');ref=json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/ref/heads/main'));assert ref['object']['sha']==commit
    cm=json.loads(fetch(f'https://api.github.com/repos/Zonglin-He/A/git/commits/{commit}'));assert cm['parents'][0]['sha']==m['base_commit'] and cm['tree']['sha']==m['expected_tree']
    def one(f):
        from urllib.parse import quote
        raw=fetch(f"https://raw.githubusercontent.com/Zonglin-He/A/{commit}/{quote(f['path'])}");assert raw==(CHECKOUT/f['path']).read_bytes() and hashlib.sha256(raw).hexdigest()==f['sha256'];return dict(**f,remote_bytes_identical=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rr=list(pool.map(one,m['files']))
    subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(CHECKOUT/'scripts/audit_decota_ln_spectrum_v1.py'),str(CHECKOUT/PUB.relative_to(ROOT))],cwd=CHECKOUT,check=True)
    subprocess.run(['git','fetch','--quiet','origin','main'],cwd=CHECKOUT,check=True);assert git('rev-parse','origin/main')==commit
    subprocess.run(['git','update-ref','refs/heads/main',commit,m['base_commit']],cwd=CHECKOUT,check=True);subprocess.run(['git','reset','--mixed',commit],cwd=CHECKOUT,check=True);assert not git('status','--porcelain')
    write(BASE/'GITHUB_REMOTE_VERIFICATION.json',dict(status='pass',commit=commit,files=rr,file_count=len(rr),bytes=sum(f['bytes'] for f in rr),time=time.time()))
    write(BASE/'FINAL_COMPLETION.json',dict(status='completed_verified_publication',commit=commit,old_payloads_checked=13824,exact_isolated_pairs=read(PUB/'EXTRACTION_AUDIT.json')['pairs'],
        new_model_calls=0,new_expert_calls=0,new_backward_calls=0,new_GT_scoring=0,projected_memory_executed=False,production_promoted=False,decision=read(PUB/'DECISION.json'),remote_verified_files=len(rr),remote_verified_bytes=sum(f['bytes'] for f in rr),time=time.time()))
    status(BASE/'STATUS.json',dict(status='completed_verified_publication',commit=commit,time=time.time()))
    archive('零新推理的全状态LN几何/精确首write效用、10合同/根与公开审计/三图目检完成；GitHub '+commit+'的'+str(len(rr))+'文件逐字节核验通过；未执行或晋升shared memory')

if __name__=='__main__':stage() if sys.argv[1]=='stage' else verify_remote(sys.argv[2])
