"""Bounded serial deployment handoff publication, not timing completion."""
import ast,hashlib,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.publish_stvg_opd_actionability_audit_v1 import git,CHECKOUT,BRANCH,put,safe,private_attachment_path,receipt_metadata
BASE=ROOT/'artifacts/stvg_opd_claim_strengthening_v1/deployment_latency/after_expanded_priority002'
OUT=ROOT/'results/stvg_opd_claim_strengthening/2026-10-13/deployment_priority002_handoff'
def run(qualification=False):
    from scripts.run_stvg_opd_deployment_priority002 import verify
    verify();assert read(BASE/'ROOT_SERIAL_HANDOFF_READBACK.json')['status']=='pass'
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    base=git('rev-parse','HEAD');assert base==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    unrelated=git('ls-files','--others','--exclude-standard').splitlines();owned=set()
    def copy(p,rel):
        if p.suffix=='.json':safe(read(p))
        elif p.suffix in {'.py','.md'}:
            text=p.read_text();assert not private_attachment_path(text)
            if p.suffix=='.py':ast.parse(text)
        dest=CHECKOUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);assert sha(dest)==sha(p);owned.add(rel)
    files=['protocols/stvg_opd_deployment_priority002.md','scripts/run_stvg_opd_deployment_priority002.py',
        'scripts/continue_stvg_opd_deployment_priority002.py','scripts/publish_stvg_opd_deployment_priority002_v1.py',
        'scripts/run_stvg_opd_deployment_latency_v1.py','scripts/finalize_stvg_opd_deployment_latency_v1.py','vg_tta/stvg_opd_deployment_profile_v1.py']
    files+= [n for n in read(BASE/'REVISION_RUNTIME.json')['pins'] if n.startswith('vg_tta/')]
    for n in sorted(set(files)):copy(ROOT/n,n)
    names=['SERIAL_HANDOFF_SCOPE.json','CPU_STORAGE_CONTRACTS.json','REVISION_RUNTIME.json','BASELINE_STANDBY_IDENTITY_RECEIPT.json',
              'LAUNCH.json','ARCHIVE_HANDOFF_RECEIPT.json','ROOT_SERIAL_HANDOFF_READBACK.json']
    if qualification:
        q=read(BASE.parent/'ROOT_QUALIFICATION_READBACK.json')
        assert q['status']=='pass' and q['qualification_unique_queries']==32 and q['actual_GPU_fits']==64
        names+=['ROOT_PRIORITY_QUALIFICATION_READBACK.json']
        p=BASE.parent/'ROOT_QUALIFICATION_READBACK.json'
        put(OUT/'receipts'/'ORIGINAL_ROOT_QUALIFICATION_READBACK.json',dict(original_private_receipt_sha256=sha(p),projection=receipt_metadata(read(p))))
        copy(OUT/'receipts'/'ORIGINAL_ROOT_QUALIFICATION_READBACK.json',str((OUT/'receipts'/'ORIGINAL_ROOT_QUALIFICATION_READBACK.json').relative_to(ROOT)))
    for n in names:
        p=BASE/n;put(OUT/'receipts'/n,dict(original_private_receipt_sha256=sha(p),projection=receipt_metadata(read(p))))
        copy(OUT/'receipts'/n,str((OUT/'receipts'/n).relative_to(ROOT)))
    manifest=OUT/('SOURCE_MANIFEST_QUALIFICATION.json' if qualification else 'SOURCE_MANIFEST.json')
    scope=('bounded original32-query/64-complete-GPU-fit qualification/root/native/expert/trace-off and source/serial handoff; not all128 timed paths or whole C completion' if qualification else
        'bounded deployment priority scheduling/source/CPU contracts/actual launch/archive handoff only; no complete qualification/timing claim')
    write(manifest,dict(status='pass',scope=scope+'; excludes self hash',
        files={n:dict(sha256=sha(CHECKOUT/n),bytes=(CHECKOUT/n).stat().st_size) for n in sorted(owned)}));copy(manifest,str((OUT/'SOURCE_MANIFEST.json').relative_to(ROOT)))
    entries=[]
    for n in sorted(owned):
        p=CHECKOUT/n;data=p.read_bytes();entries.append(dict(path=n,bytes=len(data),sha256=sha(p),blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
    with tempfile.TemporaryDirectory(prefix='deployment-priority-index-') as folder:
        env={**os.environ,'GIT_INDEX_FILE':str(Path(folder)/'index')};git('read-tree',base,env=env)
        for e in entries:
            h=git('hash-object','-w',str(CHECKOUT/e['path']));assert h==e['blob_sha'];git('update-index','--add','--cacheinfo','100644',h,e['path'],env=env)
        tree=git('write-tree',env=env)
    write(BASE/('QUALIFICATION_PUBLIC_STAGE.json' if qualification else 'PUBLIC_STAGE.json'),dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,base_commit=base,expected_tree=tree,
        files=entries,file_count=len(entries),bytes=sum(e['bytes'] for e in entries),unrelated_untracked_preserved=unrelated,
        scope=scope,time=time.time()))
    print('DEPLOYMENT_PRIORITY_BOUNDED_PUBLIC_STAGE',len(entries),sum(e['bytes'] for e in entries),tree)
if __name__=='__main__':run(len(sys.argv)>1 and sys.argv[1]=='qualification')
