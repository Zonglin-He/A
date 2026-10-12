"""Bounded original LN-only priority handoff; no completion or scientific-score claim."""
import ast,hashlib,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.publish_stvg_opd_actionability_audit_v1 import git,CHECKOUT,BRANCH,put,safe,private_attachment_path,receipt_metadata
BASE=ROOT/'artifacts/stvg_opd_mechanism_review_v1/cross_parent_LN/after_C_priority001'
OUT=ROOT/'results/stvg_opd_mechanism_review/2026-10-13/LN_after_C_priority001_handoff'
def run():
    from scripts.run_stvg_opd_LN_after_C_priority001 import verify
    verify();assert read(BASE/'ROOT_SERIAL_HANDOFF_READBACK.json')['status']=='pass'
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    base=git('rev-parse','HEAD');assert base==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    unrelated=git('ls-files','--others','--exclude-standard').splitlines();owned=set()
    def copy(p,rel):
        if p.suffix=='.json':safe(read(p))
        else:
            text=p.read_text();assert not private_attachment_path(text)
            if p.suffix=='.py':ast.parse(text)
        dest=CHECKOUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);assert sha(p)==sha(dest);owned.add(rel)
    sources=['protocols/stvg_opd_mechanism_review_v1.md','protocols/stvg_opd_LN_after_C_priority001.md',
        'scripts/stvg_opd_mechanism_LN_order_common_v1.py','scripts/run_stvg_opd_mechanism_LN_order_v1.py',
        'scripts/score_stvg_opd_mechanism_LN_order_v1.py','scripts/finalize_stvg_opd_mechanism_LN_order_v1.py',
        'scripts/run_stvg_opd_LN_after_C_priority001.py','scripts/continue_stvg_opd_LN_after_C_priority001.py','scripts/publish_stvg_opd_LN_after_C_priority001_v1.py']
    for n in sources:copy(ROOT/n,n)
    for n in ['SERIAL_HANDOFF_SCOPE.json','CPU_STORAGE_CONTRACTS.json','REVISION_RUNTIME.json','BASELINE_STANDBY_IDENTITY_RECEIPT.json','LAUNCH.json','ARCHIVE_HANDOFF_RECEIPT.json','ROOT_SERIAL_HANDOFF_READBACK.json']:
        p=BASE/n;put(OUT/'receipts'/n,dict(original_private_receipt_sha256=sha(p),projection=receipt_metadata(read(p))));copy(OUT/'receipts'/n,str((OUT/'receipts'/n).relative_to(ROOT)))
    m=OUT/'SOURCE_MANIFEST.json';write(m,dict(status='pass',scope='bounded original LN-only serial scheduling/source/real CPU storage hash contracts/actual launch/archive handoff; excludes self hash and scientific payloads; not24-GPU qualification or768 formal completion',
        files={n:dict(sha256=sha(CHECKOUT/n),bytes=(CHECKOUT/n).stat().st_size) for n in sorted(owned)}));copy(m,str(m.relative_to(ROOT)))
    entries=[]
    for n in sorted(owned):
        p=CHECKOUT/n;d=p.read_bytes();entries.append(dict(path=n,bytes=len(d),sha256=sha(p),blob_sha=hashlib.sha1(b'blob '+str(len(d)).encode()+b'\0'+d).hexdigest()))
    with tempfile.TemporaryDirectory(prefix='LN-priority-handoff-index-') as folder:
        env={**os.environ,'GIT_INDEX_FILE':str(Path(folder)/'index')};git('read-tree',base,env=env)
        for e in entries:
            h=git('hash-object','-w',str(CHECKOUT/e['path']));assert h==e['blob_sha'];git('update-index','--add','--cacheinfo','100644',h,e['path'],env=env)
        tree=git('write-tree',env=env)
    write(BASE/'PUBLIC_STAGE.json',dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,base_commit=base,expected_tree=tree,files=entries,file_count=len(entries),bytes=sum(e['bytes'] for e in entries),
        unrelated_untracked_preserved=unrelated,scope='bounded original LN-only source/CPU exact expert storage hashes/new serial handoff after genuine whole C FINAL/actual finite launch/archive only; actual24 qualifications/768 formal/root/view/public/final still required',time=time.time()))
    print('LN_ORIGINAL_SERIAL_PUBLIC_HANDOFF_STAGE',len(entries),sum(e['bytes'] for e in entries),tree)
if __name__=='__main__':run()
