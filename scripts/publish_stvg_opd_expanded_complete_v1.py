"""Publish the complete finite C Expanded-State source, scalar outcomes and view."""
import ast,gzip,hashlib,json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.stvg_opd_expanded_state_common_v1 import BASE,verify
from scripts.audit_stvg_opd_expanded_complete_public_v1 import REL
from scripts.publish_stvg_opd_actionability_audit_v1 import git,CHECKOUT,BRANCH,put,safe,private_attachment_path,receipt_metadata
OUT=ROOT/REL
SOURCES=['protocols/stvg_opd_claim_strengthening_v1.md','protocols/stvg_opd_after_p8_priority001.md',
    'protocols/stvg_opd_after_p8_cancellation_handoff002.md','protocols/stvg_opd_expanded_complete_public_v1.md',
    'scripts/stvg_opd_expanded_state_common_v1.py','scripts/run_stvg_opd_expanded_state_v1.py',
    'scripts/score_stvg_opd_expanded_state_v1.py','scripts/finalize_stvg_opd_expanded_state_v1.py',
    'scripts/run_stvg_opd_expanded_after_p8_priority001.py','scripts/continue_stvg_opd_expanded_after_p8_priority001.py',
    'scripts/pause_stvg_opd_after_p8_priority001.py','scripts/report_stvg_opd_expanded_complete_v1.py',
    'scripts/audit_stvg_opd_expanded_complete_public_v1.py','scripts/publish_stvg_opd_expanded_complete_v1.py',
    'scripts/close_stvg_opd_expanded_complete_v1.py']
def run(closing=False):
    verify();assert read(BASE/'ROOT_COMPLETE_READBACK.json')['status']=='pass'
    view=read(BASE/'ROOT_VISUAL_REVIEW.json');assert view['status']=='pass' and view['actual_pixel_view']
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    base=git('rev-parse','HEAD');assert base==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    unrelated=git('ls-files','--others','--exclude-standard').splitlines();owned=set()
    def copy(source,relative):
        data=source.read_bytes()
        if source.suffix=='.json':safe(read(source))
        elif source.suffix=='.gz':
            with gzip.open(source,'rt') as f:
                if 'jsonl' in source.name:
                    for line in f:safe(json.loads(line))
                elif '.json' in source.name:safe(json.load(f))
        elif source.suffix in {'.py','.md'}:
            assert not private_attachment_path(data.decode())
            if source.suffix=='.py':ast.parse(data.decode())
        dest=CHECKOUT/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        assert sha(dest)==sha(source);owned.add(relative)
    for n in SOURCES:copy(ROOT/n,n)
    for n in ['ROWS.jsonl.gz','DIRECT_CONTRASTS.json.gz','ALL_ARM_STATISTICS.json.gz','ALL_PARENT_CONTRASTS.csv.gz','REPORT.md','DESIGN_PUBLIC.json']:
        copy(OUT/n,str((OUT/n).relative_to(ROOT)))
    if not closing:
        for n in ['GLOBAL_PREDICTION_BARRIER.json','ROOT_QUALIFICATION_READBACK.json','CPU_SCORING_COMPLETION.json',
                  'ROOT_COMPLETE_READBACK.json','CPU_COMPLETION.json','ROOT_ANONYMOUS_SCALAR_READBACK.json','ROOT_VISUAL_REVIEW.json']:
            p=BASE/n;put(OUT/'receipts'/n,dict(original_private_receipt_sha256=sha(p),projection=receipt_metadata(read(p))))
        revision=BASE.parent/'after_P8_priority001'
        for n in ['ROOT_PRIORITY_QUALIFICATION_READBACK.json','ROOT_FIRST_FORMAL_READBACK.json','REVISION_RUNTIME.json']:
            p=revision/n;put(OUT/'receipts'/'priority'/n,dict(original_private_receipt_sha256=sha(p),projection=receipt_metadata(read(p))))
        for relative,b in view['PDF_files'].items():
            p=ROOT/relative;assert sha(p)==b['sha256'] and p.stat().st_size==b['bytes']
            copy(p,str((OUT/'figures'/p.name).relative_to(ROOT)))
        z=subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(CHECKOUT/'scripts/audit_stvg_opd_expanded_complete_public_v1.py'),str(CHECKOUT)],
            capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        assert z.returncode==0,z.stdout+z.stderr;audit=json.loads(z.stdout);assert audit['status']=='pass'
        assert audit==read(OUT/'PUBLIC_SCALAR_AUDIT.json')
        write(BASE/'PUBLIC_SCALAR_AUDIT.json',dict(audit,actual_public_checkout=True,time=time.time()))
    else:
        assert read(BASE/'FINAL_COMPLETION.json')['status']=='complete'
        for n in ['ROOT_CLOSING.json','FINAL_COMPLETION.json','ARCHIVE_VERIFICATION_RECEIPT.json','FINAL_GITHUB_RECEIPT.json']:
            p=BASE/n;put(OUT/'closing'/n,dict(original_private_receipt_sha256=sha(p),projection=receipt_metadata(read(p))))
        previous=read(BASE/'PUBLIC_STAGE.json')
        for e in previous['files']:assert sha(CHECKOUT/e['path'])==e['sha256']
        owned.update(e['path'] for e in previous['files'])
    copy(OUT/'PUBLIC_SCALAR_AUDIT.json',str((OUT/'PUBLIC_SCALAR_AUDIT.json').relative_to(ROOT)))
    for folder in ['receipts','closing']:
        if (OUT/folder).exists():
            for p in (OUT/folder).rglob('*.json'):copy(p,str(p.relative_to(ROOT)))
    rel=str((OUT/'SOURCE_MANIFEST.json').relative_to(ROOT));manifest=OUT/('SOURCE_MANIFEST_CLOSING.json' if closing else 'SOURCE_MANIFEST.json')
    write(manifest,dict(status='pass',scope='complete finite C Expanded-State anonymous outcomes/source/actual pixel-view PDF/full negative tails; excludes self hash',
        files={n:dict(sha256=sha(CHECKOUT/n),bytes=(CHECKOUT/n).stat().st_size) for n in sorted(owned) if n!=rel}));copy(manifest,rel)
    entries=[]
    for n in sorted(owned):
        p=CHECKOUT/n;data=p.read_bytes();entries.append(dict(path=n,bytes=len(data),sha256=sha(p),blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
    with tempfile.TemporaryDirectory(prefix='expanded-complete-index-') as folder:
        env={**os.environ,'GIT_INDEX_FILE':str(Path(folder)/'index')};git('read-tree',base,env=env)
        for e in entries:
            h=git('hash-object','-w',str(CHECKOUT/e['path']));assert h==e['blob_sha'];git('update-index','--add','--cacheinfo','100644',h,e['path'],env=env)
        tree=git('write-tree',env=env)
    write(BASE/('CLOSING_PUBLIC_STAGE.json' if closing else 'PUBLIC_STAGE.json'),dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,
        base_commit=base,expected_tree=tree,files=entries,file_count=len(entries),bytes=sum(e['bytes'] for e in entries),
        unrelated_untracked_preserved=unrelated,scope='complete finite C 5430 Expanded-State fits/905 historical non-development parents/two orders/three retention rules/full anonymous positive-negative-neutral-unavailable scalar outcomes/source/actual PDF pixel view; not deployment latency/LN/cost/overall completion',time=time.time()))
    print(json.dumps(dict(status='reviewed_staged',closing=closing,files=len(entries),bytes=sum(e['bytes'] for e in entries),base_commit=base,expected_tree=tree)))
if __name__=='__main__':run(len(sys.argv)>1 and sys.argv[1]=='closing')
