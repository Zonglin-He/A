"""Exact full anonymous P9-P0 results and source export, excluding raw states."""
import ast,csv,gzip,hashlib,json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_state_conditional_actionability_p0_v1 import BASE,OUT,read,write,sha,verify,rows,EPS
from scripts.publish_stvg_opd_actionability_audit_v1 import git,CHECKOUT,BRANCH,put,safe,private_attachment_path,receipt_metadata
def run(closing=False):
    verify();assert all(read(BASE/n)['status']=='pass' for n in ['ROOT_READBACK.json','ROOT_VISUAL_REVIEW.json','empty_episode_readback001/RESOLUTION_RECEIPT.json'])
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git';subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    base=git('rev-parse','HEAD');assert base==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    unrelated=git('ls-files','--others','--exclude-standard').splitlines();owned=set()
    def copy(source,relative):
        data=source.read_bytes()
        if source.suffix=='.json':safe(read(source))
        elif source.suffix=='.gz':
            with gzip.open(source,'rt') as f:
                if 'SUMMARY' in source.name:safe(json.load(f))
                else:
                    for line in f:safe(json.loads(line))
        elif source.suffix in ['.py','.md']:
            assert not private_attachment_path(data.decode())
            if source.suffix=='.py':ast.parse(data.decode())
        dest=CHECKOUT/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);assert sha(source)==sha(dest);owned.add(relative)
    sources=['protocols/stvg_opd_state_conditional_actionability_p0_v1.md','scripts/stvg_opd_state_conditional_actionability_p0_v1.py','scripts/audit_stvg_opd_state_conditional_actionability_p0_v1.py','scripts/continue_stvg_opd_state_conditional_actionability_p0_v1.py','scripts/continue_stvg_opd_state_conditional_empty001.py','scripts/report_stvg_opd_state_conditional_actionability_p0_v1.py','scripts/publish_stvg_opd_state_conditional_actionability_p0_v1.py','scripts/close_stvg_opd_state_conditional_actionability_p0_v1.py']
    for n in sources:copy(ROOT/n,n)
    for n in ['ROWS.jsonl.gz','REPORT.md','ALL_PARENT_STATE_PAIRS.csv','ALL_SIGN_REVERSALS.csv']:copy(OUT/n,str((OUT/n).relative_to(ROOT)))
    rr=rows(OUT/'ROWS.jsonl.gz');table=[]
    for r in rr:
        a=r['order1_common_U_delta'];b=r['order2_common_U_delta'];f=a is not None and b is not None and abs(a)>EPS and abs(b)>EPS and a*b<0
        table.append(dict(**r,common_U_sign_flip=f,common_U_flip_both_over1pp=f and min(abs(a),abs(b))>=.01,common_U_flip_both_over5pp=f and min(abs(a),abs(b))>=.05))
    for name,expected in [('ALL_PARENT_STATE_PAIRS.csv',table),('ALL_SIGN_REVERSALS.csv',[r for r in table if r['common_U_sign_flip']])]:
        with (OUT/name).open() as f:actual=list(csv.DictReader(f))
        assert len(actual)==len(expected)
        for x,y in zip(actual,expected):assert all(x[k]==('' if v is None else str(v)) for k,v in y.items()),(x,y)
    compressed=OUT/'SUMMARY_public.json.gz'
    if not compressed.exists():
        with compressed.open('xb') as raw:
            with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as f:f.write((OUT/'SUMMARY.json').read_bytes())
    with gzip.open(compressed,'rb') as f:assert f.read()==(OUT/'SUMMARY.json').read_bytes()
    copy(compressed,str((OUT/'SUMMARY.json.gz').relative_to(ROOT)))
    if not closing:
        for name in ['RUNTIME.json','CPU_CONTRACTS.json','GLOBAL_PAIRED_READOUT_BARRIER.json','REUSED_GT_LABEL_EXPOSURE.json','CPU_COMPLETION.json','ROOT_READBACK.json','MECHANISM_SCOPE_DECISION.json','FIGURE_EXPORT.json','ROOT_VISUAL_REVIEW.json','empty_episode_readback001/CAPTURE_RECEIPT.json','empty_episode_readback001/REVISION_RUNTIME.json','empty_episode_readback001/LAUNCH.json','empty_episode_readback001/RESOLUTION_RECEIPT.json']:
            p=BASE/name;put(OUT/'receipts'/name,dict(original_private_receipt_sha256=sha(p),projection=receipt_metadata(read(p))))
        put(OUT/'DESIGN_PUBLIC.json',dict(status='complete_locked_same_evidence_order_state_cache_diagnostic',targets=['hc2','vidstg'],parents_each=128,previous32_each=32,added96_each=96,orders=['order1','order2'],cells=512,parent_scope_pairs=512,saved_updates=1024,bootstrap=10000,seed=20261013,history_exposed=True,randomness_applicable_explicitly_reported=True,no_GT_threshold_selection=True,causal_LN_attribution=False,new_gate_created=False,new_GPU_fits=0,new_model_calls=0,new_DINO_calls=0,new_GT_scoring_calls=0,production_promoted=False,manuscript_written=False,raw_GT_frames_states_weights_payload_exported=False))
        for relative,binding in read(BASE/'ROOT_VISUAL_REVIEW.json')['PDF_files'].items():
            p=ROOT/relative;assert sha(p)==binding['sha256'];copy(p,str((OUT/'figures'/p.name).relative_to(ROOT)))
    else:
        assert read(BASE/'FINAL_COMPLETION.json')['status']=='complete'
        for name in ['ROOT_CLOSING.json','FINAL_COMPLETION.json','ARCHIVE_VERIFICATION_RECEIPT.json','FINAL_GITHUB_RECEIPT.json']:
            p=BASE/name;put(OUT/'closing'/name,dict(original_private_receipt_sha256=sha(p),projection=receipt_metadata(read(p))))
        owned.update(e['path'] for e in read(BASE/'PUBLIC_STAGE.json')['files'])
    for p in OUT.rglob('*.json'):
        if p.name not in ['SUMMARY.json','PUBLIC_SCALAR_AUDIT.json','SOURCE_MANIFEST.json','SOURCE_MANIFEST_CLOSING.json']:copy(p,str(p.relative_to(ROOT)))
    if not closing:
        z=subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(CHECKOUT/'scripts/audit_stvg_opd_state_conditional_actionability_p0_v1.py'),str(CHECKOUT)],capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        assert z.returncode==0,z.stdout+z.stderr;audit=json.loads(z.stdout);assert audit['status']=='pass'
        write(BASE/'PUBLIC_SCALAR_AUDIT.json',dict(audit,actual_public_checkout=True,time=time.time()));write(OUT/'PUBLIC_SCALAR_AUDIT.json',audit)
    else:
        prior=read(BASE/'PUBLIC_STAGE.json')
        for entry in prior['files']:assert sha(CHECKOUT/entry['path'])==entry['sha256']
        assert read(BASE/'PUBLIC_SCALAR_AUDIT.json')['status']=='pass'
    copy(OUT/'PUBLIC_SCALAR_AUDIT.json',str((OUT/'PUBLIC_SCALAR_AUDIT.json').relative_to(ROOT)))
    rel=str((OUT/'SOURCE_MANIFEST.json').relative_to(ROOT));manifest=OUT/('SOURCE_MANIFEST_CLOSING.json' if closing else 'SOURCE_MANIFEST.json')
    write(manifest,dict(status='pass',scope='complete exact finite P9-P0 anonymous paired scalars/source/all four PDF pages/actual closure whitelist; no raw states or randomness; excludes self hash',files={n:dict(sha256=sha(CHECKOUT/n),bytes=(CHECKOUT/n).stat().st_size) for n in sorted(owned) if n!=rel}));copy(manifest,rel)
    entries=[]
    for n in sorted(owned):
        p=CHECKOUT/n;data=p.read_bytes();entries.append(dict(path=n,bytes=len(data),sha256=sha(p),blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
    with tempfile.TemporaryDirectory(prefix='p9p0-public-index-') as folder:
        env={**os.environ,'GIT_INDEX_FILE':str(Path(folder)/'index')};git('read-tree',base,env=env)
        for e in entries:
            h=git('hash-object','-w',str(CHECKOUT/e['path']));assert h==e['blob_sha'];git('update-index','--add','--cacheinfo','100644',h,e['path'],env=env)
        tree=git('write-tree',env=env)
    write(BASE/('CLOSING_PUBLIC_STAGE.json' if closing else 'PUBLIC_STAGE.json'),dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,base_commit=base,expected_tree=tree,files=entries,file_count=len(entries),bytes=sum(e['bytes'] for e in entries),unrelated_untracked_preserved=unrelated,scope='finite full P9-P0 same expert evidence/common random numbers/inherited state naturally occurring contrast/all128-old32-added96/both orders/all anonymous outcomes/source/actual four PDF pages; no causal attribution or new gate',time=time.time()))
    print('P9_P0_PUBLIC_STAGE',closing,len(entries),tree)
if __name__=='__main__':run(len(sys.argv)>1 and sys.argv[1]=='closing')
