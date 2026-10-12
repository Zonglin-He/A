"""Exact anonymous P9 source/results and real scoped closure export."""
import ast,csv,gzip,hashlib,json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_evidence_actionability_audit_v1 import BASE,OUT,read,write,sha,verify,rows,parent_records,EPS
from scripts.stage_stvg_opd_o_q0_phase_public_v1 import receipt_metadata
from scripts.publish_stvg_opd_supplement_v1 import safe,private_attachment_path
CHECKOUT=ROOT.parent/'visual-grounding-public-A';BRANCH='research/stvg-opd-paper-hc2-revision-v2'
def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=CHECKOUT,text=True,env=env).strip()
def put(path,value):
    if path.exists():assert read(path)==value
    else:write(path,value)
def run(closing=False):
    verify();assert all(read(BASE/n)['status']=='pass' for n in ['ROOT_READBACK.json','ROOT_EXTERNAL_SIGNAL_READBACK.json','ROOT_VISUAL_REVIEW.json'])
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
                if 'SUMMARY' in source.name:safe(json.load(f))
                else:
                    for line in f:safe(json.loads(line))
        elif source.suffix in ['.py','.md']:
            assert not private_attachment_path(data.decode())
            if source.suffix=='.py':ast.parse(data.decode())
        dest=CHECKOUT/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);assert sha(dest)==sha(source);owned.add(relative)
    sources=['protocols/stvg_opd_evidence_actionability_audit_v1.md','scripts/stvg_opd_evidence_actionability_audit_v1.py','scripts/audit_stvg_opd_actionability_public_v1.py','scripts/continue_stvg_opd_evidence_actionability_audit_v1.py','scripts/review_stvg_opd_actionability_signals002.py','scripts/report_stvg_opd_actionability_audit_v1.py','scripts/revise_stvg_opd_actionability_figure_layout002.py','scripts/publish_stvg_opd_actionability_audit_v1.py','scripts/close_stvg_opd_actionability_audit_v1.py']
    for n in sources:copy(ROOT/n,n)
    for n in ['ROWS.jsonl.gz','REPORT.md','ALL_PARENT_UPDATES.csv','CORRECT_TEACHER_HARMFUL_UPDATES.csv','REPEATED_ORDER_DIAGNOSTICS.json']:copy(OUT/n,str((OUT/n).relative_to(ROOT)))
    rr=rows(OUT/'ROWS.jsonl.gz')
    for n,negative in [('ALL_PARENT_UPDATES.csv',False),('CORRECT_TEACHER_HARMFUL_UPDATES.csv',True)]:
        with (OUT/n).open() as f:table=list(csv.DictReader(f))
        expected=[]
        for ds in ['hc2','vidstg']:
            for scope in ['Omega0','Omega1']:
                chosen=[r for r in rr if r['dataset']==ds and r['scope']==scope]
                for p in parent_records(chosen):
                    group=[r for r in chosen if r['parent_id']==p['parent_id']];q=p['teacher_quality'];d=p['common_U_delta']
                    if negative and not(q is not None and q>=.5 and d is not None and d< -EPS):continue
                    valid=[r for r in group if r['common_U_delta'] is not None]
                    x=dict(dataset=ds,scope=scope,previous_P6=group[0]['previous_P6'],**p,order1_common_U_delta=next(r['common_U_delta'] for r in group if r['order']=='order1'),order2_common_U_delta=next(r['common_U_delta'] for r in group if r['order']=='order2'),both_orders_harmful=len(valid)==2 and all(r['common_U_delta']< -EPS for r in valid),opposite_order_effect_signs=len(valid)==2 and all(abs(r['common_U_delta'])>EPS for r in valid) and valid[0]['common_U_delta']*valid[1]['common_U_delta']<0)
                    expected.append(x)
        assert len(table)==len(expected)
        for x,y in zip(table,expected):assert all(x[k]==('' if v is None else str(v)) for k,v in y.items()),(x,y)
    compressed=OUT/'SUMMARY_public.json.gz'
    if not compressed.exists():
        with compressed.open('xb') as raw:
            with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as f:f.write((OUT/'SUMMARY.json').read_bytes())
    with gzip.open(compressed,'rb') as f:assert f.read()==(OUT/'SUMMARY.json').read_bytes()
    copy(compressed,str((OUT/'SUMMARY.json.gz').relative_to(ROOT)))
    if not closing:
        names=['RUNTIME.json','CPU_CONTRACTS.json','GLOBAL_READOUT_BARRIER.json','GT_EXPOSURE.json','CPU_COMPLETION.json','ROOT_READBACK.json','ROOT_EXTERNAL_SIGNAL_READBACK.json','CANDIDATE_DECISION.json','FINAL_FIGURE_EXPORT.json','ROOT_VISUAL_REVIEW.json']
        for name in names:
            p=BASE/name;put(OUT/'receipts'/name,dict(original_private_receipt_sha256=sha(p),projection=receipt_metadata(read(p))))
        for name in ['signals_precision_readback001/CAPTURE_RECEIPT.json','signals_precision_readback001/RESOLUTION_RECEIPT.json','figure_layout002/CAPTURE_RECEIPT.json','figure_layout002/RESOLUTION_RECEIPT.json','figure_layout002/render_bootstrap001/CAPTURE_RECEIPT.json']:
            p=BASE/name
            if p.exists():put(OUT/'recovery'/name,dict(original_private_receipt_sha256=sha(p),projection=receipt_metadata(read(p))))
        put(OUT/'DESIGN_PUBLIC.json',dict(status='complete_locked_cache_only_evidence_actionability_audit',targets=['hc2','vidstg'],parents_each=128,previous32_each=32,added96_each=96,orders=['order1','order2'],cells=512,saved_updates=1024,bootstrap=10000,seed=20261012,history_exposed=True,no_GT_threshold_selection=True,no_new_gate=True,new_GPU_fits=0,new_model_calls=0,new_DINO_calls=0,production_promoted=False,manuscript_written=False,raw_GT_frames_weights_payload_exported=False))
        for relative,binding in read(BASE/'ROOT_VISUAL_REVIEW.json')['PDF_files'].items():
            p=ROOT/relative;assert sha(p)==binding['sha256'];copy(p,str((OUT/'figures'/p.name).relative_to(ROOT)))
    else:
        assert read(BASE/'FINAL_COMPLETION.json')['status']=='complete'
        for name in ['ROOT_CLOSING.json','FINAL_COMPLETION.json','ARCHIVE_VERIFICATION_RECEIPT.json','FINAL_GITHUB_RECEIPT.json']:
            p=BASE/name;put(OUT/'closing'/name,dict(original_private_receipt_sha256=sha(p),projection=receipt_metadata(read(p))))
        owned.update(e['path'] for e in read(BASE/'PUBLIC_STAGE.json')['files'])
    for p in OUT.rglob('*.json'):
        if p.name not in ['SUMMARY.json','PUBLIC_SCALAR_AUDIT.json','SOURCE_MANIFEST.json','SOURCE_MANIFEST_CLOSING.json']:copy(p,str(p.relative_to(ROOT)))
    z=subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(CHECKOUT/'scripts/audit_stvg_opd_actionability_public_v1.py'),str(CHECKOUT)],capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
    assert z.returncode==0,z.stdout+z.stderr;audit=json.loads(z.stdout);assert audit['status']=='pass'
    if not closing:
        write(BASE/'PUBLIC_SCALAR_AUDIT.json',dict(audit,actual_public_checkout=True,time=time.time()));write(OUT/'PUBLIC_SCALAR_AUDIT.json',audit)
    copy(OUT/'PUBLIC_SCALAR_AUDIT.json',str((OUT/'PUBLIC_SCALAR_AUDIT.json').relative_to(ROOT)))
    rel=str((OUT/'SOURCE_MANIFEST.json').relative_to(ROOT));manifest=OUT/('SOURCE_MANIFEST_CLOSING.json' if closing else 'SOURCE_MANIFEST.json')
    write(manifest,dict(status='pass',scope='exact complete finite P9 anonymous positive-negative source/scalars/all five actual PDF pages/current closure whitelist; excludes self hash',files={n:dict(sha256=sha(CHECKOUT/n),bytes=(CHECKOUT/n).stat().st_size) for n in sorted(owned) if n!=rel}));copy(manifest,rel)
    entries=[]
    for n in sorted(owned):
        p=CHECKOUT/n;data=p.read_bytes();entries.append(dict(path=n,bytes=len(data),sha256=sha(p),blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
    with tempfile.TemporaryDirectory(prefix='p9-public-index-') as folder:
        env={**os.environ,'GIT_INDEX_FILE':str(Path(folder)/'index')};git('read-tree',base,env=env)
        for e in entries:
            h=git('hash-object','-w',str(CHECKOUT/e['path']));assert h==e['blob_sha'];git('update-index','--add','--cacheinfo','100644',h,e['path'],env=env)
        tree=git('write-tree',env=env)
    write(BASE/('CLOSING_PUBLIC_STAGE.json' if closing else 'PUBLIC_STAGE.json'),dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,base_commit=base,expected_tree=tree,files=entries,file_count=len(entries),bytes=sum(e['bytes'] for e in entries),unrelated_untracked_preserved=unrelated,scope='finite P9 cached evidence actionability/all128-old32-added96/twoorders/all positive-negative-unknown/readback/source/actual five PDF pages; no gate and no causal inherited-LN claim',time=time.time()))
    print('P9_PUBLIC_STAGE',closing,len(entries),tree)
if __name__=='__main__':run(len(sys.argv)>1 and sys.argv[1]=='closing')
