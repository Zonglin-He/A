"""Full anonymous measured deployment outcomes and actual scoped closing."""
import ast,hashlib,json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.run_stvg_opd_deployment_latency_v1 import BASE,PUB,verify
from scripts.audit_stvg_opd_deployment_complete_public_v1 import REL,run as audit
from scripts.publish_stvg_opd_actionability_audit_v1 import git,CHECKOUT,BRANCH,put,safe,private_attachment_path,receipt_metadata
OUT=ROOT/REL;REV=BASE/'after_expanded_priority002'
def prepare_export():
    OUT.mkdir(parents=True,exist_ok=False);q=read(BASE/'ROOT_QUALIFICATION_READBACK.json');r=read(BASE/'ROOT_COMPLETE_READBACK.json')
    assert q['status']==r['status']=='pass' and r['actual_timed_query_paths']==128
    assert sha(PUB/'PAIRED_DEPLOYMENT_SUMMARY.json')==r['statistics_sha256']
    shutil.copy2(PUB/'PAIRED_DEPLOYMENT_SUMMARY.json',OUT/'PAIRED_DEPLOYMENT_SUMMARY.json')
    context={}
    for ds in ['hc2','vidstg']:
        rr=[]
        for method in ['Frozen','Full']:
            for repeat in [0,1]:
                p=PUB/ds/method/f'repeat{repeat}.json';target=OUT/ds/method/p.name;target.parent.mkdir(parents=True,exist_ok=True)
                done=read(BASE/'timing'/ds/method/str(repeat)/'COMPLETION.json');assert sha(p)==done['timing_rows_sha256']
                shutil.copy2(p,target);rr+=read(p)['rows']
        full=[r for r in rr if r['method']=='Full'];context[ds]=dict(Full_paths=len(full),
            Full_no_DINO_paths=sum(r['actual_new_DINO_calls']==0 for r in full),Full_no_admitted_paths=sum(r['admitted_references']==0 for r in full),
            Full_no_backward_paths=sum(r['backward_steps']==0 for r in full))
    write(OUT/'MEASUREMENT_CONTEXT.json',dict(status='complete_actual_local_measurement',datasets=context,
        storage_policy='both methods original O012 frozen student storage; Full inactive frozen expert CPU staged/restored inside native-student timer',
        original_baseline_processes_stopped_with_context_resident=True,baseline_standby_identity_receipt_sha256=sha(REV/'BASELINE_STANDBY_IDENTITY_RECEIPT.json'),
        peak_memory_is_this_process_torch_allocated_not_total_system_GPU_memory=True,model_startup_reported_separately=True,
        OS_page_cache_evicted=False,empty_application_frame_cache=True,deterministic_subject_parses_prepared=True,
        Full_real_expert_called_for_valid_original_observation_scope=True,all_no_information_and_slow_paths_retained=True,
        distinct_from_pending32_audited_cost_study=True,GT_read=False))
    text=(PUB/'REPORT.md').read_text().replace('Full always calls the real frozen expert; Frozen keeps only its student model resident.',
        'Full calls the real frozen expert wherever the original Native observation scope provides a valid position; all no-information paths remain in the measurements. Frozen keeps only its student model resident.')
    text+='\nBoth methods use the exact qualified O012 native mask/frozen-storage capture. Full stages its inactive expert to CPU only during student capture and restores it before real DINO; these transfers are included in the complete timer. The original cancelled baseline remains stopped with its GPU context resident, and no competing experiment runs. Allocated/reserved peaks describe the measured process, rather than all system GPU residency.\n'
    text+='\nAll 64 qualification fits passed full original trace equality, with 4,852,728 trajectory coordinates and 134,208 trace-off output/state coordinates independently checked; qualification made 112 actual DINO calls. All 128 measured paths then passed 383,104 original prediction/state/writeback coordinate comparisons and 896 stage-sum checks. The 64 timed Full paths made 224 DINO calls: 128 HC and 96 Vid. Eight repeated Vid paths had no admitted expert observation and no detector call; four HC and eight Vid repeated Full paths had zero backward steps. They were retained without selection. These counts explain why nominal observation positions and actual expert calls differ.\n'
    text+='\nThese results close the original paired deployment profiling phase. They do not close the separate 32-query audited cost study, LN-only streams or any later method/main-experiment decision. No new GT scoring, teacher, method, gate, manuscript or production promotion was introduced.\n'
    (OUT/'REPORT.md').write_text(text)
    v=audit(ROOT);write(OUT/'PUBLIC_SCALAR_AUDIT.json',v);write(BASE/'ROOT_PUBLIC_TIMING_SCALAR_READBACK.json',dict(v,root_sha256=sha(BASE/'ROOT_COMPLETE_READBACK.json'),time=time.time()))
def run(closing=False):
    verify();assert read(BASE/'ROOT_VISUAL_REVIEW.json')['actual_pixel_view']
    if not closing:
        if OUT.exists():
            cap=read(BASE/'publication_import_resume001/CAPTURE_RECEIPT.json')
            assert cap['status']=='preserved_local_CPU_publication_import_failure'
            assert cap['original_root_sha256']==sha(BASE/'ROOT_COMPLETE_READBACK.json')
            for n,b in cap['accepted_public_scalar_outputs'].items():
                assert sha(OUT/n)==b['sha256'] and (OUT/n).stat().st_size==b['bytes']
            assert audit(ROOT)==read(OUT/'PUBLIC_SCALAR_AUDIT.json')
        else:prepare_export()
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
        dest=CHECKOUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);assert sha(p)==sha(dest);owned.add(rel)
    for n in ['protocols/stvg_opd_claim_strengthening_v1.md','protocols/stvg_opd_deployment_priority002.md',
        'scripts/run_stvg_opd_deployment_latency_v1.py','scripts/finalize_stvg_opd_deployment_latency_v1.py','scripts/run_stvg_opd_deployment_priority002.py',
        'scripts/continue_stvg_opd_deployment_priority002.py','scripts/audit_stvg_opd_deployment_complete_public_v1.py',
        'scripts/publish_stvg_opd_deployment_complete_v1.py','scripts/close_stvg_opd_deployment_complete_v1.py','vg_tta/stvg_opd_deployment_profile_v1.py']:
        copy(ROOT/n,n)
    if not closing:
        for n in ['ROOT_QUALIFICATION_READBACK.json','ROOT_COMPLETE_READBACK.json','ROOT_PUBLIC_TIMING_SCALAR_READBACK.json','CPU_COMPLETION.json','ROOT_VISUAL_REVIEW.json']:
            p=BASE/n;put(OUT/'receipts'/n,dict(original_private_receipt_sha256=sha(p),projection=receipt_metadata(read(p))))
        for n in ['REVISION_RUNTIME.json','ROOT_PRIORITY_QUALIFICATION_READBACK.json','QUALIFICATION_FINAL_GITHUB_RECEIPT.json','BASELINE_STANDBY_IDENTITY_RECEIPT.json']:
            p=REV/n;put(OUT/'receipts'/'priority'/n,dict(original_private_receipt_sha256=sha(p),projection=receipt_metadata(read(p))))
        for rel,b in read(BASE/'ROOT_VISUAL_REVIEW.json')['PDF_files'].items():
            p=ROOT/rel;assert sha(p)==b['sha256'];copy(p,str((OUT/'figures'/p.name).relative_to(ROOT)))
    else:
        assert read(BASE/'FINAL_COMPLETION.json')['status']=='complete'
        for n in ['ROOT_CLOSING.json','FINAL_COMPLETION.json','ARCHIVE_VERIFICATION_RECEIPT.json','FINAL_GITHUB_RECEIPT.json']:
            p=BASE/n;put(OUT/'closing'/n,dict(original_private_receipt_sha256=sha(p),projection=receipt_metadata(read(p))))
        for e in read(BASE/'PUBLIC_STAGE.json')['files']:
            assert sha(CHECKOUT/e['path'])==e['sha256'];owned.add(e['path'])
    for p in OUT.rglob('*'):
        if p.is_file() and p.name not in ['SOURCE_MANIFEST.json','SOURCE_MANIFEST_CLOSING.json']:copy(p,str(p.relative_to(ROOT)))
    if not closing:
        z=subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(CHECKOUT/'scripts/audit_stvg_opd_deployment_complete_public_v1.py'),str(CHECKOUT)],
            capture_output=True,text=True,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
        assert z.returncode==0,z.stdout+z.stderr;a=json.loads(z.stdout);assert a==read(OUT/'PUBLIC_SCALAR_AUDIT.json')
        write(BASE/'PUBLIC_SCALAR_AUDIT.json',dict(a,actual_public_checkout=True,time=time.time()))
    rel=str((OUT/'SOURCE_MANIFEST.json').relative_to(ROOT));manifest=OUT/('SOURCE_MANIFEST_CLOSING.json' if closing else 'SOURCE_MANIFEST.json')
    write(manifest,dict(status='pass',scope='all actual paired deployment timings/source/fresh expert/no-information paths/actual pixel-view PDF/root; excludes self hash',
        files={n:dict(sha256=sha(CHECKOUT/n),bytes=(CHECKOUT/n).stat().st_size) for n in sorted(owned) if n!=rel}));copy(manifest,rel)
    entries=[]
    for n in sorted(owned):
        p=CHECKOUT/n;data=p.read_bytes();entries.append(dict(path=n,bytes=len(data),sha256=sha(p),blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
    with tempfile.TemporaryDirectory(prefix='deployment-complete-index-') as folder:
        env={**os.environ,'GIT_INDEX_FILE':str(Path(folder)/'index')};git('read-tree',base,env=env)
        for e in entries:
            h=git('hash-object','-w',str(CHECKOUT/e['path']));assert h==e['blob_sha'];git('update-index','--add','--cacheinfo','100644',h,e['path'],env=env)
        tree=git('write-tree',env=env)
    write(BASE/('CLOSING_PUBLIC_STAGE.json' if closing else 'PUBLIC_STAGE.json'),dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,base_commit=base,expected_tree=tree,
        files=entries,file_count=len(entries),bytes=sum(e['bytes'] for e in entries),unrelated_untracked_preserved=unrelated,
        scope='complete finite 32-query/128 paired deployment timed paths/64 complete GPU qualifications/full exact-output-state/root/real PDF pixel view/all timing outcomes including no-information/source; not LN-only/32 audited cost/main completion',time=time.time()))
    print('DEPLOYMENT_COMPLETE_PUBLIC_STAGE',closing,len(entries),sum(e['bytes'] for e in entries),tree)
if __name__=='__main__':run(len(sys.argv)>1 and sys.argv[1]=='closing')
