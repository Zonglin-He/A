"""Finite anonymous export of an actual priority/qualification/first-formal handoff."""
import ast,hashlib,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.run_stvg_opd_expanded_after_p8_priority001 import BASE,EXP,verify,stopped
from scripts.stage_stvg_opd_o_q0_phase_public_v1 import receipt_metadata
from scripts.publish_stvg_opd_supplement_v1 import safe,private_attachment_path
CHECKOUT=ROOT.parent/'visual-grounding-public-A';BRANCH='research/stvg-opd-paper-hc2-revision-v2'
OUT=ROOT/'results/stvg_opd_claim_strengthening/2026-10-12/after_P8_priority001'

def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=CHECKOUT,text=True,env=env).strip()
def put(path,value):
    if path.exists():assert read(path)==value
    else:write(path,value)

def run():
    verify();stopped()
    assert read(BASE/'ROOT_PRIORITY_QUALIFICATION_READBACK.json')['status']=='pass'
    assert read(BASE/'ROOT_FIRST_FORMAL_READBACK.json')['status']=='pass'
    cancelled=ROOT/'artifacts/stvg_opd_supplement_v1/baselines/USER_CANCEL_REMAINING_FORMAL_RECEIPT.json'
    assert read(cancelled)['status']=='remaining_formal_cancelled_by_explicit_human_instruction'
    assert not read(cancelled)['formal82710_global_completed']
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    base=git('rev-parse','HEAD');assert base==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    unrelated=git('ls-files','--others','--exclude-standard').splitlines();owned=set()
    def copy(source,relative):
        data=source.read_bytes()
        if source.suffix=='.json':safe(read(source))
        elif source.suffix in ['.py','.md']:
            assert not private_attachment_path(data.decode())
            if source.suffix=='.py':ast.parse(data.decode())
        target=CHECKOUT/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);assert sha(source)==sha(target);owned.add(relative)
    sources=['protocols/stvg_opd_after_p8_priority001.md','protocols/stvg_opd_after_p8_cancellation_handoff002.md','scripts/pause_stvg_opd_after_p8_priority001.py',
             'scripts/run_stvg_opd_expanded_after_p8_priority001.py','scripts/continue_stvg_opd_expanded_after_p8_priority001.py',
             'scripts/review_stvg_opd_expanded_priority_first001.py','scripts/publish_stvg_opd_after_p8_priority001.py',
             'scripts/run_stvg_opd_expanded_state_v1.py','scripts/stvg_opd_expanded_state_common_v1.py']
    for relative in sources:copy(ROOT/relative,relative)
    for name in ['CPU_STORAGE_CONTRACTS.json','REVISION_RUNTIME.json','PRIORITY_PAUSE_RECEIPT.json','LAUNCH.json',
                 'ROOT_PRIORITY_QUALIFICATION_READBACK.json','ROOT_FIRST_FORMAL_READBACK.json']:
        path=BASE/name
        put(OUT/'receipts'/name,dict(original_private_receipt_sha256=sha(path),projection=receipt_metadata(read(path))))
    put(OUT/'receipts/ORIGINAL_ROOT_QUALIFICATION_READBACK.json',dict(original_private_receipt_sha256=sha(EXP/'ROOT_QUALIFICATION_READBACK.json'),projection=receipt_metadata(read(EXP/'ROOT_QUALIFICATION_READBACK.json'))))
    put(OUT/'receipts/USER_CANCEL_REMAINING_FORMAL_RECEIPT.json',dict(original_private_receipt_sha256=sha(cancelled),projection=receipt_metadata(read(cancelled))))
    d=read(cancelled)
    put(OUT/'DESIGN_PUBLIC.json',dict(status='actual_bounded_engineering_qualified_and_original_formal_resumed',scope='actual saved-query priority transition, same-source inactive expert host storage, original full48 GPU-repeat qualification and six accepted HC first-formal fits only; not whole expanded-state or C/N completion',
        datasets=['hc2','vidstg'],parents_HC=205,parents_Vid=700,orders=2,arms=['full','alpha0','alpha1'],
        planned_new_formal_fits=5430,qualification_arrivals=24,actual_complete_repeat_GPU_fits=48,
        first_formal_GPU_fits=6,baseline_prefix_accepted=d['current_Vid_TENT_accepted'],already_sealed_HC_stages=2,
        already_sealed_HC_arrivals=20892,remaining_baseline_formal_cancelled=True,formal82710_global_completed=False,
        baseline_resumption_requires_new_explicit_human_instruction=True,additional_baseline_tuning=False,
        original_science_source_hyperparameters_roster_unchanged=True,original_S_optimizer_memory_and_lease_preserved=True,
        actual_new_DINO_calls=0,GT_read=False,whole_expanded_formal_complete=False,whole_C_N_complete=False,
        no_future_OOM_guarantee=True,manuscript_written=False,production_promoted=False,
        raw_payload_GT_query_text_media_model_weights_exported=False))
    for path in OUT.rglob('*.json'):
        if path.name!='SOURCE_MANIFEST.json':copy(path,str(path.relative_to(ROOT)))
    rel=str((OUT/'SOURCE_MANIFEST.json').relative_to(ROOT))
    put(OUT/'SOURCE_MANIFEST.json',dict(status='pass',scope='exact current bounded original-source/CPU storage/GPU-repeat qualification/six first-formal and explicit baseline cancellation anonymous metadata whitelist; excludes self hash',files={name:dict(bytes=(CHECKOUT/name).stat().st_size,sha256=sha(CHECKOUT/name)) for name in sorted(owned)}))
    copy(OUT/'SOURCE_MANIFEST.json',rel);entries=[]
    for name in sorted(owned):
        path=CHECKOUT/name;data=path.read_bytes()
        entries.append(dict(path=name,bytes=len(data),sha256=sha(path),blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
    with tempfile.TemporaryDirectory(prefix='C-after-P8-public-index-') as folder:
        env={**os.environ,'GIT_INDEX_FILE':str(Path(folder)/'index')};git('read-tree',base,env=env)
        for e in entries:
            h=git('hash-object','-w',str(CHECKOUT/e['path']));assert h==e['blob_sha'];git('update-index','--add','--cacheinfo','100644',h,e['path'],env=env)
        tree=git('write-tree',env=env)
    write(BASE/'PUBLIC_STAGE.json',dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,base_commit=base,expected_tree=tree,files=entries,file_count=len(entries),bytes=sum(e['bytes'] for e in entries),unrelated_untracked_preserved=unrelated,scope='bounded saved-query priority transition, actual CPU storage/full48 GPU qualification/root/six first-formal/cancelled remaining formal; current5430 experiment still running',time=time.time()))
    print('C_AFTER_P8_BOUNDED_PUBLIC_STAGE',len(entries),sum(e['bytes'] for e in entries),tree)

if __name__=='__main__':run()
