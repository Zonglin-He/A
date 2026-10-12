"""Anonymous complete P7 phase publication, exact finite source whitelist."""
import ast,gzip,hashlib,os,shutil,subprocess,sys,tempfile,time,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.decota_matrix_common_v1 import read,write,sha
from scripts.stvg_opd_temporal_guided_p7_common_v1 import BASE,OUT,A,B,verify
from scripts.stage_stvg_opd_o_q0_phase_public_v1 import receipt_metadata
from scripts.publish_stvg_opd_supplement_v1 import safe,private_attachment_path
CHECKOUT=ROOT.parent/'visual-grounding-public-A';BRANCH='research/stvg-opd-paper-hc2-revision-v2'

def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=CHECKOUT,text=True,env=env).strip()

def run(phase,closing=False):
    verify();ns=A if phase=='A' else B;pub=OUT/('P7_'+phase);assert read(ns/'ROOT_READBACK.json')['status']=='pass' and read(ns/'ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git';subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    base=git('rev-parse','HEAD');assert base==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH);assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    unrelated=git('ls-files','--others','--exclude-standard').splitlines();owned=set();bindings={}
    def copy(source,relative):
        data=source.read_bytes()
        if source.suffix=='.json':safe(read(source))
        if source.suffix=='.gz':
            with gzip.open(source,'rt') as f:
                if 'SUMMARY' in source.name:safe(json.load(f))
                else:
                    for line in f:safe(json.loads(line))
        if source.suffix in ['.py','.md']:
            assert not private_attachment_path(data.decode())
            if source.suffix=='.py':ast.parse(data.decode())
        target=CHECKOUT/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target);assert sha(target)==sha(source)
        owned.add(relative);bindings[relative]=dict(sha256=sha(source),bytes=len(data))
    source_names=['protocols/stvg_opd_temporal_guided_candidate_p7_v1.md','scripts/stvg_opd_temporal_guided_p7_common_v1.py',
        'scripts/run_stvg_opd_temporal_capture_p7a_v1.py','scripts/continue_stvg_opd_temporal_guided_p7_v1.py',
        'scripts/score_stvg_opd_temporal_capture_p7a_v1.py','scripts/run_stvg_opd_temporal_guided_p7b_v1.py',
        'scripts/score_stvg_opd_temporal_guided_p7b_v1.py','scripts/report_stvg_opd_temporal_guided_p7_v1.py',
        'scripts/audit_stvg_opd_p7_public_v1.py','scripts/publish_stvg_opd_p7_phase_v1.py']
    for name in source_names:copy(ROOT/name,name)
    for name in ['ROWS.jsonl.gz','REPORT.md']:copy(pub/name,str((pub/name).relative_to(ROOT)))
    # Preserve every scalar and parent vector; deterministic compression is
    # a byte-reversible public transport, not a statistical projection.
    summary=pub/'SUMMARY_public.json.gz'
    if not summary.exists():
        with summary.open('xb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as f:f.write((pub/'SUMMARY.json').read_bytes())
    with gzip.open(summary,'rb') as f:assert f.read()==(pub/'SUMMARY.json').read_bytes()
    copy(summary,str((pub/'SUMMARY.json.gz').relative_to(ROOT)))
    if not closing:
        receipt_names=(['ROOT_QUALIFICATION.json','TEMPORAL_CAPTURE_BARRIER.json','ACTUAL_CAPTURE_PHASE_READBACK.json','GLOBAL_READOUT_BARRIER.json',
            'CPU_COMPLETION.json','ROOT_READBACK.json','PRIORITY_PAUSE_RECEIPT.json','PRIORITY_RESUMPTION_RECEIPT.json','P7_B_BRANCH_DECISION.json','ROOT_VISUAL_REVIEW.json'] if phase=='A' else
            ['RUNTIME.json','EXPERT_EVIDENCE_BARRIER.json','ROOT_QUALIFICATION.json','GLOBAL_PREDICTION_BARRIER.json','CPU_COMPLETION.json','ROOT_READBACK.json',
            'PRIORITY_PAUSE_RECEIPT.json','PRIORITY_RESUMPTION_RECEIPT.json','CANDIDATE_MECHANISM_DECISION.json','ROOT_VISUAL_REVIEW.json'])
        for name in receipt_names:
            f=ns/name;target=pub/'receipts'/name;write(target,dict(original_private_receipt_sha256=sha(f),projection=receipt_metadata(read(f))))
        for name in ['CPU_CONTRACTS.json','RUNTIME.json']:
            f=BASE/name;target=pub/'receipts'/('P7_'+name);write(target,dict(original_private_receipt_sha256=sha(f),projection=receipt_metadata(read(f))))
        write(pub/'DESIGN_PUBLIC.json',dict(status='locked_original_roster',phase='P7_'+phase,target_parents_each=128,old_P6_each=32,added_each=96,
            cells=512,orders=['order1','order2'],historically_exposed=True,independent_online_stream=False,bootstrap=10000,seed=20261012,
            original_Full_unchanged=True,production_promoted=False,new_phase_A_spatial_fits=0,new_phase_A_DINO_calls=0,
            phase_B_conditional=True,conditional_B_formal_fits=1024,temporal_parameter_training=False,bidirectional_joint_optimization=False,
            raw_payload_weights_frames_GT_geometry_exported=False))
        figures=read(ns/'ROOT_VISUAL_REVIEW.json')['PDF_files']
        for rel in figures:
            p=ROOT/rel;copy(p,str((pub/'figures'/p.name).relative_to(ROOT)))
    else:
        assert read(ns/'FINAL_COMPLETION.json')['status']=='complete'
        for name in ['ROOT_CLOSING.json','FINAL_COMPLETION.json','ARCHIVE_VERIFICATION_RECEIPT.json','FINAL_GITHUB_RECEIPT.json']:
            f=ns/name;target=pub/'closing'/name;write(target,dict(original_private_receipt_sha256=sha(f),projection=receipt_metadata(read(f))))
        old=read(ns/'PUBLIC_STAGE.json');owned.update(f['path'] for f in old['files'])
    for p in pub.rglob('*.json'):
        if p.name not in ['SUMMARY.json','PUBLIC_SCALAR_AUDIT.json','SOURCE_MANIFEST.json']:copy(p,str(p.relative_to(ROOT)))
    result=subprocess.run([str(ROOT/'.conda/tubedetr/bin/python'),'-B',str(CHECKOUT/'scripts/audit_stvg_opd_p7_public_v1.py'),str(CHECKOUT),phase],capture_output=True,text=True,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'});assert result.returncode==0,result.stdout+result.stderr
    audit=json.loads(result.stdout);assert audit['status']=='pass'
    if not closing:
        write(ns/'PUBLIC_SCALAR_AUDIT.json',dict(audit,actual_public_checkout=True,time=time.time()));write(pub/'PUBLIC_SCALAR_AUDIT.json',audit)
        write(pub/'SOURCE_MANIFEST.json',dict(status='pass',scope='exact complete source/scalar/PDF whitelist',files=bindings))
    for name in ['PUBLIC_SCALAR_AUDIT.json','SOURCE_MANIFEST.json']:copy(pub/name,str((pub/name).relative_to(ROOT)))
    entries=[]
    for relative in sorted(owned):
        p=CHECKOUT/relative;data=p.read_bytes();entries.append(dict(path=relative,bytes=len(data),sha256=sha(p),blob_sha=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()))
    with tempfile.TemporaryDirectory(prefix='p7-phase-index-') as folder:
        env={**os.environ,'GIT_INDEX_FILE':str(Path(folder)/'index')};git('read-tree',base,env=env)
        for e in entries:
            h=git('hash-object','-w',str(CHECKOUT/e['path']));assert h==e['blob_sha'];git('update-index','--add','--cacheinfo','100644',h,e['path'],env=env)
        tree=git('write-tree',env=env)
    write(ns/('CLOSING_PUBLIC_STAGE.json' if closing else 'PUBLIC_STAGE.json'),dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,
        base_commit=base,expected_tree=tree,files=entries,file_count=len(entries),bytes=sum(e['bytes'] for e in entries),unrelated_untracked_preserved=unrelated,
        scope='complete '+('closing ' if closing else '')+'P7_'+phase+' all128/old32/new96/twoorders/all positive-negative scalar/source/actual PDF view; no production promotion or full official-query expansion',time=time.time()))
    print('P7_PUBLIC_STAGED',phase,closing,len(entries),tree)

if __name__=='__main__':run(sys.argv[1],len(sys.argv)>2 and sys.argv[2]=='closing')
