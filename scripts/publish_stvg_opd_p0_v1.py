"""Stage reviewed P0 anonymous evidence and verify every public remote byte."""
import concurrent.futures,gzip,hashlib,json,shutil,subprocess,sys,time,urllib.parse,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_common_v1 import *
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
BRANCH='research/stvg-opd-paper-v1'

def git(*args):return subprocess.check_output(['git',*args],cwd=CHECKOUT).decode().strip()

def stage():
    verify();assert read(BASE/'P0_ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert read(BASE/'P0_CPU_COMPLETION.json')['status']=='pending_actual_root_visual_and_publication'
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    assert not git('status','--porcelain')
    subprocess.run(['git','fetch','--quiet','origin','main'],cwd=CHECKOUT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')
    if git('branch','--list',BRANCH):subprocess.run(['git','switch',BRANCH],cwd=CHECKOUT,check=True)
    else:subprocess.run(['git','switch','-c',BRANCH],cwd=CHECKOUT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')
    p=read(BASE/'RUNTIME_LOCK.json');cpu=read(BASE/'CPU_RUNTIME_LOCK.json');components=read(BASE/'COMPONENT_RUNTIME_LOCK.json');p1=read(BASE/'P1_RUNTIME_LOCK.json');p1rev=read(BASE/'P1_RUNTIME_LOCK_revision001.json')
    original=read(PAPER/'RUNTIME_LOCK.json');owned=set(p['pins'])|set(cpu['pins'])|set(components['pins'])|set(p1['pins'])|set(p1rev['pins'])|set(original['pins'])
    for f in sorted((PAPER/'revisions').glob('*.json')):owned.update(read(f)['pin_overrides'])
    owned.update(['scripts/audit_stvg_opd_p0_public_v1.py','scripts/publish_stvg_opd_p0_v1.py',
        'scripts/render_stvg_opd_p0_cases_v1.py','scripts/render_stvg_opd_p0_signal_chain_v1.py',
        'scripts/audit_stvg_opd_p0_dense_s_v1.py','scripts/audit_stvg_opd_p0_bytes_v1.py',
        'scripts/diagnose_stvg_opd_p0_population_v1.py','scripts/score_stvg_opd_p1_v1.py',
        'scripts/assemble_stvg_opd_table1_v1.py','scripts/test_stvg_opd_p1_cpu_v1.py',
        'docs/stvg_opd_paper_v1/EXECUTION.md','docs/STVG_OPD_P0_ROOT_REVIEW.md'])
    owned=sorted(f for f in owned if f.startswith(('scripts/','vg_tta/','methods/','protocols/','docs/')))
    for name in ['DESIGN_LOCK.json','LATER_DESIGN_LOCK.json','APPENDIX_STAGE_LOCK.json',
        'USER_SCOPE.json','TABLE2_COHORT_RESOLUTION.json','CPU_CONTRACTS.json','CPU_SCORING_CONTRACTS.json',
        'COMPONENT_CPU_CONTRACTS.json','P1_CPU_CONTRACTS.json','P1_BASELINE_BINDING.json',
        'QUALIFICATION.json','P0_ROOT_VISUAL_REVIEW.json','P0_ROOT_DECISION.json','P0_ROOT_BYTE_READBACK.json',
        'P1_RUNTIME_LOCK_revision001.json','P0_POSTSEAL_ROOT_RUNTIME.json']:
        shutil.copy2(BASE/name,PUB/name)
    write(PUB/'CODE_BINDING.json',dict(code={f:sha(ROOT/f) for f in owned},
        GPU_runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),CPU_runtime_sha256=sha(BASE/'CPU_RUNTIME_LOCK.json'),
        paper_core_main_method_unchanged=True,prepared_later_components_not_GPU_qualified=True,
        private_media_captions_labels_weights_optimizer_or_gradient_payload_exported=False))
    (PUB/'README.md').write_text('''# Fixed STVG-OPD paper suite: actual P0 confirmation\n\n[P0 report](P0_REPORT.md) contains actual 128-parent, two-order, three-arm confirmation results. Frozen is the shared matching source readout. [Root statistics](P0_ROOT_STATISTICS.json), every anonymous compressed row and all negative outcomes are retained. This cohort is historically exposed and excluded from the current parameter search; it is not a fresh unseen test.\n\nReproduce the scalar statistics and matched contrasts without private research assets:\n\n```bash\npython -B scripts/audit_stvg_opd_p0_public_v1.py results/stvg_opd_paper/2026-10-08\n```\n\nPython/NumPy suffice for this public scalar audit. GPU replay and raw official evaluation require separately obtained source checkpoints, DINO and official/private inputs. Dense readout and saved Gaussian/Adam/state arithmetic were audited postseal. The decoder Jacobian was not independently reimplemented.\n\nNo video, query caption, annotation, model weight, optimizer state, parameter vector or raw tensor cache is distributed. P1–P6 are distinct later stages; prepared code and rosters are not evidence that those stages ran. EATA's unavailable direction/source-media/Fisher stays user paused.\n''')
    files=owned+[str(f.relative_to(ROOT)) for f in PUB.rglob('*') if f.is_file()]
    assert not any(Path(f).suffix in ('.pt','.npz','.safetensors','.mp4','.sqlite') for f in files)
    for f in files:
        dst=CHECKOUT/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,dst)
        if f.endswith(('.json','.jsonl.gz')):
            raw=gzip.decompress(dst.read_bytes()) if f.endswith('.gz') else dst.read_bytes()
            for token in [b'"caption":',b'"video_path":',b'"native_boxes":',b'"GT_box":',b'"committed":',b'"gradient":',b'"optimizer_state":']:
                assert token not in raw,(f,token)
    result=subprocess.run([str(PYTHON),'-B',str(CHECKOUT/'scripts/audit_stvg_opd_p0_public_v1.py'),str(CHECKOUT/PUB.relative_to(ROOT))],cwd=CHECKOUT,check=True,capture_output=True,text=True)
    audit_result=json.loads(result.stdout);assert audit_result['status']=='pass'
    write(BASE/'P0_PUBLIC_SCALAR_AUDIT.json',audit_result)
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True)
    changed=git('diff','--cached','--name-only').splitlines();assert changed and set(changed)<=set(files)
    receipt=[dict(path=f,bytes=(CHECKOUT/f).stat().st_size,sha256=sha(CHECKOUT/f),blob_sha=git('hash-object',f)) for f in files]
    write(BASE/'P0_PUBLIC_STAGE.json',dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,
        base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),expected_tree=git('write-tree'),
        changed_files=changed,files=receipt,file_count=len(receipt),bytes=sum(r['bytes'] for r in receipt),time=time.time()))
    print('P0_PUBLIC_STAGE_READY',len(receipt),sum(r['bytes'] for r in receipt))

def fetch(url):
    error=None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'STVG-OPD-byte-audit'}),timeout=40) as r:return r.read()
        except Exception as e:error=e;time.sleep(attempt+1)
    raise error

def verify_remote(commit):
    meta=read(BASE/'P0_PUBLIC_STAGE.json')
    for branch in ['main',BRANCH]:
        ref=json.loads(fetch('https://api.github.com/repos/Zonglin-He/A/git/ref/heads/'+branch));assert ref['object']['sha']==commit
    c=json.loads(fetch(f'https://api.github.com/repos/Zonglin-He/A/git/commits/{commit}'))
    assert c['parents'][0]['sha']==meta['base_commit'] and c['tree']['sha']==meta['expected_tree']
    def one(r):
        raw=fetch(f'https://raw.githubusercontent.com/Zonglin-He/A/{commit}/'+urllib.parse.quote(r['path']))
        assert raw==(CHECKOUT/r['path']).read_bytes() and len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
        assert hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()==r['blob_sha']
        return {**r,'remote_bytes_identical':True}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,meta['files']))
    verify();assert git('rev-parse','HEAD')==commit and not git('status','--porcelain')
    write(BASE/'P0_FINAL_GITHUB_RECEIPT.json',dict(status='pass',repository='Zonglin-He/A',commit=commit,
        branches=['main',BRANCH],files=rows,file_count=len(rows),bytes=sum(r['bytes'] for r in rows),time=time.time()))
    print('P0_REMOTE_CONTENT_VERIFIED',commit,len(rows))

if __name__=='__main__':stage() if sys.argv[1]=='stage' else verify_remote(sys.argv[2])
