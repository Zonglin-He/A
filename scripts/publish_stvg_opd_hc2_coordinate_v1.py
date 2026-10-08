"""Publish the actually audited HC2 coordinate search, all trials and negative results."""
import concurrent.futures,gzip,hashlib,json,shutil,subprocess,sys,time,urllib.parse,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_hc2_coordinate_common_v1 import *
CHECKOUT=ROOT.parent/'visual-grounding-public-A';BRANCH='research/stvg-opd-hc2-coordinate-v1'

def git(*args):return subprocess.check_output(['git',*args],cwd=CHECKOUT).decode().strip()

def stage():
    verify();assert read(BASE/'ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert read(BASE/'ROOT_AUDIT_COMPLETION.json')['statistics_and_chain']=='pass'
    assert read(BASE/'ROOT_BYTE_READBACK.json')['status']=='pass'
    future=read(BASE/'NEW_CONFIGS.json');selected=read(BASE/'SELECTION_BARRIER.json')['config']
    assert future['datasets']['hc2']['config']==selected
    assert future['datasets']['vidstg']==read(BASE/'PREVIOUS_CONFIGS.json')['datasets']['vidstg']
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    assert not git('status','--porcelain');subprocess.run(['git','fetch','--quiet','origin','main'],cwd=CHECKOUT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')
    subprocess.run(['git','switch','-c',BRANCH],cwd=CHECKOUT,check=True)
    owned=set(read(BASE/'RUNTIME_LOCK.json')['pins'])
    owned.update(['scripts/publish_stvg_opd_hc2_coordinate_v1.py','docs/stvg_opd_hc2_coordinate_v1/EXECUTION.md',
        'scripts/audit_stvg_opd_hc2_coordinate_public_v1.py',
        'scripts/audit_stvg_opd_hc2_coordinate_bytes_v1.py',
        'scripts/close_stvg_opd_hc2_coordinate_and_resume_v1.py',
        'scripts/stvg_opd_paper_hc2_revision_common_v2.py','scripts/run_stvg_opd_paper_hc2_revision_v2.py',
        'scripts/continue_stvg_opd_paper_hc2_revision_v2.py','scripts/audit_stvg_opd_revision_aliases_v2.py',
        'protocols/stvg_opd_paper_hc2_revision_v2.md'])
    owned.update(['scripts/prepare_stvg_opd_revision_later_v2.py','scripts/run_stvg_opd_revision_later_v2.py',
        'scripts/continue_stvg_opd_revision_later_v2.py','scripts/stvg_opd_paper_later_common_v1.py',
        'scripts/run_stvg_opd_paper_components_v2.py','scripts/score_stvg_opd_later_phase_v1.py',
        'scripts/finalize_stvg_opd_later_phase_v1.py','scripts/test_stvg_opd_later_contracts_v1.py',
        'scripts/finalize_stvg_opd_table1_v1.py'])
    owned={f for f in owned if f.startswith(('scripts/','vg_tta/','methods/','protocols/','docs/'))}
    for name in ['DESIGN_LOCK.json','RUNTIME_LOCK.json','CPU_CONTRACTS.json','QUALIFICATION.json',
                 'USER_SCOPE.json','LAUNCH.json','SELECTION_BARRIER.json','ROOT_AUDIT_COMPLETION.json','ROOT_VISUAL_REVIEW.json','ROOT_BYTE_READBACK.json']:
        shutil.copy2(BASE/name,PUB/name)
    for d in sorted((BASE/'coordinates').iterdir()):
        dest=PUB/'coordinates'/d.name;dest.mkdir(parents=True,exist_ok=True)
        for name in ['CONFIG.json','PREDICTION_BARRIER.json','SELECTION.json']:shutil.copy2(d/name,dest/name)
    for d in sorted((BASE/'trials').glob('cfg_*')):
        dest=PUB/'trial_receipts'/d.name;dest.mkdir(parents=True,exist_ok=True)
        for name in ['CONFIG.json','PREDICTION_BARRIER.json','GT_EXPOSURE.json','CPU_COMPLETION.json','INVALID_DISPOSITION.json']:
            if (d/name).exists():shutil.copy2(d/name,dest/name)
    pause=ROOT/'artifacts/stvg_opd_paper_v1/user_hc2_coordinate_pause_20261008'
    for name in ['PAUSE_RECEIPT.json','EXACT_RESUME_STATE.json']:
        shutil.copy2(pause/name,PUB/('P1_'+name))
    (PUB/'README.md').write_text('''# HC2 OPD sequential coordinate search\n\n[Actual report](ROOT_REVIEW.md), [all 26 logical candidates](ALL_CANDIDATES.csv), [independent all-source audit](ROOT_AUDIT.json), [selected HC2 configuration](SELECTED_CONFIG.json). Every actual complete trial includes anonymous scalar rows; numerical failures are preserved and never scored as incomplete successful trials.\n\nOne finite greedy pass locks learning rate, sigma, tau, steps, then LN writeback. Each entire coordinate seals before its GT scoring and before the next coordinate is constructed. The source remains frozen VidSTG; all 32 original historically exposed HC2 validation development parent movies and both original orders are included. The 128-parent confirmation panel and full-query P1 scores do not select parameters. This is grid-best development selection, not a global optimum or independent confirmation.\n\nVidSTG configuration and OPD core are unchanged. The original paper P0 negatives and the 685-arrival old-configuration P1 prefix are preserved. The user authorized continuing the original paper after search closing; changed HC2 must use a separately locked source-reset stream. EATA's unavailable direction/media/Fisher remains user paused.\n\nNo video, caption, annotation, raw prediction tensor, weights, gradient, optimizer or 1792-state vector is public. Byte receipts and numerical/selection evidence are public. Gaussian/Adam/state arithmetic and official/dense metrics were independently audited; decoder Jacobians were not independently reimplemented. Capture costs reuse verified source inputs and are not cold end-to-end latency.\n''')
    binding={f:sha(ROOT/f) for f in sorted(owned)}
    binding['methods/decota_spatial_opd_v1/configs.json']=sha(BASE/'NEW_CONFIGS.json')
    write(PUB/'CODE_BINDING.json',dict(code=binding,
        selected_file_registration_happens_after_verified_publication=True,
        runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),private_tensor_media_annotation_exported=False))
    files=sorted(owned|{str(p.relative_to(ROOT)) for p in PUB.rglob('*') if p.is_file()})
    for f in files:
        assert Path(f).suffix not in ['.pt','.npz','.safetensors','.mp4','.sqlite','.pkl']
        dest=CHECKOUT/f;dest.parent.mkdir(parents=True,exist_ok=True)
        source=BASE/'NEW_CONFIGS.json' if f=='methods/decota_spatial_opd_v1/configs.json' else ROOT/f
        shutil.copy2(source,dest)
        if f.endswith(('.json','.jsonl.gz')):
            raw=gzip.decompress(dest.read_bytes()) if f.endswith('.gz') else dest.read_bytes()
            for token in [b'"caption":',b'"video_path":',b'"native_boxes":',b'"GT_box":',b'"committed":',b'"gradient":',b'"optimizer_state":']:
                assert token not in raw,(f,token)
    reproduced=subprocess.run([str(PYTHON),'-B',str(CHECKOUT/'scripts/audit_stvg_opd_hc2_coordinate_public_v1.py'),str(CHECKOUT/PUB.relative_to(ROOT))],cwd=CHECKOUT,capture_output=True,text=True,check=True)
    audit_result=json.loads(reproduced.stdout);assert audit_result['status']=='pass';write(BASE/'PUBLIC_SCALAR_AUDIT.json',audit_result)
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True);changed=git('diff','--cached','--name-only').splitlines();assert changed and set(changed)<=set(files)
    receipts=[dict(path=f,bytes=(CHECKOUT/f).stat().st_size,sha256=sha(CHECKOUT/f),blob_sha=git('hash-object',f)) for f in files]
    write(BASE/'PUBLIC_STAGE.json',dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,
        base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),expected_tree=git('write-tree'),
        changed_files=changed,files=receipts,file_count=len(receipts),bytes=sum(r['bytes'] for r in receipts),time=time.time()))
    print('HC2_COORDINATE_PUBLIC_STAGE',len(receipts),sum(r['bytes'] for r in receipts))

def fetch(url):
    error=None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'STVG-OPD-coordinate-audit'}),timeout=35) as r:return r.read()
        except Exception as e:error=e;time.sleep(attempt+1)
    raise error

def verify_remote(commit):
    meta=read(BASE/'PUBLIC_STAGE.json')
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
    write(BASE/'FINAL_GITHUB_RECEIPT.json',dict(status='pass',repository='Zonglin-He/A',commit=commit,
        branches=['main',BRANCH],files=rows,file_count=len(rows),bytes=sum(r['bytes'] for r in rows),time=time.time()))
    print('HC2_COORDINATE_REMOTE_BYTES_VERIFIED',commit,len(rows))

if __name__=='__main__':stage() if sys.argv[1]=='stage' else verify_remote(sys.argv[2])
