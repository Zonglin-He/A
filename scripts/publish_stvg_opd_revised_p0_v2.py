"""Explicit anonymous revised P0 export; old reports and private assets stay intact."""
import gzip,json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,PUB,PYTHON,verify,read,write,sha
CHECKOUT=ROOT.parent/'visual-grounding-public-A';BRANCH='research/stvg-opd-paper-hc2-revision-v2'

def git(*args):return subprocess.check_output(['git',*args],cwd=CHECKOUT).decode().strip()

def stage():
    verify();assert read(BASE/'P0_ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert read(BASE/'P0_ROOT_DECISION.json')['all_128_sources_reviewed']
    assert read(BASE/'P0_CPU_COMPLETION.json')['P0_postseal_scoring_complete']
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    # A preserved interrupted staging attempt may have only these owned files.
    subprocess.run(['git','fetch','--quiet','origin','main'],cwd=CHECKOUT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')
    if git('branch','--show-current')!=BRANCH:
        subprocess.run(['git','switch','-c',BRANCH],cwd=CHECKOUT,check=True)
    locks=['RUNTIME_LOCK.json','P0_POSTSEAL_ROOT_RUNTIME.json','P1_PRECISION_RUNTIME.json',
        'COMPONENT_RUNTIME_LOCK_revision001.json','LATER_CPU_RUNTIME_LOCK.json']
    owned=set()
    for name in locks:owned.update(read(BASE/name)['pins'])
    owned.update(read(BASE/'recovery/gradient_audit_001/REVISION_RUNTIME.json')['pins'])
    owned.update(['scripts/publish_stvg_opd_revised_p0_v2.py','scripts/publish_stvg_opd_p0_v1.py',
        'methods/decota_spatial_opd_v1/configs.json','methods/CURRENT_METHOD.json',
        'vg_tta/decota_spatial_opd_tunable_v1.py','vg_tta/decota_spatial_opd_tunable_audit_v1.py',
        'docs/STVG_OPD_REVISED_P0_ROOT_REVIEW.md','docs/stvg_opd_paper_hc2_revision_v2/EXECUTION.md'])
    for name in ['DESIGN_LOCK.json','LATER_DESIGN_LOCK.json','APPENDIX_STAGE_LOCK.json',
        'QUALIFICATION.json','P1_BASELINE_BINDING.json','P0_ROOT_VISUAL_REVIEW.json',
        'P0_ROOT_DECISION.json','P0_ROOT_BYTE_READBACK.json',*locks]:
        shutil.copy2(BASE/name,PUB/name)
    write(PUB/'CODE_BINDING.json',dict(code={f:sha(ROOT/f) for f in sorted(owned)},
        GPU_runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),
        precision_audit_revision_sha256=sha(BASE/'recovery/gradient_audit_001/REVISION_RUNTIME.json'),
        original_code_and_algorithm_unchanged=True,only_HC_registered_config_changed_before_P0=True,
        P2_P6_not_executed=True,private_GT_media_state_gradient_payload_exported=False))
    (PUB/'README.md').write_text('''# Revised HC2 OPD: actual P0 completion evidence\n\nRead [the actual root review](ROOT_ACTUAL_READBACK.md), [all matched statistics](P0_REPORT.md), and [all-source failure attribution](P0_POPULATION_FAILURE_ATTRIBUTION.md). All positive, negative and no-update outcomes remain in the anonymous compressed rows. HC2 uses the closed coordinate-search configuration; VidSTG evidence is exact unchanged full-history reuse. This128-parent panel has historical exposure and is not fresh unseen evaluation.\n\nThe original both-direction positive-efficacy gate fails; root decides to continue the human-authorized fixed-configuration official P1 table after full failure attribution. No P0-based retuning, roster selection, or new expert/gate is authorized. P1–P6 completion is not established here.\n\nReproduce every scalar comparison with Python/NumPy, without private media/labels/weights:\n\n```bash\npython -B scripts/audit_stvg_opd_p0_public_v1.py results/stvg_opd_paper_hc2_revision/2026-10-08\n```\n\nThe engineering supplement preserves the failed cross-precision assertion and validates same-precision autodiff plus an explicit rounding bound; no fitting algorithm or saved prefix changed. GPU replay requires separately obtained official media/checkpoints and DINO. Decoder Jacobian was not independently reimplemented. Private videos, captions, GT, gradients, states, predictions and optimizer tensor caches are excluded. EATA's unavailable direction remains user paused.\n''')
    files=sorted(owned|{str(p.relative_to(ROOT)) for p in PUB.rglob('*') if p.is_file()})
    existing_changes=set(git('diff','HEAD','--name-only').splitlines())|set(git('ls-files','--others','--exclude-standard').splitlines())
    assert existing_changes<=set(files),existing_changes-set(files)
    for f in files:
        assert f.startswith(('scripts/','vg_tta/','methods/','protocols/','docs/','results/'))
        assert Path(f).suffix not in {'.pt','.npz','.mp4','.sqlite','.safetensors'}
        target=CHECKOUT/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,target)
        if f.endswith(('.json','.jsonl.gz')):
            raw=gzip.decompress(target.read_bytes()) if f.endswith('.gz') else target.read_bytes()
            for token in [b'"caption":',b'"video_path":',b'"native_boxes":',b'"GT_box":',b'"committed":',b'"gradient":',b'"optimizer_state":']:
                assert token not in raw,(f,token)
    result=subprocess.run([str(PYTHON),'-B',str(CHECKOUT/'scripts/audit_stvg_opd_p0_public_v1.py'),str(CHECKOUT/PUB.relative_to(ROOT))],cwd=CHECKOUT,check=True,capture_output=True,text=True)
    audited=json.loads(result.stdout);assert audited['status']=='pass'
    if (BASE/'P0_PUBLIC_SCALAR_AUDIT.json').exists():assert read(BASE/'P0_PUBLIC_SCALAR_AUDIT.json')==audited
    else:write(BASE/'P0_PUBLIC_SCALAR_AUDIT.json',audited)
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True)
    changed=git('diff','--cached','--name-only').splitlines();assert changed and set(changed)<=set(files)
    receipts=[dict(path=f,bytes=(CHECKOUT/f).stat().st_size,sha256=sha(CHECKOUT/f),blob_sha=git('hash-object',f)) for f in files]
    write(BASE/'P0_PUBLIC_STAGE.json',dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,
        base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),expected_tree=git('write-tree'),
        changed_files=changed,files=receipts,file_count=len(receipts),bytes=sum(r['bytes'] for r in receipts),time=time.time()))
    print('REVISED_P0_PUBLIC_STAGE_READY',len(receipts),sum(r['bytes'] for r in receipts))

def remote(commit):
    import scripts.publish_stvg_opd_p0_v1 as old
    old.BASE=BASE;old.PUB=PUB;old.BRANCH=BRANCH;old.verify=verify
    old.verify_remote(commit)

if __name__=='__main__':stage() if sys.argv[1]=='stage' else remote(sys.argv[2])
