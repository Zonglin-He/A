"""Complete explicitly bounded anonymous P2 export and portable verification."""
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate,BASE,PUB,PYTHON,read,write,sha
from scripts.run_stvg_opd_p2_matched_gradient006 import verify
EXPORT=ROOT/'results/stvg_opd_p2_complete/2026-10-10'
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
BRANCH='research/stvg-opd-paper-hc2-revision-v2'


def git(*args):return subprocess.check_output(['git',*args],cwd=CHECKOUT,text=True).strip()


def run():
    activate();verify()
    assert read(BASE/'P2_ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert read(BASE/'P2_actual_root_readback/COMPLETION.json')['actual_arrivals']==7168
    assert read(BASE/'P2_actual_root_signal_views/COMPLETION.json')['cases']==6
    assert read(PUB/'P2/ACTUAL_ROOT_MATH_STATE_DENSE_READBACK.json')['status']=='pass'
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    unrelated=git('ls-files','--others','--exclude-standard').splitlines()
    assert all('stvg_motivation' in f.lower() for f in unrelated)
    EXPORT.mkdir(parents=True,exist_ok=True);mapped={};pins=set()
    def copy(source,target):
        assert source.is_file() and source.suffix in {'.py','.md','.json','.png','.pdf','.gz'}
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        mapped[str(source.relative_to(ROOT))]=dict(public_path=str(target.relative_to(ROOT)),sha256=sha(source),bytes=source.stat().st_size)
    for f in sorted((PUB/'P2').iterdir()):
        if f.is_file():copy(f,EXPORT/f.name)
    from scripts.stvg_opd_paper_later_common_v1 import phases
    for name in phases()['P2']:
        folder=EXPORT/'stages'/name;folder.mkdir(parents=True,exist_ok=True)
        copy(PUB/name/'SUMMARY.json',folder/'SUMMARY.json')
        copy(PUB/name/'ROWS.jsonl.gz',folder/'ROWS.jsonl.gz')
        copy(BASE/'stages'/name/'PREDICTION_BARRIER.json',EXPORT/'receipts/stages'/name/'PREDICTION_BARRIER.json')
    runtimes=['RUNTIME_LOCK.json','P1_PRECISION_RUNTIME.json','COMPONENT_RUNTIME_LOCK.json',
        'COMPONENT_RUNTIME_LOCK_revision001.json','LATER_CPU_RUNTIME_LOCK.json','P2_engineering_revision002/REVISION_RUNTIME.json',
        'recovery/P2_input_schema_003/REVISION_RUNTIME.json','recovery/P2_inline_gradient_precision_004/REVISION_RUNTIME.json',
        'recovery/P2_actual_action_reward_precision_005/REVISION_RUNTIME.json','recovery/P2_matched_gradient_precision_006/REVISION_RUNTIME.json',
        'P2_actual_root_readback/RUNTIME.json','P2_actual_root_readback/RUNTIME_revision001.json',
        'P2_actual_root_readback/RUNTIME_revision002.json','P2_actual_root_signal_views/RUNTIME.json']
    for rel in runtimes:
        p=BASE/rel;pins.update(read(p)['pins']);copy(p,EXPORT/'receipts'/rel)
    for name in ['DESIGN_LOCK.json','LATER_DESIGN_LOCK.json','APPENDIX_STAGE_LOCK.json','P2_STAGE_AUTHORIZATION.json',
        'P2_QUALIFICATION.json','P2_PREDICTION_BARRIER.json','P2_GPU_COMPLETION.json',
        'P2_CPU_SCORING_COMPLETION.json','P2_CPU_COMPLETION.json','P2_ROOT_VISUAL_REVIEW.json']:
        copy(BASE/name,EXPORT/'receipts'/name)
    for folder in ['P2_actual_root_readback','P2_actual_root_signal_views']:
        for name in ['LAUNCH.json','COMPLETION.json']:
            copy(BASE/folder/name,EXPORT/'receipts'/folder/name)
    for name in ['RELAUNCH_revision001.json','RELAUNCH_revision002.json']:
        copy(BASE/'P2_actual_root_readback'/name,EXPORT/'receipts/P2_actual_root_readback'/name)
    for folder in ['root_source_reset_001','root_immutable_resume_002']:
        copy(BASE/'P2_actual_root_readback/recovery'/folder/'CAPTURE_RECEIPT.json',EXPORT/'receipts/root_helper_recovery'/folder/'CAPTURE_RECEIPT.json')
    pins.update(['scripts/publish_stvg_opd_revised_p2_v2.py','scripts/audit_stvg_opd_later_public_v1.py',
        'docs/STVG_OPD_REVISED_P2_ROOT_REVIEW.md','methods/CURRENT_METHOD.json','methods/decota_spatial_opd_v1/configs.json'])
    metadata={f for f in pins if f.startswith('artifacts/')}
    for f in sorted(metadata):
        p=ROOT/f;assert p.is_relative_to(BASE) and p.suffix=='.json'
        copy(p,EXPORT/'receipts'/p.relative_to(BASE))
    implementation=pins-metadata
    assert all(f.startswith(('scripts/','vg_tta/','protocols/','docs/','methods/')) and (ROOT/f).is_file() for f in implementation)
    write(EXPORT/'CODE_BINDING.json',dict(status='complete_P2_actual_root_evidence_pending_remote_verification',
        files={f:sha(ROOT/f) for f in sorted(pins)},exact_metadata_projection=mapped,
        anonymous_logical_rows=7168,new_formal_fits=5632,exact_original_stream_reuse_rows=1536,parent_sources=256,
        actual_math_state_dense_sha256=sha(PUB/'P2/ACTUAL_ROOT_MATH_STATE_DENSE_READBACK.json'),
        actual_visual_review_sha256=sha(BASE/'P2_ROOT_VISUAL_REVIEW.json'),private_payloads_exported=False,paper_suite_complete=False))
    (EXPORT/'README.md').write_text('''# Fixed OPD P2: complete mechanism evidence

Read [the actual root review](ACTUAL_ROOT_REVIEW.md), complete paired controls in
[ROOT_STATISTICS.json](ROOT_STATISTICS.json), [all parent effects](ALL_PARENT_EFFECTS.json),
[negative attribution](FAILURE_STRATA.json) and [actual costs](COST.json).
All 7,168 anonymous logical rows, including failures and no-ops, are under `stages`.
5,632 new formal fits and 1,536 exact original complete-stream aliases globally sealed before GT.

Recompute all aggregate, paired bootstrap, tails, strata and cost arithmetic:

```bash
python -B scripts/audit_stvg_opd_later_public_v1.py results/stvg_opd_p2_complete/2026-10-10
```

Full beats shuffled feedback in both directions/settings; advantages over Fixed Rollout
depend on the setting. Full-minus-Direct intervals include zero in all four setting means.
The historical 128-parent cohorts are not fresh unseen panels. All severe failures and
wrong-expert attraction remain. Root actually read every saved fit/math/state/dense result
and viewed four plot pairs and six distinct private RGB cases. Public arithmetic is
independently reproducible; private inference/full decoder Jacobian is not reproduced here.
Raw RGB/query/caption/GT coordinates, boxes/actions/weights/fit/gradient/Adam arrays and
credentials are excluded. P3–P6 retain their own actual execution/root/public/archive gates.
EATA and all historical paused queues remain paused.
''')
    files=sorted(implementation|{str(f.relative_to(ROOT)) for f in EXPORT.rglob('*') if f.is_file()})
    forbidden=[b'"caption":',b'"video_path":',b'"native_boxes":',b'"GT_box":',b'"committed":',b'"gradient":',b'"optimizer_state":',b'"raw_logits":']
    for rel in files:
        assert Path(rel).suffix in {'.py','.json','.md','.gz','.png','.pdf'}
        raw=(ROOT/rel).read_bytes()
        if rel.endswith('.gz'):raw=gzip.decompress(raw)
        if rel.endswith(('.json','.gz')):
            for token in forbidden:assert token not in raw,(rel,token)
        dest=CHECKOUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    audited=json.loads(subprocess.check_output([str(PYTHON),'-B',str(CHECKOUT/'scripts/audit_stvg_opd_later_public_v1.py'),str(CHECKOUT/EXPORT.relative_to(ROOT))],cwd=CHECKOUT,text=True))
    assert audited['status']=='pass' and audited['logical_rows']==7168 and audited['scalar_comparisons']==239436
    write(BASE/'P2_PUBLIC_SCALAR_AUDIT.json',dict(audited,actual_public_checkout=True,time=time.time()))
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True)
    changed=git('diff','--cached','--name-only').splitlines();assert changed and set(changed)<=set(files)
    records=[dict(path=f,bytes=(CHECKOUT/f).stat().st_size,sha256=sha(CHECKOUT/f),blob_sha=git('hash-object',f)) for f in files]
    write(BASE/'P2_PUBLIC_STAGE.json',dict(status='reviewed_staged',scope='complete actual P2 evidence only',repository='Zonglin-He/A',
        branch=BRANCH,base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),expected_tree=git('write-tree'),
        changed_files=changed,files=records,file_count=len(records),bytes=sum(v['bytes'] for v in records),
        preserved_unrelated_untracked=unrelated,portable_scalar_audit=audited,paper_suite_complete=False,time=time.time()))
    print(json.dumps(dict(status='ready',files=len(records),bytes=sum(v['bytes'] for v in records),changed_files=len(changed),scalar_comparisons=audited['scalar_comparisons'])))


if __name__=='__main__':run()
