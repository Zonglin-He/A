"""Complete explicitly bounded anonymous P3 export and portable verification."""
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate,BASE,PUB,PYTHON,read,write,sha
from scripts.run_stvg_opd_p3_scope_precision001 import verify
EXPORT=ROOT/'results/stvg_opd_p3_complete/2026-10-10'
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
BRANCH='research/stvg-opd-paper-hc2-revision-v2'


def git(*args):return subprocess.check_output(['git',*args],cwd=CHECKOUT,text=True).strip()


def run():
    activate();verify()
    assert read(BASE/'P3_ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert read(BASE/'P3_actual_root_readback/COMPLETION.json')['actual_arrivals']==7168
    assert read(BASE/'P3_actual_root_signal_views/COMPLETION.json')['cases']==6
    assert read(PUB/'P3/ACTUAL_ROOT_MATH_STATE_DENSE_READBACK.json')['status']=='pass'
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    unrelated=git('ls-files','--others','--exclude-standard').splitlines()
    assert all('stvg_motivation' in f.lower() or f.startswith('results/stvg_opd_p3_scope_launch/') for f in unrelated)
    EXPORT.mkdir(parents=True,exist_ok=True);mapped={};pins=set()
    def copy(source,target):
        assert source.is_file() and source.suffix in {'.py','.md','.json','.png','.pdf','.gz'}
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        mapped[str(source.relative_to(ROOT))]=dict(public_path=str(target.relative_to(ROOT)),sha256=sha(source),bytes=source.stat().st_size)
    for f in sorted((PUB/'P3').iterdir()):
        if f.is_file():copy(f,EXPORT/f.name)
    from scripts.stvg_opd_paper_later_common_v1 import phases
    for name in phases()['P3']:
        folder=EXPORT/'stages'/name;folder.mkdir(parents=True,exist_ok=True)
        copy(PUB/name/'SUMMARY.json',folder/'SUMMARY.json')
        copy(PUB/name/'ROWS.jsonl.gz',folder/'ROWS.jsonl.gz')
        copy(BASE/'stages'/name/'PREDICTION_BARRIER.json',EXPORT/'receipts/stages'/name/'PREDICTION_BARRIER.json')
    runtimes=['RUNTIME_LOCK.json','P1_PRECISION_RUNTIME.json','COMPONENT_RUNTIME_LOCK.json',
        'COMPONENT_RUNTIME_LOCK_revision001.json','LATER_CPU_RUNTIME_LOCK.json','P2_engineering_revision002/REVISION_RUNTIME.json',
        'recovery/P2_input_schema_003/REVISION_RUNTIME.json','recovery/P2_inline_gradient_precision_004/REVISION_RUNTIME.json',
        'recovery/P2_actual_action_reward_precision_005/REVISION_RUNTIME.json','recovery/P2_matched_gradient_precision_006/REVISION_RUNTIME.json',
        'P3_scope_engineering_revision001/REVISION_RUNTIME.json',
        'P3_actual_root_readback/RUNTIME.json','P3_actual_root_signal_views/RUNTIME.json']
    for rel in runtimes:
        p=BASE/rel;pins.update(read(p)['pins']);copy(p,EXPORT/'receipts'/rel)
    for name in ['DESIGN_LOCK.json','LATER_DESIGN_LOCK.json','APPENDIX_STAGE_LOCK.json','P3_STAGE_AUTHORIZATION.json',
        'P3_QUALIFICATION.json','P3_PREDICTION_BARRIER.json','P3_GPU_COMPLETION.json',
        'P3_CPU_SCORING_COMPLETION.json','P3_CPU_COMPLETION.json','P3_ROOT_VISUAL_REVIEW.json']:
        copy(BASE/name,EXPORT/'receipts'/name)
    for folder in ['P3_actual_root_readback','P3_actual_root_signal_views']:
        for name in ['LAUNCH.json','COMPLETION.json']:
            copy(BASE/folder/name,EXPORT/'receipts'/folder/name)
    pins.update(['scripts/publish_stvg_opd_revised_p3_v2.py','scripts/audit_stvg_opd_later_public_v1.py',
        'scripts/verify_stvg_opd_public_remote_v1.py','scripts/report_stvg_opd_revised_p3_root_v2.py','docs/STVG_OPD_REVISED_P3_ROOT_REVIEW.md','methods/CURRENT_METHOD.json','methods/decota_spatial_opd_v1/configs.json'])
    metadata={f for f in pins if f.startswith('artifacts/')}
    for f in sorted(metadata):
        p=ROOT/f;assert p.is_relative_to(BASE) and p.suffix=='.json'
        copy(p,EXPORT/'receipts'/p.relative_to(BASE))
    result_evidence={f for f in pins if f.startswith('results/')}
    for f in sorted(result_evidence):
        p=ROOT/f;assert p.parent==PUB/'P3' and p.is_file()
        assert f in mapped and mapped[f]['sha256']==sha(p)
    implementation=pins-metadata-result_evidence
    assert all(f.startswith(('scripts/','vg_tta/','protocols/','docs/','methods/')) and (ROOT/f).is_file() for f in implementation)
    write(EXPORT/'CODE_BINDING.json',dict(status='complete_P3_actual_root_evidence_pending_remote_verification',
        files={f:sha(ROOT/f) for f in sorted(pins)},exact_metadata_projection=mapped,
        anonymous_logical_rows=7168,new_formal_fits=5376,exact_original_stream_reuse_rows=1792,parent_sources=256,
        actual_math_state_dense_sha256=sha(PUB/'P3/ACTUAL_ROOT_MATH_STATE_DENSE_READBACK.json'),
        actual_visual_review_sha256=sha(BASE/'P3_ROOT_VISUAL_REVIEW.json'),private_payloads_exported=False,paper_suite_complete=False))
    (EXPORT/'README.md').write_text('''# Fixed OPD P3: complete parameter-scope evidence

Read [the actual root review](ACTUAL_ROOT_REVIEW.md), complete paired controls in
[ROOT_STATISTICS.json](ROOT_STATISTICS.json), [all parent effects](ALL_PARENT_EFFECTS.json),
[negative attribution](FAILURE_STRATA.json) and [actual costs](COST.json).
All 7,168 anonymous logical rows are under `stages`; 5,376 new formal fits and 1,792 exact
complete-stream Full aliases globally sealed before GT. No qualification is a formal row.

```bash
python -B scripts/audit_stvg_opd_later_public_v1.py results/stvg_opd_p3_complete/2026-10-10
```

Full exceeds Query-only clearly in HC2 same-domain and VidSTG cross-domain means.
Incremental Full-minus-LN-only is clear only for VidSTG cross-domain.
Full-minus-alpha0 is positive for HC2 cross-domain; other three setting intervals contain
zero. Full inherited delta is a within-trajectory decomposition, not Full-minus-alpha0.
Complete negative tails and all condition/order results are retained. All saved scope/math/
1792 state/dense and source statistics were actually checked; four plots and six private
RGB cases with three complete scope controls per case were actually viewed/read.
The exposed 128-parent cohorts are not fresh. Public arithmetic is reproducible;
private model inference/full decoder Jacobians are not reproduced by this package.
No private RGB/query/caption/GT geometry/box/action/weights/fit/gradient/Adam payload is
exported. Original P4-P6 still require their actual root/view/public/archive closing;
EATA and historical paused queues remain paused. No retuning or method promotion.
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
    write(BASE/'P3_PUBLIC_SCALAR_AUDIT.json',dict(audited,actual_public_checkout=True,time=time.time()))
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True)
    changed=git('diff','--cached','--name-only').splitlines();assert changed and set(changed)<=set(files)
    records=[dict(path=f,bytes=(CHECKOUT/f).stat().st_size,sha256=sha(CHECKOUT/f),blob_sha=git('hash-object',f)) for f in files]
    write(BASE/'P3_PUBLIC_STAGE.json',dict(status='reviewed_staged',scope='complete actual P3 evidence only',repository='Zonglin-He/A',
        branch=BRANCH,base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),expected_tree=git('write-tree'),
        changed_files=changed,files=records,file_count=len(records),bytes=sum(v['bytes'] for v in records),
        preserved_unrelated_untracked=unrelated,portable_scalar_audit=audited,paper_suite_complete=False,time=time.time()))
    print(json.dumps(dict(status='ready',files=len(records),bytes=sum(v['bytes'] for v in records),changed_files=len(changed),scalar_comparisons=audited['scalar_comparisons'])))


if __name__=='__main__':run()
