"""Reviewed complete original P6 anonymous export; no private model or GT payloads."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_p6_existing_common001 import verify,BASE,PUB,NS,PYTHON,read,write,sha
EXPORT=ROOT/'results/stvg_opd_p6_complete/2026-10-10'
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
BRANCH='research/stvg-opd-paper-hc2-revision-v2'


def git(*args):return subprocess.check_output(['git',*args],cwd=CHECKOUT,text=True).strip()


def safe_json(value):
    forbidden={'caption','video_path','native_boxes','GT_box','GT_boxes','committed','gradient','optimizer_state',
               'raw_logits','hidden_states','actual_actions','actual_samples','actions','sample_logits','sample_boxes',
               'query_text','GT_geometry','frame_pixels','Fisher','weights'}
    if isinstance(value,dict):
        assert not set(value)&forbidden,set(value)&forbidden
        for v in value.values():safe_json(v)
    elif isinstance(value,list):
        for v in value:safe_json(v)


def run():
    runtime=verify();assert read(BASE/'P6_ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert read(BASE/'P6_PREDICTION_BARRIER.json')['logical_outputs']==256
    assert read(PUB/'P6/ROOT_MATH_STATE_DENSE_READBACK.json')['counts']['formal_queries']==64
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    unrelated=git('ls-files','--others','--exclude-standard').splitlines()
    assert all('stvg_motivation' in p.lower() for p in unrelated)
    EXPORT.mkdir(parents=True,exist_ok=True);mapped={}
    def copy(source,target):
        assert source.is_file() and source.suffix in {'.py','.md','.json','.png','.pdf'}
        if source.suffix=='.json':safe_json(read(source))
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        assert source.read_bytes()==target.read_bytes()
        mapped[str(source.relative_to(ROOT))]=dict(public_path=str(target.relative_to(EXPORT)),bytes=source.stat().st_size,sha256=sha(source))
    for source in sorted((PUB/'P6').iterdir()):
        assert source.is_file() and source.suffix in {'.json','.md','.png','.pdf'}
        copy(source,EXPORT/source.name)
    for rel in ['DESIGN_LOCK.json','LATER_DESIGN_LOCK.json','P5_ROOT_CLOSING_RECEIPT.json','P5_FINAL_ARCHIVE_SNAPSHOT_RECEIPT.json',
        'P5_COMPLETE_CLOSING_GITHUB_RECEIPT.json','P6_QUALIFICATION.json','P6_PREDICTION_BARRIER.json',
        'P6_GPU_COMPLETION.json','P6_CPU_COMPLETION.json','P6_ROOT_VISUAL_REVIEW.json']:
        copy(BASE/rel,EXPORT/'receipts'/rel)
    receipts=['REVISION_RUNTIME.json','REVISION_RUNTIME_revision002.json','CPU_CONTRACTS.json',
        'CPU_ORIGINAL_INPUT_PREFLIGHT.json','LAUNCH.json','GT_EXPOSURE.json','COMPLETION.json',
        'actual_root_signal_views/RUNTIME.json','actual_root_signal_views/LAUNCH.json','actual_root_signal_views/COMPLETION.json',
        'actual_root_signal_views/VISUAL_LABEL_REVISION.json','actual_root_signal_views/PRIVATE_RGB_READBACK.json',
        'actual_root_signal_views/PRIVATE_RGB_GT_SUPPORT_REVISION.json',
        'qualification/hc2/GPU_QUALIFICATION.json','qualification/vidstg/GPU_QUALIFICATION.json']
    for rel in receipts:copy(NS/rel,EXPORT/'receipts/P6_existing_temporal_revision001'/rel)
    for source in sorted((NS/'recovery').rglob('*')):
        if source.is_file() and source.suffix in {'.py','.json'}:
            copy(source,EXPORT/'receipts/P6_existing_temporal_revision001'/source.relative_to(NS))
    implementation=set(runtime['pins'])|set(read(NS/'actual_root_signal_views/RUNTIME.json')['pins'])
    implementation.update(['scripts/render_stvg_opd_p6_existing_root001.py','scripts/revise_stvg_opd_p6_bar_labels001.py',
        'scripts/render_stvg_opd_p6_GT_support002.py','scripts/report_stvg_opd_p6_existing_root001.py',
        'scripts/audit_stvg_opd_p6_export001.py','scripts/publish_stvg_opd_p6_existing_root001.py',
        'scripts/verify_stvg_opd_public_remote_v1.py','docs/STVG_OPD_REVISED_P6_ROOT_REVIEW.md',
        'methods/decota_spatial_opd_v1/configs.json','vg_tta/tastvg_oracle_event5_v1.py'])
    assert all(p.startswith(('scripts/','methods/','protocols/','docs/','external/TA-STVG/models/','vg_tta/')) for p in implementation)
    for p in implementation:
        assert (ROOT/p).is_file() and (ROOT/p).suffix in {'.py','.json','.md'}
        if (ROOT/p).suffix=='.json':safe_json(read(ROOT/p))
    (EXPORT/'README.md').write_text('''# Fixed P6: existing temporal alternatives and offline supervised diagnostic

Read [the actual complete root review](ACTUAL_ROOT_REVIEW.md), every anonymous
query/parent result in [ROWS](ROWS.json), all paired contrasts and negative tails
in [STATISTICS](STATISTICS.json), and six descriptive signal chains. The original
64 queries / 256 deployment outputs all globally sealed before GT. Four genuine
qualification arrivals / eight complete GPU fits remain excluded from formal.
The separate 64 CPU GT-head fits use postseal supervision and are not deployable
baselines. Source/config/roster/state histories are unchanged; no new temporal
branch, parameter selection or method promotion follows from these results.

```bash
python -B scripts/audit_stvg_opd_p6_export001.py results/stvg_opd_p6_complete/2026-10-10
```

Temporal arms keep Native boxes, SpatialOPD keeps Native time. SpatialOPD reuses
each matched original complete P0 fit with its full 128-query predecessor LN
history; it is not a fresh 32-query stream or a state-history-matched causal arm.
The original fixed 32 parents per target are historically exposed. All 10,000
paired-parent intervals are conditional on this roster and not multiplicity
adjusted. Recorded timings retain recording/cache conditions, not cold latency.

All six final report PNG/PDF pairs and six original-pixel RGB cases were actually
viewed. Display-only label/GT-support frame revisions preserve the original
images, scores, predictions, official GT geometry and case selection. Private
RGB/query/caption/GT arrays/native boxes/actions/logits/prefix/weights/gradient/
Adam/fit/cache payloads are excluded. Only their exact hash/count bindings and
anonymous actual scalar results are public. Preserved unsealed CPU helper
corrections are metadata/signature/display-path changes, not new model failures.

Public arithmetic independently reproduces all anonymous statistics and byte
bindings. It does not recreate private media or prove the complete decoder
Jacobian/CUDA kernels/future numerical safety. No new DINO calls occur in P6.
P6 closure additionally requires actual remote verification and archive receipts;
whole-suite completion requires all P1-P6 closures and actual FINAL_COMPLETION.
EATA and all historical paused queues remain paused.
''')
    public_files={str(p.relative_to(EXPORT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in EXPORT.rglob('*') if p.is_file()}
    write(EXPORT/'CODE_BINDING.json',dict(status='complete_P6_actual_root_evidence_pending_remote_verification',
        files={p:sha(ROOT/p) for p in sorted(implementation)},public_files=public_files,exact_source_projection=mapped,
        formal_queries=64,logical_deployment_outputs=256,offline_CPU_GT_head_fits=64,
        actual_full_root_sha256=sha(PUB/'P6/ROOT_MATH_STATE_DENSE_READBACK.json'),actual_visual_review_sha256=sha(BASE/'P6_ROOT_VISUAL_REVIEW.json'),
        private_payloads_exported=False,main_method_changed=False,paper_suite_complete=False))
    files=sorted(implementation|{str(p.relative_to(ROOT)) for p in EXPORT.rglob('*') if p.is_file()})
    for rel in files:
        p=ROOT/rel;assert p.suffix in {'.py','.json','.md','.png','.pdf'}
        if p.suffix=='.json':safe_json(read(p))
        target=CHECKOUT/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    audited=json.loads(subprocess.check_output([str(PYTHON),'-B',str(CHECKOUT/'scripts/audit_stvg_opd_p6_export001.py'),str(CHECKOUT/EXPORT.relative_to(ROOT))],cwd=CHECKOUT,text=True))
    assert audited['status']=='pass' and audited['queries']==64 and audited['logical_deployment_outputs']==256
    write(BASE/'P6_PUBLIC_SCALAR_AUDIT.json',dict(audited,actual_public_checkout=True,time=time.time()))
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True)
    changed=git('diff','--cached','--name-only').splitlines();assert changed and set(changed)<=set(files)
    records=[dict(path=f,bytes=(CHECKOUT/f).stat().st_size,sha256=sha(CHECKOUT/f),blob_sha=git('hash-object',f)) for f in files]
    write(BASE/'P6_PUBLIC_STAGE.json',dict(status='reviewed_staged',scope='complete original actual P6 deployment and separately supervised diagnostic evidence',repository='Zonglin-He/A',
        branch=BRANCH,base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),expected_tree=git('write-tree'),changed_files=changed,
        files=records,file_count=len(records),bytes=sum(v['bytes'] for v in records),preserved_unrelated_untracked=unrelated,portable_scalar_audit=audited,paper_suite_complete=False,time=time.time()))
    print(json.dumps(dict(status='ready',files=len(records),bytes=sum(v['bytes'] for v in records),changed_files=len(changed),scalar_comparisons=audited['scalar_comparisons'])))


if __name__=='__main__':run()
