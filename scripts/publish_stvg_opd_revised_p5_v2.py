"""Complete explicitly bounded anonymous P5 export and portable verification."""
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate,BASE,PUB,PYTHON,read,write,sha
from scripts.run_stvg_opd_p5_budget_precision001 import verify
EXPORT=ROOT/'results/stvg_opd_p5_complete/2026-10-10'
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
BRANCH='research/stvg-opd-paper-hc2-revision-v2'


def git(*args):return subprocess.check_output(['git',*args],cwd=CHECKOUT,text=True).strip()


def run():
    activate();verify()
    assert read(BASE/'P5_ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert read(BASE/'P5_actual_root_readback_revision004/COMPLETION.json')['actual_arrivals']==2560
    assert read(BASE/'P5_actual_root_signal_views/COMPLETION.json')['cases']==6
    assert read(PUB/'P5/ACTUAL_ROOT_MATH_STATE_DENSE_READBACK.json')['status']=='pass'
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    unrelated=git('ls-files','--others','--exclude-standard').splitlines()
    assert all('stvg_motivation' in f.lower() or f.startswith(('results/stvg_opd_p3_scope_launch/','results/stvg_opd_p4_joint_launch/','results/stvg_opd_p5_budget_launch/')) for f in unrelated)
    EXPORT.mkdir(parents=True,exist_ok=True);mapped={};pins=set()
    def copy(source,target):
        assert source.is_file() and source.suffix in {'.py','.md','.json','.png','.pdf','.gz'}
        compressed=source.suffix=='.json' and source.stat().st_size>2*2**20
        if compressed:target=target.with_name(target.name+'.gz')
        target.parent.mkdir(parents=True,exist_ok=True)
        if compressed:
            raw=source.read_bytes();target.write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))
            assert gzip.decompress(target.read_bytes())==raw
        else:shutil.copy2(source,target)
        mapped[str(source.relative_to(ROOT))]=dict(public_path=str(target.relative_to(ROOT)),sha256=sha(source),bytes=source.stat().st_size,
            public_sha256=sha(target),public_bytes=target.stat().st_size,encoding='gzip_exact_original_bytes' if compressed else 'identity')
    for f in sorted((PUB/'P5').iterdir()):
        if f.is_file():copy(f,EXPORT/f.name)
    from scripts.stvg_opd_paper_later_common_v1 import phases
    for name in phases()['P5']:
        folder=EXPORT/'stages'/name;folder.mkdir(parents=True,exist_ok=True)
        copy(PUB/name/'SUMMARY.json',folder/'SUMMARY.json')
        copy(PUB/name/'ROWS.jsonl.gz',folder/'ROWS.jsonl.gz')
        copy(BASE/'stages'/name/'PREDICTION_BARRIER.json',EXPORT/'receipts/stages'/name/'PREDICTION_BARRIER.json')
    runtimes=['RUNTIME_LOCK.json','P1_PRECISION_RUNTIME.json','COMPONENT_RUNTIME_LOCK.json',
        'COMPONENT_RUNTIME_LOCK_revision001.json','LATER_CPU_RUNTIME_LOCK.json','P2_engineering_revision002/REVISION_RUNTIME.json',
        'recovery/P2_input_schema_003/REVISION_RUNTIME.json','recovery/P2_inline_gradient_precision_004/REVISION_RUNTIME.json',
        'recovery/P2_actual_action_reward_precision_005/REVISION_RUNTIME.json','recovery/P2_matched_gradient_precision_006/REVISION_RUNTIME.json',
        'P3_scope_engineering_revision001/REVISION_RUNTIME.json','P4_joint_engineering_revision001/REVISION_RUNTIME.json',
        'P5_budget_engineering_revision001/REVISION_RUNTIME.json','P5_actual_root_readback/RUNTIME.json',
        'P5_actual_root_readback_revision001/RUNTIME.json','P5_actual_root_readback_revision002/RUNTIME.json',
        'P5_actual_root_readback_revision003/RUNTIME.json','P5_actual_root_readback_revision004/RUNTIME.json','P5_actual_root_signal_views/RUNTIME.json']
    for rel in runtimes:
        p=BASE/rel;pins.update(read(p)['pins']);copy(p,EXPORT/'receipts'/rel)
    for name in ['DESIGN_LOCK.json','LATER_DESIGN_LOCK.json','APPENDIX_STAGE_LOCK.json','P5_STAGE_AUTHORIZATION.json',
        'P5_QUALIFICATION.json','P5_PREDICTION_BARRIER.json','P5_GPU_COMPLETION.json',
        'P5_CPU_SCORING_COMPLETION.json','P5_CPU_COMPLETION.json','P5_ROOT_VISUAL_REVIEW.json']:
        copy(BASE/name,EXPORT/'receipts'/name)
    for folder in ['P5_actual_root_readback_revision004','P5_actual_root_signal_views']:
        for name in ['LAUNCH.json','COMPLETION.json']:
            copy(BASE/folder/name,EXPORT/'receipts'/folder/name)
    for rel in ['P5_actual_root_readback/recovery/input_receipt_schema_001/CAPTURE_RECEIPT.json',
        'P5_actual_root_readback_revision001/recovery/immutable_statistics_002/CAPTURE_RECEIPT.json',
        'P5_actual_root_readback_revision002/recovery/qualified_input_container_003/CAPTURE_RECEIPT.json',
        'P5_actual_root_readback_revision003/recovery/original_vid_runtime_binding_004/CAPTURE_RECEIPT.json']:
        copy(BASE/rel,EXPORT/'receipts'/rel)
    pins.update(['scripts/publish_stvg_opd_revised_p5_v2.py','scripts/audit_stvg_opd_p5_public_v2.py','scripts/audit_stvg_opd_p5_export_v2.py',
        'scripts/verify_stvg_opd_public_remote_v1.py','scripts/report_stvg_opd_revised_p5_root_v2.py','docs/STVG_OPD_REVISED_P5_ROOT_REVIEW.md','methods/CURRENT_METHOD.json','methods/decota_spatial_opd_v1/configs.json'])
    metadata={f for f in pins if f.startswith('artifacts/')}
    for f in sorted(metadata):
        p=ROOT/f;assert p.is_relative_to(ROOT/'artifacts') and p.suffix=='.json'
        copy(p,EXPORT/'receipts'/p.relative_to(BASE) if p.is_relative_to(BASE) else EXPORT/'receipts/original_pre_revision'/p.relative_to(ROOT/'artifacts/stvg_opd_paper_v1'))
    result_evidence={f for f in pins if f.startswith('results/')}
    for f in sorted(result_evidence):
        p=ROOT/f;assert p.parent in {PUB/'P5',*[PUB/n for n in phases()['P5']]} and p.is_file()
        assert f in mapped and mapped[f]['sha256']==sha(p)
    implementation=pins-metadata-result_evidence
    assert all(f.startswith(('scripts/','vg_tta/','protocols/','docs/','methods/')) and (ROOT/f).is_file() for f in implementation)
    write(EXPORT/'CODE_BINDING.json',dict(status='complete_P5_actual_root_evidence_pending_remote_verification',
        files={f:sha(ROOT/f) for f in sorted(pins)},exact_metadata_projection=mapped,
        anonymous_logical_rows=2560,new_formal_fits=2048,exact_original_stream_reuse_rows=512,parent_sources=256,
        actual_math_state_dense_sha256=sha(PUB/'P5/ACTUAL_ROOT_MATH_STATE_DENSE_READBACK.json'),
        actual_visual_review_sha256=sha(BASE/'P5_ROOT_VISUAL_REVIEW.json'),private_payloads_exported=False,paper_suite_complete=False))
    (EXPORT/'README.md').write_text("""# Fixed OPD P5: complete observation budget, cost and mechanism evidence

Read [the actual root review](ACTUAL_ROOT_REVIEW.md), all original stages in
[ROOT_STATISTICS.json](ROOT_STATISTICS.json), every paired K contrast and fixed-order
result in [the budget readback](ACTUAL_ROOT_BUDGET_CONTRASTS.json), all parent effects,
negative attribution, recorded costs, feedback rows and six descriptive signal chains.
All 2,560 anonymous rows are under `stages`: 128 historical parents per target, one
query per parent, two locked source-blocked orders. Eight cross-domain clean streams
use Uniform K=1/2/4/8; two original pre-GT unified-configuration appendix streams stay
separate. The 2,048 new formal fits and 512 exact complete original K4 aliases all
sealed before GT. Qualification outputs are never scored or spliced into formal.
Original main Uniform4 and dataset-specific configurations remain unchanged.
Large anonymous root JSON files use deterministic gzip; CODE_BINDING binds both
original uncompressed bytes/SHA and public compressed bytes/SHA independently.

```bash
python -B scripts/audit_stvg_opd_p5_export_v2.py results/stvg_opd_p5_complete/2026-10-10
```

Complete parent confidence intervals, all pairwise budget differences, both orders,
negative tails, strata and actual recorded cost remain. Confidence intervals use
10,000 paired parent draws conditional on the historical roster and fixed histories;
they are not multiplicity-adjusted. Current/inherited effects describe each budget's
own trajectory and are not an alpha0 causal contrast. Eight report plot PNG/PDF pairs
and six private RGB sheets were actually viewed. Private RGB must match original
captured pixel SHA. Recorded timings include synchronous numerical recording;
K4 aliases retain original timing and cached expert receipts are not cold latency.
Original CPU helper failures remain preserved and separately pinned: missing legacy
receipt bytes, an attempted duplicate immutable statistics write, the mistaken
SHA comparison of different K4 input containers, and the original pre-revision
Vid K4 runtime binding. Strict legacy source/native/pixel/
full-expert comparison is backed by the exact original NPZ hash; all new formal input
SHA comparisons remain exact. No fit, optimizer, scientific runtime or prediction
bytes changed.

Public arithmetic reproduces anonymous statistics and byte bindings; it does not
recreate private inference or certify the entire decoder Jacobian/CUDA kernels.
No private RGB/query/caption/GT geometry/box/action/weights/fit/gradient/Adam payload
is exported. Original P6 still requires actual implementation, qualification, all
deployment seals, CPU/root/view/public/archive closing. EATA and historical paused
queues remain paused. No result-driven retuning or method promotion.
""")
    files=sorted(implementation|{str(f.relative_to(ROOT)) for f in EXPORT.rglob('*') if f.is_file()})
    forbidden=[b'"caption":',b'"video_path":',b'"native_boxes":',b'"GT_box":',b'"committed":',b'"gradient":',b'"optimizer_state":',b'"raw_logits":']
    for rel in files:
        assert Path(rel).suffix in {'.py','.json','.md','.gz','.png','.pdf'}
        raw=(ROOT/rel).read_bytes()
        if rel.endswith('.gz'):raw=gzip.decompress(raw)
        if rel.endswith(('.json','.gz')):
            for token in forbidden:assert token not in raw,(rel,token)
        dest=CHECKOUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    audited=json.loads(subprocess.check_output([str(PYTHON),'-B',str(CHECKOUT/'scripts/audit_stvg_opd_p5_export_v2.py'),str(CHECKOUT/EXPORT.relative_to(ROOT))],cwd=CHECKOUT,text=True))
    assert audited['status']=='pass' and audited['logical_rows']==2560 and audited['original_population_scalar_comparisons']==read(PUB/'P5/ACTUAL_ROOT_POPULATION_READBACK.json')['scalar_comparisons']
    write(BASE/'P5_PUBLIC_SCALAR_AUDIT.json',dict(audited,actual_public_checkout=True,time=time.time()))
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True)
    changed=git('diff','--cached','--name-only').splitlines();assert changed and set(changed)<=set(files)
    records=[dict(path=f,bytes=(CHECKOUT/f).stat().st_size,sha256=sha(CHECKOUT/f),blob_sha=git('hash-object',f)) for f in files]
    write(BASE/'P5_PUBLIC_STAGE.json',dict(status='reviewed_staged',scope='complete actual P5 evidence only',repository='Zonglin-He/A',
        branch=BRANCH,base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),expected_tree=git('write-tree'),
        changed_files=changed,files=records,file_count=len(records),bytes=sum(v['bytes'] for v in records),
        preserved_unrelated_untracked=unrelated,portable_scalar_audit=audited,paper_suite_complete=False,time=time.time()))
    print(json.dumps(dict(status='ready',files=len(records),bytes=sum(v['bytes'] for v in records),changed_files=len(changed),scalar_comparisons=audited['scalar_comparisons'])))


if __name__=='__main__':run()
