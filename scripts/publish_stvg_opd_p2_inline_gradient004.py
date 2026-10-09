"""Stage only approved anonymous code and real bounded numerical receipts."""
import json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_opd_p2_inline_gradient004 import REC,verify
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,read,write,sha
OUT=ROOT/'results/stvg_opd_p2_inline_gradient_recovery/2026-10-09'
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
BRANCH='research/stvg-opd-paper-hc2-revision-v2'


def git(*args):return subprocess.check_output(['git',*args],cwd=CHECKOUT,text=True).strip()


def run():
    verify()
    for f in ['ROOT_QUALIFICATION_READBACK.json','ROOT_ORIGINAL_PREFIX_READBACK.json',
              'ROOT_RESUME_READBACK.json','ENGINEERING_ROOT_REVIEW.json']:
        assert read(REC/f)['status']=='pass'
    prior=read(BASE/'recovery/P2_input_schema_003/FINAL_GITHUB_RECEIPT.json')
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    head=git('rev-parse','HEAD')
    assert head==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert head==prior['commit']
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    untracked=git('ls-files','--others','--exclude-standard').splitlines();mapped={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        mapped[str(source.relative_to(ROOT))]=dict(public_path=str(target.relative_to(ROOT)),
            sha256=sha(source),bytes=source.stat().st_size)
    for f in ['CAPTURE_RECEIPT.json','REPRODUCTION_RECEIPT.json','REPRODUCTION_RUNTIME.json',
              'REPRODUCTION_RUNTIME_revision001.json','CPU_DIAGNOSIS.json','CPU_CONTRACTS.json',
              'REVISION_RUNTIME.json','ROOT_RUNTIME.json','ROOT_RUNTIME_revision001.json',
              'GPU_QUALIFICATION.json','ROOT_QUALIFICATION_READBACK.json','ROOT_ORIGINAL_PREFIX_READBACK.json',
              'ROOT_RECEIPT_DISPATCH_CONTRACTS.json','FIRST_FORMAL_FIT_BITWISE.json',
              'ROOT_RESUME_READBACK.json','ENGINEERING_ROOT_REVIEW.json','RELAUNCH.json']:
        copy(REC/f,OUT/'receipts'/f)
    for folder in ['failed_query','ordinary_controls']:
        copy(REC/'qualification'/folder/'GPU_QUALIFICATION.json',OUT/'qualification'/folder/'GPU_QUALIFICATION.json')
    for folder in ['reproduction_helper_namespace_001','root_receipt_helper_001']:
        copy(REC/folder/'CAPTURE_RECEIPT.json',OUT/'receipts'/(folder+'_CAPTURE.json'))
    code=set(read(REC/'REVISION_RUNTIME.json')['pins'])|{
        'scripts/reproduce_stvg_opd_p2_inline_gradient004.py',
        'scripts/review_stvg_opd_p2_inline_gradient004.py','scripts/publish_stvg_opd_p2_inline_gradient004.py'}
    root=read(REC/'ROOT_RESUME_READBACK.json')
    write(OUT/'CODE_BINDING.json',dict(status='actually_qualified_and_formally_resumed',
        scope='bounded P2 in-loop audit repair only; P2 phase and paper remain incomplete',
        code={f:sha(ROOT/f) for f in sorted(code)},exact_metadata_projection=mapped,
        actual_GPU_qualification_fits=26,accepted_snapshot_predictions=root['accepted_snapshot_predictions'],
        accepted_snapshot_bytes=root['accepted_snapshot_bytes'],original_prediction_bytes_unchanged=True,
        original_predictions=286,original_prediction_bytes=299637020,
        original_absolute_guard_failure_retained=True,loss_gradient_Adam_and_scientific_configuration_unchanged=True,
        qualification_new_DINO_calls=0,qualification_formal_predictions_accepted=0,
        GT_read=False,P2_global_seal=False,P2_complete=False,paper_suite_complete=False))
    (OUT/'README.md').write_text('''# Fixed P2: original in-loop mean-gradient failure and qualified continuation

P1 is fully root/view/public/archive closed. P2 retains its original four-arm,
two-source128-parent design, double cross-clean orders and five same-domain5%
physical conditions. This directory closes only a bounded numerical engineering
recovery. It does not contain P2 GT efficacy scores or close P2 or the paper.

The previous input-schema003 stream completed HC cross-clean1024 logical rows
(256 real Direct fits plus768 exact complete P0 aliases). In HC same-domain
frame_drop_5, first-order arrival7/query3051 frozen-rollout failed the original
inline mean-gradient absolute2e-5 guard in its second round. Direct and shuffled
for that arrival had already been accepted. All286 existing new formal fits,
299637020 bytes, their immutable receipts,156 opaque matched-input snapshots,
original code, scientific locks, status and logs were preserved. No partial-arm
GT score was computed. The failed process did not serialize its incomplete fit.

One actual GPU replay reconstructed the full saved online prefix and reproduced
the original assertion with zero new formal acceptance, zero new DINO and no GT.
Its incomplete-fit witness was saved. Original GPU derivative versus analytic
expression discrepancy3.0517578125e-5 remains a failed2e-5 guard. Independent
CPU32 autodiff differs from the GPU derivative by2.384185791015625e-7; matching
the installed CUDA scalar reciprocal/multiplication path matches it exactly.
The original mixed-precision absolute failures are retained as failures.

The additive in-loop check changes only one assertion test in a process-local
AST copy of the original fitter. AST restoration proves all other nodes remain
identical. The original loss, Gaussian samples, rewards, softmax weights,
autograd gradient, Adam,1792 state, readout and all parameter/input/science bytes
are unchanged. The already qualified native-logit chart and allocator remain.
Independent CPU32 retains2e-5, CPU64 autodiff verifies the closed formula at
2e-5, and both original float32 branches must satisfy explicit gamma48 arithmetic
bounds. The original observed analytic discrepancy is never replaced with zero.

Twenty valid CPU contracts and17 wrong/malformed rejection contracts passed.
Actual GPU qualification executed26 fits: the failed query completed all40
rounds twice; both source configurations ran original/new complete controls for
the first two predeclared qualification arrivals under all three Gaussian arms.
Ordinary full results match the original and prior saved qualifications exactly.
The failed-query repeated complete results match1304 tensors/265316 coordinates
and1883 scalars. Its saved original reproduction prefix/state/failed derivative
matches58 tensors/11079 coordinates and36 scalars. These comparisons use actual
serialized replay/qualification fits; they are not comparisons to unavailable
dead-process memory and do not prove a full decoder Jacobian.

Actual root independently recomputed the saved complete qualification
Gaussian/IoU/softmax/gradient/Adam/chart/reset/writeback dictionaries, old/current
receipt dispatch and all286 original accepted P2 fits. Unknown revisions, wrong
runtime and a rewritten math dictionary were rejected. One finite controller
then resumed the original missing suffix. First formal acceptance required exact
qualified full-fit equality. The root resumption receipt records an actual saved
prefix snapshot and rechecks all original prediction/receipt/input bytes, full
1792 inheritance, source/order resets, query/Adam resets and Native WHEN.
The extra CPU audit time remains inside the original fit wall-time field and is
separately recorded, so costs must not be read as pure GPU device time.

An unsealed original-replay helper import-order error and a later root receipt
duplicate-keyword serialization error were preserved and separately repaired.
They did not change GPU fitting, qualification, predictions, scores or scientific
locks. Later helper pins do not rewrite their originals.

All P2 deploy arms, conditions, orders and both directions still must globally
seal before GT scoring. Full phase root mathematics/state/dense/statistics,
10000 paired bootstrap, tails/cost/cases, actual plot review, complete anonymous
publication and archive closure remain required before P3. EATA and historical
paused queues remain paused. Qualification and resumption do not establish
future numerical/OOM safety or method efficacy. Private RGB/query/caption/GT,
weights, box/actions, fit payloads, gradients and Adam arrays are excluded.
''')
    files=sorted(code|{str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file()})
    forbidden=[b'"caption":',b'"video_path":',b'"native_boxes":',b'"GT_box":',
        b'"committed":',b'"gradient":',b'"optimizer_state":',b'"raw_logits":',
        b'"autograd_mean_gradient":',b'"mean_before":',b'"samples":',b'"weights":']
    for f in files:
        assert f.startswith(('scripts/','vg_tta/','protocols/','results/')) and Path(f).suffix in ['.py','.md','.json']
        if f.endswith('.json'):
            raw=(ROOT/f).read_bytes()
            for token in forbidden:assert token not in raw,(f,token)
        p=CHECKOUT/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,p)
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True)
    changed=git('diff','--cached','--name-only').splitlines();assert changed and set(changed)<=set(files)
    assert git('ls-files','--others','--exclude-standard').splitlines()==untracked
    records=[dict(path=f,sha256=sha(CHECKOUT/f),bytes=(CHECKOUT/f).stat().st_size,
        blob_sha=git('hash-object',f)) for f in files]
    write(REC/'PUBLIC_STAGE.json',dict(status='reviewed_staged',scope='bounded P2 inline precision repair/26 actual qualifications/root resumption',
        repository='Zonglin-He/A',branch=BRANCH,base_commit=head,base_tree=git('rev-parse','HEAD^{tree}'),
        expected_tree=git('write-tree'),files=records,changed_files=changed,file_count=len(records),
        bytes=sum(r['bytes'] for r in records),preserved_unrelated_untracked=untracked,
        GT_read=False,P2_complete=False,paper_suite_complete=False,time=time.time()))
    print(json.dumps(dict(status='ready',files=len(records),changed=len(changed),bytes=sum(r['bytes'] for r in records))))


if __name__=='__main__':run()
