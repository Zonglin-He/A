"""Stage only anonymous numerical-repair code and actual scalar/hash receipts."""
import json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_opd_p2_matched_gradient006 import REC,verify,original5
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,read,write,sha
OUT=ROOT/'results/stvg_opd_p2_matched_gradient_recovery/2026-10-10'
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
BRANCH='research/stvg-opd-paper-hc2-revision-v2'

def git(*args):return subprocess.check_output(['git',*args],cwd=CHECKOUT,text=True).strip()

def run():
    verify()
    for f in ['ROOT_QUALIFICATION_READBACK.json','ROOT_ORIGINAL_PREFIX_READBACK.json','ROOT_RESUME_READBACK.json','ENGINEERING_ROOT_REVIEW.json']:
        assert read(REC/f)['status']=='pass'
    prior=read(original5.REC/'FINAL_GITHUB_RECEIPT.json')
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    head=git('rev-parse','HEAD')
    assert head==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)==prior['commit']
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    untracked=git('ls-files','--others','--exclude-standard').splitlines();mapped={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        mapped[str(source.relative_to(ROOT))]=dict(public_path=str(target.relative_to(ROOT)),sha256=sha(source),bytes=source.stat().st_size)
    receipts=['CAPTURE_RECEIPT.json','REPRODUCTION_RECEIPT.json','REPRODUCTION_RUNTIME.json','CPU_DIAGNOSIS.json',
        'CPU_KERNEL_DIAGNOSIS_SOURCE_PIN.json','CPU_CONTRACTS.json','REVISION_RUNTIME.json','GPU_QUALIFICATION.json',
        'ROOT_QUALIFICATION_READBACK.json','ROOT_ORIGINAL_ALL_FITS_READBACK.json','ROOT_ORIGINAL_PREFIX_READBACK.json',
        'ROOT_RECEIPT_DISPATCH_CONTRACTS.json','FIRST_FORMAL_FIT_BITWISE.json','ROOT_RESUME_READBACK.json',
        'ENGINEERING_ROOT_REVIEW.json','RELAUNCH.json']
    for f in receipts:copy(REC/f,OUT/'receipts'/f)
    for folder in ['failed_query','ordinary_controls']:
        copy(REC/'qualification'/folder/'GPU_QUALIFICATION.json',OUT/'qualification'/folder/'GPU_QUALIFICATION.json')
    code=set(read(REC/'REVISION_RUNTIME.json')['pins'])|{'scripts/reproduce_stvg_opd_p2_matched_gradient006.py','scripts/publish_stvg_opd_p2_matched_gradient006.py'}
    root=read(REC/'ROOT_RESUME_READBACK.json');q=read(REC/'ROOT_QUALIFICATION_READBACK.json')
    write(OUT/'CODE_BINDING.json',dict(status='actually_qualified_and_formally_resumed',scope='bounded P2 CPU numerical audit repair; not P2 efficacy/phase root or paper completion',
        code={f:sha(ROOT/f) for f in sorted(code)},exact_metadata_projection=mapped,
        actual_GPU_diagnostic_replays=1,actual_GPU_qualification_fits=q['actual_GPU_fits'],
        accepted_snapshot_predictions=root['accepted_snapshot_predictions'],accepted_snapshot_bytes=root['accepted_snapshot_bytes'],
        original_predictions=2410,original_prediction_bytes=3242898557,original_prediction_bytes_unchanged=True,
        original_dead_failed_fit_serialized=False,no_dead_memory_bitwise_claim=True,
        original004_CPU32_guard_failure_retained=True,matched_threshold_unchanged=2e-5,
        loss_actions_rewards_weights_gradients_Adam_and_scientific_configuration_unchanged=True,
        qualification_new_DINO_calls=0,qualification_formal_predictions_accepted=0,
        GT_read=False,P2_complete=False,paper_suite_complete=False))
    (OUT/'README.md').write_text('''# Fixed P2: CUDA reduction precision audit and qualified continuation

This closes a bounded numerical diagnosis/qualification/resumption, not P2
accuracy evaluation or paper completion. P1 and the finite Figure1 are already
root/view/public/archive closed. All P2 scientific arms, source checkpoints,
original historical parent cohorts, orders, conditions and fixed parameters
remain unchanged. Human-paused EATA and historical queues remain paused.

The previous actual-action005 controller stopped inside HC same-domain
exposure_5 order1 arrival26/query1151 frozen-rollout fitting. Original GPU
autograd versus original analytic32 passed2e-5, but the independent prior004
CPU32 check differed by one float32 ULP(.0001220703125). All2410 accepted new
formal fits3242898557bytes,679 inputs, receipts, original code, locks and logs
were preserved. The original dead-process failed fit was not serialized.
One unchanged original GPU full-prefix replay saved the actual incomplete
five completed rounds and sixth-round failing derivative. No dead-memory
bitwise equality is claimed. It accepted no prediction, new DINO call or GT.

Installed pinned PyTorch134179474539648ba7dee1317959529fbd0e7f89 Reduce.cuh
matches the official source byte-for-byte. For contiguous(N,32,4) input reduced
on its middle dimension, each CUDA thread uses four accumulator registers,
indices i,i+4,... and sequential accumulator0+1,+2,+3;32 inputs do not split
across warps. CPU default sum uses another order. Revision006 changes only the
independent CPU32 audit broadcast-gradient reduction backward to that pinned
order. On the actual failure, it matches the recorded GPU derivative exactly.
The failed004CPU check remains recorded as failed. The original2e-5 threshold,
independent CPU64 autodiff/formula check, original analytic32 and gamma48 bounds
remain unchanged. This is not a full independent decoder Jacobian or CUDA
transcendental proof, and does not guarantee every future input will pass.

Twenty valid CPU contracts and17 wrong/malformed derivative rejections passed.
Two real complete40-round failed-query fits match each other through the full
scientific result and all audits. The first five completed rounds, initial/
current state, original sampled actions and sixth-round failed derivative
match the saved actual unchanged replay. Twelve predeclared ordinary cells,
two arrivals in each source for all three Gaussian arms, each run a real
original/new full fit (24 more fits). Ordinary original/new and historical
qualification fits match; source/expert process hashes and input hashes pass.
Qualification accepts zero formal predictions, new DINO calls or GT.

Root independently read all13 complete new qualification fits, Gaussian/
original-action-IoU/softmax/gradient/Adam/chart/1792/reset/LN-writeback chains,
all2410 original complete saved fits and immutable input/prediction/receipt
bytes. Saved original/004/005/006/P0 dictionaries dispatch by pinned revision;
unknown revisions, wrong runtime, changed actions and rewritten mathematical
dictionaries are rejected. The first missing frozen-rollout26 requires real
qualified complete-fit bitwise equality before acceptance. A finite controller
continues only the original missing suffix. Actual root resumption counts and
hashes are in ROOT_RESUME_READBACK.json, not an assumption of future completion.

No fitting action/reward/weights/loss/gradient/Adam/state, source/input/cohort,
frame count, step, parameter, WHEN/readout or old runtime changes. No clipping,
skipped query, retuning, partial-arm GT score or qualification prediction is
admitted. Audit CPU time remains inside recorded fit wall time. All P2 arms,
directions, orders and conditions must globally seal before original GT scoring
and phase root statistics/state/dense/bootstrap/tails/cost/cases/real figure
view/anonymous negative-results/public/archive closure. Then fixed P3-P6 follow.
Private videos/captions/GT geometry, weights, arrays, raw fits/actions/gradients/
Adam and caches are excluded. Public receipts contain scalar checks and hashes.
''')
    files=sorted(code|{str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file()})
    forbidden=[b'"caption":',b'"video_path":',b'"native_boxes":',b'"GT_box":',b'"committed":',
        b'"gradient":',b'"optimizer_state":',b'"raw_logits":',b'"autograd_mean_gradient":',b'"mean_before":',
        b'"samples":',b'"weights":',b'"actions":']
    for f in files:
        assert f.startswith(('scripts/','vg_tta/','protocols/','results/')) and Path(f).suffix in ['.py','.md','.json']
        if f.endswith('.json'):
            for token in forbidden:assert token not in (ROOT/f).read_bytes(),(f,token)
        p=CHECKOUT/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,p)
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True)
    changed=git('diff','--cached','--name-only').splitlines();assert changed and set(changed)<=set(files)
    assert git('ls-files','--others','--exclude-standard').splitlines()==untracked
    records=[dict(path=f,sha256=sha(CHECKOUT/f),bytes=(CHECKOUT/f).stat().st_size,blob_sha=git('hash-object',f)) for f in files]
    write(REC/'PUBLIC_STAGE.json',dict(status='reviewed_staged',scope='bounded P2 CUDA-reduction CPU audit repair/26 real qualification/root resumption',
        repository='Zonglin-He/A',branch=BRANCH,base_commit=head,base_tree=git('rev-parse','HEAD^{tree}'),expected_tree=git('write-tree'),
        files=records,changed_files=changed,file_count=len(records),bytes=sum(r['bytes'] for r in records),
        preserved_unrelated_untracked=untracked,GT_read=False,P2_complete=False,paper_suite_complete=False,time=time.time()))
    print(json.dumps(dict(status='ready',files=len(records),changed=len(changed),bytes=sum(r['bytes'] for r in records))))

if __name__=='__main__':run()
