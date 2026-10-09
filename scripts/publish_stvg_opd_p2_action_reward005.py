"""Stage anonymous code and actual bounded original-action recovery receipts."""
import json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_opd_p2_action_reward005 import REC,verify
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,read,write,sha
OUT=ROOT/'results/stvg_opd_p2_actual_action_reward_recovery/2026-10-10'
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
BRANCH='research/stvg-opd-paper-hc2-revision-v2'


def git(*args):return subprocess.check_output(['git',*args],cwd=CHECKOUT,text=True).strip()


def run():
    verify()
    for f in ['ROOT_QUALIFICATION_READBACK.json','ROOT_ORIGINAL_PREFIX_READBACK.json',
              'ROOT_RESUME_READBACK.json','ENGINEERING_ROOT_REVIEW.json']:
        assert read(REC/f)['status']=='pass'
    prior=read(BASE/'recovery/P2_inline_gradient_precision_004/FINAL_GITHUB_RECEIPT.json')
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    head=git('rev-parse','HEAD')
    assert head==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)==prior['commit']
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    untracked=git('ls-files','--others','--exclude-standard').splitlines();mapped={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        mapped[str(source.relative_to(ROOT))]=dict(public_path=str(target.relative_to(ROOT)),
            sha256=sha(source),bytes=source.stat().st_size)
    for f in ['CAPTURE_RECEIPT.json','REPRODUCTION_RECEIPT.json','REPRODUCTION_RUNTIME.json',
              'CPU_DIAGNOSIS.json','CPU_ACTUAL_ACTION_DIAGNOSIS.json','CPU_CONTRACTS.json',
              'REVISION_RUNTIME.json','GPU_QUALIFICATION.json','ROOT_QUALIFICATION_READBACK.json',
              'ROOT_ORIGINAL_PREFIX_READBACK.json','ROOT_RECEIPT_DISPATCH_CONTRACTS.json',
              'FIRST_FORMAL_FIT_BITWISE.json','ROOT_RESUME_READBACK.json','ENGINEERING_ROOT_REVIEW.json','RELAUNCH.json']:
        copy(REC/f,OUT/'receipts'/f)
    for folder in ['failed_query','ordinary_controls']:
        copy(REC/'qualification'/folder/'GPU_QUALIFICATION.json',OUT/'qualification'/folder/'GPU_QUALIFICATION.json')
    copy(REC/'reproduction_expert_representation_helper_001/CAPTURE_RECEIPT.json',
         OUT/'receipts/reproduction_expert_representation_helper_001_CAPTURE.json')
    copy(REC/'CPU_DIAGNOSIS_SERIALIZATION_ATTEMPT.json',OUT/'receipts/CPU_DIAGNOSIS_SERIALIZATION_ATTEMPT.json')
    code=set(read(REC/'REVISION_RUNTIME.json')['pins'])|{
        'scripts/reproduce_stvg_opd_p2_action_reward005.py','scripts/publish_stvg_opd_p2_action_reward005.py'}
    root=read(REC/'ROOT_RESUME_READBACK.json')
    write(OUT/'CODE_BINDING.json',dict(status='actually_qualified_and_formally_resumed',
        scope='bounded P2 original action readback audit only; P2 phase and paper remain incomplete',
        code={f:sha(ROOT/f) for f in sorted(code)},exact_metadata_projection=mapped,
        actual_GPU_diagnostic_replays=1,actual_GPU_qualification_fits=26,
        accepted_snapshot_predictions=root['accepted_snapshot_predictions'],accepted_snapshot_bytes=root['accepted_snapshot_bytes'],
        original_predictions=467,original_prediction_bytes=535577607,original_prediction_bytes_unchanged=True,
        original_complete_failed_fit_serialized_before_exit=True,original_CPU32_sigmoid_guard_failure_retained=True,
        original_GPU_action_reward_loss_gradient_Adam_and_scientific_configuration_unchanged=True,
        qualification_new_DINO_calls=0,qualification_formal_predictions_accepted=0,
        GT_read=False,P2_global_seal=False,P2_complete=False,paper_suite_complete=False))
    (OUT/'README.md').write_text('''# Fixed P2: complete original CPU32 sigmoid/reward failure and qualified continuation

P1 is fully root/view/public/archive closed. P2 keeps the original Direct L1/GIoU,
shuffled-feedback, fixed-rollout and full on-policy arms, the same two source
configurations, original128 historical parent cohorts, two cross-clean orders
and five same-domain5% physical conditions. This directory closes only a bounded
engineering diagnosis/qualification/resumption. It is not P2 efficacy scoring,
P2 root closure or paper completion. The earlier inline-gradient004 recovery
and its immutable results remain intact; this is a separate later failure.

The complete HC same-domain frame_drop_5 order1 arrival52/query2676 on-policy
40-round fit finished before the independent CPU32 sigmoid/IoU3e-6 audit failed.
The worker serialized the complete failed fit before exiting. All467 accepted
new formal P2 fits535577607bytes, immutable receipts, inputs, code, locks and
logs were preserved. No GT or partial-arm efficacy scoring was performed.

One actual complete GPU replay with read-only same-call original action logging
matched the original serialized complete fit (1304 tensors276556coordinates
1891scalars, prior audit CPU wall-time alone excluded). Independent CPU32
corner/area/IoU geometry on those original GPU actions matches all40 rewards
exactly. CPU32 sigmoid readout caused eight original reward audit failures;
they are retained as failures. This comparison is to the actual serialized
complete original fit, and does not imply equality to any older unavailable
dead-process memory.

An unsealed replay helper stopped after fitting at a fresh-Tensor versus
cached-list expert representation guard. Its complete replay was already
serialized. Official full pack_expert canonicalization recovered277 scalar
expert values exactly on CPU without another diagnostic GPU replay. That
helper did not reach its final source/expert process hash checks; the actual
qualification below separately executed and passed those checks. A CPU-only
diagnosis JSON scalar serialization interruption was also preserved and fixed.

Revision005 returns the same original GPU sigmoid tensor intact and logs a
detached CPU copy solely for audit. It changes no fitting action/reward, weights,
Gaussian sample, loss, gradient, Adam,1792 state, source reset, inherited LN,
query reset, Native WHEN, output, frame budget, parameter or step. Independent
CPU geometry uses the actual original actions, retaining3e-6. Action readout
is separately checked against stable float64 sigmoid under the existing native
2*float32-epsilon bound; outward per-operation float32 geometry intervals cover
the observed cross-precision difference. This is not a proof of CUDA
transcendental kernels or an independent full decoder Jacobian.

Twenty valid CPU contracts and21 wrong/malformed rejections passed. Twenty-six
real qualification fits followed: two complete failed-query40-round fits;
two ordinary original qualification arrivals in each source configuration for
all three Gaussian arms, with an actual original/new full fit in every cell.
All ordinary old/new and historical complete fits match exactly. New complete
failed-query fits match each other, including same-call action readback and
all prior inline checks; they also match the original complete serialized fit.
Qualification accepted zero formal predictions, made zero new DINO calls and
read no GT. Source and expert process hashes remained unchanged.

Root actually recomputed the complete qualification Gaussian/IoU/softmax,
gradient/Adam/chart/1792-state/query-reset/LN-writeback dictionaries, all467
original accepted prefix fits and their input/receipt hashes and full state
chains. Original, inline004 and new005 receipts are dispatched by saved
revision; unknown revisions, wrong runtime, changed actions and rewritten math
dictionaries are rejected. A single finite controller resumed the original
missing suffix. First formal on-policy52 required exact qualification equality
before acceptance, and the root resumption receipt records an actual bounded
prefix readback. Extra CPU action-readback and in-loop audit time remain in
original fit wall time and are also explicit fields for truthful cost analysis.

All P2 deployment arms, directions, conditions and orders still must globally
seal before the original GT scorer/finalizer. Full phase root statistics,
dense/mathematics/state chains,10000 paired bootstrap, tails, costs, signal
cases, actual figures, anonymous publication and archive closing remain
required before P3. EATA and historical paused queues remain paused. No future
numerical/OOM safety or method efficacy claim is made. Private RGB, query,
caption, GT geometry, actions/samples/boxes, weights, fit/gradient/Adam payloads
are excluded from this export.
''')
    files=sorted(code|{str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file()})
    forbidden=[b'"caption":',b'"video_path":',b'"native_boxes":',b'"GT_box":',
        b'"committed":',b'"gradient":',b'"optimizer_state":',b'"raw_logits":',
        b'"autograd_mean_gradient":',b'"mean_before":',b'"samples":',b'"weights":',b'"actions":']
    for f in files:
        assert f.startswith(('scripts/','vg_tta/','protocols/','results/')) and Path(f).suffix in ['.py','.md','.json']
        if f.endswith('.json'):
            raw=(ROOT/f).read_bytes()
            for token in forbidden:assert token not in raw,(f,token)
        p=CHECKOUT/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,p)
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True)
    changed=git('diff','--cached','--name-only').splitlines();assert changed and set(changed)<=set(files)
    assert git('ls-files','--others','--exclude-standard').splitlines()==untracked
    records=[dict(path=f,sha256=sha(CHECKOUT/f),bytes=(CHECKOUT/f).stat().st_size,blob_sha=git('hash-object',f)) for f in files]
    write(REC/'PUBLIC_STAGE.json',dict(status='reviewed_staged',scope='bounded P2 original action readback repair/26 real qualification/root resumption',
        repository='Zonglin-He/A',branch=BRANCH,base_commit=head,base_tree=git('rev-parse','HEAD^{tree}'),
        expected_tree=git('write-tree'),files=records,changed_files=changed,file_count=len(records),
        bytes=sum(r['bytes'] for r in records),preserved_unrelated_untracked=untracked,
        GT_read=False,P2_complete=False,paper_suite_complete=False,time=time.time()))
    print(json.dumps(dict(status='ready',files=len(records),changed=len(changed),bytes=sum(r['bytes'] for r in records))))


if __name__=='__main__':run()
