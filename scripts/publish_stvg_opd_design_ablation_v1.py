"""Whitelist actual anonymous results/code; GitHub write follows reviewed stage."""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_design_ablation_common_v1 import BASE,PUB,OLDPUB,PYTHON,read,write,sha,verify
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
BRANCH='research/stvg-opd-paper-hc2-revision-v2'


def git(*args): return subprocess.check_output(['git',*args],cwd=CHECKOUT,text=True).strip()


def safe(value):
    bad={'caption','query_text','video_path','native_boxes','GT_boxes','GT_box','GT_geometry',
         'committed','gradient','optimizer_state','raw_logits','hidden_states','actions','actual_actions',
         'actual_samples','sample_logits','sample_boxes','frame_pixels','weights','Fisher'}
    if isinstance(value,dict):
        forbidden=set(value)&bad
        # Three existing archive receipts call the literal check/snapshot/check
        # command list "actions". Accept only that exact metadata schema; GPU
        # action arrays and all other payload uses of this key remain rejected.
        if 'actions' in forbidden and set(value)=={'status','scope','actions','log_sha256','time'}:
            assert value['status']=='pass' and value['actions']==['check','snapshot','check']
            assert isinstance(value['scope'],str)
            assert isinstance(value['log_sha256'],str) and len(value['log_sha256'])==64
            assert all(c in '0123456789abcdef' for c in value['log_sha256'])
            forbidden.remove('actions')
        assert not forbidden,forbidden
        for v in value.values(): safe(v)
    elif isinstance(value,list):
        for v in value: safe(v)
    elif isinstance(value,str):
        assert '/.codex/attachments/' not in value


def source_projection(path):
    text=path.read_text(); changes=[]
    if path.name=='stvg_opd_design_ablation_common_v1.py':
        tree=ast.parse(text); assignments=[n for n in tree.body if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='ATTACHMENTS' for t in n.targets)]
        assert len(assignments)==1
        node=assignments[0]; lines=text.splitlines(keepends=True)
        lines[node.lineno-1:node.end_lineno]=['AUTHORIZATION_ATTACHMENT_SHA256 = '+repr(read(BASE/'DESIGN_LOCK.json')['authorization_attachments_sha256'])+'\n']
        text=''.join(lines)
        old='authorization_attachments_sha256=[sha(p) for p in ATTACHMENTS]'
        assert text.count(old)==1
        text=text.replace(old,'authorization_attachments_sha256=AUTHORIZATION_ATTACHMENT_SHA256')
        changes=['local private human-attachment paths replaced by the same prelocked attachment hashes; scientific functions unchanged']
    assert '/.codex/attachments/' not in text
    compile(text,str(path),'exec')
    return text,changes


def run():
    verify(); assert read(BASE/'ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert read(BASE/'actual_root/ROOT_ALL_FITS_READBACK.json')['counts']['complete_fits']==5376
    assert read(BASE/'actual_fixed_case_views002/COMPLETION.json')['cases']==6
    assert read(BASE/'ROOT_INTERPRETATION.json')['all_negative_results_retained']
    assert read(BASE/'GLOBAL_PREDICTION_BARRIER.json')['new_formal_fits']==5376
    assert read(BASE/'CPU_COMPLETION.json')['status']=='pending_actual_root_visual_and_publication'
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    unrelated=git('ls-files','--others','--exclude-standard').splitlines()
    assert all('stvg_motivation' in p.lower() for p in unrelated)
    mappings={}; owned=set()
    def localcopy(source,target):
        assert source.suffix in {'.json','.jsonl','.gz','.md','.tex','.png','.pdf','.py'}
        if source.suffix=='.json': safe(read(source))
        target.parent.mkdir(parents=True,exist_ok=True)
        if source.resolve()!=target.resolve(): shutil.copy2(source,target)
        assert sha(source)==sha(target)
        mappings[str(source.relative_to(ROOT))]=dict(path=str(target.relative_to(ROOT)),sha256=sha(target),bytes=target.stat().st_size)
        owned.add(str(target.relative_to(ROOT)))
    # Entire anonymous new rows, statistics, costs and plots, including negatives.
    for path in sorted(PUB.rglob('*')):
        if path.name in {'fixed_historical_case_signals.png','fixed_historical_case_signals.pdf'}:
            continue  # Original clipped labels remain locally bound by their capture.
        if path.is_file() and not any(x in path.parts for x in ['receipts','historical','projection']):
            localcopy(path,path)
    for phase in ['P2','P3']:
        for dataset in ['hc2','vidstg']:
            for setting in ['cross_clean','same_5percent']:
                name=f'{phase}_{dataset}_{setting}'
                localcopy(OLDPUB/name/'ROWS.jsonl.gz',PUB/'historical'/name/'ROWS.jsonl.gz')
    receipts=['DESIGN_LOCK.json','HUMAN_SCOPE_CONFIRMATION.json','RUNTIME_LOCK.json','CPU_CONTRACTS.json','LAUNCH.json','ROOT_QUALIFICATION_READBACK.json',
        'GLOBAL_PREDICTION_BARRIER.json','GT_EXPOSURE.json','CPU_SCORING_COMPLETION.json','CPU_COMPLETION.json','ROOT_VISUAL_REVIEW.json','ROOT_INTERPRETATION.json',
        'ARCHIVE_PREPARE_RECEIPT.json','ARCHIVE_QUALIFICATION_RECEIPT.json',
        'CPU_projection_revision002/REVISION_RUNTIME.json','CPU_projection_revision002/ORIGINAL_TABLE_READBACK.json',
        'CPU_projection_revision002/ARCHIVE_VERIFICATION_RECEIPT.json',
        'CPU_projection_revision003/REVISION_RUNTIME.json','CPU_projection_revision003/COMPLETION.json',
        'actual_root/RUNTIME.json','actual_root/ROOT_ALL_FITS_READBACK.json',
        'actual_root/COVERAGE_FIGURE_VISUAL_READBACK.json','actual_root/PORTABLE_STATISTICS_PRECHECK.json',
        'actual_fixed_case_views/CASE_DIAGNOSTIC_LOCK.json','actual_fixed_case_views/RUNTIME.json',
        'actual_fixed_case_views/STATIC_INTERFACE_REVIEW.json',
        'actual_fixed_case_views002/CASE_DIAGNOSTIC_LOCK.json','actual_fixed_case_views002/RUNTIME.json',
        'actual_fixed_case_views002/COMPLETION.json','actual_fixed_case_views002/ACTUAL_LAUNCH.json',
        'fixed_case_signal_visual003/CAPTURE_RECEIPT.json','fixed_case_signal_visual003/RUNTIME.json',
        'fixed_case_signal_visual003/COMPLETION.json',
        'CPU_actual_root_continuation002/RUNTIME.json','CPU_actual_root_continuation002/LAUNCH.json',
        'CPU_actual_root_continuation002/RECOVERY_NOTICE.json',
        'CPU_actual_root_continuation003/RUNTIME.json','CPU_actual_root_continuation003/LAUNCH.json',
        'CPU_actual_root_continuation003/COMPLETION.json',
        'recovery/CPU_root_status_writer_003/CAPTURE_RECEIPT.json',
        'recovery/public_precheck_package_004/CAPTURE_RECEIPT.json',
        'recovery/CPU_existing_table_projection_001/CAPTURE_RECEIPT.json',
        'ARCHIVE_CORE_SCOPE_RECEIPT.json','ARCHIVE_CPU_CONTINUATION003_RECEIPT.json',
        'CASE_DIAGNOSTIC_PREPARE.json','ARCHIVE_FIXED_CASE_LOCK_RECEIPT.json',
        'ARCHIVE_HCCROSS_STAGE_RECEIPT.json','ARCHIVE_HC_COMPLETE_RECEIPT.json',
        'ARCHIVE_THREE_GPU_STAGES_RECEIPT.json','ARCHIVE_GLOBAL_BARRIER_RECEIPT.json',
        'ARCHIVE_POSTSEAL_CPU_STATISTICS_RECEIPT.json','ARCHIVE_ACTUAL_ROOT_VIEWS_RECEIPT.json']
    receipts += [str(p.relative_to(BASE)) for p in sorted((BASE/'actual_root').glob('PREFIX_READBACK*.json'))]
    for name in read(BASE/'DESIGN_LOCK.json')['stages']:
        receipts += [f'stages/{name}/PREDICTION_BARRIER.json',f'stages/{name}/CPU_COMPLETION.json',
                     f'qualification/{name}/PREDICTION_BARRIER.json']
    for rel in receipts: localcopy(BASE/rel,PUB/'receipts'/rel)
    for p in sorted((BASE/'first_formal').rglob('*.json')): localcopy(p,PUB/'receipts'/p.relative_to(BASE))
    for name in read(BASE/'DESIGN_LOCK.json')['stages']:
        control=read(BASE/'qualification'/name/'BITWISE_CONTROLS.json')
        projected={k:v for k,v in control.items() if k!='records'}
        projected['original_complete_control_receipt_sha256']=sha(BASE/'qualification'/name/'BITWISE_CONTROLS.json')
        projected['records']=[{k:v for k,v in row.items() if k!='native_head_VJP'} for row in control['records']]
        projected['native_head_VJP_values_private']=True
        path=PUB/'receipts/qualification'/name/'BITWISE_CONTROLS_META.json'
        safe(projected); write(path,projected); owned.add(str(path.relative_to(ROOT)))
    implementations=['protocols/stvg_opd_design_ablation_v1.md',
        'scripts/stvg_opd_design_ablation_common_v1.py','scripts/run_stvg_opd_design_ablation_v1.py',
        'scripts/continue_stvg_opd_design_ablation_v1.py','scripts/score_stvg_opd_design_ablation_v1.py',
        'scripts/finalize_stvg_opd_design_ablation_v1.py','scripts/finalize_stvg_opd_design_ablation_cpu002.py',
        'scripts/finalize_stvg_opd_design_ablation_cpu003.py',
        'scripts/review_stvg_opd_design_ablation_v1.py','scripts/continue_stvg_opd_design_ablation_CPU_root002.py',
        'scripts/continue_stvg_opd_design_ablation_CPU_root003.py',
        'scripts/render_stvg_opd_design_ablation_fixed_cases_v1.py','scripts/render_stvg_opd_design_ablation_fixed_cases_v2.py',
        'scripts/render_stvg_opd_design_ablation_signal_visual003.py',
        'scripts/report_stvg_opd_design_ablation_root_v1.py',
        'scripts/audit_stvg_opd_design_ablation_public_v1.py','scripts/publish_stvg_opd_design_ablation_v1.py',
        'scripts/close_stvg_opd_design_ablation_v1.py',
        'scripts/stage_stvg_opd_design_ablation_closing_v1.py',
        'scripts/verify_stvg_opd_public_remote_v1.py']
    source_bindings={}
    for rel in implementations:
        p=ROOT/rel; target=CHECKOUT/rel; target.parent.mkdir(parents=True,exist_ok=True)
        if p.suffix=='.py':
            text,changes=source_projection(p); target.write_text(text)
        else:
            text=p.read_text(); assert '/.codex/attachments/' not in text
            shutil.copy2(p,target); changes=[]
        source_bindings[rel]=dict(actual_local_sha256=sha(p),public_sha256=sha(target),transport_changes=changes)
        owned.add(rel)
    qualifier=read(BASE/'ROOT_QUALIFICATION_READBACK.json')
    barriers=[read(BASE/'stages'/name/'PREDICTION_BARRIER.json') for name in read(BASE/'DESIGN_LOCK.json')['stages']]
    write(PUB/'ACTUAL_DINO_CALL_ACCOUNTING.json',dict(status='pass',
        qualification_actual_new_DINO_calls=qualifier['actual_new_DINO_calls'],
        formal_actual_new_DINO_calls=sum(b['actual_new_DINO_calls'] for b in barriers),
        total_actual_new_DINO_calls=qualifier['actual_new_DINO_calls']+sum(b['actual_new_DINO_calls'] for b in barriers),
        state_arms_reuse_original_expert=True,shared_observation_capture_counted_once=True,
        formal_fit_arrivals=5376,qualification_arrivals_excluded_from_formal=72))
    (PUB/'README.md').write_text('''# Three fixed component ablations and event-coverage Figure 3

The complete component table combines the preserved P2/P3 controls with three
new interventions: whole observed-video Uniform4, persistent query residual
with per-query Adam reset, and full LayerNorm carry with query/Adam reset.
All 5,376 new formal fits, both directions, conditions and fixed orders sealed
before labels. Seventy-two qualification arrivals / 144 actual full fits stay
excluded from formal predictions. Main source checkpoints, hyperparameters,
frame grid, sparse budget, reward/optimizer and Native WHEN are unchanged.

Read [the actual root review](ACTUAL_ROOT_REVIEW.md), [the component table](COMPONENT_TABLE.md),
all per-condition/order contrasts, parent effects and negative tails in
[ALL_STATISTICS](ALL_STATISTICS.json), and the fixed-bin [coverage analysis](COVERAGE_ANALYSIS.json).
The cohort uses the original 128 historically exposed parents per target; this
is not a fresh population claim. Confidence intervals use 10,000 paired-parent
draws and fixed seed 20261011, without multiplicity adjustment. Coverage is
postseal descriptive grouping, not tIoU or a causal coverage intervention.
Both arms keep their own online LayerNorm trajectory, so their final contrast
also includes inherited effects. Averaging the two fixed orders does not
establish robustness over arbitrary stream orders.

```bash
python -B scripts/audit_stvg_opd_design_ablation_public_v1.py results/stvg_opd_design_ablation/2026-10-11
```

Two historical table display values are corrected from full sealed row
readback. No old score or prediction is rewritten. The original failed CPU
projection, the CPU002 dynamic-status failure and the additive CPU003 correction have separate bindings. The GPU
runtime is preserved. Original P1-P6 closure remains unchanged and separate.
Actual expert calls, reused observations, valid-GT quality denominators and
recorded fitting times are retained; cached fitting costs are not cold latency.
Six completed P3 cases and their display frames were fixed before the new GT
read, then compared with all three new controls. Their historical success/harm
roles are not newly selected positive/negative examples. Only scalar signals
are public; actual RGB comparison sheets remain private. Signal curves use
each arm's admitted evidence, so they do not share the frame-level denominator.
The final signal PNG/PDF uses visual003 to shorten clipped axis labels. All 24
saved curves and case selection are unchanged; the original provisional images
remain locally preserved and bound by the display-only capture receipt.

Private RGB/video, query/caption, labels/box geometry, native outputs, actions,
gradients, optimizer/LN/prefix tensors, weights and raw fit/input caches are
excluded. Only anonymous scalars, code/configuration, hashes and scope/count
receipts are public. Common orchestration transports the same human attachment
hashes as constants instead of publishing private attachment paths; its original
and public file hashes are explicit. Private mathematical/scientific runtime
pins bind the actual execution, not a claim that an asset-free checkout can
rerun the GPU experiment. The portable audit independently checks all public
arithmetic, not the complete decoder Jacobian or CUDA transcendental safety.
Final close additionally requires exact remote bytes and archive receipts.
''')
    owned.update({str((PUB/n).relative_to(ROOT)) for n in ['README.md','ACTUAL_DINO_CALL_ACCOUNTING.json']})
    binding=dict(status='actual_all_results_and_root_view_pending_remote_verification',
        source_code_bindings=source_bindings,exact_scalar_receipt_projection=mappings,
        private_payloads_exported=False,original_suite_unchanged=True,production_promotion=False,
        root_math_state_readback_sha256=sha(BASE/'actual_root/ROOT_ALL_FITS_READBACK.json'),
        root_visual_review_sha256=sha(BASE/'ROOT_VISUAL_REVIEW.json'),
        scientific_runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'))
    write(PUB/'CODE_BINDING.json',binding); owned.add(str((PUB/'CODE_BINDING.json').relative_to(ROOT)))
    for rel in sorted(owned):
        if rel in source_bindings: continue
        p=ROOT/rel; q=CHECKOUT/rel; q.parent.mkdir(parents=True,exist_ok=True)
        if p.suffix=='.json': safe(read(p))
        shutil.copy2(p,q)
    audit=json.loads(subprocess.check_output([str(PYTHON),'-B','scripts/audit_stvg_opd_design_ablation_public_v1.py',str(PUB.relative_to(ROOT))],cwd=CHECKOUT,text=True))
    assert audit['status']=='pass' and audit['new_formal_rows']==5376
    write(BASE/'PUBLIC_SCALAR_AUDIT.json',dict(audit,actual_public_checkout=True,time=time.time()))
    for rel in sorted(owned):
        p=CHECKOUT/rel
        assert p.suffix in {'.py','.md','.json','.gz','.tex','.png','.pdf'}
        if p.suffix=='.json': safe(read(p))
    subprocess.run(['git','add','--',*sorted(owned)],cwd=CHECKOUT,check=True)
    changed=git('diff','--cached','--name-only').splitlines(); assert changed and set(changed)<=owned
    records=[dict(path=rel,bytes=(CHECKOUT/rel).stat().st_size,sha256=sha(CHECKOUT/rel),blob_sha=git('hash-object',rel)) for rel in sorted(owned)]
    write(BASE/'PUBLIC_STAGE.json',dict(status='reviewed_staged',scope='all actual fixed three component ablations, complete negative results and offline event-coverage Figure3',
        repository='Zonglin-He/A',branch=BRANCH,base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),
        expected_tree=git('write-tree'),changed_files=changed,files=records,file_count=len(records),
        bytes=sum(r['bytes'] for r in records),preserved_unrelated_untracked=unrelated,portable_audit=audit,
        experiment_complete=False,time=time.time()))
    print(json.dumps(dict(status='ready',file_count=len(records),bytes=sum(r['bytes'] for r in records),audit=audit)))


if __name__=='__main__': run()
