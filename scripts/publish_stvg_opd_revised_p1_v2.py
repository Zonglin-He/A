"""Explicit complete anonymous P1 export; preserve unrelated public staging."""
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate,BASE,PUB,PYTHON,verify,read,write,sha

EXPORT=ROOT/'results/stvg_opd_p1_complete/2026-10-09'
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
BRANCH='research/stvg-opd-paper-hc2-revision-v2'


def git(*args):return subprocess.check_output(['git',*args],cwd=CHECKOUT).decode().strip()


def stage():
    activate();verify()
    assert read(BASE/'P1_ROOT_VISUAL_REVIEW.json')['status']=='pass'
    assert read(BASE/'P1_actual_root_readback/COMPLETION.json')['actual_arrivals']==41355
    assert read(PUB/'TABLE1_ACTUAL_ROOT_MATH_STATE_DENSE_READBACK.json')['status']=='pass'
    assert read(BASE/'P1_actual_root_signal_views/COMPLETION.json')['cases']==6
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert git('branch','--show-current')==BRANCH
    assert not git('diff','--cached','--name-only'),'Preserve and inspect unrelated staged edits'
    before_tracked=set(git('diff','HEAD','--name-only').splitlines())
    before_untracked=set(git('ls-files','--others','--exclude-standard').splitlines())
    EXPORT.mkdir(parents=True,exist_ok=True)
    for file in sorted(PUB.glob('TABLE1*')):
        if file.is_file():shutil.copy2(file,EXPORT/file.name)
    for ds in ['hc2','vidstg']:
        target=EXPORT/('P1_'+ds);target.mkdir(exist_ok=True)
        shutil.copy2(PUB/('P1_'+ds)/'SUMMARY.json',target/'SUMMARY.json')
    # Full matched baseline + OPD rows, including all failures and no-ops.
    from scripts.finalize_stvg_opd_table1_v1 import load_methods
    total=0
    for ds in ['hc2','vidstg']:
        for method,rows in load_methods(ds).items():
            target=EXPORT/'anonymous_rows'/ds/method;target.mkdir(parents=True,exist_ok=True)
            raw=''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows).encode()
            (target/'ROWS.jsonl.gz').write_bytes(gzip.compress(raw,mtime=0));total+=len(rows)
    assert total==258576
    locks=['RUNTIME_LOCK.json','DESIGN_LOCK.json','P1_BASELINE_BINDING.json','P1_PRECISION_RUNTIME.json',
           'P1_PREDICTION_BARRIER.json','P1_GPU_COMPLETION.json','P1_CPU_COMPLETION.json']
    pins=set(read(BASE/'RUNTIME_LOCK.json')['pins'])|set(read(BASE/'P1_PRECISION_RUNTIME.json')['pins'])
    for folder in ['P1_actual_root_readback','P1_actual_root_signal_views']:
        pins.update(read(BASE/folder/'RUNTIME.json')['pins'])
        dest=EXPORT/'receipts'/folder;dest.mkdir(parents=True,exist_ok=True)
        for name in ['RUNTIME.json','LAUNCH.json','COMPLETION.json']:
            shutil.copy2(BASE/folder/name,dest/name)
    for lock in locks:
        dest=EXPORT/'receipts'/lock;dest.parent.mkdir(exist_ok=True)
        shutil.copy2(BASE/lock,dest)
    for recovery in ['recovery/gradient_audit_001','recovery/P1_softmax_precision_002',
                     'recovery/P1_box_chart_003/authorized_revision003','recovery/P1_cuda_memory_004',
                     'recovery/P1_reward_precision_005']:
        runtime=BASE/recovery/'REVISION_RUNTIME.json'
        pins.update(read(runtime)['pins'])
        dest=EXPORT/'audit_revision_pins'/Path(recovery).name;dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(runtime,dest/'REVISION_RUNTIME.json')
    pins.update(['scripts/publish_stvg_opd_revised_p1_v2.py','scripts/audit_stvg_opd_table1_public_v2.py',
                 'docs/STVG_OPD_REVISED_P1_ROOT_REVIEW.md','methods/CURRENT_METHOD.json',
                 'methods/decota_spatial_opd_v1/configs.json'])
    assert all((ROOT/f).is_file() for f in pins)
    # Root runtime pins refer to the same five protocol/barrier metadata files
    # already copied above. Publish their exact bytes in receipts, while retaining
    # the original source path and digest in the binding; never export artifacts
    # recursively or relax the public top-level whitelist.
    metadata_names={'DESIGN_LOCK.json','P1_PREDICTION_BARRIER.json','P1_GPU_COMPLETION.json',
                    'P1_CPU_COMPLETION.json','P1_BASELINE_BINDING.json'}
    metadata_pins={f for f in pins if f.startswith('artifacts/')}
    assert metadata_pins=={str((BASE/name).relative_to(ROOT)) for name in metadata_names}
    metadata_projection={f:str((EXPORT/'receipts'/Path(f).name).relative_to(ROOT)) for f in metadata_pins}
    for src,dest in metadata_projection.items():assert sha(ROOT/src)==sha(ROOT/dest)
    binding=dict(status='pinned_implementation_and_actual_P1_results',
        files={f:sha(ROOT/f) for f in sorted(pins)},anonymous_logical_rows=total,
        metadata_projection=metadata_projection,
        original_scientific_runtime_sha256=sha(BASE/'RUNTIME_LOCK.json'),
        actual_population_readback_sha256=sha(PUB/'TABLE1_ACTUAL_ROOT_POPULATION_READBACK.json'),
        actual_math_state_dense_sha256=sha(PUB/'TABLE1_ACTUAL_ROOT_MATH_STATE_DENSE_READBACK.json'),
        actual_visual_review_sha256=sha(BASE/'P1_ROOT_VISUAL_REVIEW.json'),
        RGB_caption_GT_geometry_weights_predictions_fit_gradient_Adam_exported=False,
        P2_P6_complete=False)
    if (EXPORT/'CODE_BINDING.json').exists():assert read(EXPORT/'CODE_BINDING.json')==binding
    else:write(EXPORT/'CODE_BINDING.json',binding)
    (EXPORT/'README.md').write_text('''# Fixed OPD: complete P1 official-query evidence

Read [the actual root review](TABLE1_ACTUAL_ROOT_REVIEW.md), [all-table results](TABLE1_SUMMARY.json),
[complete parent effects](TABLE1_ALL_PARENT_EFFECTS.json) and [negative attribution](TABLE1_FAILURE_STRATA.json).
All 258,576 matched anonymous baseline/OPD rows are under `anonymous_rows`, including failures and no-ops.
OPD has 41,355 real globally sealed predictions; baselines reuse exactly matched already completed streams.

Recompute every table, paired source-bootstrap, tails, strata, cost and deterministic case selection with Python/NumPy:

```bash
python -B scripts/audit_stvg_opd_table1_public_v2.py results/stvg_opd_p1_complete/2026-10-09
```

The source-only cross-domain protocol, 32-development-parent exclusion, historical exposure,
fixed Native WHEN and final-round readout are documented in the root review. In-domain supervised
reference and HC-only EATA supplement are separately identified; EATA's missing direction remains paused.
VidSTG improvement over DINO Refine is inconclusive. All negative tails and expert-feedback failures remain.

The real root read every saved mathematical/state/dense record, and actually viewed three plot pairs
and six private RGB cases. Public scalar rows reproduce aggregate arithmetic; they cannot independently
replay the private complete decoder Jacobian or model-fit tensors. Raw media/query/GT coordinates,
weights, stochastic actions, predicted boxes, state/gradient/Adam payloads and credentials are excluded.
Numerical/allocator repair originals and thresholds are preserved in the registered revision records.
P1 closure does not finish the authorized paper suite; P2–P6 retain their own real qualification/root/public gates.
''')
    files=sorted((pins-metadata_pins)|{str(f.relative_to(ROOT)) for f in EXPORT.rglob('*') if f.is_file()})
    assert before_tracked<=set(files),('Unrelated tracked edits require review',before_tracked-set(files))
    excluded=before_untracked-set(files)
    # The earlier same-domain Figure v5 interrupted stage is deliberately preserved.
    for f in excluded:
        assert 'stvg_motivation' in f.lower(),(f,'Unknown unrelated untracked asset; preserve before publication')
    forbidden=[b'"caption":',b'"video_path":',b'"native_boxes":',b'"GT_box":',b'"committed":',
               b'"gradient":',b'"optimizer_state":',b'"raw_logits":']
    for rel in files:
        assert rel.startswith(('scripts/','vg_tta/','methods/','protocols/','docs/','results/'))
        assert Path(rel).suffix not in {'.pt','.npz','.npy','.mp4','.sqlite','.safetensors'}
        dest=CHECKOUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
        if rel.endswith(('.json','.jsonl.gz')):
            raw=gzip.decompress(dest.read_bytes()) if rel.endswith('.gz') else dest.read_bytes()
            for token in forbidden:assert token not in raw,(rel,token)
    audited=json.loads(subprocess.check_output([str(PYTHON),'-B',str(CHECKOUT/'scripts/audit_stvg_opd_table1_public_v2.py'),
        str(CHECKOUT/EXPORT.relative_to(ROOT))],cwd=CHECKOUT,text=True))
    assert audited['status']=='pass' and audited['logical_rows']==258576
    write(BASE/'P1_PUBLIC_SCALAR_AUDIT.json',audited)
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True)
    changed=git('diff','--cached','--name-only').splitlines();assert changed and set(changed)<=set(files)
    records=[dict(path=f,bytes=(CHECKOUT/f).stat().st_size,sha256=sha(CHECKOUT/f),blob_sha=git('hash-object',f)) for f in files]
    write(BASE/'P1_PUBLIC_STAGE.json',dict(status='reviewed_staged',repository='Zonglin-He/A',branch=BRANCH,
        base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),expected_tree=git('write-tree'),
        changed_files=changed,files=records,file_count=len(records),bytes=sum(v['bytes'] for v in records),
        preserved_unrelated_untracked=sorted(excluded),portable_scalar_audit=audited,time=time.time()))
    print(json.dumps(dict(status='ready',files=len(records),bytes=sum(v['bytes'] for v in records),
                         changed_files=len(changed),scalar_comparisons=audited['scalar_comparisons'])))


if __name__=='__main__':stage()
