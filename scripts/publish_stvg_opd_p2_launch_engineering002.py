"""Explicit anonymous P1 closing / real P2 qualification-and-launch export."""
import json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_stvg_opd_p2_engineering002 import REC,verify_revision
from scripts.stvg_opd_paper_hc2_revision_common_v2 import BASE,read,write,sha
OUT=ROOT/'results/stvg_opd_p2_launch/2026-10-09'
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
BRANCH='research/stvg-opd-paper-hc2-revision-v2'


def git(*a):return subprocess.check_output(['git',*a],cwd=CHECKOUT,text=True).strip()


def run():
    verify_revision();q=read(BASE/'P2_QUALIFICATION.json')
    assert q['status']=='pass' and q['actual_GPU_fits']==192 and not q['GT_read']
    assert read(REC/'ENGINEERING_ROOT_QUALIFICATION.json')['status']=='pass'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    unrelated=git('ls-files','--others','--exclude-standard').splitlines()
    assert all('stvg_motivation' in f.lower() for f in unrelated)
    pins=set()
    for path in [BASE/'COMPONENT_RUNTIME_LOCK.json',BASE/'COMPONENT_RUNTIME_LOCK_revision001.json',
                 BASE/'LATER_CPU_RUNTIME_LOCK.json',REC/'REVISION_RUNTIME.json']:
        pins.update(read(path)['pins'])
    pins.add('scripts/publish_stvg_opd_p2_launch_engineering002.py')
    mapped={}
    def copy(source,dest):
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        mapped[str(source.relative_to(ROOT))]=dict(public_path=str(dest.relative_to(ROOT)),sha256=sha(source),bytes=source.stat().st_size)
    for name in ['LATER_DESIGN_LOCK.json','APPENDIX_STAGE_LOCK.json','COMPONENT_RUNTIME_LOCK.json',
                 'COMPONENT_RUNTIME_LOCK_revision001.json','LATER_CPU_RUNTIME_LOCK.json','P1_STAGE_AUTHORIZATION.json',
                 'P2_STAGE_AUTHORIZATION.json','P2_QUALIFICATION.json']:
        copy(BASE/name,OUT/'receipts'/name)
    for name in ['REVISION_RUNTIME.json','CPU_STREAM_CONTRACTS.json','CPU_SAVED_RECEIPT_CONTRACTS.json',
                 'LAUNCH.json','ENGINEERING_ROOT_QUALIFICATION.json','ACTUAL_LAUNCH_ARCHIVE_VERIFICATION_RECEIPT.json']:
        copy(REC/name,OUT/'receipts'/name)
    for r in q['stages']:
        folder=REC/'qualification'/r['stage']
        for name in ['BITWISE_CONTROLS.json','ROOT_READBACK.json']:
            copy(folder/name,OUT/'qualification'/r['stage']/name)
    p1=ROOT/'results/stvg_opd_p1_complete/2026-10-09/receipts'
    for name in ['P1_ROOT_CLOSING_RECEIPT.json','P1_ARCHIVE_VERIFICATION_RECEIPT.json','P1_FINAL_GITHUB_RECEIPT.json']:
        copy(BASE/name,p1/name)
    write(OUT/'CODE_BINDING.json',dict(status='actual_P2_qualification_and_launch_not_P2_efficacy_or_paper_closure',
        code={f:sha(ROOT/f) for f in sorted(pins)},exact_metadata_projection=mapped,
        actual_GPU_qualification_fits=192,qualification_arrivals=96,GT_read=False,
        original_science_unchanged=True,formal_P2_global_seal=False,paper_suite_complete=False))
    (OUT/'README.md').write_text('''# Original fixed P2: real qualification and continuation

P1 actually completed all root/public/archive duties. Its full anonymous rows,
independent public audit and all positive/negative results are in
`results/stvg_opd_p1_complete/2026-10-09`. The additional exact closing receipts
record this gate; P1 completion is not completion of the paper.

P2 retains the original Direct L1/GIoU, Shuffled Feedback, Fixed Rollout and Full
OPD controls on both locked 128-parent panels, cross-domain clean two orders and
the five same-domain physical 5% burst conditions. Native WHEN, original expert,
configuration, source reset, inherited LN and last-round output remain fixed.
The joint-arm chart and precision checks reuse the authorized P1 numerical
repairs in separately pinned runtime; Direct retains its original fitter/audit.

Actual qualification covered all four stages, arms and physical conditions:
96 locked qualification arrivals, 192 actual complete old/new GPU fits with
all stored tensor/state/action/gradient/Adam/readout coordinates bitwise equal.
Every saved qualification fit had an actual CPU root input/math/reset/inherited
state/writeback/chart readback. Qualification predicts no formal result and
is never inserted into the formal stream. The frozen source and expert hash
checks passed. Native head VJP checks do not prove a full decoder Jacobian.

The finite controller actually launched formal P2 after this gate. No P2 score
or benefit is reported here. All original deployment arms and both directions
must seal globally before GT; full root statistics/math/dense/state/cost/cases,
actual plot review, verified anonymous publication and archive closing precede
P3. EATA and old paused queues remain paused. The current allocator is not a
guarantee against every future OOM. Private media, query/caption, GT geometry,
weights, box/action/fit/gradient/Adam arrays are excluded.
''')
    files=sorted(pins|{str(f.relative_to(ROOT)) for f in OUT.rglob('*') if f.is_file()}|
        {str((p1/name).relative_to(ROOT)) for name in ['P1_ROOT_CLOSING_RECEIPT.json','P1_ARCHIVE_VERIFICATION_RECEIPT.json','P1_FINAL_GITHUB_RECEIPT.json']})
    forbidden=[b'"caption":',b'"video_path":',b'"native_boxes":',b'"GT_box":',b'"committed":',b'"gradient":',b'"optimizer_state":',b'"raw_logits":']
    for rel in files:
        assert rel.startswith(('scripts/','vg_tta/','protocols/','results/'))
        assert Path(rel).suffix in {'.py','.md','.json'}
        if rel.endswith('.json'):
            raw=(ROOT/rel).read_bytes()
            for token in forbidden:assert token not in raw,(rel,token)
        dest=CHECKOUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True)
    changed=git('diff','--cached','--name-only').splitlines();assert changed and set(changed)<=set(files)
    records=[dict(path=f,sha256=sha(CHECKOUT/f),bytes=(CHECKOUT/f).stat().st_size,blob_sha=git('hash-object',f)) for f in files]
    write(REC/'PUBLIC_STAGE.json',dict(status='reviewed_staged',scope='actual P1 closing and actual P2 qualification/launch only',
        repository='Zonglin-He/A',branch=BRANCH,base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),
        expected_tree=git('write-tree'),files=records,changed_files=changed,file_count=len(records),bytes=sum(v['bytes'] for v in records),
        preserved_unrelated_untracked=unrelated,GT_read=False,paper_suite_complete=False,time=time.time()))
    print(json.dumps(dict(status='ready',files=len(records),changed=len(changed),bytes=sum(v['bytes'] for v in records))))


if __name__=='__main__':run()
