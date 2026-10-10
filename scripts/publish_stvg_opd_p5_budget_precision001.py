"""Explicitly bounded original P5 qualification/resumption anonymous export."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate,BASE,read,write,sha
activate()
from scripts.run_stvg_opd_p5_budget_precision001 import verify,REC
from scripts.stvg_opd_paper_later_common_v1 import phases
EXPORT=ROOT/'results/stvg_opd_p5_budget_launch/2026-10-10'
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
BRANCH='research/stvg-opd-paper-hc2-revision-v2'


def git(*args):return subprocess.check_output(['git',*args],cwd=CHECKOUT,text=True).strip()


def run():
    verify();qualified=read(BASE/'P5_QUALIFICATION.json');prefix=read(REC/'ROOT_RESUME_READBACK.json')
    assert qualified['status']=='pass' and qualified['qualification_arrivals']==20 and qualified['actual_GPU_fits']==40
    assert prefix['status']=='pass' and prefix['GT_read'] is False and prefix['counts']['actual_qualified_formal_pairs_bitwise']==2
    assert read(BASE/'P4_ROOT_CLOSING_RECEIPT.json')['status']=='complete'
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    untracked=git('ls-files','--others','--exclude-standard').splitlines()
    owned=set(read(REC/'REVISION_RUNTIME.json')['pins'])|set(read(REC/'ROOT_RESUME_RUNTIME.json')['pins'])
    owned.update(['scripts/publish_stvg_opd_p5_budget_precision001.py','protocols/stvg_opd_p5_budget_precision001.md'])
    unrelated=[f for f in untracked if f not in owned and not f.startswith(str(EXPORT.relative_to(ROOT))+'/')]
    assert all('stvg_motivation' in f.lower() for f in unrelated)
    EXPORT.mkdir(parents=True,exist_ok=True);mapped={};pins=set()
    def copy(p,dest):
        assert p.is_file() and p.suffix=='.json'
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        mapped[str(p.relative_to(ROOT))]=dict(public_path=str(dest.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))
    for rel in ['P4_ROOT_CLOSING_RECEIPT.json','P4_FINAL_GITHUB_RECEIPT.json','P4_ARCHIVE_VERIFICATION_RECEIPT.json',
        'P4_FINAL_ARCHIVE_SNAPSHOT_RECEIPT.json','P5_STAGE_AUTHORIZATION.json','P5_QUALIFICATION.json',
        'LATER_DESIGN_LOCK.json','COMPONENT_RUNTIME_LOCK_revision001.json']:
        copy(BASE/rel,EXPORT/'receipts'/rel)
    for rel in ['REVISION_RUNTIME.json','CPU_CONTRACTS.json',
        'LAUNCH.json','ROOT_QUALIFICATION_READBACK.json','ROOT_RESUME_RUNTIME.json','ROOT_RESUME_READBACK.json']:
        copy(REC/rel,EXPORT/'receipts/P5_budget_engineering_revision001'/rel)
    for name in phases()['P5']:
        copy(REC/'qualification'/name/'BITWISE_CONTROLS.json',EXPORT/'receipts/qualification'/name/'BITWISE_CONTROLS.json')
    for r in prefix['first_formal_receipts']:
        p=BASE/r['path'];assert sha(p)==r['sha256']
        copy(p,EXPORT/'receipts'/p.relative_to(REC))
    for rel in ['REVISION_RUNTIME.json','ROOT_RESUME_RUNTIME.json']:pins.update(read(REC/rel)['pins'])
    pins.update(['scripts/verify_stvg_opd_public_remote_v1.py','scripts/publish_stvg_opd_p5_budget_precision001.py','protocols/stvg_opd_p5_budget_precision001.md'])
    assert all(p.startswith(('scripts/','vg_tta/','protocols/')) for p in pins)
    write(EXPORT/'CODE_BINDING.json',dict(status='actual_qualification_and_resumption_pending_remote_verification',
        scope='bounded original P5 parameter-scope engineering qualification/resumption; not P5 phase or paper closing',
        pins={f:sha(ROOT/f) for f in sorted(pins)},exact_metadata_projection=mapped,
        qualification_arrivals=20,actual_GPU_fits=40,formal_qualified_pairs_checked=2,
        accepted_prefix_predictions=prefix['counts']['accepted_formal_predictions'],
        qualified_math_dictionaries=20,qualified_rounds=440,qualified_state_coordinates=35840,
        prefix_counts=prefix['counts'],GT_read=False,P5_phase_complete=False,paper_suite_complete=False,
        private_payloads_exported=False,time=time.time()))
    (EXPORT/'README.md').write_text("""# Original P5: actual budget qualification and formal start

The preceding P4 must actually close its full root/view/public/archive obligations.
P5 preserves the original K=1/2/4/8 cross-domain clean suite, both fixed orders and
128 historical parent sources per target, one query per parent. The original two
pre-GT unified-configuration appendix streams stay separate. 2,560 logical arrivals
comprise 2,048 new formal fits and 512 exact original K4 complete-stream aliases.
No qualification prediction is accepted as formal, and no old payload is rewritten.

All ten stages' first two fixed order1 arrivals are qualified against the actually
qualified P4 numerical recorder: 20 arrival pairs, 40 complete real GPU fits.
Root reads all 20 dictionaries, 440 rounds, 35,840 state coordinates, actual input
DINO capture counts, query/Adam reset, LN inheritance/writeback, process source/expert
hashes and native-head-output VJP. This is not a claim that historical failed
absolute guards passed or that an entire decoder Jacobian was independently proved.

First two new formal fits must match qualification before acceptance. A bounded
first prefix readback checks reset/state/writeback/input/Native/bitwise and twice
reads original SHA evidence. It does not close P5 or the paper. All budgets,
directions/orders and unified appendix deploy predictions globally seal before GT;
actual complete source/statistics/math/state/dense/bootstrap/tail/cost/observed-
unobserved/signal/case/view/anonymous publication/remote bytes/archive follows.
No retuning, new experts, algorithm, training, scorer, gate or memory is introduced.
P6 remains an actual later task. EATA and historical paused queues stay paused.
No RGB/query/GT geometry/fit/action/weights/gradient/Adam payload is exported.
""")
    files=sorted(pins|{str(p.relative_to(ROOT)) for p in EXPORT.rglob('*') if p.is_file()})
    prohibited=[b'"caption":',b'"video_path":',b'"native_boxes":',b'"GT_box":',b'"committed":',b'"gradient":',b'"optimizer_state":',b'"raw_logits":',b'"weights":']
    def check_sample_metadata(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k=='samples':assert isinstance(v,int) and v==32,'Only original scalar sample budget allowed'
                check_sample_metadata(v)
        elif isinstance(value,list):
            for v in value:check_sample_metadata(v)
    for rel in files:
        assert Path(rel).suffix in {'.py','.md','.json'}
        data=(ROOT/rel).read_bytes()
        if rel.endswith('.json'):
            for token in prohibited:assert token not in data,(rel,token)
            check_sample_metadata(json.loads(data))
        dest=CHECKOUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,dest)
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True)
    changed=git('diff','--cached','--name-only').splitlines();assert changed and set(changed)<=set(files)
    records=[dict(path=f,bytes=(CHECKOUT/f).stat().st_size,sha256=sha(CHECKOUT/f),blob_sha=git('hash-object',f)) for f in files]
    write(REC/'PUBLIC_STAGE.json',dict(status='reviewed_staged',scope='bounded original P5 qualification/resumption plus actual P4 closing receipts',
        repository='Zonglin-He/A',branch=BRANCH,base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),
        expected_tree=git('write-tree'),changed_files=changed,files=records,file_count=len(records),
        bytes=sum(r['bytes'] for r in records),preserved_unrelated_untracked=unrelated,
        GT_read=False,P5_phase_complete=False,paper_suite_complete=False,time=time.time()))
    print(json.dumps(dict(status='ready',files=len(records),bytes=sum(r['bytes'] for r in records),changed_files=len(changed))))


if __name__=='__main__':run()
