"""Explicitly bounded original P3 qualification/resumption anonymous export."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate,BASE,read,write,sha
activate()
from scripts.run_stvg_opd_p3_scope_precision001 import verify,REC
from scripts.stvg_opd_paper_later_common_v1 import phases
EXPORT=ROOT/'results/stvg_opd_p3_scope_launch/2026-10-10'
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
BRANCH='research/stvg-opd-paper-hc2-revision-v2'


def git(*args):return subprocess.check_output(['git',*args],cwd=CHECKOUT,text=True).strip()


def run():
    verify();qualified=read(BASE/'P3_QUALIFICATION.json');prefix=read(REC/'ROOT_RESUME_READBACK.json')
    assert qualified['status']=='pass' and qualified['qualification_arrivals']==96 and qualified['actual_GPU_fits']==192
    assert prefix['status']=='pass' and prefix['GT_read'] is False and prefix['counts']['actual_qualified_formal_pairs_bitwise']==6
    assert read(BASE/'P2_ROOT_CLOSING_RECEIPT.json')['status']=='complete'
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    untracked=git('ls-files','--others','--exclude-standard').splitlines()
    owned=set(read(REC/'REVISION_RUNTIME.json')['pins'])|set(read(REC/'ROOT_RESUME_RUNTIME.json')['pins'])
    owned.update(['scripts/publish_stvg_opd_p3_scope_precision001.py','protocols/stvg_opd_p3_scope_precision001.md'])
    unrelated=[f for f in untracked if f not in owned and not f.startswith(str(EXPORT.relative_to(ROOT))+'/')]
    assert all('stvg_motivation' in f.lower() for f in unrelated)
    EXPORT.mkdir(parents=True,exist_ok=True);mapped={};pins=set()
    def copy(p,dest):
        assert p.is_file() and p.suffix=='.json'
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        mapped[str(p.relative_to(ROOT))]=dict(public_path=str(dest.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))
    for rel in ['P2_ROOT_CLOSING_RECEIPT.json','P2_FINAL_GITHUB_RECEIPT.json','P2_ARCHIVE_VERIFICATION_RECEIPT.json',
        'P2_FINAL_ARCHIVE_SNAPSHOT_RECEIPT.json','P3_STAGE_AUTHORIZATION.json','P3_QUALIFICATION.json',
        'LATER_DESIGN_LOCK.json','COMPONENT_RUNTIME_LOCK_revision001.json']:
        copy(BASE/rel,EXPORT/'receipts'/rel)
    for rel in ['REVISION_RUNTIME.json','CPU_CONTRACTS.json','CPU_CONTRACTS_revision002.json',
        'LAUNCH.json','ROOT_QUALIFICATION_READBACK.json','ROOT_RESUME_RUNTIME.json','ROOT_RESUME_READBACK.json']:
        copy(REC/rel,EXPORT/'receipts/P3_scope_engineering_revision001'/rel)
    for name in phases()['P3']:
        copy(REC/'qualification'/name/'BITWISE_CONTROLS.json',EXPORT/'receipts/qualification'/name/'BITWISE_CONTROLS.json')
    for folder in ['public_schema_scalar_samples_001','public_stage_immutable_binding_002']:
        copy(REC/'recovery'/folder/'CAPTURE_RECEIPT.json',EXPORT/'receipts/helper_recovery'/folder/'CAPTURE_RECEIPT.json')
    for r in prefix['first_formal_receipts']:
        p=BASE/r['path'];assert sha(p)==r['sha256']
        copy(p,EXPORT/'receipts'/p.relative_to(REC))
    for rel in ['REVISION_RUNTIME.json','ROOT_RESUME_RUNTIME.json']:pins.update(read(REC/rel)['pins'])
    pins.update(['scripts/publish_stvg_opd_p3_scope_precision001.py','protocols/stvg_opd_p3_scope_precision001.md'])
    assert all(p.startswith(('scripts/','vg_tta/','protocols/')) for p in pins)
    write(EXPORT/'CODE_BINDING.json',dict(status='actual_qualification_and_resumption_pending_remote_verification',
        scope='bounded original P3 parameter-scope engineering qualification/resumption; not P3 phase or paper closing',
        pins={f:sha(ROOT/f) for f in sorted(pins)},exact_metadata_projection=mapped,
        qualification_arrivals=96,actual_GPU_fits=192,formal_qualified_pairs_checked=6,
        accepted_prefix_predictions=prefix['counts']['accepted_formal_predictions'],
        qualified_math_dictionaries=96,qualified_rounds=2400,qualified_state_coordinates=172032,
        prefix_counts=prefix['counts'],GT_read=False,P3_phase_complete=False,paper_suite_complete=False,
        private_payloads_exported=False,time=time.time()))
    (EXPORT/'README.md').write_text('''# Original P3: actual scope qualification and first formal resumption

The preceding P2 is actually closed; its final root/public/archive receipts are under
`receipts`. Read its complete negative findings and paired controls at
`results/stvg_opd_p2_complete/2026-10-10/ACTUAL_ROOT_REVIEW.md`.

P3 retains query-only/LN-only/alpha-zero joint/Full, fixed original source/configuration,
independent streams and Native WHEN. The separately pinned CPU audit records the actual
sampled action and checks the declared active optimizer scope and every frozen state.
No GPU loss, sampling, reward, gradient, Adam, frames, steps or parameters are altered.

All 96 original qualification arrivals, comprising 192 actual complete raw/new GPU fits,
passed exact scientific tensor/scalar comparison, process source/expert hashes and
native-head-output VJP checks. Actual CPU root readback checked all 96 math dictionaries,
2,400 rounds and 172,032 state coordinates. Qualification accepted zero formal
predictions, used zero new DINO observations and read no GT.

Before acceptance, the first two fits of each of the three new formal arms must equal
their actual qualified fits. Root independently read an explicit accepted prefix,
including those six full pairs, all saved math/state/reset/writeback/Native/input
identities and immutable bytes. Its count is a snapshot, not a complete P3 result.

Full only aliases exact complete P0/P2 source/input/config/history streams. All 7,168
logical P3 rows must globally seal before GT/scoring, then actual root/view/public/archive
closing precedes P4. This publication closes only bounded qualification/resumption;
P3–P6 and the paper remain unfinished. Complete decoder Jacobians and future numerical
or memory safety are not proven. All old failures and original locks remain preserved.
Raw media/query/GT/box/action/fit/weights/gradient/Adam arrays and secrets are excluded.
EATA and historical paused queues remain paused.
''')
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
    write(REC/'PUBLIC_STAGE.json',dict(status='reviewed_staged',scope='bounded original P3 qualification/resumption plus actual P2 closing receipts',
        repository='Zonglin-He/A',branch=BRANCH,base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),
        expected_tree=git('write-tree'),changed_files=changed,files=records,file_count=len(records),
        bytes=sum(r['bytes'] for r in records),preserved_unrelated_untracked=unrelated,
        GT_read=False,P3_phase_complete=False,paper_suite_complete=False,time=time.time()))
    print(json.dumps(dict(status='ready',files=len(records),bytes=sum(r['bytes'] for r in records),changed_files=len(changed))))


if __name__=='__main__':run()
