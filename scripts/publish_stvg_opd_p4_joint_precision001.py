"""Explicitly bounded original P4 qualification/resumption anonymous export."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_paper_hc2_revision_common_v2 import activate,BASE,read,write,sha
activate()
from scripts.run_stvg_opd_p4_joint_precision001 import verify,REC
from scripts.stvg_opd_paper_later_common_v1 import phases
EXPORT=ROOT/'results/stvg_opd_p4_joint_launch/2026-10-10'
CHECKOUT=ROOT.parent/'visual-grounding-public-A'
BRANCH='research/stvg-opd-paper-hc2-revision-v2'


def git(*args):return subprocess.check_output(['git',*args],cwd=CHECKOUT,text=True).strip()


def run():
    verify();qualified=read(BASE/'P4_QUALIFICATION.json');prefix=read(REC/'ROOT_RESUME_READBACK.json')
    assert qualified['status']=='pass' and qualified['qualification_arrivals']==64 and qualified['actual_GPU_fits']==128
    assert prefix['status']=='pass' and prefix['GT_read'] is False and prefix['counts']['actual_qualified_formal_pairs_bitwise']==2
    assert read(BASE/'P3_ROOT_CLOSING_RECEIPT.json')['status']=='complete'
    assert git('remote','get-url','origin')=='https://github.com/Zonglin-He/A.git'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    untracked=git('ls-files','--others','--exclude-standard').splitlines()
    owned=set(read(REC/'REVISION_RUNTIME.json')['pins'])|set(read(REC/'ROOT_RESUME_RUNTIME.json')['pins'])
    owned.update(['scripts/publish_stvg_opd_p4_joint_precision001.py','protocols/stvg_opd_p4_joint_precision001.md'])
    unrelated=[f for f in untracked if f not in owned and not f.startswith(str(EXPORT.relative_to(ROOT))+'/')]
    assert all('stvg_motivation' in f.lower() for f in unrelated)
    EXPORT.mkdir(parents=True,exist_ok=True);mapped={};pins=set()
    def copy(p,dest):
        assert p.is_file() and p.suffix=='.json'
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        mapped[str(p.relative_to(ROOT))]=dict(public_path=str(dest.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))
    for rel in ['P3_ROOT_CLOSING_RECEIPT.json','P3_FINAL_GITHUB_RECEIPT.json','P3_ARCHIVE_VERIFICATION_RECEIPT.json',
        'P3_FINAL_ARCHIVE_SNAPSHOT_RECEIPT.json','P4_STAGE_AUTHORIZATION.json','P4_QUALIFICATION.json',
        'LATER_DESIGN_LOCK.json','COMPONENT_RUNTIME_LOCK_revision001.json']:
        copy(BASE/rel,EXPORT/'receipts'/rel)
    for rel in ['REVISION_RUNTIME.json','CPU_CONTRACTS.json',
        'LAUNCH.json','ROOT_QUALIFICATION_READBACK.json','ROOT_RESUME_RUNTIME.json','ROOT_RESUME_READBACK.json']:
        copy(REC/rel,EXPORT/'receipts/P4_joint_engineering_revision001'/rel)
    for name in phases()['P4']:
        copy(REC/'qualification'/name/'BITWISE_CONTROLS.json',EXPORT/'receipts/qualification'/name/'BITWISE_CONTROLS.json')
    for r in prefix['first_formal_receipts']:
        p=BASE/r['path'];assert sha(p)==r['sha256']
        copy(p,EXPORT/'receipts'/p.relative_to(REC))
    for rel in ['REVISION_RUNTIME.json','ROOT_RESUME_RUNTIME.json']:pins.update(read(REC/rel)['pins'])
    pins.update(['scripts/verify_stvg_opd_public_remote_v1.py','scripts/publish_stvg_opd_p4_joint_precision001.py','protocols/stvg_opd_p4_joint_precision001.md'])
    assert all(p.startswith(('scripts/','vg_tta/','protocols/')) for p in pins)
    write(EXPORT/'CODE_BINDING.json',dict(status='actual_qualification_and_resumption_pending_remote_verification',
        scope='bounded original P4 parameter-scope engineering qualification/resumption; not P4 phase or paper closing',
        pins={f:sha(ROOT/f) for f in sorted(pins)},exact_metadata_projection=mapped,
        qualification_arrivals=64,actual_GPU_fits=128,formal_qualified_pairs_checked=2,
        accepted_prefix_predictions=prefix['counts']['accepted_formal_predictions'],
        qualified_math_dictionaries=64,qualified_rounds=1600,qualified_state_coordinates=114688,
        prefix_counts=prefix['counts'],GT_read=False,P4_phase_complete=False,paper_suite_complete=False,
        private_payloads_exported=False,time=time.time()))
    (EXPORT/'README.md').write_text('''# Original P4: actual robustness qualification and first formal resumption

The preceding P3 is actually closed; its full negative results remain under
`results/stvg_opd_p3_complete/2026-10-10`. P4 follows the original same-domain clean plus
five physical burst families at 2.5/5/10%, independent source reset per condition.
HC2 has 237 parent movies, one clip/query each; VidSTG 732 parents, one query each:
969 x 16 = 15,504 formal joint fits. Native WHEN, Uniform4/admitted Top1 DINO, source
weights, frames, per-query Adam/residual reset, configured LN writeback and final readout
are unchanged. No training, new expert, retuning or new algorithm is introduced.

Each condition/source's first two fixed arrivals are qualified with two real complete
fits: the already qualified P3 numerical recorder and the separately pinned P4 recorder.
All scientific tensors/scalars match, and actual process source/expert hashes and
native-head VJP are checked. This is 64 qualification arrivals and 128 real GPU fits.
The old numerical control is the qualified P3 hook, not a claim that historical failed
original absolute precision guards now pass. Root checks all 64 fit dictionaries,
1600 stored rounds and 114688 full-state coordinates; qualification accepts zero formal
predictions and reads no GT. Sparse DINO capture is counted from actual input receipts.

First formal fits must match the qualification before acceptance; bounded root prefix
readback checks the first two pairs and a snapshot of full state/reset/writeback/Native/
input hashes twice. This does not close all P4 or the paper. All 15,504 deployment outputs
must globally seal before GT/CPU scoring, then actual root/view/anonymous full negative
publication/archive closing precedes P5. Full decoder Jacobians and future numerical
or memory safety are not proven. All previous locks/failures remain immutable.
Raw RGB/query/GT geometry/fit/weights/action/gradient/Adam payloads are excluded.
EATA and historical paused queues remain paused; monitoring stays ACTIVE.
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
    write(REC/'PUBLIC_STAGE.json',dict(status='reviewed_staged',scope='bounded original P4 qualification/resumption plus actual P3 closing receipts',
        repository='Zonglin-He/A',branch=BRANCH,base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),
        expected_tree=git('write-tree'),changed_files=changed,files=records,file_count=len(records),
        bytes=sum(r['bytes'] for r in records),preserved_unrelated_untracked=unrelated,
        GT_read=False,P4_phase_complete=False,paper_suite_complete=False,time=time.time()))
    print(json.dumps(dict(status='ready',files=len(records),bytes=sum(r['bytes'] for r in records),changed_files=len(changed))))


if __name__=='__main__':run()
