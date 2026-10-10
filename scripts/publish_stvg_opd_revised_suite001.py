"""Explicit final original suite public companion; closure requires remote verification."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_p6_existing_common001 import BASE,NS,PYTHON,read,write,sha
from scripts.publish_stvg_opd_p6_existing_root001 import CHECKOUT,BRANCH,git,safe_json
from scripts.assemble_stvg_opd_revised_suite001 import DEST


def run():
    assert read(BASE/'PAPER_SUITE_ROOT_ASSEMBLY.json')['phase_count']==6
    assert read(BASE/'P6_FINAL_ARCHIVE_SNAPSHOT_RECEIPT.json')['status']=='pass'
    subprocess.run(['git','fetch','--quiet','origin','main',BRANCH],cwd=CHECKOUT,check=True)
    assert git('rev-parse','HEAD')==git('rev-parse','origin/main')==git('rev-parse','origin/'+BRANCH)
    assert not git('diff','HEAD','--name-only') and not git('diff','--cached','--name-only')
    other=git('ls-files','--others','--exclude-standard').splitlines();assert all('stvg_motivation' in f.lower() for f in other)
    for source in (BASE/'PAPER_SUITE_ROOT/recovery').rglob('*'):
        if source.is_file():
            assert source.suffix in {'.py','.json'}
            target=DEST/'receipts/PAPER_SUITE_ROOT/recovery'/source.relative_to(BASE/'PAPER_SUITE_ROOT/recovery');target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    shutil.copy2(BASE/'PAPER_SUITE_ROOT_ASSEMBLY.json',DEST/'receipts/PAPER_SUITE_ROOT_ASSEMBLY.json')
    owned=['scripts/assemble_stvg_opd_revised_suite001.py','scripts/audit_stvg_opd_suite_export001.py','scripts/publish_stvg_opd_revised_suite001.py',
        'scripts/verify_stvg_opd_public_remote_v1.py','docs/STVG_OPD_REVISED_SUITE_ROOT_REVIEW.md']
    (DEST/'README.md').write_text('''# Original fixed P1–P6 experimental suite

Read [the full result index](ACTUAL_SUITE_REVIEW.md), the six unabridged phase
reports and [the actual closure inventory](PHASE_CLOSURE_INVENTORY.json).
All positive and negative results, source cohorts, precision repairs, formal
global-seal GT barriers, true root views and original public byte verification
receipts remain intact. Code/data/complete anonymous results stay at each linked
immutable original phase commit. No new model/scorer/GT/fit is run by this index.

```bash
python -B scripts/audit_stvg_opd_suite_export001.py results/stvg_opd_paper_suite_complete/2026-10-10
```

This companion verifies original phase closing contracts and anonymous public
byte bindings; it does not independently repeat every private mathematical or
inference audit. All historical counts/aliases and different units stay separate.
Negative Direct/DINO/ablation/budget/temporal contrasts remain. Original method,
scientific settings and rosters are unchanged. EATA and old queues stay paused.
Private media/query/caption/GT geometry/weights/logits/actions/Adam/fit/cache
assets are excluded. Final suite close requires fresh remote verification of
this companion, final archive maintenance and a true FINAL_COMPLETION receipt.
''')
    write(DEST/'CODE_BINDING.json',dict(status='actual_six_phase_root_companion_pending_final_remote_and_archive',
        source_files={f:sha(ROOT/f) for f in owned},public_files={str(p.relative_to(DEST)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in DEST.rglob('*') if p.is_file()},
        actual_root_assembly_sha256=sha(BASE/'PAPER_SUITE_ROOT_ASSEMBLY.json'),private_payloads_exported=False,paper_suite_complete=False))
    files=sorted(set(owned)|{str(p.relative_to(ROOT)) for p in DEST.rglob('*') if p.is_file()})
    for rel in files:
        p=ROOT/rel;assert p.suffix in {'.py','.md','.json'}
        if p.suffix=='.json':safe_json(read(p))
        q=CHECKOUT/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    audited=json.loads(subprocess.check_output([str(PYTHON),'-B',str(CHECKOUT/'scripts/audit_stvg_opd_suite_export001.py'),str(CHECKOUT/DEST.relative_to(ROOT))],cwd=CHECKOUT,text=True));assert audited['status']=='pass'
    write(BASE/'PAPER_SUITE_PUBLIC_AUDIT.json',dict(audited,actual_public_checkout=True,time=time.time()))
    subprocess.run(['git','add','--',*files],cwd=CHECKOUT,check=True)
    changed=git('diff','--cached','--name-only').splitlines();assert changed and set(changed)<=set(files)
    records=[dict(path=f,bytes=(CHECKOUT/f).stat().st_size,sha256=sha(CHECKOUT/f),blob_sha=git('hash-object',f)) for f in files]
    write(BASE/'PAPER_SUITE_PUBLIC_STAGE.json',dict(status='reviewed_staged',scope='original P1-P6 actual phase closing and complete all-negative result companion',repository='Zonglin-He/A',branch=BRANCH,
        base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),expected_tree=git('write-tree'),changed_files=changed,files=records,
        file_count=len(records),bytes=sum(f['bytes'] for f in records),portable_audit=audited,paper_suite_complete=False,time=time.time()))
    print(json.dumps(dict(status='ready',files=len(records),bytes=sum(f['bytes'] for f in records),comparisons=audited['comparisons'])))


if __name__=='__main__':run()
