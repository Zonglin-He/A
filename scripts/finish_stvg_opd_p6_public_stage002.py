"""Finish the reviewed P6 public whitelist after its single ignored official .py file."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stvg_opd_p6_existing_common001 import BASE,NS,PYTHON,read,write,sha
from scripts.publish_stvg_opd_p6_existing_root001 import EXPORT,CHECKOUT,BRANCH,git,safe_json


def run():
    binding=read(EXPORT/'CODE_BINDING.json')
    files=sorted(set(binding['files'])|{str(p.relative_to(ROOT)) for p in EXPORT.rglob('*') if p.is_file()})
    current=git('diff','--cached','--name-only').splitlines();assert set(current)<=set(files)
    rec=NS/'recovery/CPU_public_ignored_source_004';rec.mkdir(parents=True,exist_ok=True)
    original=ROOT/'scripts/publish_stvg_opd_p6_existing_root001.py'
    shutil.copy2(original,rec/'publish_stvg_opd_p6_existing_root001_original.py')
    shutil.copy2(EXPORT/'CODE_BINDING.json',rec/'CODE_BINDING_original.json')
    write(rec/'CAPTURE_RECEIPT.json',dict(status='preserved_unsealed_CPU_public_stage',scope='single reviewed official source py ignored by inherited external gitignore; no scientific change',
        original_code_sha256=sha(original),original_code_binding_sha256=sha(EXPORT/'CODE_BINDING.json'),original_staged_tree=git('write-tree'),
        original_staged_count=len(current),original_public_scalar_audit_sha256=sha(BASE/'P6_PUBLIC_SCALAR_AUDIT.json'),
        exact_reviewed_source='external/TA-STVG/models/net_utils.py',source_sha256=sha(ROOT/'external/TA-STVG/models/net_utils.py'),
        recovery='force-stage only this exact reviewed source file; preserve all other package and scientific bytes',
        new_GPU_calls=0,new_GT_read=False,old_prediction_or_runtime_changed=False,time=time.time()))
    for p in rec.iterdir():
        dest=EXPORT/'receipts/P6_existing_temporal_revision001/recovery/CPU_public_ignored_source_004'/p.name
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    helper='scripts/finish_stvg_opd_p6_public_stage002.py';files=sorted(set(files)|{helper}|{str(p.relative_to(ROOT)) for p in EXPORT.rglob('*') if p.is_file()})
    for rel in files:
        p=ROOT/rel;assert p.suffix in {'.py','.json','.md','.png','.pdf'}
        if p.suffix=='.json':safe_json(read(p))
        dest=CHECKOUT/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    subprocess.run(['git','add','--',*[p for p in files if not p.startswith('external/')]],cwd=CHECKOUT,check=True)
    subprocess.run(['git','add','-f','--','external/TA-STVG/models/net_utils.py'],cwd=CHECKOUT,check=True)
    audited=json.loads(subprocess.check_output([str(PYTHON),'-B',str(CHECKOUT/'scripts/audit_stvg_opd_p6_export001.py'),str(CHECKOUT/EXPORT.relative_to(ROOT))],cwd=CHECKOUT,text=True))
    old=read(BASE/'P6_PUBLIC_SCALAR_AUDIT.json');assert old['status']=='pass' and audited['scalar_comparisons']==old['scalar_comparisons']
    changed=git('diff','--cached','--name-only').splitlines();assert changed and set(changed)<=set(files)
    records=[dict(path=f,bytes=(CHECKOUT/f).stat().st_size,sha256=sha(CHECKOUT/f),blob_sha=git('hash-object',f)) for f in files]
    write(BASE/'P6_PUBLIC_STAGE.json',dict(status='reviewed_staged',scope='complete original actual P6 deployment and separately supervised diagnostic evidence',repository='Zonglin-He/A',
        branch=BRANCH,base_commit=git('rev-parse','HEAD'),base_tree=git('rev-parse','HEAD^{tree}'),expected_tree=git('write-tree'),changed_files=changed,
        files=records,file_count=len(records),bytes=sum(v['bytes'] for v in records),portable_scalar_audit=audited,
        unsealed_helper_capture_sha256=sha(rec/'CAPTURE_RECEIPT.json'),paper_suite_complete=False,time=time.time()))
    print(json.dumps(dict(status='ready',files=len(records),bytes=sum(v['bytes'] for v in records),changed_files=len(changed),scalar_comparisons=audited['scalar_comparisons'])))


if __name__=='__main__':run()
